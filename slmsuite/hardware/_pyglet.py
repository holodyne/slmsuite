"""
Hidden abstract classes for pyglet windowing in slmsuite.

Provides :class:`_Window` (a :mod:`pyglet` window subclass for SLM display),
:class:`_ViewerWindow` (a resizable :class:`_Window` for :meth:`.live`),
:class:`_WindowThread` (a dedicated thread per window for event dispatch), and
:class:`_WindowManager` (a singleton coordinating all window threads).

All :mod:`pyglet` and ``OpenGL`` calls are executed on the window's dedicated
thread to satisfy OS thread-affinity requirements (especially Win32, where a
window's message queue is bound to the thread that created it). This prevents
windows from freezing between :meth:`~slmsuite.hardware.slms.slm.SLM.set_phase`
calls.
"""
import contextlib
import os
import sys
import time
import ctypes
import threading
import queue
import atexit
import numpy as np
from packaging.version import Version

try:
    import pyglet
    import pyglet.gl as gl
    from pyglet.window import key, mouse
    from pyglet.window import Window as __Window

    # Helper to get display/canvas depending on pyglet version
    PYGLET_VERSION = Version(getattr(pyglet, '__version__', '0'))

    def get_pyglet_display():
        """
        Get the :mod:`pyglet` display object, which handles OS-dependent display management.

        Returns
        -------
        pyglet.display.Display or pyglet.canvas.Display
            The platform display object.
        """
        if PYGLET_VERSION >= Version('2.1.0'):
            return pyglet.display.get_display()
        else:
            return pyglet.canvas.get_display()
except Exception:
    pyglet = None
    gl = None
    key = mouse = None
    __Window = object
    PYGLET_VERSION = None
    def get_pyglet_display():
        raise ImportError("pyglet not installed.")

# Optional cupy, for GPU frames and page-locked host memory.
try:
    import cupy as cp
    from cupyx import zeros_pinned
except ImportError:
    cp = None
    zeros_pinned = None

from slmsuite._logging import make_logger
from slmsuite import __version__ as SLMSUITE_VERSION

logger = make_logger(__name__)

# Window creation clears the process-global gl.current_context to build an unshared
# context, and switch_to() restores it: both take this lock so that neither undoes the
# other. Reentrant, since pyglet calls switch_to() from inside window creation.
_creation_lock = threading.RLock()

# CUDA driver API, for OpenGL interop; cupy exposes no interop bindings of its own.
try:
    _cuda = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
    _cuda.cuGraphicsGLRegisterBuffer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_uint
    ]
    _cuda.cuGraphicsResourceGetMappedPointer_v2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p
    ]
    for _name in ("cuGraphicsMapResources", "cuGraphicsUnmapResources"):
        getattr(_cuda, _name).argtypes = [
            ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
        ]
    _cuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    _cuda.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
    _cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    _cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    _cuda.cuGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
    _cuda.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
except Exception:
    _cuda = None


def _cu(name, *args):
    """Call a CUDA driver function, raising with the resolved status name on failure."""
    result = getattr(_cuda, name)(*args)
    if result:
        message = ctypes.c_char_p()
        _cuda.cuGetErrorName(result, ctypes.byref(message))
        raise RuntimeError("{} failed: {}".format(name, message.value))


def _stream():
    """The stream that :mod:`cupy` is currently queueing work onto."""
    return ctypes.c_void_p(cp.cuda.get_current_stream().ptr)


class _PixelBuffer(object):
    """
    An ``OpenGL`` pixel buffer registered with ``CUDA``, writable as a :mod:`cupy` array.

    Lets phase data reach the display without ever crossing PCIe: :meth:`map` hands back a
    :mod:`cupy` view of the buffer's device memory, and :meth:`~_Window.render` uploads it to
    the texture entirely on the GPU.

    Important
    ~~~~~~~~~
    Construction and :meth:`release` require the ``OpenGL`` context, so they must run on the
    window thread. :meth:`map` and :meth:`unmap` must instead run on the thread that writes
    the data, which must hold the ``CUDA`` primary context.
    """

    def __init__(self, shape, device):
        """Allocate a pixel buffer of ``shape`` on ``device`` and register it with ``CUDA``."""
        if _cuda is None or cp is None:
            raise RuntimeError("CUDA driver or cupy unavailable.")
        if device is None:
            raise RuntimeError("No CUDA device to register against.")
        if not gl.base.gl_info.have_version(3, 0):
            raise RuntimeError("Pixel buffers require OpenGL 3.0+.")

        self.shape = shape
        self.buffer = gl.GLuint()
        self.device = device
        self.context = None
        self.resource = None
        self.mapped = False

        try:
            # Registering needs a context on this thread; the primary one is what cupy uses.
            context = ctypes.c_void_p()
            _cu("cuDevicePrimaryCtxRetain", ctypes.byref(context), self.device)
            self.context = context
            _cu("cuCtxSetCurrent", context)

            gl.glGenBuffers(1, ctypes.byref(self.buffer))
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.buffer.value)
            gl.glBufferData(
                gl.GL_PIXEL_UNPACK_BUFFER, int(np.prod(shape)), None, gl.GL_STREAM_DRAW
            )
            gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

            resource = ctypes.c_void_p()
            _cu("cuGraphicsGLRegisterBuffer", ctypes.byref(resource), self.buffer.value, 0)
            self.resource = resource

            # Blank the undefined buffer, on the writer's device, so alpha starts opaque.
            with cp.cuda.Device(self.device):
                frame = self.map()
                frame.fill(0)
                frame[:, :, 3] = 255
                self.unmap()
        except BaseException:
            self.release()
            raise

    def map(self):
        """Map the buffer into ``CUDA`` and return a :mod:`cupy` view of it."""
        device = cp.cuda.runtime.getDevice()
        if device != self.device:
            raise RuntimeError("Frame is registered to device {}, but cupy is on device {}."
                .format(self.device, device))

        # Registering happened on the window thread, so this one may hold no context yet.
        current = ctypes.c_void_p()
        _cu("cuCtxGetCurrent", ctypes.byref(current))
        if not current.value:
            _cu("cuCtxSetCurrent", self.context)

        # Mapping synchronizes against one stream, which must be the one the writes go to.
        _cu("cuGraphicsMapResources", 1, ctypes.byref(self.resource), _stream())
        self.mapped = True

        try:
            pointer = ctypes.c_void_p()
            size = ctypes.c_size_t()
            _cu(
                "cuGraphicsResourceGetMappedPointer_v2",
                ctypes.byref(pointer), ctypes.byref(size), self.resource
            )
        except BaseException:
            # Else the buffer stays mapped and every later frame fails.
            self.unmap()
            raise

        # The device pointer is not guaranteed to survive a remap, so rewrap every time.
        memory = cp.cuda.UnownedMemory(pointer.value, size.value, self)
        return cp.ndarray(self.shape, cp.uint8, cp.cuda.MemoryPointer(memory, 0))

    def unmap(self):
        """Release the buffer back to ``OpenGL``. Idempotent."""
        if not self.mapped:
            return
        _cu("cuGraphicsUnmapResources", 1, ctypes.byref(self.resource), _stream())
        self.mapped = False

    def release(self):
        """Unregister and delete the buffer. Must run on the window thread."""
        # Separate guards, so a failure in one step still frees the others.
        if self.resource is not None:
            try:
                # Unregistering under a live mapping frees memory cupy still points at.
                self.unmap()
            except Exception as e:
                logger.debug("Unmapping the interop frame failed: %s", e)
            try:
                _cu("cuGraphicsUnregisterResource", self.resource)
            except Exception as e:
                logger.debug("Unregistering the interop frame failed: %s", e)
        if self.buffer.value:
            try:
                gl.glDeleteBuffers(1, ctypes.byref(self.buffer))
            except Exception as e:
                logger.debug("Deleting the interop buffer failed: %s", e)
        if self.context is not None:
            try:
                _cu("cuDevicePrimaryCtxRelease", self.device)
            except Exception as e:
                logger.debug("Releasing the CUDA context failed: %s", e)

        self.resource, self.context, self.mapped = None, None, False
        self.buffer = gl.GLuint()


# Win32 ``EnumDisplayDevicesW`` flag requesting the device interface path.
_EDD_GET_DEVICE_INTERFACE_NAME = 0x00000001

# Class for Windows-only display information.
class _DISPLAY_DEVICEW(ctypes.Structure):
    """Win32 ``DISPLAY_DEVICEW``, used by :func:`_screen_id`."""
    _fields_ = [
        ("cb",              ctypes.c_ulong),
        ("DeviceName",      ctypes.c_wchar * 32),
        ("DeviceString",    ctypes.c_wchar * 128),
        ("StateFlags",      ctypes.c_ulong),
        ("DeviceID",        ctypes.c_wchar * 128),
        ("DeviceKey",       ctypes.c_wchar * 128),
    ]

def _screen_id(screen):
    """
    Get a stable, OS-level identifier for a monitor.

    Unlike a screen's index or geometry, this does not change when other
    monitors are attached or when the desktop is rearranged: on Windows it is
    the monitor's device interface path, which is tied to the physical display
    output. This is used to keep track of which display belongs to which SLM
    across display hotplugs (see
    :meth:`~slmsuite.hardware.slms.texasinstruments.PLM.open_all`).

    Parameters
    ----------
    screen : pyglet screen object
        Screen to identify.

    Returns
    -------
    str
        Stable identifier, e.g. ``"DISPLAY#DLP03C9#5&4c0ed3&1&UID4353"``. Falls
        back to the screen geometry on platforms where no such identifier is
        available.
    """
    device_name = getattr(screen, "_device_name", None)

    if sys.platform == "win32" and device_name is not None:
        try:
            device = _DISPLAY_DEVICEW()
            device.cb = ctypes.sizeof(device)
            if ctypes.windll.user32.EnumDisplayDevicesW(
                device_name,
                0,
                ctypes.byref(device),
                _EDD_GET_DEVICE_INTERFACE_NAME):
                # e.g. '\\?\DISPLAY#DLP03C9#5&4c0ed3&1&UID4353#{e6f07b5f-ee97-...}'
                # Drop interface GUID.
                device_id = device.DeviceID.split("#{")[0]      
                if device_id.startswith("\\\\?\\"):
                    # Drop interface prefix.
                    device_id = device_id[4:]                   
                if device_id:
                    return device_id
        except Exception as e:
            logger.debug("Could not resolve monitor identifier: %s", e)

    return "{}x{}+{}+{}".format(screen.width, screen.height,
                                screen.x, screen.y)

def _screen_ids():
    """
    Get the identifiers of all currently-attached screens.

    Returns
    -------
    set of str
        See :func:`_screen_id`.
    """
    return {_screen_id(screen) for screen in get_pyglet_display().get_screens()}

def _screen_index(screen_id):
    """
    Find the index of the screen with the given :func:`_screen_id`.

    Parameters
    ----------
    screen_id : str
        Identifier previously returned by :func:`_screen_id`.

    Returns
    -------
    int OR None
        The current index of the matching screen, or ``None`` if it is not attached.
    """
    for index, screen in enumerate(get_pyglet_display().get_screens()):
        if _screen_id(screen) == screen_id:
            return index
    return None

def _wait_for_new_screen(known_ids, timeout_s=60, interval_s=1):
    """
    Poll until a screen outside ``known_ids`` is attached.

    Parameters
    ----------
    known_ids : set of str
        Identifiers of the screens that were already attached, from :func:`_screen_ids`.
    timeout_s, interval_s : float
        Maximum time to wait, and sleep interval between polls, in seconds. Displays
        can take a surprisingly long and variable time to enumerate after a hotplug,
        so ``timeout_s`` is generous by default.

    Returns
    -------
    str OR None
        Identifier of the new screen, or ``None`` if none appeared in time.

    Raises
    ------
    RuntimeError
        If several screens appear at once, as they cannot then be told apart.
    """
    deadline = time.time() + timeout_s
    candidate = None

    while time.time() < deadline:
        time.sleep(interval_s)
        new_ids = _screen_ids() - known_ids

        if len(new_ids) > 1:
            raise RuntimeError(
                "Several displays appeared at once ({}); cannot tell them apart."
                .format(sorted(new_ids))
            )
        elif len(new_ids) == 1:
            # A monitor's identifier changes while the OS finishes enumerating
            # it (:func:`_screen_id` falls back to geometry until the monitor's
            # device interface exists), so only accept one that holds across
            # two polls.
            new_id = new_ids.pop()
            if new_id == candidate:
                return new_id
            candidate = new_id
        else:
            candidate = None

    return None


def _wait_for_screens_settled(timeout_s=20, settle_s=2, interval_s=0.5):
    """
    Poll until the set of attached screens has stopped changing.

    Useful after detaching displays, which the OS does not do instantaneously.

    Parameters
    ----------
    timeout_s, settle_s, interval_s : float
        Maximum time to wait, time the screen set must be unchanged for, and sleep
        interval between polls, in seconds.
    """
    deadline = time.time() + timeout_s
    screen_ids = _screen_ids()
    settled = time.time()

    while time.time() < deadline:
        time.sleep(interval_s)
        current_ids = _screen_ids()

        if current_ids != screen_ids:
            screen_ids = current_ids
            settled = time.time()
        elif time.time() - settled >= settle_s:
            return

    logger.warning("Displays did not settle within %s seconds.", timeout_s)


class _Window(__Window):
    """
    A :mod:`pyglet` window subclass for displaying SLM phase patterns.

    Wraps a fullscreen (or windowed) ``OpenGL`` surface with a texture-based
    rendering pipeline. Phase data is written into an RGBA :attr:`frame`,
    uploaded to an ``OpenGL`` texture via ``glTexSubImage2D``, and displayed
    via double-buffered vsync'd flips.

    The frame is claimed with :meth:`acquire`, filled, and handed over with :meth:`commit`
    before :meth:`render` displays it. Since there is only one, the writer must let a
    :meth:`render` finish before starting the next frame.

    Supports both ``OpenGL`` 3.0+ (programmable shader pipeline, pyglet 2.0+)
    and ``OpenGL`` 2.0 (fixed-function pipeline, pyglet < 2.0).

    Important
    ~~~~~~~~~
    All methods on this class except :meth:`info`, :meth:`acquire` and :meth:`commit` must
    be called from the thread that created the window. Use :class:`_WindowThread` to ensure
    thread affinity.

    Attributes
    ----------
    shape : (int, int)
        The ``(height, width)`` of the window in pixels.
    mode : {"interop", "pinned", "pageable"}
        Which frame storage :meth:`_setup_frame` settled on.
    frame : numpy.ndarray or None
        Host RGBA frame of shape ``(height, width, 4)`` and dtype ``uint8``.
        ``None`` in ``"interop"`` mode.
    cframe : ctypes array or None
        A ctypes view into :attr:`frame` for passing to ``OpenGL``.
    pixel_buffer : _PixelBuffer or None
        The device-side frame; ``None`` outside ``"interop"`` mode.
    texture : pyglet.gl.GLuint
        Handle to the ``OpenGL`` texture object.
    """

    # How long the window thread may sleep before pumping OS events again. A display
    # that takes no input only needs to stay under Win32's "Not Responding" threshold.
    event_period = 1.0

    def __init__(self, shape, screen=None, caption="", interop=None, device=None):
        """
        Create a :mod:`pyglet` window on the specified screen.

        Parameters
        ----------
        shape : (int, int) or None
            If ``None``, creates a fullscreen window. Otherwise, creates a
            windowed display with ``(height, width)`` pixels.
        screen : pyglet screen object or None
            Target screen. If ``None``, uses the default screen.
        caption : str
            Window title (visible in windowed mode).
        interop : bool or None
            Whether to render from ``CUDA``-mapped device memory. ``None`` uses it when
            available.
        device : int or None
            ``CUDA`` device that will write the frames, which is not necessarily the one
            current on this thread.
        """
        self.interop = interop
        self.device = device
        self.mode = None
        self.frame = None
        self.cframe = None
        self.pixel_buffer = None

        # Make the window and do basic setup.
        if screen is None:
            display = get_pyglet_display()
            screen = display.get_default_screen()

        if shape is None:   # Fullscreen
            super().__init__(
                screen=screen,
                fullscreen=True,
                vsync=True,
                caption=caption
            )
            self.set_mouse_visible(False)
            self.flip()
        else:
            super().__init__(
                screen=screen,
                width=shape[1],
                height=shape[0],
                resizable=True,
                fullscreen=False,
                vsync=True,
                caption=caption,
                style=pyglet.window.Window.WINDOW_STYLE_DEFAULT
            )
            self.set_visible(False)
            self.flip()

        self.shape = (self.height, self.width)

        try:
            # Icons. Currently hardcoded. Feel free to implement custom icons.
            path, _ = os.path.split(os.path.realpath(__file__))
            path = os.path.join(
                path, '..', '..', 'docs', 'source', 'static', 'slmsuite-notext-'
            )
            img16x16 =      pyglet.image.load(path + '16x16.png')
            img32x32 =      pyglet.image.load(path + '32x32.png')
            img512x512 =    pyglet.image.load(path + '512x512.png')
            self.set_icon(img16x16, img32x32, img512x512)
        except Exception as e:
            logger.warning("Failed to set window icon: %s", e)

    # Event handlers: consume all events to prevent OS default behavior
    # (modal drag loops, window resizing, accidental close) that would
    # interfere with SLM display or cause window freezing.

    def on_mouse_press(self, x, y, button, modifiers):
        """Consume mouse press to prevent OS modal drag loops on SLM windows."""
        return True

    def on_mouse_release(self, x, y, button, modifiers):
        """Consume mouse release to prevent interference with SLM display."""
        return True

    def on_mouse_motion(self, x, y, dx, dy):
        """Consume mouse motion to prevent interference with SLM display."""
        return True

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Consume mouse drag to prevent OS window move/resize behavior."""
        return True

    def on_key_press(self, symbol, modifiers):
        """Consume key press to prevent interference with SLM display."""
        return True

    def on_key_release(self, symbol, modifiers):
        """Consume key release to prevent interference with SLM display."""
        return True

    def on_resize(self, width, height):
        """Prevent window resizing. SLM dimensions are fixed at initialization."""
        return True

    def on_expose(self):
        """Consume expose event. Rendering is controlled via :meth:`render`."""
        return True

    def on_draw(self):
        """Suppress automatic redraws. Rendering is manual via :meth:`render`."""
        return True

    def on_close(self):
        """Allow the close button to stop the event loop."""
        self.has_exit = True

    def switch_to(self):
        """
        Activate this window's ``OpenGL`` context, if it has one.

        Guards against a :mod:`pyglet` race on window creation.
        ``Window._create()`` calls ``SetWindowPos`` to move the new fullscreen
        window onto the SLM's monitor, which sends ``WM_DPICHANGED``
        *synchronously* whenever that monitor has a different DPI than the
        previous one. Pyglet's handler for that message calls ``switch_to()``,
        but the ``OpenGL`` canvas is only attached to the context a few lines
        later in ``_create()``. The unguarded call raises ``RuntimeError:
        Canvas has not been attached`` inside the Win32 window procedure, which
        Python prints as an ignored ctypes callback exception.

        Also makes ``switch_to()`` a no-op after :meth:`close`, when the
        context has been destroyed.

        Taken under the module's creation lock, since this is what writes the
        process-global ``gl.current_context`` that window creation needs cleared.
        """
        context = getattr(self, "context", None)
        if context is None or context.canvas is None:
            return
        with _creation_lock:
            super().switch_to()

    @contextlib.contextmanager
    def current(self):
        """
        Hold this window's ``OpenGL`` context current for a whole draw.

        ``gl.current_context`` is process-global, so another thread creating or
        destroying a window mid-draw would clear it and every remaining call of the
        draw would raise ``GLException``.
        """
        with _creation_lock:
            self.switch_to()
            yield

    def dispatch_events(self):
        """
        Process pending OS events for this window.

        On Windows, overrides the parent :meth:`dispatch_events` to bypass
        pyglet's ``platform_event_loop.start()`` thread check. Pyglet 2.x
        requires ``start()`` to be called from the thread that imported
        :mod:`pyglet.app`, but SLM windows run on dedicated background threads.
        We perform the Win32 message pump directly, which is safe on the
        window's creator thread.

        On Linux and macOS, the parent implementation works correctly from
        background threads, so we delegate to it directly.
        """
        if sys.platform == "win32":
            from pyglet.libs.win32 import _user32
            from pyglet.libs.win32 import constants
            from pyglet.libs.win32.types import MSG

            self._allow_dispatch_event = True
            self.dispatch_pending_events()

            msg = MSG()
            while _user32.PeekMessageW(
                ctypes.byref(msg), 0, 0, 0, constants.PM_REMOVE
            ):
                _user32.TranslateMessage(ctypes.byref(msg))
                _user32.DispatchMessageW(ctypes.byref(msg))
            self._allow_dispatch_event = False
        else:
            super().dispatch_events()

    def _bring_to_front(self):
        """
        Make this window always-on-top using platform-specific APIs.

        Called once after window creation on the window's owning thread.
        Uses :func:`SetWindowPos` with ``HWND_TOPMOST`` on Windows,
        ``_NET_WM_STATE_ABOVE`` on Linux/X11, and
        ``NSFloatingWindowLevel`` on macOS. Falls back to
        :meth:`~pyglet.window.Window.activate` on unknown platforms.

        The window rises without being activated, so the keyboard stays with whatever
        opened it.
        """
        if sys.platform == "win32":
            try:
                from pyglet.libs.win32 import _user32, constants
                _user32.SetWindowPos(
                    self._hwnd, constants.HWND_TOPMOST,
                    0, 0, 0, 0,
                    constants.SWP_NOMOVE | constants.SWP_NOSIZE | constants.SWP_NOACTIVATE
                )
            except (ImportError, AttributeError):
                pass
        elif sys.platform == "linux":
            try:
                # _set_wm_state is defined on pyglet's XlibWindow and sets
                # _NET_WM_STATE_ABOVE via XChangeProperty + ClientMessage.
                self._set_wm_state("_NET_WM_STATE_ABOVE")
            except Exception:
                try:
                    self.activate()
                except Exception:
                    pass
        elif sys.platform == "darwin":
            try:
                # NSFloatingWindowLevel = 3 — above normal windows.
                self._nswindow.setLevel_(3)
            except Exception:
                try:
                    self.activate()
                except Exception:
                    pass
        else:
            try:
                self.activate()
            except Exception:
                pass

    def _setup_frame(self, shape, B):
        """
        Allocate the frame, degrading ``interop`` to ``pinned`` to ``pageable``.

        Reallocates, so it must not run while a frame is being written or rendered.
        """
        self._release_frame()

        if self.interop is not False:
            try:
                self.pixel_buffer = _PixelBuffer(shape + (B,), self.device)
                self.mode = "interop"
            except Exception as e:
                if self.interop:
                    raise RuntimeError(
                        "interop=True requested, but unavailable: {}".format(e)
                    )
                logger.debug("Interop frame unavailable: %s", e)

        if self.pixel_buffer is None:
            try:
                if zeros_pinned is None:
                    raise RuntimeError("cupy unavailable.")
                self.frame = zeros_pinned(shape + (B,), dtype=np.uint8)
                self.mode = "pinned"
            except Exception as e:
                logger.debug("Pinned frame unavailable: %s", e)
                self.frame = np.zeros(shape + (B,), dtype=np.uint8)
                self.mode = "pageable"

            self.frame[:, :, 3] = 255  # Opaque alpha
            self.cframe = (gl.GLubyte * int(shape[0] * shape[1] * B)).from_buffer(self.frame)

        logger.debug("Frame: '%s'.", self.mode)

    def _release_frame(self):
        """Free the frame. Must run on the window thread."""
        try:
            if self.pixel_buffer is not None:
                self.pixel_buffer.release()
        finally:
            self.frame, self.cframe, self.pixel_buffer, self.mode = None, None, None, None

    def acquire(self):
        """
        Claim the frame for writing.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            The RGBA array to write into.
        """
        # Read once; the window thread may release the frame concurrently.
        pixel_buffer, frame = self.pixel_buffer, self.frame
        if pixel_buffer is None and frame is None:
            raise RuntimeError("Window has closed; its frame is released.")

        return pixel_buffer.map() if pixel_buffer is not None else frame

    def commit(self):
        """Hand the written frame back to ``OpenGL``, ready for :meth:`render`."""
        if self.pixel_buffer is not None:
            self.pixel_buffer.unmap()

    def _setup_context(self):
        """
        Sets up a ``TRIANGLE_STRIP`` quad drawn by the default :mod:`pyglet` blit shader.

        Raises
        ------
        RuntimeError
            If no compatible ``OpenGL`` context is available.
        """
        shape = self.shape

        if self.context.get_info().have_version(3, 0):
            # Channels: R+G+B+A=4
            B = 4

            self._setup_frame(shape, B)

            # Setup the texture
            self.texture = gl.GLuint()
            gl.glGenTextures(1, ctypes.byref(self.texture))
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)

            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

            # Malloc the OpenGL memory, blanked so nothing is displayed before the first frame
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8,
                shape[1], shape[0],
                0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                (gl.GLubyte * int(shape[0] * shape[1] * B))()
            )

            # Use the default pyglet shader; this is required in 2.0+.
            self.shader = pyglet.graphics.get_default_blit_shader()
            self.shader.use()

            # Also allocate the quadrangle using pyglet 2.0+ formalism.
            self.batch = pyglet.graphics.Batch()
            self.vertex_list = self.shader.vertex_list(
                4,
                gl.GL_TRIANGLE_STRIP,
                self.batch,
                # Vertex positions (x, y, z)
                position=('f',
                    [
                        0.,  float(shape[0]), 0.,
                        0., 0., 0.,
                        float(shape[1]), float(shape[0]), 0.,
                        float(shape[1]), 0., 0.,
                    ]
                ),
                # Texture coordinates (u, v, r); v selected to match matplotlib
                # imshow convention (top-left origin)
                tex_coords= ('f',
                    [
                        0., 0., 0.,
                        0., 1., 0.,
                        1., 0., 0.,
                        1., 1., 0.,
                    ]
                )
            )

            # Cleanup.
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glFlush()
        else:
            raise RuntimeError("A compatible OpenGL 3.0+ context is required.")

    def render(self):
        """
        Upload the current frame to the ``OpenGL`` texture and display it.

        This method:

        1.  Activates this window's ``OpenGL`` context via :meth:`current()`.
        2.  Uploads the frame to the GPU texture with ``glTexSubImage2D``.
        3.  Draws the textured quad to the back buffer.
        4.  Calls ``flip()`` to swap front/back buffers (blocks on vsync).
        5.  Calls ``dispatch_events()`` for additional event processing.

        Important
        ~~~~~~~~~
        Must be called from the same thread that created the window.
        Practically, this means calling from :meth:`~_WindowThread.submit`.
        """
        with self.current():
            # In interop the source is device memory, addressed as an offset into the bound buffer.
            bound = self.pixel_buffer is not None
            if bound:
                gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer.buffer.value)
                source = 0
            else:
                source = self.cframe

            try:
                self._blit(source)
            finally:
                # A binding left behind would turn every later upload's pointer into an offset.
                if bound:
                    gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

            # Display the other side of the double buffer.
            # (with vsync enabled, this will block until the next frame is ready to display).
            self.flip()

        self.dispatch_events()

    def _blit(self, source):
        """
        Upload ``source`` to the texture and draw it. See :meth:`render`.

        A ``None`` source redraws whatever the texture already holds, which is how
        :meth:`_ViewerWindow._redraw` pans and zooms without touching host memory.
        """
        shape = self.shape

        self.shader.use()

        # Bind texture.
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)
        if source is not None:
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0,
                shape[1], shape[0],
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                source
            )

        # Draw the quad.
        self.vertex_list.draw(gl.GL_TRIANGLE_STRIP)

    @staticmethod
    def info(verbose=True):
        """
        Get information about the available displays, their indexes, and their sizes.

        Parameters
        ----------
        verbose : bool
            Whether or not to print display information.

        Returns
        -------
        list of (int, (int, int, int, int), bool, bool, str) tuples
            The number (int), geometry of each display ((int, int, int, int)),
            whether it is the main or mirrored display (bool, bool), and a
            stable identifier for the display (str; see :func:`_screen_id`).
        """
        # Note: in pyglet, the display is the full arrangement of screens,
        # unlike the terminology in other SLM subclasses
        display = get_pyglet_display()

        screens = display.get_screens()
        default = display.get_default_screen()
        windows = display.get_windows()

        def parse_screen(screen):
            return (
                "x={}, y={}, width={}, height={}"
                .format(screen.x, screen.y, screen.width, screen.height)
            )
        def parse_screen_int(screen):
            return (screen.x, screen.y, screen.width, screen.height)
        def parse_window(window):
            x, y = window.get_location()
            return (
                "x={}, y={}, width={}, height={}"
                .format(x, y, window.width, window.height)
            )

        default_str = parse_screen(default)

        window_strs = []
        for window in windows:
            window_strs.append(parse_window(window))

        if verbose:
            print('Display Positions:')
            print('#,  Position,  Identifier')

        screen_list = []

        for x, screen in enumerate(screens):
            screen_str = parse_screen(screen)
            screen_id = _screen_id(screen)

            # main_bool is True if this screen is the default (main) display.
            main_bool = False
            # window_bool is True if this screen has a window mirrored on it.
            window_bool = screen_str in window_strs

            if screen_str == default_str:
                main_bool = True
                screen_str += ' (main)'
            if window_bool:
                screen_str += ' (has ScreenMirrored)'

            if verbose:
                print('{},  {},  {}'.format(x, screen_str, screen_id))

            screen_list.append((
                x,
                parse_screen_int(screen),
                main_bool,
                window_bool,
                screen_id
            ))

        return screen_list


class _ViewerWindow(_Window):
    """
    A resizable :class:`_Window` displaying camera or SLM data for :meth:`.live`.

    The whole image lives in the texture at its native resolution and the viewer's
    region of interest is a texture-coordinate rectangle, so zoom and pan cost no host
    work and the window is free to be any size. Mouse and keyboard input are forwarded
    to the viewer rather than consumed.

    Important
    ~~~~~~~~~
    Every method here runs on the window thread, including the input handlers, which
    :class:`_WindowThread` dispatches. Requests that need the main thread (anything
    touching widgets) are posted to the viewer instead of applied here.

    Attributes
    ----------
    viewer : ~slmsuite.hardware._viewer._ViewerObject
        Viewer whose state is displayed and edited.
    shape : (int, int)
        The ``(height, width)`` of the *image*, unlike :class:`_Window`, where it is
        the size of the window.
    """

    # Input is only as smooth as the rate the thread pumps it at. Windows rounds a
    # wait up to its 15.6 ms timer tick, which this asks for and so lands on.
    event_period = 1 / 120.

    def __init__(self, shape, screen=None, caption="", *, viewer, image_shape, **kwargs):
        """
        Create a window of ``shape`` pixels showing an ``image_shape`` texture.

        Parameters
        ----------
        shape : (int, int)
            Initial ``(height, width)`` of the window.
        viewer : ~slmsuite.hardware._viewer._ViewerObject
            Viewer to read the region of interest from and forward input to.
        image_shape : (int, int)
            ``(height, width)`` of the image, which sizes the texture.
        **kwargs
            See :meth:`_Window.__init__`, which also documents ``screen`` and ``caption``.
        """
        self.viewer = viewer
        self.vertex_list = None
        self._quad = (0., 0., 1., 1.)
        self._dirty = False
        self._sized = (1, 1)

        kwargs["interop"] = False       # Frames are colorized on the host.
        super().__init__(shape, screen, caption, **kwargs)

        self.shape = (int(image_shape[0]), int(image_shape[1]))

    def _setup_context(self):
        """Set up the ``OpenGL`` context, then fit the quad to the window."""
        super()._setup_context()

        if not self.context.get_info().have_version(3, 0):
            raise RuntimeError("The pyglet viewer requires OpenGL 3.0+.")

        # A two-tap filter drops isolated spots when a sensor-sized image is minified
        # into a small window, so the whole mip chain is averaged instead.
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR
        )
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # Else the letterbox margins alternate stale swap-chain contents at 60 Hz.
        gl.glClearColor(0., 0., 0., 1.)

        self._refresh()

    def _bring_to_front(self):
        """Show the window above the others, leaving the keyboard where it was."""
        if sys.platform == "win32":
            # pyglet's own set_visible() ends in SetForegroundWindow, which would pull
            # focus out of the editor that opened the viewer.
            from pyglet.libs.win32 import _user32, constants
            _user32.SetWindowPos(
                self._hwnd, constants.HWND_TOPMOST,
                0, 0, 0, 0,
                constants.SWP_NOMOVE | constants.SWP_NOSIZE
                | constants.SWP_SHOWWINDOW | constants.SWP_NOACTIVATE
            )
            self._visible = True
            self.dispatch_event("on_show")
        else:
            self.set_visible(True)      # _Window hides windowed displays.
            super()._bring_to_front()

    def _refresh(self):
        """Fit the region of interest to the window and mark it for redraw."""
        if self.vertex_list is None:
            return

        ih, iw = self.shape
        x0, y0, x1, y1 = self.viewer.state["roi"]
        width, height = self.get_size()

        scale = min(width / (x1 - x0), height / (y1 - y0))
        qw, qh = (x1 - x0) * scale, (y1 - y0) * scale
        qx, qy = (width - qw) / 2, (height - qh) / 2
        self._quad = (qx, qy, qw, qh)

        self.vertex_list.position[:] = (
            qx, qy + qh, 0.,    qx, qy, 0.,
            qx + qw, qy + qh, 0.,   qx + qw, qy, 0.,
        )
        # v runs top-down, matching the top-left origin of the image.
        u0, u1, v0, v1 = x0 / iw, x1 / iw, y0 / ih, y1 / ih
        self.vertex_list.tex_coords[:] = (
            u0, v0, 0.,     u0, v1, 0.,
            u1, v0, 0.,     u1, v1, 0.,
        )

        self.viewer.state["geometry"] = self._geometry()
        self._dirty = True

    def _redraw(self, source=None):
        """
        Draw the texture, uploading ``source`` into it first if one is given.

        Unlike :meth:`_Window.render`, this does not dispatch events. The handlers
        that redraw are themselves called from :meth:`_Window.dispatch_events`, and a
        nested pump would strand the outer one's messages until the next frame.
        """
        self._dirty = False
        with self.current():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            self._blit(source)
            if source is not None:
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)   # _blit leaves the texture bound.
            self.flip()

    def render(self):
        """Upload the current frame to the texture and display it. See :meth:`_redraw`."""
        self._refresh()     # The viewer moves the region from its own thread, not this one.
        self._redraw(self.cframe)

    def _fraction(self, x, y):
        """Map window pixels to a ``(fx, fy)`` fraction of the displayed quad."""
        qx, qy, qw, qh = self._quad
        return (x - qx) / qw, 1. - (y - qy) / qh    # pyglet's y is bottom-origin.

    @staticmethod
    def _inside(fraction):
        """Whether a :meth:`_fraction` falls on the image rather than the letterbox."""
        return all(0 <= f <= 1 for f in fraction)

    def _geometry(self):
        """The window's ``(x, y, width, height)`` in the whole pixels the OS deals in."""
        return tuple(int(v) for v in tuple(self.get_location()) + tuple(self.get_size()))

    def _maximized(self):
        """Whether the OS is holding the window at a size that is not ours to choose."""
        if sys.platform == "win32":
            from pyglet.libs.win32 import _user32
            return bool(_user32.IsZoomed(self._hwnd))
        return self.width >= self.screen.width and self.height >= self.screen.height

    def dispatch_events(self):
        """Pump OS events, then settle the whole burst of them with one draw."""
        super().dispatch_events()
        if self._dirty and not self.has_exit:
            self._redraw()

    def on_expose(self):
        """Redraw when the window is uncovered."""
        self._dirty = True
        return True

    def on_move(self, x, y):
        """Remember where the window was put, so the next viewer opens there."""
        self.viewer.state["geometry"] = self._geometry()
        return True

    def on_resize(self, width, height):
        """Hold the window to the image's aspect, then refit the quad into it."""
        # A maximized window belongs to the OS; the quad letterboxes inside it instead.
        if self.vertex_list is not None and not self._maximized():
            ih, iw = self.shape
            w0, h0 = self._sized

            # Whichever axis the drag moved proportionally further drives the other,
            # so grabbing any edge or corner resizes the way it looks like it should.
            if abs(width - w0) * h0 >= abs(height - h0) * w0:
                snap = (width, max(1, round(width * ih / iw)))
            else:
                snap = (max(1, round(height * iw / ih)), height)

            self._sized = snap
            # A pixel of rounding is left to the letterbox, else this never settles.
            if abs(snap[0] - width) > 1 or abs(snap[1] - height) > 1:
                self.set_size(*snap)    # Re-enters here with the corrected size.
                return True

        self._refresh()
        return True

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """Scroll-wheel zoom toward the cursor, while the viewer allows zooming."""
        fraction = self._fraction(x, y)
        if scroll_y and self.viewer.state["zoom"] and self._inside(fraction):
            self.viewer._zoom(*fraction, scroll_y > 0)
            self._refresh()
        return True

    def on_mouse_press(self, x, y, button, modifiers):
        """Grab for a drag-pan, or restore the full image on a right-click."""
        fraction = self._fraction(x, y)
        if self._inside(fraction):
            if button == mouse.RIGHT:
                self.viewer._reset_roi()
                self._refresh()
            else:
                self.viewer._grab(*fraction)
        return True

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Click-drag pan. The cursor is not clamped to the quad, but the region is."""
        if self.viewer._pan(*self._fraction(x, y)):
            self._refresh()
        return True

    def on_mouse_release(self, x, y, button, modifiers):
        """Print the source pixel under the cursor, unless the click was a drag."""
        fraction = self._fraction(x, y)
        if not self.viewer._dragged and self._inside(fraction):
            self.viewer._post(("print", np.round(self.viewer._to_source(*fraction)).astype(int)))
        self.viewer._release()
        return True

    def on_key_press(self, symbol, modifiers):
        """
        Shortcuts standing in for the widgets, which a plain script cannot host.

        See :meth:`~slmsuite.hardware._viewer._Viewable.live` for the bindings. The
        color scaling ones are camera-only, matching the widgets they stand in for.
        """
        if symbol == key.ESCAPE:
            # Not _WindowThread.close(), which would join this very thread.
            self.has_exit = True
        elif symbol == key.R:
            self.viewer._reset_roi()
            self._refresh()
        elif symbol == key.C:
            self.viewer._post(("cmap",))
        elif not self.viewer.parent.is_slm:
            if symbol == key.X:
                self.viewer._post(("crosshair",))
            elif symbol == key.L:
                self.viewer._post(("log",))
            elif symbol == key.A:
                self.viewer._post(("autorange",))
        return True


class _WindowThread(object):
    """
    Manages a dedicated :class:`~threading.Thread` for a single :class:`_Window`.

    Each :class:`_WindowThread` creates its :class:`_Window` on a background
    daemon thread and continuously dispatches OS events to prevent the window
    from freezing. Commands from the main thread (e.g. rendering via
    :meth:`~_Window.render`) are submitted via :meth:`submit` and executed on
    the window thread.

    Important
    ~~~~~~~~~
    On Windows, window message queues are bound to the thread that created the
    window. All :mod:`pyglet` and ``OpenGL`` calls for a given window **must**
    happen on that window's thread. The main thread communicates via a
    thread-safe :class:`~queue.Queue`.

    Note
    ~~~~
    The thread runs as a daemon and will be terminated automatically when the
    main program exits. :meth:`close` provides graceful cleanup by closing
    the window, stopping the thread, and removing itself from the manager.

    Attributes
    ----------
    window : _Window or None
        The :class:`_Window` managed by this thread. ``None`` before the
        thread has finished initialization.
    """

    def __init__(
        self, shape, screen, caption, manager=None, window_class=None, on_close=None, **kwargs
    ):
        """
        Create a :class:`_Window` on a dedicated background thread.

        The constructor blocks until the window has been created and its
        ``OpenGL`` context initialized on the background thread, or until
        a timeout of 10 seconds is reached.

        Parameters
        ----------
        shape : (int, int) or None
            Window shape as ``(height, width)``, or ``None`` for fullscreen.
        screen : pyglet screen object
            Target screen for the window.
        caption : str
            Window title.
        manager : _WindowManager or None
            The :class:`_WindowManager` that owns this thread. If provided,
            :meth:`close` will automatically remove this thread from the
            manager.
        window_class : type or None
            :class:`_Window` subclass to construct, e.g. :class:`_ViewerWindow`.
            Defaults to :class:`_Window`.
        on_close : callable or None
            Called on the window thread once the window is gone, however it went.
        **kwargs
            See :meth:`_Window.__init__` for permissible options.

        Raises
        ------
        RuntimeError
            If the window thread fails to start within 10 seconds.
        Exception
            Re-raises any exception that occurred during window creation
            on the background thread.
        """
        self._command_queue = queue.Queue()
        self._command_event = threading.Event()
        self._submit_lock = threading.Lock()
        self._window = None
        self._running = False
        self._ready = threading.Event()
        self._error = None
        self._manager = manager
        self._window_class = _Window if window_class is None else window_class
        self._on_close = on_close

        # Store creation params; the device is sampled here, on the thread that writes frames.
        self._init_args = (shape, screen, caption)
        self._init_kwargs = kwargs
        if cp is not None and "device" not in kwargs:
            try:
                self._init_kwargs["device"] = cp.cuda.runtime.getDevice()
            except Exception:
                pass  # No usable GPU.
        self._start()

    def _start(self):
        """Start the background thread and wait for window creation."""
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="slmsuite-pyglet-{}".format(self._init_args[2])
        )
        self._thread.start()

        if not self._ready.wait(timeout=10.0):
            # Else the half-built thread runs on, unregistered and so beyond atexit.
            self.close()
            raise RuntimeError(
                "Window thread failed to start within 10s: {}".format(self._error)
            )
        if self._error is not None:
            raise self._error

    def _loop(self):
        """
        Main loop running on the background thread.

        1.  Creates the :class:`_Window` and initializes its ``OpenGL`` context.
        2.  Signals readiness to the main thread.
        3.  Enters an infinite loop that:

            a.  Processes commands from the main thread (via :attr:`_command_queue`).
            b.  Dispatches OS events for the window to prevent freezing.
            c.  Waits up to :attr:`~_Window.event_period` for new commands (via
                :attr:`_command_event`), waking instantly when one is submitted.
        """
        # Phase 1: Create window and OpenGL context on this thread.
        try:
            shape, screen, caption = self._init_args

            # Two pyglet issues must be worked around when creating windows on
            # background threads (Windows-specific, harmless no-op elsewhere):
            #
            # 1. WGL extension function pointers (like wglChoosePixelFormatARB) are
            #    thread-local on Windows. Pyglet's global gl_info singleton was
            #    populated during import on the main thread, so have_context()
            #    returns True, but wglGetProcAddress fails on this thread.
            #    Fix: temporarily clear _have_context to force the standard
            #    ChoosePixelFormat API (non-ARB path) which always works.
            #
            # 2. Pyglet tries to share the new GL context with gl.current_context
            #    (the main thread's context) via wglShareLists, which fails across
            #    threads. Fix: temporarily clear current_context so the new window
            #    creates an independent context.
            # Both workarounds edit process-global state, so they are serialized
            # against any other thread building a window.
            with _creation_lock:
                _saved_have_context = None
                try:
                    from pyglet.gl import gl_info as _gli
                    _saved_have_context = _gli._gl_info._have_context
                    _gli._gl_info._have_context = False
                except AttributeError:
                    pass  # Non-WGL platform

                gl.current_context = None

                try:
                    self._window = self._window_class(
                        shape, screen, caption, **self._init_kwargs
                    )
                finally:
                    # Else every later window in the process takes the non-ARB path.
                    # gl.current_context stays as the new window left it: _setup_context
                    # needs that context current to compile its shaders.
                    if _saved_have_context is not None:
                        _gli._gl_info._have_context = _saved_have_context
                        _gli.set_active_context()

                self._window._setup_context()

            # Bring window to front / set always-on-top (cross-platform).
            self._window._bring_to_front()
        except Exception as e:
            self._error = e
            # Else a half-built window is stranded on screen with nothing left to close it.
            self._teardown()
            self._ready.set()
            return

        self._ready.set()

        # Phase 2: Event loop — process commands and dispatch events.
        while self._running and not self._window.has_exit:
            # Drain all pending commands from the main thread.
            while True:
                try:
                    cmd = self._command_queue.get_nowait()
                    func, args, kwargs, future = cmd
                    try:
                        result = func(*args, **kwargs)
                        future['result'] = result
                        future['error'] = None
                    except Exception as e:
                        future['result'] = None
                        future['error'] = e
                    finally:
                        future['event'].set()
                except queue.Empty:
                    break

            if not self._running:
                break

            # Dispatch OS events to keep the window responsive.
            # This calls PeekMessageW on Win32, preventing the OS from
            # marking the window as "Not Responding".
            try:
                self._window.dispatch_events()
            except Exception:
                pass

            # Wait for a command, or for this window's own event-dispatch period.
            self._command_event.wait(timeout=self._window.event_period)
            self._command_event.clear()

        # Cleanup: stop accepting work and fail any still-queued futures so no waiter
        # blocks forever.
        with self._submit_lock:
            self._running = False
            while True:
                try:
                    _, _, _, future = self._command_queue.get_nowait()
                except queue.Empty:
                    break
                future['error'] = RuntimeError("Window thread exited before the command ran.")
                future['event'].set()

        self._teardown()
        if self._manager is not None:
            self._manager.remove_thread(self)
        if self._on_close is not None:
            self._on_close()

    def _teardown(self):
        """Free the frame and close the window. Must run on the window thread."""
        if self._window is None:
            return
        # Separate guards, so a failed release still lets the window close.
        try:
            self._window._release_frame()
        except Exception:
            pass
        try:
            self._window.close()
        except Exception:
            pass

    def submit(self, func, *args, **kwargs):
        """
        Submit a callable for execution on the window thread.

        This is the primary mechanism for the main thread to perform
        ``OpenGL`` operations (rendering, context changes, etc.) on the
        correct thread. The call returns immediately with a future dict;
        use :meth:`wait` to block until completion.

        Parameters
        ----------
        func : callable
            Function to execute on the window thread. Called as
            ``func(*args, **kwargs)``.
        *args
            Positional arguments for ``func``.
        **kwargs
            Keyword arguments for ``func``.

        Returns
        -------
        dict
            A future with keys ``'event'`` (:class:`threading.Event`),
            ``'result'``, and ``'error'``. Pass to :meth:`wait` to block
            until completion and retrieve the result.

        Raises
        ------
        RuntimeError
            If the window thread is not running.
        """
        future = {'event': threading.Event(), 'result': None, 'error': None}

        # Guard + enqueue atomically against the loop's stop/drain (see run() cleanup).
        # Also reject if the window has exited (e.g. user closed it).
        with self._submit_lock:
            if not self.running:
                raise RuntimeError("Window thread is not running.")
            self._command_queue.put((func, args, kwargs, future))

        self._command_event.set()
        return future

    @staticmethod
    def wait(future, timeout=None):
        """
        Block until a submitted future completes.

        Parameters
        ----------
        future : dict
            Future returned by :meth:`submit`.
        timeout : float or None
            Seconds to wait before raising. ``None`` waits forever.

        Returns
        -------
        object
            The return value of the submitted callable.

        Raises
        ------
        TimeoutError
            If the command did not finish within ``timeout``.
        Exception
            Re-raises any exception that occurred during execution on
            the window thread.
        """
        if not future['event'].wait(timeout):
            raise TimeoutError("Window thread did not finish within {} s.".format(timeout))
        if future['error'] is not None:
            raise future['error']
        return future['result']

    @property
    def window(self):
        """_Window or None: The managed window instance."""
        return self._window

    @property
    def running(self):
        """bool: Whether the thread is still alive and willing to accept work."""
        return self._running and self._window is not None and not self._window.has_exit

    def close(self):
        """
        Stop the event loop and join the thread.

        The loop handles window close and manager deregistration on exit.
        Safe to call multiple times.
        """
        self._running = False
        # Else the loop sleeps out its full dispatch timeout before noticing.
        self._command_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)


class _WindowManager(object):
    """
    Singleton that manages the lifecycle of all :class:`_WindowThread` instances.

    Provides centralized creation and cleanup of window threads. Registered as
    an :func:`atexit` handler to ensure all windows are closed gracefully when
    the program exits.

    Note
    ~~~~
    Use :meth:`get_instance` to obtain the singleton. Do not instantiate directly.

    Attributes
    ----------
    _threads : list of _WindowThread
        All active window threads managed by this instance.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """
        Get or create the singleton :class:`_WindowManager`.

        Returns
        -------
        _WindowManager
            The singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._threads = []
        self._threads_lock = threading.Lock()

        try:
            myappid = 'holodyne.slmsuite.viewer.' + SLMSUITE_VERSION
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except:
            pass

        atexit.register(self.shutdown)

    def create_window(self, shape, screen, caption, **kwargs):
        """
        Create a new :class:`_Window` on its own dedicated thread.

        Parameters
        ----------
        shape : (int, int) or None
            Window shape as ``(height, width)``, or ``None`` for fullscreen.
        screen : pyglet screen object
            Target screen for the window.
        caption : str
            Window title.
        **kwargs
            See :meth:`_WindowThread.__init__` and :meth:`_Window.__init__` for
            permissible options.

        Returns
        -------
        _WindowThread
            The thread managing the new window.

        Raises
        ------
        RuntimeError
            If the window thread fails to start.
        """
        wt = _WindowThread(shape, screen, caption, manager=self, **kwargs)
        with self._threads_lock:
            self._threads.append(wt)
        return wt

    def remove_thread(self, wt):
        """
        Remove a :class:`_WindowThread` from management.

        Parameters
        ----------
        wt : _WindowThread
            The thread to remove.
        """
        with self._threads_lock:
            try:
                self._threads.remove(wt)
            except ValueError:
                pass

    def shutdown(self):
        """
        Shut down all managed window threads.

        Called automatically via :func:`atexit`. Closes all windows and
        joins all threads.
        """
        with self._threads_lock:
            threads_copy = list(self._threads)
        for wt in threads_copy:
            try:
                wt.close()
            except Exception:
                pass
