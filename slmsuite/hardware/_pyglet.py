"""
Hidden abstract classes for pyglet windowing in slmsuite.

Provides :class:`_Window` (a :mod:`pyglet` window subclass for SLM display),
:class:`_WindowThread` (a dedicated thread per window for event dispatch), and
:class:`_WindowManager` (a singleton coordinating all window threads).

All :mod:`pyglet` and ``OpenGL`` calls are executed on the window's dedicated
thread to satisfy OS thread-affinity requirements (especially Win32, where a
window's message queue is bound to the thread that created it). This prevents
windows from freezing between :meth:`~slmsuite.hardware.slms.slm.SLM.set_phase`
calls.
"""
import os
import ctypes
import threading
import queue
import atexit
import numpy as np
from packaging.version import Version

try:
    import pyglet
    import pyglet.gl as gl
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
except:
    pyglet = None
    gl = None
    __Window = object
    PYGLET_VERSION = None
    def get_pyglet_display():
        raise ImportError("pyglet not installed.")

# Try to import CuPy for optimized GPU transfers
try:
    import cupy as cp
except ImportError:
    cp = None

from slmsuite._logging import make_logger

logger = make_logger(__name__)


class _Window(__Window):
    """
    A :mod:`pyglet` window subclass for displaying SLM phase patterns.

    Wraps a fullscreen (or windowed) ``OpenGL`` surface with a texture-based
    rendering pipeline. Phase data is written into an RGBA :attr:`buffer`,
    uploaded to an ``OpenGL`` texture via ``glTexSubImage2D``, and displayed
    via double-buffered vsync'd flips.

    Supports both ``OpenGL`` 3.0+ (programmable shader pipeline, pyglet 2.0+)
    and ``OpenGL`` 2.0 (fixed-function pipeline, pyglet < 2.0).

    Important
    ~~~~~~~~~
    All methods on this class (except :meth:`info`) must be called from the
    same thread that created the window. Use :class:`_WindowThread` to ensure
    thread affinity.

    Attributes
    ----------
    shape : (int, int)
        The ``(height, width)`` of the window in pixels.
    buffer : numpy.ndarray
        RGBA buffer of shape ``(height, width, 4)`` and dtype ``uint8``.
        Write grayscale data to channels 0-2 before calling :meth:`render`.
    cbuffer : ctypes array
        A ctypes view into :attr:`buffer` for passing to ``OpenGL``.
    texture : pyglet.gl.GLuint
        Handle to the ``OpenGL`` texture object.
    """

    def __init__(self, shape, screen=None, caption=""):
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
        """
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
        import sys

        if sys.platform == "win32":
            import ctypes
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
        """
        import sys

        if sys.platform == "win32":
            try:
                from pyglet.libs.win32 import _user32, constants
                _user32.SetWindowPos(
                    self._hwnd, constants.HWND_TOPMOST,
                    0, 0, 0, 0,
                    constants.SWP_NOMOVE | constants.SWP_NOSIZE
                )
            except (ImportError, AttributeError):
                pass
        elif sys.platform == "linux":
            try:
                # _set_wm_state is defined on pyglet's XlibWindow and sets
                # _NET_WM_STATE_ABOVE via XChangeProperty + ClientMessage.
                self._set_wm_state("_NET_WM_STATE_ABOVE")
            except (AttributeError, Exception):
                try:
                    self.activate()
                except Exception:
                    pass
        elif sys.platform == "darwin":
            try:
                # NSFloatingWindowLevel = 3 — above normal windows.
                self._nswindow.setLevel_(3)
            except (AttributeError, Exception):
                try:
                    self.activate()
                except Exception:
                    pass
        else:
            try:
                self.activate()
            except Exception:
                pass

    def _setup_context(self):
        """
        Initialize the ``OpenGL`` context, buffers, and texture.

        Detects the available ``OpenGL`` version and sets up the rendering
        pipeline accordingly:

        -   **GL 3.0+** (pyglet 2.0+): Programmable shader pipeline with
            ``TRIANGLE_STRIP`` quad and the default pyglet blit shader.
        -   **GL 2.0** (pyglet < 2.0): Fixed-function pipeline with
            ``Projection2D``, power-of-2 padded textures, and ``QUADS``.

        Raises
        ------
        RuntimeError
            If no compatible ``OpenGL`` context is available.
        """
        shape = self.shape

        if gl.base.gl_info.have_version(3,0):       # Pyglet >= 2.0.0
            # Channels: R+G+B+A=4
            B = 4

            # Setup buffers (texbuffer is power of 2 padded to init the memory in OpenGL)
            self.buffer = np.zeros(shape + (B,), dtype=np.uint8)
            self.buffer[:, :, 3] = 255  # Opaque alpha
            N = int(shape[0] * shape[1] * B)
            self.cbuffer = (gl.GLubyte * N).from_buffer(self.buffer)

            # Setup the texture
            self.texture = gl.GLuint()
            gl.glGenTextures(1, ctypes.byref(self.texture))
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)

            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

            # Malloc the OpenGL memory
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8,
                shape[1], shape[0],
                0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE,
                self.cbuffer
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
        elif gl.base.gl_info.have_version(2,0):     # Pyglet < 2.0.0
            # Set the viewpoint.
            proj = pyglet.window.Projection2D()
            proj.set(shape[1], shape[0], shape[1], shape[0])

            # Setup shapes.
            texture_shape = tuple(
                np.power(2, np.ceil(np.log2(shape))).astype(np.int64)
            )
            self.tex_shape_ratio = (
                float(shape[0])/float(texture_shape[0]),
                float(shape[1])/float(texture_shape[1])
            )
            B = 4

            # Setup buffers (texbuffer is power of 2 padded to init the memory in OpenGL).
            self.buffer = np.zeros(shape + (B,), dtype=np.uint8)
            N = int(shape[0] * shape[1] * B)
            self.cbuffer = (gl.GLubyte * N).from_buffer(self.buffer)

            texbuffer = np.zeros(texture_shape + (B,), dtype=np.uint8)
            Nt = int(texture_shape[0] * texture_shape[1] * B)
            texcbuffer = (gl.GLubyte * Nt).from_buffer(texbuffer)

            # Setup the texture.
            gl.glEnable(gl.GL_TEXTURE_2D)
            self.texture = gl.GLuint()
            gl.glGenTextures(1, ctypes.byref(self.texture))
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)

            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_GENERATE_MIPMAP, gl.GL_FALSE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

            # Malloc the OpenGL memory.
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8,
                texture_shape[1], texture_shape[0],
                0, gl.GL_BGRA, gl.GL_UNSIGNED_BYTE,
                texcbuffer
            )

            # Make sure we can write to a subset of the memory (as we will do in the future).
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0,
                shape[1], shape[0],
                gl.GL_BGRA, gl.GL_UNSIGNED_BYTE,
                self.cbuffer
            )

            # Cleanup.
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glFlush()
        else:
            raise RuntimeError("Could not find a compatible GL context.")

    def render(self):
        """
        Upload the current :attr:`buffer` contents to the ``OpenGL`` texture
        and display them.

        This method:

        1.  Activates this window's ``OpenGL`` context via ``switch_to()``.
        2.  Uploads :attr:`buffer` data to the GPU texture with ``glTexSubImage2D``.
        3.  Draws the textured quad to the back buffer.
        4.  Calls ``flip()`` to swap front/back buffers (blocks on vsync).
        5.  Calls ``dispatch_events()`` for additional event processing.

        Important
        ~~~~~~~~~
        Must be called from the same thread that created the window.
        Practically, this means calling from :meth:`~_WindowThread.submit`.
        """
        self.switch_to()

        shape = self.shape

        if gl.base.gl_info.have_version(3,0):       # Pyglet >= 2.0.0
            self.shader.use()

            # Bind texture.
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0,
                shape[1], shape[0],
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                self.cbuffer
            )

            # Draw the quad.
            self.vertex_list.draw(gl.GL_TRIANGLE_STRIP)
        elif gl.base.gl_info.have_version(2,0):     # Pyglet < 2.0.0
            # Setup texture variables.
            x1 = 0
            y1 = 0
            x2 = shape[1]
            y2 = shape[0]

            xa = 0
            ya = 0
            xb = self.tex_shape_ratio[1]
            yb = self.tex_shape_ratio[0]

            array = (gl.GLfloat * 32)(
                xa, ya, 0., 1.,         # tex coord,
                x1, y1, 0., 1.,         # real coord, ...
                xb, ya, 0., 1.,
                x2, y1, 0., 1.,
                xb, yb, 0., 1.,
                x2, y2, 0., 1.,
                xa, yb, 0., 1.,
                x1, y2, 0., 1.
            )

            # Update the texture with the cbuffer.
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.value)
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0,
                shape[1], shape[0],
                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                self.cbuffer
            )

            # Blit the texture.
            gl.glPushClientAttrib(gl.GL_CLIENT_VERTEX_ARRAY_BIT)
            gl.glInterleavedArrays(gl.GL_T4F_V4F, 0, array)
            gl.glDrawArrays(gl.GL_QUADS, 0, 4)
            gl.glPopClientAttrib()

        # Display the other side of the double buffer.
        # (with vsync enabled, this will block until the next frame is ready to display).
        self.flip()
        self.dispatch_events()

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
        list of (int, (int, int, int, int), bool, bool) tuples
            The number (int), geometry of each display ((int, int, int, int)), and whether
            it is the main or mirrored display (bool, bool).
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
            print('#,  Position')

        screen_list = []

        for x, screen in enumerate(screens):
            screen_str = parse_screen(screen)

            # main_bool is True if this screen is the default (main) display.
            main_bool = False
            # window_bool is True if this screen has a window mirrored on it.
            window_bool = False

            if screen_str == default_str:
                main_bool = True
                screen_str += ' (main)'
            if screen_str in window_strs:
                window_bool = True
                screen_str += ' (has ScreenMirrored)'

            if verbose:
                print('{},  {}'.format(x, screen_str))

            screen_list.append((
                x,
                parse_screen_int(screen),
                main_bool,
                window_bool
            ))

        return screen_list


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

    def __init__(self, shape, screen, caption, manager=None):
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
        self._window = None
        self._running = False
        self._ready = threading.Event()
        self._error = None
        self._manager = manager
        # Store creation params for the window thread to use.
        self._init_args = (shape, screen, caption)
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
            c.  Waits up to 1s for new commands (via :attr:`_command_event`),
                waking instantly when a command is submitted. The 1s timeout
                ensures periodic event dispatch to stay well below the ~5s
                Win32 "Not Responding" threshold.
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
            _saved_have_context = None
            try:
                from pyglet.gl import gl_info as _gli
                _saved_have_context = _gli._gl_info._have_context
                _gli._gl_info._have_context = False
            except AttributeError:
                pass  # Non-WGL platform

            gl.current_context = None

            self._window = _Window(shape, screen, caption)

            # Restore gl_info state. Do NOT restore gl.current_context —
            # the new window's context is now current on this thread, and
            # _setup_context() needs it to compile shaders.
            if _saved_have_context is not None:
                _gli._gl_info._have_context = _saved_have_context
                _gli.set_active_context()

            self._window._setup_context()

            # Bring window to front / set always-on-top (cross-platform).
            self._window._bring_to_front()
        except Exception as e:
            self._error = e
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

            # Wait for a command or 1s timeout (for periodic event dispatch).
            self._command_event.wait(timeout=1.0)
            self._command_event.clear()

        # Cleanup: close the window and deregister from the manager.
        try:
            self._window.close()
        except Exception:
            pass
        if self._manager is not None:
            self._manager.remove_thread(self)

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
        if not self._running:
            raise RuntimeError("Window thread is not running.")

        future = {'event': threading.Event(), 'result': None, 'error': None}
        self._command_queue.put((func, args, kwargs, future))
        self._command_event.set()
        return future

    @staticmethod
    def wait(future):
        """
        Block until a submitted future completes.

        Parameters
        ----------
        future : dict
            Future returned by :meth:`submit`.

        Returns
        -------
        object
            The return value of the submitted callable.

        Raises
        ------
        Exception
            Re-raises any exception that occurred during execution on
            the window thread.
        """
        future['event'].wait()
        if future['error'] is not None:
            raise future['error']
        return future['result']

    @property
    def window(self):
        """_Window or None: The managed window instance."""
        return self._window

    def close(self):
        """
        Stop the event loop and join the thread.

        The loop handles window close and manager deregistration on exit.
        Safe to call multiple times.
        """
        self._running = False
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
        atexit.register(self.shutdown)

    def create_window(self, shape, screen, caption):
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

        Returns
        -------
        _WindowThread
            The thread managing the new window.

        Raises
        ------
        RuntimeError
            If the window thread fails to start.
        """
        wt = _WindowThread(shape, screen, caption, manager=self)
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
