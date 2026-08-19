"""
Projects data onto the SLM's virtual display, using the :mod:`pyglet` library.

Each :class:`ScreenMirrored` instance creates a fullscreen :mod:`pyglet` window
on a dedicated background thread. This thread continuously dispatches OS events
to prevent window freezing, while rendering commands are submitted from the main
thread via a thread-safe queue.
"""
import time
import warnings

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware._pyglet import _Window, _WindowManager, _WindowThread, get_pyglet_display
from slmsuite._logging import make_logger

logger = make_logger(__name__)

try:
    import pyglet
except ImportError:
    pyglet = None
    warnings.warn("pyglet not installed. Install to use ScreenMirrored SLMs.")

try:
    import cupy as cp
except ImportError:
    cp = None

class ScreenMirrored(SLM):
    """
    Wraps a :mod:`pyglet` window for displaying data to an SLM.

    .. warning::
        Version `2.1.9` of `pyglet` introduced a bug that leaves the SLM display
        zeroed even after phase data has been applied. Please use version `2.1.8` or earlier
        until this is resolved in a future release.

    Important
    ~~~~~~~~~
    Many SLM manufacturers provide an SDK for interfacing with their devices.
    Using a python wrapper for these SDKs is recommended, instead of or in supplement to this class,
    as there often is functionality additional to a mirrored screen
    (e.g. USB for changing settings) along with device-specific optimizations.

    Note
    ~~~~
    There are a variety of python packages that support blitting images onto a fullscreen display.

    -   `Simple DirectMedia Layer (SDL) <https://www.libsdl.org/>`_ wrappers:

        - :mod:`pygame` (`link <https://www.pygame.org/docs/>`__),
          which also supports OpenGL. Only supports one screen.
        - :mod:`sdl2` (`readthedocs <https://pysdl2.readthedocs.io/en/latest/>`__)
          through the ``PySDL2`` package. Requires additional libraries.

    -   `Open Graphics Library (OpenGL) <https://www.opengl.org/>`_ wrappers:

        - :mod:`moderngl` (`readthedocs <https://moderngl.readthedocs.io/en/latest/>`__),
          an OpenGL wrapper focusing on a pythonic interface for core OpenGL functions.
        - :mod:`OpenGL` (`link <http://pyopengl.sourceforge.net/documentation/index.html>`__)
          through the ``PyOpenGL``/``PyOpenGL_accelerate`` package, a very light OpenGL wrapper.
        - :mod:`pyglet` (`readthedocs <https://pyglet.readthedocs.io/en/latest/>`__),
          a light OpenGL wrapper.

    -   GUI Library wrappers:

        - :mod:`gi` (`readthedocs <https://pygobject.readthedocs.io/en/latest/>`__),
          through the ``PyGObject`` package wrapping ``GTK`` and other GUI libraries.
        - :mod:`pyqt6` (`link <https://riverbankcomputing.com/software/pyqt/>`__),
          through the ``PyQt6`` package wrapping the version 6 ``Qt`` GUI library.
        - :mod:`tkinter` (`link <https://docs.python.org/3/library/tkinter.html>`__),
          included in standard ``python``, wrapping the ``Tcl``/``Tk`` GUI library.
        - :mod:`wx` (`link <https://docs.wxpython.org/>`__),
          through the ``wxPython`` package wrapping the ``wxWidgets`` GUI library.
          :mod:`slmpy` (`GitHub <https://github.com/wavefrontshaping/slmPy>`__) uses :mod:`wx`.

    :mod:`slmsuite` uses :mod:`pyglet` as the default display package.
    :mod:`pyglet` is generally more capable than the mentioned SDL wrappers while immediately supporting
    features such as detecting connected displays which low-level packages like :mod:`OpenGL` and
    :mod:`moderngl` do not have. :mod:`pyglet` allows us to interact more directly with the display
    hardware without the additional overhead that is found in GUI libraries.
    Most importantly, :mod:`pyglet` is well documented.

    However, it might be worthwhile in the future to look back into SDL options, as SDL surfaces
    are closer to the pixels than OpenGL textures, so greater speed might be achievable (even without
    loading data to the GPU as a texture).

    GPU Optimization
    ~~~~~~~~~~~~~~~~
    Grayscale data is expanded to RGBA and delivered to the display by whichever of the
    following the hardware supports, in order of preference:

    -   ``"interop"``, which writes into ``OpenGL`` memory mapped into ``CUDA`` and never
        crosses PCIe. Requires an NVIDIA driver and ``OpenGL`` 3.0+.
    -   ``"pinned"``, page-locked host memory for fast DMA. Requires a ``CUDA`` device.
    -   ``"pageable"``, ordinary host memory, which always works.

    See ``interop`` in :meth:`__init__` and :attr:`~slmsuite.hardware._pyglet._Window.mode`.
    Constructing with ``gpu=True`` (see :meth:`.SLM.__init__`) additionally keeps the phase
    pipeline on the GPU, avoiding a host round-trip before the expansion.

    Important
    ~~~~~~~~~
    :class:`ScreenMirrored` uses a double-buffered and vertically synchronized (vsync) ``OpenGL``
    context. This is to prevent "tearing" resulting from data being modified during a display write:
    rather, all monitor writes are synchronized such that clean frames are always displayed.
    This feature is similar to the ``isImageLock`` flag in :mod:`slmpy`, but is implemented a bit
    closer to the hardware.

    Threading Model
    ~~~~~~~~~~~~~~~
    Each :class:`ScreenMirrored` window is created on its own dedicated background
    thread via :class:`~slmsuite.hardware._pyglet._WindowThread`. This allows
    the background threads to handle OS events and independent event
    dispatch/vsync timing for multi-SLM support.

    The main thread communicates with those window threads via
    :meth:`~slmsuite.hardware._pyglet._WindowThread.submit`, which queues a command and
    returns. Since a single frame is shared with the window thread, each
    :meth:`.set_phase` waits for the previous render regardless of ``block``, and for its
    own only when ``block=True``.

    Note
    ~~~~
    Windows are created in fullscreen mode by default and are not intended for user
    interaction - they exist solely to display phase patterns to the SLM hardware.
    Event handling is implemented purely to prevent freezing, not to enable interactivity.

    Attributes
    ----------
    window : _Window
        Fullscreen window used to send information to the SLM.
    display_number : int
        Number of the display that this SLM is mirrored onto.
    display_shape : (int, int)
        Shape of the mirrored display in pixels, as (height, width).
    """

    def __init__(
        self,
        display_number,
        bitdepth=8,
        wav_um=1,
        pitch_um=(8,8),
        slm_resolution=None,
        gpu=None,
        interop=None,
        **kwargs
    ):
        """
        Initializes a :mod:`pyglet` window for displaying data to an SLM.

        The window is created on a dedicated background thread to ensure
        continuous event dispatch and prevent freezing.

        Caution
        ~~~~~~~
        An SLM designed at 1064 nm can be used for an application at 780 nm by passing
        ``wav_um=.780`` and ``wav_design_um=1.064``,
        thus causing the SLM to use only a fraction (780/1064)
        of the full dynamic range. Be sure these values are correct.
        Note that there are some performance losses from using this modality (see :meth:`.set_phase()`).

        Caution
        ~~~~~~~
        There is some subtlety to
        `complex display setups with Linux <https://pyglet.readthedocs.io/en/latest/modules/canvas.html>`_.
        Working outside the default display is currently not implemented.

        Parameters
        ----------
        display_number : int
            Monitor number for frame to be instantiated upon.
        bitdepth : int
            Bitdepth of the SLM. Defaults to 8.

            Caution
            ~~~~~~~
            This class currently supports SLMs with 8-bit precision or less.
            In the future, this class will also support 16-bit SLMs using RG color.
        wav_um : float
            Wavelength of operation in microns. Defaults to 1 μm.
        pitch_um : (float, float)
            Pixel pitch in microns. Defaults to 8 micron square pixels.
        slm_resolution : tuple of int or None
            SLM resolution as ``(width, height)``, for when the SLM's
            active area differs from the display resolution (e.g. PLM).
            Defaults to ``None``, which uses the display's native resolution.

            Caution
            ~~~~~~~
            This should normally be left as ``None`` unless the SLM has a
            different shape than the display. Note that different SLM and
            screen resolutions are not generally supported unless explicitly
            implemented in the associated SLM class.
        gpu : bool or None
            Whether to store and process data with :mod:`cupy` (see :attr:`xp`).
            ``None`` uses :mod:`cupy` if it is installed. Defaults to ``False``.
            If this feature is enabled, the SLM will attempt to use
            :mod:`cupy`-OpenGL interop if available.
        interop : bool or None
            Whether to write phase data straight into an ``OpenGL`` pixel buffer, avoiding
            a transfer across PCIe. ``None`` (the default) uses interop when ``gpu`` is
            enabled and it is available, ``True`` requires it, and ``False`` forbids it in
            favor of pinned host staging.

            Caution
            ~~~~~~~
            Interop is only usable from the thread ``OpenGL`` is current on -- in practice
            the thread that initialized :mod:`pyglet`. Mapping the buffer from any other
            thread fails with ``CUDA_ERROR_INVALID_GRAPHICS_CONTEXT``. Pass
            ``interop=False`` when :meth:`set_phase` will be called from a worker thread,
            as a real-time control loop does.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """
        if pyglet is None:
            raise ImportError("pyglet not installed. Install to use ScreenMirrored SLMs.")

        logger.debug("Initializing pyglet...")

        # Display/screen enumeration is read-only and thread-safe in pyglet 2.x.
        display = get_pyglet_display()
        screens = display.get_screens()
        logger.debug("Searching for window with display_number=%s...", display_number)

        if len(screens) <= display_number:
            raise ValueError("Could not find display_number={}; only {} displays"
                .format(display_number, len(screens)))

        screen_info = ScreenMirrored.info(verbose=False)

        if screen_info[display_number][3]:
            raise ValueError(
                "ScreenMirrored window already created on display_number={}"
                .format(display_number))

        if screen_info[display_number][2]:
            logger.warning("display_number=%s is the main display.", display_number)

        logger.debug("Creating window...")

        screen = screens[display_number]
        self.display_number = display_number
        # Store as (height, width) for consistency with shape convention.
        self.display_shape = (screen.height, screen.width)

        # Use custom slm_resolution if provided, else use display resolution.
        # slm_resolution is (width, height) per SLM.__init__ convention.
        if slm_resolution is None:
            slm_resolution = (screen.width, screen.height)

        # The display buffer is uint8 RGBA; >8-bit data would be silently truncated.
        if bitdepth > 8:
            raise NotImplementedError(
                "ScreenMirrored currently supports 8-bit SLMs or less; "
                "16-bit (RG color) packing is not yet implemented."
            )

        super().__init__(
            resolution=slm_resolution,
            bitdepth=bitdepth,
            wav_um=wav_um,
            pitch_um=pitch_um,
            gpu=gpu,
            **kwargs
        )
        gpu = self.xp is cp

        # Interop is only meaningful on the GPU backend, so `gpu` is the default answer;
        # an explicit `interop` overrides it. See the caution in the docstring for why a
        # caller writing from a worker thread must pass False.
        if interop is None:
            interop = gpu

        # Create the window on a dedicated background thread.
        try:
            time.sleep(0.2) # Short delay
            wm = _WindowManager.get_instance()
            self._window_thread = wm.create_window(None, screen, self.name, interop=interop)
            self.window = self._window_thread.window
        except Exception:
            self.logger.error("Window creation failed.")
            raise

        self.logger.debug("Window creation successful. Mode='%s'.", self.window.mode)
        if self.window.mode != "interop":
            self.logger.info("Mode is '%s'; cupy-GL interop not available.", self.window.mode)

        # Warn the user if wav_um > wav_design_um
        if self.phase_scaling > 1:
            self.logger.warning(
                "Wavelength %s μm is inaccessible to this SLM with design wavelength %s μm",
                self.wav_um, self.wav_design_um,
            )

        # Variable to keep track of the last thread future.
        self._window_thread_future = None

        # Staging array for expanding a GPU display to RGBA before a single transfer.
        self._display_rgba = None
        
    def _log_detail(self):
        """Identify which display this SLM is mirrored onto. See :meth:`._Loggable._log_detail`."""
        return "on display {}".format(self.display_number)

    def _set_phase_hw(self, display, execute=True, block=True):
        """
        Writes phase data from `display` to the screen via the window's
        dedicated thread.

        The expansion to RGBA happens on the main thread, then the ``OpenGL`` render is
        submitted to the window thread. By default the main thread blocks until rendering
        is complete.

        Parameters
        ----------
        display : numpy.ndarray or cupy.ndarray
            Integer data to display on the SLM. See :meth:`.SLM._set_phase_hw`.
        execute : bool
            Whether to actually send the image to the SLM. See :meth:`.SLM._set_phase_hw`.
        block : bool
            Whether to block the thread until this image is fully rendered.
            See :meth:`.SLM._set_phase_hw`. The *previous* image is always waited on,
            as the two share one frame.
        """
        # Let any outstanding render reach the screen before its frame is overwritten.
        self._wait()

        if not execute:
            return

        frame = self.window.acquire()
        try:
            self._pack(display, frame)
        finally:
            self.window.commit()

        self._window_thread_future = self._window_thread.submit(self.window.render)

        if block:
            self._wait()

    def _wait(self, timeout=None):
        """Block until any outstanding render has finished."""
        if self._window_thread_future is not None:
            future, self._window_thread_future = self._window_thread_future, None
            _WindowThread.wait(future, timeout)

    def _pack(self, display, frame):
        """Expand grayscale or per-channel ``display`` into an RGBA ``frame``."""
        if cp is not None and isinstance(frame, cp.ndarray):
            # Interop: write into OpenGL memory directly, so no transfer at all.
            target = frame
            display = cp.asarray(display)
        elif cp is not None and isinstance(display, cp.ndarray):
            if self._display_rgba is None or self._display_rgba.shape != frame.shape:
                self._display_rgba = cp.zeros(frame.shape, dtype=cp.uint8)
                self._display_rgba[:, :, 3] = 255  # Opaque alpha
            target = self._display_rgba
        else:
            target = frame

        # Per-channel writes outpace a single broadcast into [:, :, :3].
        for c in range(3):
            target[:, :, c] = display if display.ndim == 2 else display[c % len(display)]

        if target is not frame:
            target.get(out=frame)

    def close(self):
        """
        Closes the SLM window and stops its background thread.

        See :class:`.SLM`.
        """
        # Let a non-blocking final frame reach the screen, but never fail or stall the close.
        try:
            self._wait(timeout=1)
        except Exception:
            pass
        self._window_thread.close()

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
            The number and geometry of each display, whether it is the main or
            a mirrored display, and a stable identifier for the display
            (related to the physical connection port) which (unlike the number)
            survives other displays being attached or detached.
        """
        if pyglet is None:
            raise ImportError("pyglet not installed. Install to use ScreenMirrored SLMs.")

        return _Window.info(verbose=verbose)
