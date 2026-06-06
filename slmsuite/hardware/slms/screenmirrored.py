"""
Projects data onto the SLM's virtual display, using the :mod:`pyglet` library.

Each :class:`ScreenMirrored` instance creates a fullscreen :mod:`pyglet` window
on a dedicated background thread. This thread continuously dispatches OS events
to prevent window freezing, while rendering commands are submitted from the main
thread via a thread-safe queue.
"""
import time
import warnings
import numpy as np

from slmsuite.hardware.slms.slm import SLM
from slmsuite.hardware._pyglet import _Window, _WindowManager, _WindowThread, get_pyglet_display

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
    This class supports GPU arrays when CuPy is available. Phase data can stay on the GPU
    throughout the processing pipeline, with only the final display step transferring to CPU.
    When pinned memory is available, direct CUDA memcpy to pinned host memory is used for
    faster DMA transfers compared to standard ``cp.asnumpy()``.

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

    This main thread communicates with those window threads via
    :meth:`~slmsuite.hardware._pyglet._WindowThread.submit`, which blocks until
    the command completes on the window thread.

    Note
    ~~~~
    Windows are created in fullscreen mode by default and are not intended for user
    interaction - they exist solely to display phase patterns to the SLM hardware.
    Event handling is implemented purely to prevent freezing, not to enable interactivity.

    Attributes
    ----------
    window : _Window
        Fullscreen window used to send information to the SLM.
    display_shape : (int, int)
        Shape of the mirrored display in pixels, as (height, width).
    """

    def __init__(
        self,
        display_number,
        bitdepth=8,
        wav_um=1,
        pitch_um=(8,8),
        verbose=True,
        slm_shape=None,
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
        verbose : bool
            Whether or not to print extra information.
        slm_shape : tuple of int or None
            SLM resolution as ``(width, height)``, for when the SLM's
            active area differs from the display resolution (e.g. PLM).
            Defaults to ``None``, which uses the display's native resolution.
            
            Caution
            ~~~~~~~
            This should normally be left as ``None`` unless the SLM has a
            different shape than the display. Note that different SLM and
            screen resolutions are not generally supported unless explicitly
            implemented in the associated SLM class.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """
        if pyglet is None:
            raise ImportError("pyglet not installed. Install to use ScreenMirrored SLMs.")

        if verbose:
            print("Initializing pyglet... ", end="")

        # Display/screen enumeration is read-only and thread-safe in pyglet 2.x.
        display = get_pyglet_display()
        screens = display.get_screens()
        if verbose:
            print("success")
            print("Searching for window with display_number={}... "
                    .format(display_number), end="")

        if len(screens) <= display_number:
            raise ValueError("Could not find display_number={}; only {} displays"
                .format(display_number, len(screens)))

        screen_info = ScreenMirrored.info(verbose=False)

        if screen_info[display_number][3]:
            raise ValueError(
                "ScreenMirrored window already created on display_number={}"
                .format(display_number))

        if verbose and screen_info[display_number][2]:
            print("warning: this is the main display... ", end="")

        if verbose:
            print("success")
            print("Creating window... ", end="")

        screen = screens[display_number]
        # Store as (height, width) for consistency with shape convention.
        self.display_shape = (screen.height, screen.width)

        # Use custom slm_shape if provided, else use display resolution.
        # slm_shape is (width, height) per SLM.__init__ convention.
        if slm_shape is None:
            slm_shape = self.display_shape

        super().__init__(
            slm_shape,
            bitdepth=bitdepth,
            wav_um=wav_um,
            pitch_um=pitch_um,
            **kwargs
        )

        # Create the window on a dedicated background thread.
        # The _WindowThread handles window creation, OpenGL context setup,
        # and continuous event dispatch on the same thread.
        try:
            time.sleep(0.2) # Short delay
            wm = _WindowManager.get_instance()
            self._window_thread = wm.create_window(None, screen, self.name)
            self.window = self._window_thread.window
        except Exception as e:
            if verbose:
                print("Window creation failed")
            raise

        if verbose:
            print("Window creation successful")

        # Warn the user if wav_um > wav_design_um
        if self.phase_scaling > 1:
            print(
                "Warning: Wavelength {} μm is inaccessible to this SLM with "
                "design wavelength {} μm".format(self.wav_um, self.wav_design_um)
            )

        # Variable to keep track of the last thread future.
        self._window_thread_future = None

    def _set_phase_hw(self, display, execute=True, block=True):
        """
        Writes phase data from `display` to the screen via the window's
        dedicated thread.

        The GPU→CPU transfer (if needed) happens on the main thread,
        then the buffer copy and ``OpenGL`` render are submitted to the window
        thread. By default the main thread blocks until rendering is complete.

        Parameters
        ----------
        display : numpy.ndarray or cupy.ndarray
            Integer data to display on the SLM. See :meth:`.SLM._set_phase_hw`.
        execute : bool
            Whether to actually send the image to the SLM. See :meth:`.SLM._set_phase_hw`.
        block : bool
            Whether to block the thread until the image is fully rendered.
            See :meth:`.SLM._set_phase_hw`.
        """
        # GPU→CPU transfer happens on main thread (no OpenGL needed).
        if cp is not None and isinstance(display, cp.ndarray):
            display = cp.asnumpy(display)

        # Submit render to the window's dedicated thread.
        if execute:
            self._window_thread_future = self._window_thread.submit(
                self._render, 
                self.window,
                display
            )
        if block and self._window_thread_future is not None:
            _WindowThread.wait(self._window_thread_future)

    @staticmethod
    def _render(window, display):
        """Copy grayscale data to RGBA buffer and render on window thread."""
        window.switch_to()
        # 3x writes faster than single broadcast
        # (buffer[:,:,:3] = display[:,:,np.newaxis])
        window.buffer[:,:,0] = display # R
        window.buffer[:,:,1] = display # G
        window.buffer[:,:,2] = display # B
        window.render()

    def close(self):
        """
        Closes the SLM window and stops its background thread.

        See :class:`.SLM`.
        """
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
        list of (int, (int, int, int, int), bool, bool) tuples
            The number, geometry of each display.
        """
        if pyglet is None:
            raise ImportError("pyglet not installed. Install to use ScreenMirrored SLMs.")

        return _Window.info(verbose=verbose)
