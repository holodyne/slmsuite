
import asyncio
import html
import io
import os
import queue
import struct
import sys
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL

from slmsuite.holography.analysis import _center, image_centroids, image_remove_field
from slmsuite.misc.xp import as_numpy
from slmsuite.misc.files import generate_path, save_h5

_DISPLAY_BACKENDS = ("ipython", "pyglet")
_WIDGET_BACKENDS = ("ipython",)

_CROSSHAIR_OPTIONS = [
    ("None", "none"),
    ("Center", "center"),
    ("Centroid", "centroid"),
    ("Center+Centroid", "center+centroid"),
]

# What each keyboard shortcut cycles its state entry through. The colormap options
# are per-viewer, so they are read from the state instead.
_CYCLES = {
    "cmap": None,
    "crosshair": [value for _, value in _CROSSHAIR_OPTIONS],
    "log": [False, True],
}

# ipyevents always suppresses the page scroll for a watched wheel event, so "wheel"
# is only appended to this list while zoom is enabled.
_MOUSE_EVENTS = ["mousedown", "mousemove", "mouseup", "mouseleave", "click", "dblclick"]

# State a closing viewer leaves on its hardware, for the next one to open with. "live" is
# left out so that reopening never starts polling on its own.
_REMEMBERED = ("cmap", "scale", "zoom", "crosshair", "log", "range", "geometry")

# Mouse events are throttled to this period, roughly the rate at which frames can be
# drawn: a megapixel frame takes tens of milliseconds to encode.
_RENDER_PERIOD_S = .033

# Frames are encoded as 8-bit palette PNGs: the colormap is sent once as a palette
# and each pixel is a single index, which is what keeps mouse zoom and pan tracking
# the cursor.
_LEVELS = 253       # Indices 0..252 hold image data.
_TRANSPARENT = 253  # Reserved for nan.
_DARK, _LIGHT = 254, 255    # Reserved for crosshairs.


def _is_dark(palette):
    """Which ``palette`` entries a crosshair must contrast against by drawing light."""
    dark = palette[:, :3].astype(np.float32) @ np.float32([.299, .587, .114]) < 128
    dark[_TRANSPARENT] = False      # Nan shows the page behind it, which is light.
    return dark


def _png(index, palette):
    """Encode palette indices as 8-bit ``PNG`` bytes."""
    buff = io.BytesIO()
    image = PIL.Image.frombytes("P", (index.shape[1], index.shape[0]), index.tobytes())
    image.putpalette(palette[:, :3].tobytes())
    image.save(buff, format="png", compress_level=1, transparency=_TRANSPARENT)
    return buff.getvalue()


def _ipython():
    """The running IPython kernel able to host widgets, or ``None``."""
    try:
        from IPython import get_ipython
        import ipywidgets     # noqa: F401
    except ImportError:
        return None
    return get_ipython()


def _save_dialog(file_path):
    """Ask the user where to save, defaulting to ``file_path``. Empty if cancelled."""
    if sys.platform != "win32":
        raise NotImplementedError(f"No save dialog is implemented for '{sys.platform}'.")

    import win32con     # pywin32
    import win32gui

    try:
        return win32gui.GetSaveFileNameW(
            InitialDir=os.path.dirname(file_path),
            File=os.path.basename(file_path),
            DefExt="png",
            Title="Save view",
            Filter="PNG image\0*.png\0HDF5 raw data\0*.h5\0All files\0*.*\0",
            Flags=win32con.OFN_OVERWRITEPROMPT,
        )[0]
    except Exception:
        return ""       # The dialog raises rather than returns when cancelled.


def _clipboard_image(png, name):
    """Copy ``png`` bytes to the system clipboard as an image named ``name``."""
    if sys.platform != "win32":
        raise NotImplementedError(f"No clipboard copy is implemented for '{sys.platform}'.")

    import win32clipboard      # pywin32

    # The clipboard's device-independent bitmap is a BMP without its 14-byte header.
    bmp = io.BytesIO()
    PIL.Image.open(io.BytesIO(png)).convert("RGB").save(bmp, format="bmp")
    png_format = win32clipboard.RegisterClipboardFormat("PNG")

    # A file drop names a file rather than carrying data, so the image is staged in
    # the temp directory under a fixed name that later copies overwrite.
    file_path = os.path.join(tempfile.gettempdir(), name + ".png")
    with open(file_path, "wb") as f:
        f.write(png)
    drop = (
        struct.pack("<IiiII", 20, 0, 0, 0, 1)   # DROPFILES: wide paths at offset 20.
        + (file_path + "\0\0").encode("utf-16-le")
    )

    win32clipboard.OpenClipboard()
    try:
        # Each format is offered: applications take whichever they prefer.
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, bmp.getvalue()[14:])
        win32clipboard.SetClipboardData(png_format, png)
        win32clipboard.SetClipboardData(win32clipboard.CF_HDROP, drop)
    finally:
        win32clipboard.CloseClipboard()


class _Viewable:

    def live(self, activate=None, widgets=True, backend="ipython", **kwargs):
        """
        Creates and displays a live viewer.
          - When used with a camera, the viewer displays the last image:
            the result of :meth:`get_image()` or the last image of :meth:`get_images()`
            **whenever these methods are called**.
            Averaging and HDR are displayed with the same color scaling as without.
          - When used with an SLM, the viewer displays the phase pattern currently on the
            SLM: the phase pattern passed to :meth:`set_phase()` or
            the last phase pattern passed to :meth:`set_phases()`
            **whenever these methods are called**.

        The view itself is drawn either inside the notebook (``backend="ipython"``) or in
        a resizable window of its own (``backend="pyglet"``), which also works from a
        plain script, where no notebook is available to draw into.

        If ``True`` is passed to the ``widgets`` argument, this viewer is accompanied by
        a series of `IPython widgets
        <https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html>`_
        in the form of sliders and buttons
        for controlling the color scale, colormap, viewer scale, and live viewing,
        along with buttons to copy the current view to the clipboard or save it to a
        file, either as a ``.png`` of the view or an ``.h5`` of the raw data.
        By toggling the ``Live`` widget button, a loop is created that continuously
        polls the camera for new images.
        This viewer can be used as a realtime camera monitor within the jupyter notebook.
        However, note that any user-execution will block the monitoring loop.
        Regardless, any image polling during the blocked period will still update the viewer,
        which provides useful active feedback for what is happening during the
        execution.
        ``Live`` mode is ignored for SLMs.

        The viewer also supports zooming into a region of interest. With the ``Zoom``
        widget enabled, scroll the mouse wheel to zoom in/out toward the cursor,
        click-drag to pan, and double-click to restore the full image; disabling
        ``Zoom`` also restores the full image. Clicking prints the source-image pixel
        coordinate under the cursor. These mouse interactions require the optional
        :mod:`ipyevents` package
        (``pip install ipyevents``); without it the viewer still functions normally.
        A ``pyglet`` window needs no such package, and zooms by default since it has
        no page to scroll over. It restores the full image on a right-click and carries
        keyboard shortcuts (``esc`` close, ``r`` reset, ``c`` colormap, ``x`` crosshairs,
        ``l`` logarithmic, ``a`` autorange) so that it is usable without any widgets.
        The shortcuts that change color scaling take effect on the next frame. The
        ``Scale`` widget sizes the window itself, which keeps the image's aspect ratio.
        Closing the window closes the viewer, as toggling :meth:`live` would.

        A viewer leaves its settings on the hardware when it closes, so calling
        :meth:`live` again reopens with the same colormap, scaling, and crosshairs, and
        with the window where and how big it last was. Pass an argument to override any
        of them.

        This limitation is imposed by the
        Python Global Interpreter Lock (GIL) which restricts operation to a single thread,
        especially operation connecting to a diverse set of camera and SLM hardware.
        We use :mod:`asyncio` to allow the realtime monitoring loop to be
        interrupted by user-execution (e.g. running a cell in jupyter),
        blocking until the execution is finished.

        Parameters
        ----------
        activate : bool OR None
            If ``True``, creates a live viewer in the current cell,
            destroying any other attached viewer.
            If ``False``, destroys any other attached viewer.
            If ``None``, toggles the live viewer, destroying any attached viewer or
            creating one in the current cell if none is attached. Defaults to ``None``.
        widgets : bool OR str OR None
            The backend hosting the sliders and controls used to hone the display
            properties; one of ``"ipython"``. ``True`` selects the default and ``False``
            displays no controls. Widgets need a notebook, so a ``"pyglet"`` view opened
            from a script goes without.
        backend : str
            The backend drawing the view itself: ``"ipython"`` or ``"pyglet"``.
            The default is ``"ipython"``.
        **kwargs
            Options passed to the :class:`_ViewerObject` to customize the default settings.
            These features will be made less hidden in the future.
            Most things are customizable via these keywords. For instance, the user can pass
            a custom list of colormaps to appear in the widget dropdown as ``cmap_options=``.
        """
        # Destroying a viewer needs no backend, so it is settled before they are parsed.
        if self.viewer is not None and not activate:
            self.viewer.close()
            self.viewer = None
            return
        if activate is False:
            return

        if backend not in _DISPLAY_BACKENDS:
            raise ValueError(
                f"'{backend}' not recognized; "
                f"the .live() backend must be one of {_DISPLAY_BACKENDS}."
            )

        if widgets is True:
            widgets = _WIDGET_BACKENDS[0]
        elif widgets is False:
            widgets = None
        if widgets is not None and widgets not in _WIDGET_BACKENDS:
            raise ValueError(
                f"'{widgets}' not recognized; "
                f"the .live() widget backend must be one of {_WIDGET_BACKENDS}."
            )

        if _ipython() is None:
            if backend == "ipython":
                raise ImportError(
                    "The 'ipython' backend needs jupyter, ipywidgets, and a running kernel; "
                    "pass backend='pyglet' to view from a script."
                )
            widgets = None      # A window stands alone; its shortcuts replace the widgets.

        if self.viewer is not None:
            self.viewer.close()

        # Else a viewer that fails to build is left attached and closed.
        self.viewer = None
        self.viewer = _ViewerObject(
            self,
            widgets,
            backend,
            **kwargs
        )


class _ViewerObject:
    """
    Hidden class holding the state behind :meth:`._Viewable.live`.

    The view is drawn by a display backend (:class:`_ViewerDisplayIPython` or
    :class:`_ViewerDisplayPyglet`), which owns everything about *how* pixels reach the
    user. This class owns *what* is shown: the color scaling, the region of interest,
    the widgets, and the polling loop.
    """
    def __init__(
        self,
        parent,
        widgets,
        backend="ipython",
        live=False,
        min=None,
        max=None,
        log=None,
        cmap=True,
        scale=None,
        cmap_options=None,
        crosshair=None,
        zoom=None,
    ):
        self.parent = parent
        self.backend = backend

        # The last viewer on this hardware left its settings behind. Anything the caller
        # did not ask for is taken from there before falling back to a default.
        memory = getattr(parent, "_viewer_memory", None) or {}

        def recall(key, value, default):
            return memory.get(key, default) if value is None else value

        # Parse range.
        if min is None and max is None and "range" in memory:
            range_ = list(memory["range"])
        else:
            if min is None:
                min = 0
            if max is None:
                max = self.parent.bitresolution-1
            range_ = [np.min([min, max]), np.max([min, max])]

        # Parse scale
        scale = 2 ** np.round(np.log2(recall("scale", scale, 1)))

        # Parse colormap options.
        if cmap_options is None:
            if self.parent.is_slm:
                cmap_options = ["twilight", "twilight_shifted", "gray", "hsv"]
            else:
                cmap_options = [
                    "default", "gray", "Blues", "turbo",
                    'viridis', 'plasma', 'inferno', 'magma', 'cividis'
                ]

        if cmap is True: cmap = recall("cmap", None, cmap_options[0])
        if cmap is False: cmap = "gray"
        if cmap not in cmap_options and isinstance(cmap, str) and "cmap" in memory:
            cmap = cmap_options[0]      # A remembered colormap the new options lack.

        self.state = {
            "live" : live,
            "range" : range_,
            "log" : bool(recall("log", log, False)),
            "cmap" : cmap,
            "scale" : scale,
            "cmap_options" : cmap_options,
            "crosshair" : recall("crosshair", crosshair, "none"),
            "zoom" : bool(recall("zoom", zoom, backend == "pyglet")),
            "geometry" : memory.get("geometry"),
        }

        self.task = None
        self.closed = False
        self._drag = None
        self._dragged = False

        # Requests from the window thread, which may not touch the widgets itself.
        self._posts = queue.Queue()
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None   # A script applies posted requests before the next frame.

        # Region of interest (crop) in source-image pixels: [x0, y0, x1, y1].
        H, W = self.parent.shape[0], self.parent.shape[1]
        self.state["roi"] = [0.0, 0.0, float(W), float(H)]

        self.last_image = as_numpy(
            self.parent._viewer_frame if self.parent.is_slm else self.parent.last_image
        )

        self.widgets = {}
        self.state_keys = []
        self._lut = self._lut_key = None
        self._rgba = self._rgba_key = None

        if widgets: self.init_widgets()

        if backend == "pyglet":
            self.display = _ViewerDisplayPyglet(self)
        else:
            self.display = _ViewerDisplayIPython(self)

        if "output" in self.widgets:
            from IPython.display import display
            display(self.widgets["output"])

    def _quantize(self, img):
        """Window raw counts to the display range and quantize them to palette indices."""
        # Single-precision is ample for an 8-bit display and halves the memory
        # traffic of the windowing below.
        img = np.asarray(img, dtype=np.float32)

        # The scalars are plain floats so that they do not promote img back to double.
        r = [float(v) for v in self.state["range"]]
        d = max(r[1] - r[0], 1.)
        img = np.clip(img, r[0], r[1]) * (1. / d) - r[0] / d

        if not self.parent.is_slm and self.state["log"]:
            img = np.log10(1 + img * d) / np.log10(1 + d)

        # Windowing leaves img in [0, 1]; clipping keeps float error out of the
        # reserved entries above _LEVELS.
        nan = np.isnan(img)
        with np.errstate(invalid="ignore"):
            index = np.clip(img * (_LEVELS - 1) + .5, 0, _LEVELS - 1).astype(np.uint8)

        if nan.any():
            index[nan] = _TRANSPARENT

        return index

    def _palette(self):
        """The current colormap as a ``(256, 4)`` ``uint8`` table, cached against it."""
        cmap = self.state["cmap"]
        if cmap != self._rgba_key:
            # Match the colormap aliases understood by :meth:`_gray2rgb`.
            resolved = plt.rcParams["image.cmap"] if cmap == "default" else cmap
            if resolved == "grayscale":
                resolved = "gray"

            rgba = np.empty((256, 4), np.uint8)
            rgba[:_LEVELS, :3] = (
                255 * plt.get_cmap(resolved, _LEVELS)(np.linspace(0, 1, _LEVELS))[:, :3]
            ).astype(np.uint8)
            rgba[_LEVELS:, :3] = ((0, 0, 0), (0, 0, 0), (255, 255, 255))
            rgba[:, 3] = 255
            rgba[_TRANSPARENT, 3] = 0

            self._rgba_key = cmap
            self._rgba = rgba
        return self._rgba

    def _index_lut(self, dtype):
        """:meth:`_quantize` tabulated over every raw count, cached against the scaling."""
        key = (dtype.str, tuple(self.state["range"]), self.state["log"])
        if key != self._lut_key:
            # Tabulated in unsigned order, which is how the image view indexes it.
            counts = np.arange(1 << (8 * dtype.itemsize), dtype="u%d" % dtype.itemsize)
            self._lut_key = key
            self._lut = self._quantize(counts.view(dtype))
        return self._lut

    def parse(self, region=None):
        """
        Color the last image into palette indices.

        Parameters
        ----------
        region : list of float OR None
            Source-image ``[x0, y0, x1, y1]`` to crop to, downsampled to fit the display
            box. ``None`` returns the whole image at full resolution, for the backends
            that crop on the GPU instead.

        Returns
        -------
        numpy.ndarray
            ``(h, w)`` ``uint8`` indices into :meth:`_palette`.
        """
        is_cam = not self.parent.is_slm

        if self.last_image is None:
            # No frame has been captured yet (e.g. a fresh camera, or one whose
            # last_image was invalidated by a WOI/shape change). Render a blank
            # frame
            dtype = getattr(self.parent, "dtype", np.float32)
            self.last_image = np.zeros(self.parent.shape, dtype=dtype)

        H, W = self.last_image.shape[0], self.last_image.shape[1]

        if region is None:
            x0, y0, x1, y1 = 0, 0, W, H
            img = self.last_image
        else:
            # Crop to the requested region of interest (ROI). The size is rounded
            # once and the origin clamped to it, so a pan cannot resize the crop.
            cw = int(np.clip(round(region[2] - region[0]), 1, W))
            ch = int(np.clip(round(region[3] - region[1]), 1, H))
            x0 = int(np.clip(round(region[0]), 0, W - cw))
            y0 = int(np.clip(round(region[1]), 0, H - ch))
            x1, y1 = x0 + cw, y0 + ch
            src = self.last_image[y0:y1, x0:x1]

            # Downsample only if the crop has more pixels than the display box can
            # show; otherwise show the exact, full-resolution source pixels. The raw
            # crop is a plain integer slice, so it stays stable under panning (no
            # interpolation), unlike a resampled version.
            Bw = max(1, int(round(W * self.state["scale"])))
            Bh = max(1, int(round(H * self.state["scale"])))
            ch, cw = src.shape[0], src.shape[1]
            f = min(1.0, Bw / cw, Bh / ch)
            if f < 1.0:
                img = cv2.resize(
                    np.ascontiguousarray(src, dtype=np.float32),
                    (max(1, round(cw * f)), max(1, round(ch * f))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                img = src

        crosshairs = self.state["crosshair"].split("+") if is_cam else []

        if "centroid" in crosshairs:
            # Field removal subtracts a median, which would wrap an unsigned image.
            gray = np.asarray(img, dtype=np.float32)
            if len(gray.shape) != 2:
                gray = np.mean(gray, axis=-1)
            centroid = np.squeeze(image_centroids(image_remove_field([gray], deviations=None)))

        # An integer image spans few enough counts to tabulate the whole scaling.
        if img.dtype.kind in "ui" and img.dtype.itemsize <= 2:
            index = self._index_lut(img.dtype)[img.view("u%d" % img.dtype.itemsize)]
        else:
            index = self._quantize(img)

        is_dark = _is_dark(self._palette())

        # Add crosshairs: solid at the sensor center, dashed at the median-subtracted centroid.
        if "center" in crosshairs:
            self._crosshair(
                index,
                is_dark,
                (_center(W) - x0 + .5) * index.shape[1] / (x1 - x0) - .5,
                (_center(H) - y0 + .5) * index.shape[0] / (y1 - y0) - .5,
            )
        if "centroid" in crosshairs:
            self._crosshair(
                index,
                is_dark,
                _center(index.shape[1]) + centroid[0],
                _center(index.shape[0]) + centroid[1],
                dashed=True,
            )

        return index

    @staticmethod
    def _crosshair(index, is_dark, x, y, dashed=False):
        """
        Draw a horizontal and vertical line through the displayed pixel ``(x, y)``.

        Each pixel is set to the reserved black or white palette entry, whichever
        contrasts with the color underneath. Cyclic colormaps such as
        ``"twilight"`` are light at both ends, so choosing by luminance is what
        keeps the line visible there.
        """
        H, W = index.shape[0], index.shape[1]
        dash = slice(None, None, 2 if dashed else None)

        # Round half up. A nan position, which a centroid over nan data produces,
        # is pushed out of view so that it is simply not drawn.
        cx = int(np.floor(x + .5)) if np.isfinite(x) else -1
        cy = int(np.floor(y + .5)) if np.isfinite(y) else -1

        # Each line is drawn only if it falls within the view.
        if 0 <= cx < W:
            index[dash, cx] = np.where(is_dark[index[dash, cx]], _LIGHT, _DARK)
        if 0 <= cy < H:
            index[cy, dash] = np.where(is_dark[index[cy, dash]], _LIGHT, _DARK)

    def _print(self, message):
        """
        Show a message in the viewer's output area, replacing whatever was there.

        Only the latest message is kept: a fault that recurs on every mouse event
        would otherwise grow the output area without bound, slowing the frontend.
        """
        if "output" in self.widgets:
            self.widgets["output"].value = html.escape(str(message))
        else:
            print("slmsuite viewer:", message)

    def render(self, img=None):
        """Store new data, apply anything posted from the window thread, and draw."""
        try:
            if img is not None:
                # Stored before anything that might not draw, so that a skipped frame
                # costs a draw and not the data itself.
                previous = None if self.last_image is None else self.last_image.shape
                self.last_image = as_numpy(img)     # The renderers are host-only.
                if previous is not None and self.last_image.shape != previous:
                    self._reset_roi()      # Else the crop points outside the new image.

            self._apply()
            if not self.closed:     # A closed window tears the viewer down from _apply.
                self.display.render()
        except Exception as e:
            self._print(str(e))

    def _post(self, request):
        """Queue a request from the window thread for the main thread, which owns the widgets."""
        self._posts.put(request)
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self.render)
            except RuntimeError:
                pass    # The loop has closed; the next frame applies the request instead.

    def _apply(self):
        """Apply posted requests. Every state request cycles one option."""
        while True:
            try:
                key, *rest = self._posts.get_nowait()
            except queue.Empty:
                return

            if key == "print":
                self._print(rest[0])
            elif key == "close":
                self.parent.live(activate=False)
            elif key == "autorange":
                self._autorange()
            else:
                options = _CYCLES[key] or list(self.state["cmap_options"])
                # A value outside its options, such as a custom cmap=, starts at the top.
                current = self.state[key]
                index = options.index(current) + 1 if current in options else 0
                self._set(key, options[index % len(options)])

    def _set(self, key, value):
        """
        Set a state entry and the widget that shows it.

        The widget's observer is detached across the write: it would otherwise call
        :meth:`update` and draw a frame that the caller is about to draw anyway.
        """
        self.state[key] = value
        widget = self.widgets.get(key)

        if widget is not None:
            widget.unobserve(self.update, "value")
            try:
                widget.value = value
            finally:
                widget.observe(self.update, "value")

    def update(self, event):
        # A widget callback that raises is swallowed by ipywidgets, so a fault here
        # would silently stop the viewer from responding at all.
        try:
            for key in self.state_keys:
                self.state[key] = self.widgets[key].value

            self.render()
        except Exception as e:
            self._print(str(e))

    def live(self, event=None):
        if self.parent.is_slm:
            raise ValueError("Live viewing is not supported for SLMs.")

        state = self.state["live"] = self.widgets["live"].value
        self.widgets["live"].button_style = "success" if state else ""

        loop = asyncio.get_running_loop()

        if self.task is not None:
            try:
                self.task.cancel()
            except Exception:
                pass

        if not state:
            self.task = None
        else:
            self.task = loop.create_task(self.live_loop())

    async def live_loop(self):
        while self.state["live"]:
            self.parent.get_image()     # SLMs are not allowed to have gotten here.
            await asyncio.sleep(0.01)

    def _to_source(self, fx, fy):
        """Map a fraction of the view to ``(sx, sy)`` source-image pixels."""
        x0, y0, x1, y1 = self.state["roi"]
        return x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)

    def _grab(self, fx, fy):
        """Start a click-drag pan at a fraction of the view."""
        self._drag = {"pointer": self._to_source(fx, fy), "roi": list(self.state["roi"])}
        self._dragged = False

    def _release(self):
        """End a click-drag pan. Whether it moved is left for the click to read."""
        self._drag = None

    def _zoom(self, fx, fy, inward):
        """Rescale the ROI about a fraction of the view, zooming in if ``inward``."""
        H, W = self._source_shape()
        x0, y0, x1, y1 = self.state["roi"]
        w, h = x1 - x0, y1 - y0
        sx, sy = self._to_source(fx, fy)

        # Width alone carries the zoom and height follows the image aspect, so the
        # region never letterboxes; the clamp keeps its shorter side above 8 px.
        factor = 0.8 if inward else 1.25
        w = float(np.clip(w * factor, 8 * max(1., W / H), W))
        h = w * H / W

        x0 = float(np.clip(sx - fx * w, 0, W - w))
        y0 = float(np.clip(sy - fy * h, 0, H - h))
        self.state["roi"] = [x0, y0, x0 + w, y0 + h]

    def _pan(self, fx, fy):
        """Translate the ROI to keep the grabbed pixel under the cursor. True if it moved."""
        if self._drag is None:
            return False

        x0d, y0d, x1d, y1d = self._drag["roi"]
        gx, gy = self._drag["pointer"]
        w, h = x1d - x0d, y1d - y0d

        H, W = self._source_shape()
        x0 = float(np.clip(gx - fx * w, 0, W - w))
        y0 = float(np.clip(gy - fy * h, 0, H - h))

        roi = [x0, y0, x0 + w, y0 + h]
        if roi == self.state["roi"]:
            return False

        self._dragged = True
        self.state["roi"] = roi
        return True

    def _source_shape(self):
        """``(H, W)`` of the image the ROI indexes, which is what :meth:`parse` crops."""
        if self.last_image is None:
            return self.parent.shape[0], self.parent.shape[1]
        return self.last_image.shape[0], self.last_image.shape[1]

    def _reset_roi(self):
        """Reset the ROI to show the full image."""
        H, W = self._source_shape()
        self.state["roi"] = [0.0, 0.0, float(W), float(H)]
        self._drag = None

    def save(self, event=None):
        """Save the view as a ``.png``, or the raw data and metadata as an ``.h5``."""
        try:
            file_path = _save_dialog(
                generate_path(".", self.parent.name + "-view", extension="png")
            )
            if file_path:
                if os.path.splitext(file_path)[1] == ".h5":
                    save_h5(file_path, self.parent.pickle())
                else:
                    with open(file_path, "wb") as f:
                        f.write(self.display.png())
                self._print(file_path)
        except Exception as e:
            self._print(str(e))

    def copy(self, event=None):
        """Copy the current view to the system clipboard."""
        try:
            _clipboard_image(self.display.png(), self.parent.name + "-view")
            self._print("Copied to clipboard.")
        except Exception as e:
            self._print(str(e))

    def _autorange(self):
        """Window the color scale to the extremes of the last image."""
        if "range" in self.widgets:
            self.widgets["range"].max = self.parent.bitresolution - 1

        if self.last_image is not None:
            # Both the slider and the image are in raw counts.
            range = np.array([np.min(self.last_image), np.max(self.last_image)])
            range = np.clip(np.rint(range), 0, self.parent.bitresolution - 1).astype(int)
            self._set("range", [int(r) for r in range])

    def autorange(self, event):
        self._autorange()
        self.render()

    def _row(self, *keys):
        """A row of the named widgets, skipping any that this backend does not provide."""
        from ipywidgets import HBox
        return HBox([self.widgets[key] for key in keys if key in self.widgets])

    def init_widgets(self):
        from ipywidgets import (
            HTML,
            Button,
            Checkbox,
            Dropdown,
            FloatLogSlider,
            IntRangeSlider,
            Layout,
            ToggleButton,
        )

        item_layout = Layout(width="auto")
        grow_layout = Layout(width="auto", flex="1 1 auto")  # Absorbs the leftover row width.

        self.widgets = {
            "name" : HTML(
                value=f"<b>{self.parent.name}</b>",
                description="Viewing",
                tooltip="Name of the hardware.",
                layout=item_layout,
            ),
            "cmap" : Dropdown(
                options=self.state["cmap_options"],
                value=self.state["cmap"],
                description="Colormap",
                tooltip="Choose the colormap to use for display.",
                layout=item_layout,
            ),
            "save" : Button(
                description="Save",
                tooltip="Save the view as a .png, or the raw data as an .h5.",
                layout=(Layout(width="100px") if self.parent.is_slm else Layout(width="50%")),
            ),
            "copy" : Button(
                description="Copy",
                tooltip="Copy the current view to the system clipboard.",
                layout=(Layout(width="100px") if self.parent.is_slm else Layout(width="50%")),
            ),
            # An Output widget would be the natural home for these messages, but it
            # does not reliably render inside a container in every frontend.
            "output": HTML(
                value="",
                tooltip="Clicked coordinates, saved file paths, and viewer errors.",
            )
        }

        self.state_keys = ["cmap"]

        self.widgets.update({
            "scale" : FloatLogSlider(
                value=self.state["scale"],
                base=2,
                min=-3, # 12.5%
                max=3,  # 800%
                step=1,
                description="Scale",
                tooltip="Scale the view by powers of two.",
                layout=Layout(width="300px"),
                continuous_update=False,
            ),
            "zoom" : Checkbox(
                value=self.state["zoom"],
                description="Zoom",
                tooltip=(
                    "Enable scroll-wheel zoom and click-drag pan; double-click restores "
                    "the full image. Disable to see the whole image again."
                ),
                layout=item_layout,
                indent=False,   # Else a description-width gutter pads the box.
            ),
        })
        self.state_keys += ["scale", "zoom"]

        # Extra widgets for cameras, not relevant for SLMs.
        if not self.parent.is_slm:
            self.widgets.update({
                "live" : ToggleButton(
                    value=self.state["live"],
                    description="Live",
                    tooltip="Toggle an asyncio loop to poll images from the hardware.",
                    layout=item_layout,
                    button_style=("success" if self.state["live"] else ""),
                    disabled=self.parent.is_slm
                ),
                "range" : IntRangeSlider(
                    value=self.state["range"],
                    min=0,
                    max=self.parent.bitresolution-1,
                    step=1,
                    description="Range",
                    tooltip="Color scale of the plot.",
                    layout=grow_layout,
                ),
                "autorange" : Button(
                    description="AutoRange",
                    tooltip="Scale the plot to the minimum and maximum of the current image.",
                    layout=item_layout,
                ),
                "log" : Checkbox(
                    value=self.state["log"],
                    description="Logarithmic",
                    tooltip="Toggle logarithmic scaling of the current plot.",
                    layout=item_layout,
                    indent=False,
                ),
                "crosshair" : Dropdown(
                    options=_CROSSHAIR_OPTIONS,
                    value=self.state["crosshair"],
                    description="Crosshairs",
                    tooltip=(
                        "Overlay a solid crosshair on the center of the view and/or a dashed "
                        "crosshair on the median-subtracted centroid (center of mass) of the view."
                    ),
                    layout=item_layout,
                ),
            })
            self.state_keys += ["live", "range", "log", "crosshair"]

        for k, w in self.widgets.items():
            if k == "autorange":
                w.on_click(self.autorange)
            elif k == "save":
                w.on_click(self.save)
            elif k == "copy":
                w.on_click(self.copy)
            elif k == "live":
                w.observe(self.live, "value")
            # Only state-bearing widgets redraw; "zoom" is the display's to observe.
            elif k in self.state_keys and k != "zoom":
                w.observe(self.update, "value")

        from IPython.display import display
        from ipywidgets import HBox, VBox

        if self.parent.is_slm:
            self.widgets["layout"] = VBox([
                self._row("name", "cmap", "scale", "zoom", "save", "copy"),
            ])
        else:
            # The controls on the right are sized to their text; the rest of the width
            # goes to the left, where the sliders need the room to be usable.
            box_layout1 = Layout(
                display="flex",
                flex_flow="auto",
                align_items="stretch",
                flex="1 1 auto",
            )
            box_layout2 = Layout(
                display="flex",
                flex_flow="auto",
                align_items="stretch",
                flex="0 0 auto",
                width="200px",
            )

            self.widgets["layout"] = HBox([
                VBox(
                    [
                        self._row("name", "scale", "zoom"),
                        self._row("cmap", "log", "crosshair"),
                        self._row("range"),
                    ],
                    layout=box_layout1,
                ),
                VBox(
                    [
                        self.widgets["live"],
                        self._row("save", "copy"),
                        self.widgets["autorange"],
                    ],
                    layout=box_layout2,
                )
            ])

        display(self.widgets["layout"])

    def close(self):
        self.closed = True
        try:
            self.task.cancel()
            self.task = None
        except Exception:
            pass

        self.display.close()
        self.parent._viewer_memory = {
            key: self.state[key] for key in _REMEMBERED if self.state.get(key) is not None
        }
        for w in self.widgets.values():
            w.close()


class _ViewerDisplayIPython:
    """
    Draws the view as a palette ``PNG`` in an :mod:`ipywidgets` image, inside the notebook.

    Mouse interaction arrives as :mod:`ipyevents` DOM events, which are converted into the
    fractional view coordinates that :class:`_ViewerObject` does its region math in.
    """

    def __init__(self, viewer):
        from IPython.display import HTML, display
        from ipywidgets import Image

        self.viewer = viewer
        self._events = None

        self.image = Image(value=b"", format="png")
        # Render the image with nearest-neighbor upscaling so that, when zoomed in,
        # individual source pixels appear as crisp blocks rather than a blurred,
        # smoothly-interpolated patch.
        self.image.add_class("slmsuite-viewer-pixelated")
        display(HTML(
            "<style>.slmsuite-viewer-pixelated {"
            " image-rendering: -moz-crisp-edges;"
            " image-rendering: crisp-edges;"
            " image-rendering: pixelated;"
            " }</style>"
        ))
        self.render()
        self._attach_events()
        display(self.image)

    def render(self):
        """Size the box to the hardware, then encode the cropped view for the frontend."""
        # A fixed box size means ROI zoom changes the content rather than the layout.
        H, W = self.viewer._source_shape()
        scale = self.viewer.state["scale"]
        self.image.layout.width = f"{int(W * scale)}px"
        self.image.layout.height = f"{int(H * scale)}px"

        self.image.value = _png(
            self.viewer.parse(self.viewer.state["roi"]), self.viewer._palette()
        )

    def png(self):
        """The current view as ``PNG`` bytes."""
        return self.image.value

    def _attach_events(self):
        """Attach ipyevents mouse handlers to the image widget, if available."""
        if "zoom" in self.viewer.widgets:
            self.viewer.widgets["zoom"].observe(self._set_zoom, "value")

        try:
            from ipyevents import Event
        except ImportError:
            self.viewer._print(
                "Install 'ipyevents' (pip install ipyevents) to enable "
                "scroll-wheel zoom and click-drag pan in the viewer."
            )
            return

        # Events are throttled to the rate at which frames can be drawn, else a drag
        # queues up in the kernel and falls further behind the cursor as it lasts.
        self._events = Event(
            source=self.image,
            prevent_default_action=True,
            throttle_or_debounce="throttle",
            wait=int(1e3 * _RENDER_PERIOD_S),
        )
        self._events.on_dom_event(self._on_dom_event)
        self._set_zoom()

    def _set_zoom(self, event=None):
        """Enable or disable mouse zoom and pan, restoring the full image."""
        viewer = self.viewer
        if "zoom" in viewer.widgets:
            viewer.state["zoom"] = viewer.widgets["zoom"].value
        if self._events is not None:
            self._events.watched_events = (
                _MOUSE_EVENTS + ["wheel"] if viewer.state["zoom"] else list(_MOUSE_EVENTS)
            )
        viewer._reset_roi()
        self.render()

    def _on_dom_event(self, event):
        """Dispatch ipyevents DOM events for scroll-zoom, drag-pan, and coordinate readout."""
        viewer = self.viewer
        try:
            etype = event.get("type")
            zoom = viewer.state["zoom"]
            fx = event["relativeX"] / event["boundingRectWidth"]
            fy = event["relativeY"] / event["boundingRectHeight"]

            if zoom and etype == "wheel":
                # Wheel up (deltaY < 0) zooms in; wheel down zooms out.
                if event.get("deltaY", 0):
                    viewer._zoom(fx, fy, event["deltaY"] < 0)
                    self.render()
            elif zoom and etype == "mousedown":
                viewer._grab(fx, fy)
            elif etype == "mousemove":
                if viewer._pan(fx, fy):
                    self.render()
            elif etype in ("mouseup", "mouseleave"):
                viewer._release()
            elif zoom and etype == "dblclick":
                viewer._reset_roi()
                self.render()
            elif etype == "click" and not viewer._dragged:
                viewer._print(np.round(viewer._to_source(fx, fy)).astype(int))
        except Exception as e:
            viewer._print(str(e))

    def close(self):
        if self._events is not None:
            self._events.close()
        self.image.close()


class _ViewerDisplayPyglet:
    """
    Draws the view into a resizable :class:`~slmsuite.hardware._pyglet._ViewerWindow`.

    The whole image is uploaded to the window's texture and the region of interest is a
    texture-coordinate rectangle, so zoom and pan are drawn by the window's own thread
    without any work here. A frame waits for its predecessor to reach the screen, but
    only briefly: a window held in an OS modal loop is skipped rather than allowed to
    throttle the hardware behind it.
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self.thread = None
        self._future = None
        self._index = None
        self._planes = None
        self._scale = None

        if "zoom" in viewer.widgets:
            viewer.widgets["zoom"].observe(self._set_zoom, "value")

        self.render()

    def _size(self):
        """Window ``(height, width)`` at the current scale, in the image's aspect ratio."""
        H, W = self._index.shape

        # One scale on both axes, never so small that the window cannot be grabbed.
        scale = max(self.viewer.state["scale"], 64. / W, 64. / H)

        return round(H * scale), round(W * scale)

    def _open(self):
        """Create the window and its thread, sized to the current image."""
        from slmsuite.hardware._pyglet import (
            _ViewerWindow,
            _WindowManager,
            get_pyglet_display,
        )

        H, W = self._index.shape

        # Open no larger than the desktop, in the slider's powers of two, and show that
        # on the slider rather than silently capping every larger value it can reach.
        screen = get_pyglet_display().get_default_screen()
        fit = min(.8 * screen.width / W, .8 * screen.height / H)
        if self.viewer.state["scale"] > fit:
            self.viewer._set("scale", 2. ** np.floor(np.log2(fit)))

        self._scale = self.viewer.state["scale"]

        # A viewer closed earlier on this hardware left the frame the user chose.
        geometry = self.viewer.state.get("geometry")
        size = (geometry[3], geometry[2]) if geometry else self._size()

        self.thread = _WindowManager.get_instance().create_window(
            size,
            get_pyglet_display().get_default_screen(),
            self.viewer.parent.name,
            window_class=_ViewerWindow,
            viewer=self.viewer,
            image_shape=(H, W),
            on_close=self._closed,
        )

        if geometry:
            self.thread.submit(self.thread.window.set_location, geometry[0], geometry[1])

    def _closed(self):
        """Tear the viewer down once the window is gone, as toggling :meth:`.live` would."""
        self.viewer._post(("close",))

    def _set_zoom(self, event=None):
        """Follow the Zoom widget, restoring the full image whenever it is switched off."""
        self.viewer.state["zoom"] = self.viewer.widgets["zoom"].value
        if not self.viewer.state["zoom"]:
            self.viewer._reset_roi()
        self.viewer.render()

    def render(self):
        """Color the whole image into the window's frame and hand it to the window thread."""
        if self.thread is not None and not self.thread.running:
            self.viewer.parent.live(activate=False)     # The user closed the window.
            return

        if self._future is not None:
            try:
                self.thread.wait(self._future, timeout=_RENDER_PERIOD_S)
            except TimeoutError:
                return      # The window thread is wedged; skip rather than block on it.
            except Exception:
                self._future = None     # Else the failed frame is re-raised forever.
                raise

        self._index = self.viewer.parse()

        # The first frame, or one that a WOI change has resized.
        opened = self.thread is None or self._index.shape != self.thread.window.shape
        if opened:
            self.close()
            self._open()
        elif self.viewer.state["scale"] != self._scale:
            self._scale = self.viewer.state["scale"]
            height, width = self._size()
            self.thread.submit(self.thread.window.set_size, width, height)

        window = self.thread.window
        if self._planes is None:
            self._planes = [np.empty(self._index.shape, np.uint8) for _ in range(4)]

        rgba = self.viewer._palette()
        frame = window.acquire()
        try:
            for channel, plane in enumerate(self._planes):
                cv2.LUT(self._index, rgba[:, channel].copy(), dst=plane)
            cv2.merge(self._planes, dst=frame)
        finally:
            window.commit()
        self._future = self.thread.submit(window.render)

        if opened:
            # Else a caller that writes once, as an SLM does, never draws the frame.
            self.thread.wait(self._future, timeout=10)

    def png(self):
        """The current view, cropped to the region of interest, as ``PNG`` bytes."""
        return _png(self.viewer.parse(self.viewer.state["roi"]), self.viewer._palette())

    def close(self):
        if self.thread is not None:
            self.thread.close()
            self.thread = None
        self._future = None
        self._planes = None
