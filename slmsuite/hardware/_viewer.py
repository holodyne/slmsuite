
import asyncio
import html
import io
import os
import struct
import sys
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL

from slmsuite.holography.analysis import _center, image_centroids, image_remove_field
from slmsuite.misc.files import generate_path, save_h5

_CROSSHAIR_OPTIONS = [
    ("None", "none"),
    ("Center", "center"),
    ("Centroid", "centroid"),
    ("Center+Centroid", "center+centroid"),
]

# ipyevents always suppresses the page scroll for a watched wheel event, so "wheel"
# is only appended to this list while zoom is enabled.
_MOUSE_EVENTS = ["mousedown", "mousemove", "mouseup", "mouseleave", "click", "dblclick"]

# Mouse events are throttled to this period, roughly the rate at which frames can be
# drawn: a megapixel frame takes tens of milliseconds to encode.
_RENDER_PERIOD_S = .033

# Frames are encoded as 8-bit palette PNGs: the colormap is sent once as a palette
# and each pixel is a single index, instead of four RGBA bytes. This is ~12x faster
# to encode than the equivalent RGBA image at full resolution, and ~20x faster when
# zoomed in, where the cost is otherwise dominated by building a bitresolution-sized
# colormap that does not shrink with the crop. That headroom is what keeps mouse
# zoom and pan tracking the cursor.
_LEVELS = 253       # Indices 0..252 hold image data.
_TRANSPARENT = 253  # Reserved for nan.
_DARK, _LIGHT = 254, 255    # Reserved for crosshairs.
_palette_cache = {}


def _viewer_palette(cmap):
    """
    Cached palette for the display path.

    Returns the 256-entry RGB palette as ``bytes`` alongside a boolean array
    marking which entries are dark, used to draw crosshairs in a contrasting
    color.
    """
    # Colormap objects define __eq__ without __hash__, so they cannot be keys.
    key = cmap if isinstance(cmap, str) else id(cmap)

    if key not in _palette_cache:
        # Match the colormap aliases understood by :meth:`_gray2rgb`.
        if cmap == "default":
            cmap = plt.rcParams["image.cmap"]
        elif cmap == "grayscale":
            cmap = "gray"

        colors = (
            255 * plt.get_cmap(cmap, _LEVELS)(np.linspace(0, 1, _LEVELS))[:, :3]
        ).astype(np.uint8)
        luminance = colors.astype(np.float32) @ np.float32([.299, .587, .114])

        _palette_cache[key] = (
            bytes(np.vstack((colors, (0, 0, 0), (0, 0, 0), (255, 255, 255))).astype(np.uint8).ravel()),
            np.concatenate((luminance < 128, (False, True, False))),  # Black is dark, white is not.
        )

    return _palette_cache[key]


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
        Creates and displays an IPython viewer.
          - When used with a camera, the viewer displays the last image:
            the result of :meth:`get_image()` or the last image of :meth:`get_images()`
            **whenever these methods are called**.
            Averaging and HDR are displayed with the same color scaling as without.
          - When used with an SLM, the viewer displays the phase pattern currently on the
            SLM: the phase pattern passed to :meth:`set_phase()` or
            the last phase pattern passed to :meth:`set_phases()`
            **whenever these methods are called**.

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
        widgets : bool
            If ``True``, also displays sliders and controls used to hone the display properties.
        backend : str
            Placeholder option for different types of viewers.
            The default is ``"ipython"``.
        **kwargs
            Options passed to the :class:`_ViewerObject` to customize the default settings.
            These features will be made less hidden in the future.
            Most things are customizable via these keywords. For instance, the user can pass
            a custom list of colormaps to appear in the widget dropdown as ``cmap_options=``.
        """
        if backend != "ipython":
            raise ValueError(
                f"'{backend}' not recognized; "
                "'ipython' is currently the only supported .live() backend."
            )

        try:
            from IPython.display import display
            from ipywidgets import Image
        except ImportError:
            raise ImportError("jupyter must be installed to use .live().")

        if (self.viewer is None and activate is None) or activate:
            if self.viewer is not None:
                self.viewer.close()

            self.viewer = _ViewerObject(
                self,
                widgets,
                backend,
                **kwargs
            )
        elif self.viewer is not None and (activate is None or not activate):
            self.viewer.close()
            self.viewer = None


class _ViewerObject:
    """
    Hidden class for live viewing enabled by ipython widgets.
    """
    def __init__(
        self,
        parent,
        widgets,
        backend="ipython",
        live=False,
        min=None,
        max=None,
        log=False,
        cmap=True,
        scale=1,
        border=None,
        cmap_options=None,
        center_crosshair=False,
        centroid_crosshair=False,
        zoom=False,
    ):
        self.parent = parent
        self.backend = backend

        # Parse range.
        if min is None:
            min = 0
        if max is None:
            max = self.parent.bitresolution-1
        range_ = [np.min([min, max]), np.max([min, max])]

        # Parse scale
        scale = 2 ** np.round(np.log2(scale))

        # Parse colormap options.
        if cmap_options is None:
            if self.parent.is_slm:
                cmap_options = ["twilight", "twilight_shifted", "gray", "hsv"]
            else:
                cmap_options = [
                    "default", "gray", "Blues", "turbo",
                    'viridis', 'plasma', 'inferno', 'magma', 'cividis'
                ]

        if cmap is True: cmap = cmap_options[0]
        if cmap is False: cmap = "gray"

        # Parse crosshairs into the "+"-delimited form used by the dropdown.
        crosshair = "+".join(
            name for name, enabled
            in (("center", center_crosshair), ("centroid", centroid_crosshair))
            if enabled
        ) or "none"

        self.state = {
            "backend" : backend,
            "live" : live,
            "range" : range_,
            "log" : bool(log),
            "cmap" : cmap,
            "scale" : scale,
            "border" : border,
            "cmap_options" : cmap_options,
            "crosshair" : crosshair,
            "zoom" : bool(zoom),
        }

        self.task = None
        self._drag = None
        self._dragged = False
        self._events = None

        # Region of interest (crop) in source-image pixels: [x0, y0, x1, y1].
        H, W = self.parent.shape[0], self.parent.shape[1]
        self.state["roi"] = [0.0, 0.0, float(W), float(H)]

        self.widgets = {}
        if widgets: self.init_widgets()
        self.init_image()

    def parse(self, img=None):
        is_cam = not self.parent.is_slm

        if img is not None:
            self.last_image = img
        if self.last_image is None:
            return  # Nothing to render.

        # Crop to the current region of interest (ROI).
        H, W = self.last_image.shape[0], self.last_image.shape[1]
        x0, y0, x1, y1 = np.rint(self.state["roi"]).astype(int)
        x0, x1 = np.clip([x0, x1], 0, W)
        y0, y1 = np.clip([y0, y1], 0, H)
        if x1 - x0 < 1: x1 = min(W, x0 + 1)
        if y1 - y0 < 1: y1 = min(H, y0 + 1)
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
            # Single-precision is ample for an 8-bit display and halves the memory
            # traffic of the windowing below.
            img = np.asarray(src, dtype=np.float32)

        crosshairs = self.state["crosshair"].split("+") if is_cam else []

        if "centroid" in crosshairs:
            gray = img if len(img.shape) == 2 else np.mean(img, axis=-1)
            centroid = np.squeeze(image_centroids(image_remove_field([gray], deviations=None)))

        # Window the raw counts to the display range.
        # The scalars are plain floats so that they do not promote img back to double.
        r = [float(v) for v in self.state["range"]]
        d = max(r[1] - r[0], 1.)
        img = np.clip(img, r[0], r[1]) * (1. / d) - r[0] / d

        if is_cam and self.state["log"]:
            img = np.log10(1 + img * d) / np.log10(1 + d)

        # Quantize to palette indices. Windowing leaves img in [0, 1]; clipping
        # keeps float error out of the reserved entries above _LEVELS.
        index = np.clip(img * (_LEVELS - 1) + .5, 0, _LEVELS - 1).astype(np.uint8)
        palette, is_dark = _viewer_palette(self.state["cmap"])

        nan = np.isnan(img)
        if nan.any():
            index[nan] = _TRANSPARENT

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

        buff = io.BytesIO()
        image = PIL.Image.fromarray(index, mode="P")
        image.putpalette(palette)
        image.save(buff, format="png", compress_level=1, transparency=_TRANSPARENT)

        return buff.getvalue()

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
        try:
            self.image.value = self.parse(img)
        except Exception as e:
            self._print(str(e))

    def update(self, event):
        # A widget callback that raises is swallowed by ipywidgets, so a fault here
        # would silently stop the viewer from responding at all.
        try:
            for key in self.state_keys:
                self.state[key] = self.widgets[key].value

            self._resize_display()
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

    def _event_to_source(self, event):
        """Map an ipyevents DOM mouse event to ``(sx, sy)`` source-image pixels."""
        x0, y0, x1, y1 = self.state["roi"]
        fx = event["relativeX"] / event["boundingRectWidth"]
        fy = event["relativeY"] / event["boundingRectHeight"]
        return x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)

    def _zoom(self, event):
        """Scroll-wheel zoom: rescale the ROI about the cursor position."""
        delta = event.get("deltaY", 0)
        if not delta:
            return

        H, W = self.last_image.shape[0], self.last_image.shape[1]
        x0, y0, x1, y1 = self.state["roi"]
        w, h = x1 - x0, y1 - y0
        fx = event["relativeX"] / event["boundingRectWidth"]
        fy = event["relativeY"] / event["boundingRectHeight"]
        sx, sy = x0 + fx * w, y0 + fy * h

        # Wheel up (deltaY < 0) zooms in; wheel down zooms out. The factor is
        # clamped so the ROI stays within [8 px, full image] while preserving
        # the full-image aspect ratio (same factor for both axes).
        factor = 0.8 if delta < 0 else 1.25
        factor = min(factor, W / w, H / h)
        factor = max(factor, 8.0 / w, 8.0 / h)
        # Keep the ROI dimensions integral so the integer-sliced crop is exactly
        # the same size at every pan position (no ±1px breathing).
        w = float(np.clip(round(w * factor), 8, W))
        h = float(np.clip(round(w * H / W), 8, H))

        x0 = float(np.clip(sx - fx * w, 0, W - w))
        y0 = float(np.clip(sy - fy * h, 0, H - h))
        self.state["roi"] = [x0, y0, x0 + w, y0 + h]
        self.render()

    def _pan(self, event):
        """Click-drag pan: translate the ROI to keep the grabbed pixel under the cursor."""
        x0d, y0d, x1d, y1d = self._drag["roi"]
        gx, gy = self._drag["pointer"]
        w, h = x1d - x0d, y1d - y0d

        H, W = self.last_image.shape[0], self.last_image.shape[1]
        fx = event["relativeX"] / event["boundingRectWidth"]
        fy = event["relativeY"] / event["boundingRectHeight"]
        x0 = float(np.clip(gx - fx * w, 0, W - w))
        y0 = float(np.clip(gy - fy * h, 0, H - h))

        roi = [x0, y0, x0 + w, y0 + h]
        if roi != self.state["roi"]:
            self._dragged = True
            self.state["roi"] = roi
            self.render()

    def _reset_roi(self, event=None):
        """Reset the ROI to show the full image."""
        H, W = self.parent.shape[0], self.parent.shape[1]
        self.state["roi"] = [0.0, 0.0, float(W), float(H)]
        self._drag = None
        self.render()

    def _set_zoom(self, event=None):
        """Enable or disable mouse zoom and pan, restoring the full image."""
        if "zoom" in self.widgets:
            self.state["zoom"] = self.widgets["zoom"].value
        if self._events is not None:
            self._events.watched_events = (
                _MOUSE_EVENTS + ["wheel"] if self.state["zoom"] else list(_MOUSE_EVENTS)
            )
        self._reset_roi()

    def _on_dom_event(self, event):
        """Dispatch ipyevents DOM events for scroll-zoom, drag-pan, and coordinate readout."""
        try:
            etype = event.get("type")
            zoom = self.state["zoom"]
            if zoom and etype == "wheel":
                self._zoom(event)
            elif zoom and etype == "mousedown":
                sx, sy = self._event_to_source(event)
                self._drag = {"pointer": (sx, sy), "roi": list(self.state["roi"])}
                self._dragged = False
            elif etype == "mousemove":
                if self._drag is not None:
                    self._pan(event)
            elif etype in ("mouseup", "mouseleave"):
                self._drag = None
            elif zoom and etype == "dblclick":
                self._reset_roi()
            elif etype == "click" and not self._dragged:
                sx, sy = self._event_to_source(event)
                self._print(np.round([sx, sy]).astype(int))
        except Exception as e:
            self._print(str(e))

    def _resize_display(self):
        """Fix the display box size so ROI zoom changes content, not widget size."""
        H, W = self.parent.shape[0], self.parent.shape[1]
        s = self.state["scale"]
        # The existing layout is edited rather than replaced, else every update
        # orphans a widget model in the frontend.
        self.image.layout.width = f"{int(W * s)}px"
        self.image.layout.height = f"{int(H * s)}px"

    def _attach_events(self):
        """Attach ipyevents mouse handlers to the image widget, if available."""
        try:
            from ipyevents import Event
        except ImportError:
            self._print(
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
                        f.write(self.image.value)
                self._print(file_path)
        except Exception as e:
            self._print(str(e))

    def copy(self, event=None):
        """Copy the current view to the system clipboard."""
        try:
            _clipboard_image(self.image.value, self.parent.name + "-view")
            self._print("Copied to clipboard.")
        except Exception as e:
            self._print(str(e))

    def autorange(self, event):
        if self.last_image is not None:
            # Both the slider and the image are in raw counts.
            range = np.array([np.min(self.last_image), np.max(self.last_image)])
            range = np.clip(np.rint(range), 0, self.parent.bitresolution - 1).astype(int)
            self.state["range"] = self.widgets["range"].value = [int(r) for r in range]

        self.render()

    def init_image(self):
        from IPython.display import HTML, display
        from ipywidgets import Image

        if self.parent.is_slm:
            self.last_image = self.parent.phase
        else:
            self.last_image = self.parent.last_image

        self.image = Image(
            value=self.parse(self.last_image),
            format="png"
        )
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
        self._resize_display()
        self._attach_events()
        display(self.image)

        if "output" in self.widgets:
            display(self.widgets["output"])

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
            "scale" : FloatLogSlider(
                value=self.state["scale"],
                base=2,
                min=-3, # 12.5%
                max=3,  # 800%
                step=1,
                description="Scale",
                tooltip="Scale the image by powers of two.",
                layout=Layout(width="300px"),
                continuous_update=False,
            ),
            "zoom" : Checkbox(
                value=self.state["zoom"],
                description="Zoom",
                tooltip=(
                    "Enable scroll-wheel zoom and click-drag pan; double-click restores "
                    "the full image. Disable to scroll the notebook over the viewer."
                ),
                layout=item_layout,
                indent=False,   # Else a description-width gutter pads the box.
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

        self.state_keys = ["cmap", "scale", "zoom"]

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
            elif k == "zoom":
                w.observe(self._set_zoom, "value")
            # Only the state-bearing widgets drive a redraw; the name and message
            # areas carry a value trait of their own that must not trigger one.
            elif k in self.state_keys:
                w.observe(self.update, "value")

        from IPython.display import display
        from ipywidgets import HBox, VBox

        if self.parent.is_slm:
            self.widgets["layout"] = VBox([
                HBox([
                    self.widgets["name"],
                    self.widgets["cmap"],
                    self.widgets["scale"],
                    self.widgets["zoom"],
                    self.widgets["save"],
                    self.widgets["copy"],
                ]),
                # self.widgets["output"],
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
                        HBox([
                            self.widgets["name"],
                            self.widgets["scale"],
                            self.widgets["zoom"],
                        ]),
                        HBox([
                            self.widgets["cmap"],
                            self.widgets["log"],
                            self.widgets["crosshair"],
                        ]),
                        HBox([
                            self.widgets["range"],
                        ]),
                        # self.widgets["output"],
                    ],
                    layout=box_layout1,
                ),
                VBox(
                    [
                        self.widgets["live"],
                        HBox([self.widgets["save"], self.widgets["copy"]]),
                        self.widgets["autorange"],
                    ],
                    layout=box_layout2,
                )
            ])

        display(self.widgets["layout"])

    def close(self):
        try:
            self.task.cancel()
            self.task = None
        except Exception:
            pass

        for w in self.widgets.values():
            w.close()
        if self._events is not None:
            self._events.close()
        self.image.close()
