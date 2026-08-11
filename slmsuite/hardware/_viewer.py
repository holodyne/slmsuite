
import asyncio
import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL

from slmsuite.holography.analysis import _center, image_centroids, image_remove_field

_CROSSHAIR_OPTIONS = [
    ("None", "none"),
    ("Center", "center"),
    ("Centroid", "centroid"),
    ("Center+Centroid", "center+centroid"),
]

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
        for controlling the color scale, colormap, viewer scale, and live viewing.
        By toggling the ``Live`` widget button, a loop is created that continuously
        polls the camera for new images.
        This viewer can be used as a realtime camera monitor within the jupyter notebook.
        However, note that any user-execution will block the monitoring loop.
        Regardless, any image polling during the blocked period will still update the viewer,
        which provides useful active feedback for what is happening during the
        execution.
        ``Live`` mode is ignored for SLMs.

        The viewer also supports zooming into a region of interest: scroll the mouse
        wheel to zoom in/out toward the cursor, click-drag to pan, and double-click
        (or press the ``Reset View`` button) to restore the full image. Clicking
        prints the source-image pixel coordinate under the cursor. These mouse
        interactions require the optional :mod:`ipyevents` package
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
            if hasattr(img, "get"):
                self.last_image = img.get()
            else:
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

    def render(self, img=None):
        try:
            self.image.value = self.parse(img)
        except Exception as e:
            # Only the latest error is kept. A fault that recurs on every mouse
            # event would otherwise grow the output widget without bound, which
            # slows the whole notebook frontend long after the drag has ended.
            if "output" in self.widgets:
                with self.widgets["output"]:
                    self.widgets["output"].clear_output(wait=True)
                    print(str(e))
            else:
                print("slmsuite viewer render error:", str(e))

    def update(self, event):
        with self.widgets["output"]:
            self.widgets["output"].clear_output(wait=True)
        for key in self.state_keys:
            self.state[key] = self.widgets[key].value

        self._resize_display()
        self.render()

    def live(self, event=None):
        if self.parent.is_slm:
            raise ValueError("Live viewing is not supported for SLMs.")

        state = self.state["live"] = self.widgets["live"].value

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
        H, W = self.last_image.shape[0], self.last_image.shape[1]
        self.state["roi"] = [0.0, 0.0, float(W), float(H)]
        self._drag = None
        self.render()

    def _on_dom_event(self, event):
        """Dispatch ipyevents DOM events for scroll-zoom, drag-pan, and coordinate readout."""
        try:
            etype = event.get("type")
            if etype == "wheel":
                self._zoom(event)
            elif etype == "mousedown":
                sx, sy = self._event_to_source(event)
                self._drag = {"pointer": (sx, sy), "roi": list(self.state["roi"])}
                self._dragged = False
            elif etype == "mousemove":
                if self._drag is not None:
                    self._pan(event)
            elif etype in ("mouseup", "mouseleave"):
                self._drag = None
            elif etype == "dblclick":
                self._reset_roi()
            elif etype == "click" and not self._dragged:
                sx, sy = self._event_to_source(event)
                coord = np.round([sx, sy]).astype(int)
                if "output" in self.widgets:
                    with self.widgets["output"]:
                        self.widgets["output"].clear_output(wait=True)
                        print(coord)
        except Exception as e:
            if "output" in self.widgets:
                with self.widgets["output"]:
                    self.widgets["output"].clear_output(wait=True)
                    print(str(e))

    def _resize_display(self):
        """Fix the display box size so ROI zoom changes content, not widget size."""
        from ipywidgets import Layout
        H, W = self.parent.shape[0], self.parent.shape[1]
        s = self.state["scale"]
        self.image.layout = Layout(
            width=f"{int(W * s)}px",
            height=f"{int(H * s)}px",
        )

    def _attach_events(self):
        """Attach ipyevents mouse handlers to the image widget, if available."""
        try:
            from ipyevents import Event
        except ImportError:
            if "output" in self.widgets:
                with self.widgets["output"]:
                    print(
                        "Install 'ipyevents' (pip install ipyevents) to enable "
                        "scroll-wheel zoom and click-drag pan in the viewer."
                    )
            return

        # Events are throttled to roughly the rate at which frames can be drawn.
        # Faster than that and mousemove events queue up in the kernel, so a drag
        # falls further and further behind the cursor the longer it lasts.
        self._events = Event(
            source=self.image,
            watched_events=[
                "wheel", "mousedown", "mousemove", "mouseup",
                "mouseleave", "click", "dblclick",
            ],
            prevent_default_action=True,
            throttle_or_debounce="throttle",
            wait=33,
        )
        self._events.on_dom_event(self._on_dom_event)

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

    def init_widgets(self):
        from ipywidgets import (
            HTML,
            Button,
            Checkbox,
            Dropdown,
            FloatLogSlider,
            IntRangeSlider,
            Layout,
            Output,
            ToggleButton,
        )

        item_layout = Layout(width="auto")
        range_layout = Layout(width="70%")

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
                layout=(Layout(width="30%") if self.parent.is_slm else item_layout),
                continuous_update=False,
            ),
            "reset" : Button(
                description="Reset View",
                tooltip="Reset zoom and pan to show the full image.",
                layout=item_layout,
            ),
            "output": Output()
        }

        self.state_keys = ["cmap", "scale"]

        # Extra widgets for cameras, not relevant for SLMs.
        if not self.parent.is_slm:
            self.widgets.update({
                "live" : ToggleButton(
                    value=self.state["live"],
                    description="Live",
                    tooltip="Toggle an asyncio loop to poll images from the hardware.",
                    layout=item_layout,
                    disabled=self.parent.is_slm
                ),
                "range" : IntRangeSlider(
                    value=self.state["range"],
                    min=0,
                    max=self.parent.bitresolution-1,
                    step=1,
                    description="Range",
                    tooltip="Color scale of the plot.",
                    layout=range_layout,
                    continuous_update=False,
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
            elif k == "reset":
                w.on_click(self._reset_roi)
            elif k == "live":
                w.observe(self.live, "value")
            else:
                w.observe(self.update, "value")

        from IPython.display import display
        from ipywidgets import HBox, VBox

        if self.parent.is_slm:
            self.widgets["layout"] = VBox([
                HBox([
                    self.widgets["name"],
                    self.widgets["cmap"],
                    self.widgets["scale"],
                    self.widgets["reset"],
                ]),
                self.widgets["output"],
            ])
        else:
            box_layout1 = Layout(
                display="flex",
                flex_flow="auto",
                align_items="stretch",
                width="70%"
            )
            box_layout2 = Layout(
                display="flex",
                flex_flow="auto",
                align_items="stretch",
                width="30%"
            )

            self.widgets["layout"] = HBox([
                VBox(
                    [
                        HBox([
                            self.widgets["name"],
                        ]),
                        HBox([
                            self.widgets["cmap"],
                            self.widgets["log"],
                            self.widgets["crosshair"],
                        ]),
                        HBox([
                            self.widgets["range"],
                        ]),
                        self.widgets["output"],
                    ],
                    layout=box_layout1,
                ),
                VBox(
                    [
                        self.widgets["live"],
                        self.widgets["scale"],
                        self.widgets["autorange"],
                        self.widgets["reset"],
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
