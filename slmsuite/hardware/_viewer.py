
import asyncio
import numpy as np
from scipy.ndimage import zoom
import PIL
import io
from slmsuite.holography.analysis import image_centroids, image_remove_field
from slmsuite.holography.analysis.files import _gray2rgb

class _Viewable(object):

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

        This limitation is imposed by the
        Python Global Interpreter Lock (GIL) which restricts operation to a single thread,
        especially operation connecting to a diverse set of camera and SLM hardware.
        We use :mod:`asyncio` to allow the realtime monitoring loop to be
        interrupted by user-execution (e.g. running a cell in jupyter),
        blocking until the execution is finished.

        Running multiple viewers live at once might not play nicely right now.

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
            from ipywidgets import Image
            from IPython.display import display
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


class _ViewerObject(object):
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
            crosshair=False,
            centroid=False,
        ):
        self.parent = parent
        self.backend = backend

        # Parse range.
        if min is None:
            min = 0
        if max is None:
            max = self.parent.bitresolution-1
        range = [min, max]
        range = [np.min(range), np.max(range)]

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

        self.state = {
            "backend" : backend,
            "live" : live,
            "range" : range,
            "log" : bool(log),
            "cmap" : cmap,
            "scale" : scale,
            "border" : border,
            "cmap_options" : cmap_options,
            "center_crosshair" : crosshair,
            "centroid_crosshair" : centroid,
        }

        self.task = None
        self.widgets = {}
        if widgets: self.init_widgets()
        self.init_image()

    def parse(self, img=None):
        is_cam = not self.parent.is_slm

        if img is not None:
            self.prev_img = img
        if self.prev_img is None:
            return  # Nothing to render.

        # Downscaling can happen before intensive operations.
        if self.state["scale"] < 1:
            img = zoom(
                self.prev_img,
                [self.state["scale"], self.state["scale"]] + ([1] if len(self.prev_img.shape) == 3 else []),
                order=1
            )
        else:
            img = np.copy(self.prev_img)

        if is_cam and self.state["centroid_crosshair"]:
            img_median_subtract = image_remove_field([img], deviations=None)
            cx, cy = np.rint(
                (
                    np.squeeze(image_centroids(img_median_subtract)) + np.flip(img.shape) / 2
                ) * (self.state["scale"] if self.state["scale"] > 1 else 1)
            ).astype(int)

        # Scale intensity of image
        r = np.array(self.state["range"]).astype(img.dtype)
        img = np.clip(img, r[0], r[1])
        img -= r[0]
        d = r[1] - r[0]

        if is_cam and self.state["log"]:
            # clip to avoid log(0)
            img = (np.log10(np.clip(img, 1, np.inf)) / np.log10(d+1))

        # Make image color
        rgb = _gray2rgb(
            img,
            cmap=self.state["cmap"],
            lut=d,
            normalize=False,
            border=self.state["border"]
        )

        # Upscaling can happen after intensive operations.
        if self.state["scale"] > 1:
            rgb = zoom(rgb, (1, self.state["scale"], self.state["scale"], 1), order=0)

        # Add crosshair at the median-subtracted centroid (center of mass) position.
        if is_cam and self.state["centroid_crosshair"]:
            rgb[:, :, cx, :3] = 255 - rgb[:, :, cx, :3]
            rgb[:, cy, :, :3] = 255 - rgb[:, cy, :, :3]

        # Finally, add crosshair in the center.
        if is_cam and self.state["center_crosshair"]:
            rgb[:, :, int(rgb.shape[2]/2), :3] = 127 - rgb[:, :, int(rgb.shape[2]/2), :3]
            rgb[:, int(rgb.shape[1]/2), :, :3] = 127 - rgb[:, int(rgb.shape[1]/2), :, :3]

        buff = io.BytesIO()
        rgb = PIL.Image.fromarray(rgb[0])
        rgb.save(buff, format="png")

        return buff.getvalue()

    def render(self, img=None):
        try:
            self.image.value = self.parse(img)
        except Exception as e:
            with self.widgets["output"]:
                print(str(e))

    def update(self, event):
        with self.widgets["output"]:
            self.widgets["output"].clear_output(wait=True)
        for key in self.state_keys:
            self.state[key] = self.widgets[key].value

        self.render()

    def live(self, event=None):
        if self.parent.is_slm:
            raise ValueError("Live viewing is not supported for SLMs.")

        state = self.state["live"] = self.widgets["live"].value

        loop = asyncio.get_running_loop()

        if self.task is not None:
            try:
                self.task.cancel()
            except:
                pass

        if not state:
            self.task = None
        else:
            self.task = loop.create_task(self.live_loop())

    async def live_loop(self):
        while self.state["live"]:
            self.parent.get_image()     # SLMs are not allowed to have gotten here.
            await asyncio.sleep(0.01)

    def on_click(self, event):
        coord = np.array([event["x"], event["y"]])
        with self.widgets["output"]:
            self.widgets["output"].clear_output(wait=True)
            print(np.round(coord / self.state["scale"]).astype(int))

    def autorange(self, event):
        if self.prev_img is not None:
            range = [np.min(self.prev_img), np.max(self.prev_img)]
            self.state["range"] = self.widgets["range"].value = range

        self.render()

    def init_image(self):
        from ipywidgets import Image
        from IPython.display import display

        self.prev_img = np.zeros(self.parent.shape, dtype=self.parent.dtype)

        self.image = Image(
            value=self.prev_img,
            format="png"
        )
        self.image.on_click = self.on_click
        display(self.image)

    def init_widgets(self):
        from ipywidgets import HTML, IntRangeSlider, ToggleButton, Button, Checkbox, Dropdown, FloatLogSlider, Output, Layout

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
                "center_crosshair" : Checkbox(
                    value=self.state["center_crosshair"],
                    description="Center Crosshair",
                    tooltip="Toggle a crosshair centered on the image.",
                    layout=item_layout,
                ),
                "centroid_crosshair" : Checkbox(
                    value=self.state["centroid_crosshair"],
                    description="Centroid Crosshair",
                    tooltip="Toggle a crosshair at the median-subtracted centroid (center of mass) of the image.",
                    layout=item_layout,
                ),
            })
            self.state_keys += ["live", "range", "log", "center_crosshair", "centroid_crosshair"]

        for k, w in self.widgets.items():
            if k == "autorange":
                w.on_click(self.autorange)
            elif k == "live":
                w.observe(self.live, "value")
            else:
                w.observe(self.update, "value")

        from ipywidgets import HBox, VBox
        from IPython.display import display

        if self.parent.is_slm:
            self.widgets["layout"] = VBox([
                HBox([
                    self.widgets["name"],
                    self.widgets["cmap"],
                    self.widgets["scale"],
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
                            self.widgets["center_crosshair"],
                            self.widgets["centroid_crosshair"],
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
                    ],
                    layout=box_layout2,
                )
            ])

        display(self.widgets["layout"])

    def close(self):
        try:
            self.task.cancel()
            self.task = None
        except:
            pass

        for w in self.widgets.values():
            w.close()
        self.image.close()

        del self
