"""
Unit tests for the live viewer, its region-of-interest math, and its display backends.
"""
import threading
import time

import numpy as np
import pytest

from slmsuite.hardware._viewer import _ViewerObject
from slmsuite.hardware.cameras.simulated import SimulatedCamera

try:
    from slmsuite.hardware._pyglet import get_pyglet_display
    get_pyglet_display().get_default_screen()
    HAS_DISPLAY = True
except Exception:
    HAS_DISPLAY = False

needs_display = pytest.mark.skipif(not HAS_DISPLAY, reason="No display to open a window on.")


@pytest.fixture
def viewer(camera_small):
    """A widget-free viewer drawing into the notebook display."""
    pytest.importorskip("ipywidgets")
    view = _ViewerObject(camera_small, None, "ipython")
    yield view
    view.close()


@pytest.fixture
def widget_viewer(camera_small):
    """A viewer with the full ipywidgets control set attached."""
    pytest.importorskip("ipywidgets")
    view = _ViewerObject(camera_small, "ipython", "ipython")
    yield view
    view.close()


@pytest.fixture
def oblong(slm_small):
    """A camera that is not square, so aspect and letterbox assertions can bite."""
    cam = SimulatedCamera(slm_small, resolution=(96, 64), bitdepth=8)
    yield cam
    cam.close()


@pytest.fixture
def window(camera_small):
    """The window thread behind a camera's pyglet viewer."""
    camera_small.live(backend="pyglet")
    yield camera_small.viewer.display.thread
    camera_small.live(activate=False)


def count_draws(viewer):
    """Record one entry per frame the display actually draws."""
    drawn = []
    render = viewer.display.render

    def counted():
        drawn.append(np.max(viewer.last_image))
        render()

    viewer.display.render = counted
    return drawn


class TestViewer:
    """Backend selection, coloring, and region math, none of which need a window."""

    def test_live(self, camera_small, subtests):
        """.live() names its backends and refuses the ones it cannot draw into."""
        with subtests.test("an unknown display backend is rejected"):
            with pytest.raises(ValueError, match="not recognized"):
                camera_small.live(backend="qt")

        with subtests.test("an unknown widget backend is rejected"):
            with pytest.raises(ValueError, match="not recognized"):
                camera_small.live(widgets="qt")

        with subtests.test("a rejected backend attaches nothing"):
            assert camera_small.viewer is None

        with subtests.test("the notebook backend needs a kernel to draw into"):
            with pytest.raises(ImportError, match="pyglet"):
                camera_small.live(backend="ipython")

    def test_parse(self, viewer, subtests):
        """A cropped view is the same pixels as the corresponding slice of the full view."""
        H, W = viewer.parent.shape[0], viewer.parent.shape[1]
        viewer.state["range"] = [0, 255]
        img = (np.arange(H * W).reshape(H, W) % 256).astype(np.uint8)
        viewer.last_image = img

        full = viewer.parse()
        assert full.shape == (H, W) and full.dtype == np.uint8

        with subtests.test("a region is a slice of the whole"):
            region = [W // 8, H // 4, W // 2, 3 * H // 4]
            crop = viewer.parse(region)
            assert np.array_equal(crop, full[region[1]:region[3], region[0]:region[2]])

        with subtests.test("a region larger than the display box is downsampled"):
            viewer.state["scale"] = .25
            assert viewer.parse([0., 0., float(W), float(H)]).shape == (H // 4, W // 4)
            viewer.state["scale"] = 1

        with subtests.test("the integer lookup table matches the float pipeline"):
            for log in (False, True):
                viewer.state["log"] = log
                assert np.array_equal(viewer.parse(), viewer._quantize(img))
            viewer.state["log"] = False

        with subtests.test("the lookup table follows a change of range"):
            viewer.state["range"] = [10, 60]
            assert np.array_equal(viewer.parse(), viewer._quantize(img))
            viewer.state["range"] = [0, 255]

        with subtests.test("nan takes the reserved transparent index"):
            nan = img.astype(np.float32)
            nan[0, 0] = np.nan
            viewer.last_image = nan
            index = viewer.parse()
            rgba = viewer._palette()
            assert rgba[index[0, 0]][3] == 0
            viewer.last_image = img
            assert (np.take(rgba, viewer.parse(), axis=0)[..., 3] == 255).all()

        with subtests.test("crosshairs take the reserved contrast indices"):
            viewer.state["crosshair"] = "center+centroid"
            assert (viewer.parse() >= 254).any()

    def test_render(self, viewer, subtests):
        """Each render draws once and keeps its data, whatever else is pending."""
        drawn = count_draws(viewer)
        img = np.full(viewer.parent.shape, 30, np.uint8)

        with subtests.test("a plain render draws once"):
            viewer.render(img)
            assert drawn == [30]

        with subtests.test("a pending request does not cost the caller's frame"):
            viewer._post(("log",))
            viewer.render(np.full(viewer.parent.shape, 40, np.uint8))
            assert drawn == [30, 40], "the frame was drawn twice, or drawn stale"
            assert viewer.state["log"] is True
            assert np.max(viewer.last_image) == 40
            viewer.state["log"] = False

    def test_zoom(self, viewer, oblong, subtests):
        """Scroll-zoom keeps the region inside the image, at aspect, and reversible."""
        for parent in (viewer.parent, oblong):
            viewer.parent = parent
            H, W = parent.shape[0], parent.shape[1]
            viewer.last_image = np.zeros(parent.shape, np.uint8)
            viewer._reset_roi()
            full = [0., 0., float(W), float(H)]

            with subtests.test(f"{W}x{H}: zooming in and back out round-trips"):
                viewer._zoom(.5, .5, True)
                assert viewer.state["roi"] != full
                viewer._zoom(.5, .5, False)
                assert viewer.state["roi"] == full

            with subtests.test(f"{W}x{H}: every depth stays inside, at 8 px or more"):
                for _ in range(40):
                    viewer._zoom(.31, .77, True)
                    x0, y0, x1, y1 = viewer.state["roi"]
                    assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
                    assert x1 - x0 >= 8 and y1 - y0 >= 8
                    assert (x1 - x0) / (y1 - y0) == pytest.approx(W / H, rel=.15)

            with subtests.test(f"{W}x{H}: zooming out clamps at the full image"):
                for _ in range(40):
                    viewer._zoom(.31, .77, False)
                assert viewer.state["roi"] == full

    def test_pan(self, viewer, subtests):
        """Click-drag translates the region without resizing it or leaving the image."""
        H, W = viewer.parent.shape[0], viewer.parent.shape[1]
        viewer._zoom(.5, .5, True)
        zoomed = list(viewer.state["roi"])

        viewer._grab(.5, .5)
        with subtests.test("a pan back to the grabbed point is not a move"):
            assert viewer._pan(.5, .5) is False

        with subtests.test("a pan translates only"):
            assert viewer._pan(.9, .9) is True
            x0, y0, x1, y1 = viewer.state["roi"]
            assert (x1 - x0, y1 - y0) == (zoomed[2] - zoomed[0], zoomed[3] - zoomed[1])
            assert viewer._dragged

        with subtests.test("a pan past the edge clamps to the image"):
            viewer._grab(.5, .5)
            assert viewer._pan(-5., -5.) is True
            assert viewer.state["roi"][2:] == [float(W), float(H)]

        with subtests.test("releasing ends the drag"):
            viewer._release()
            assert viewer._drag is None
            assert viewer._pan(.1, .1) is False

        with subtests.test("reset restores the full image"):
            viewer._reset_roi()
            assert viewer.state["roi"] == [0., 0., float(W), float(H)]

    def test_to_source(self, viewer):
        """View fractions map onto source pixels through the current region."""
        H, W = viewer.parent.shape[0], viewer.parent.shape[1]
        assert viewer._to_source(0., 0.) == (0., 0.)
        assert viewer._to_source(1., 1.) == (float(W), float(H))

        viewer.state["roi"] = [10., 20., 50., 52.]
        assert viewer._to_source(.5, .25) == (30., 28.)


class TestViewerWidgets:
    """The ipywidgets control set, and the notebook mouse path that drives it."""

    def test_widgets(self, widget_viewer, subtests):
        """The control set covers the state the notebook backend exposes."""
        widgets = widget_viewer.widgets

        with subtests.test("the notebook backend sizes and gates its own view"):
            assert {"scale", "zoom"} <= set(widgets)
            assert {"scale", "zoom"} <= set(widget_viewer.state_keys)

        with subtests.test("a camera gets the color-scaling controls"):
            assert {"live", "range", "log", "crosshair", "autorange"} <= set(widgets)

        with subtests.test("a widget drives the state it shows"):
            widgets["cmap"].value = "gray"
            assert widget_viewer.state["cmap"] == "gray"

        with subtests.test("the scale widget resizes the box, not the region"):
            H, W = widget_viewer.parent.shape[0], widget_viewer.parent.shape[1]
            widgets["scale"].value = 2
            assert widget_viewer.display.image.layout.width == f"{2 * W}px"
            assert widget_viewer.state["roi"] == [0., 0., float(W), float(H)]
            widgets["scale"].value = 1

    def test_post(self, widget_viewer, subtests):
        """A request from another thread reaches the state and the widget that shows it."""
        drawn = count_draws(widget_viewer)

        with subtests.test("a colormap request advances the dropdown and its widget"):
            expected = widget_viewer.state["cmap_options"][1]
            posted = threading.Thread(target=widget_viewer._post, args=(("cmap",),))
            posted.start()
            posted.join(timeout=5)

            widget_viewer.render()
            assert widget_viewer.state["cmap"] == expected
            assert widget_viewer.widgets["cmap"].value == expected

        with subtests.test("mirroring to the widget does not draw a second frame"):
            assert len(drawn) == 1

        with subtests.test("autorange windows to the data and moves the slider"):
            widget_viewer.last_image = np.full(widget_viewer.parent.shape, 7, np.uint8)
            widget_viewer._post(("autorange",))
            widget_viewer.render()
            assert widget_viewer.state["range"] == [7, 7]
            assert list(widget_viewer.widgets["range"].value) == [7, 7]

        with subtests.test("a message reaches the output widget"):
            widget_viewer._post(("print", "hello"))
            widget_viewer.render()
            assert widget_viewer.widgets["output"].value == "hello"

        with subtests.test("an unknown request is reported, not swallowed silently"):
            widget_viewer._post(("nonsense",))
            widget_viewer.render()
            assert "nonsense" in widget_viewer.widgets["output"].value

    def test_dom_events(self, widget_viewer, subtests):
        """ipyevents mouse events drive the region the same way the window does."""
        display = widget_viewer.display
        H, W = widget_viewer.parent.shape[0], widget_viewer.parent.shape[1]
        full = [0., 0., float(W), float(H)]

        def event(etype, fx=.5, fy=.5, **extra):
            return dict(
                {"type": etype, "relativeX": fx * 400, "boundingRectWidth": 400,
                 "relativeY": fy * 320, "boundingRectHeight": 320},
                **extra,
            )

        with subtests.test("the wheel does nothing while Zoom is off"):
            display._on_dom_event(event("wheel", deltaY=-1))
            assert widget_viewer.state["roi"] == full

        widget_viewer.widgets["zoom"].value = True

        with subtests.test("the wheel zooms once enabled"):
            display._on_dom_event(event("wheel", deltaY=-1))
            assert widget_viewer.state["roi"] != full

        with subtests.test("drag pans without resizing"):
            zoomed = list(widget_viewer.state["roi"])
            display._on_dom_event(event("mousedown", .5, .5))
            display._on_dom_event(event("mousemove", .8, .8))
            panned = widget_viewer.state["roi"]
            assert panned != zoomed
            assert panned[2] - panned[0] == pytest.approx(zoomed[2] - zoomed[0])
            display._on_dom_event(event("mouseup", .8, .8))

        with subtests.test("double-click restores the full image"):
            display._on_dom_event(event("dblclick"))
            assert widget_viewer.state["roi"] == full

        with subtests.test("a click reads out the source pixel under the cursor"):
            widget_viewer._dragged = False
            display._on_dom_event(event("click", .25, .75))
            assert widget_viewer.widgets["output"].value.strip("[]").split() == \
                [str(W // 4), str(3 * H // 4)]

        with subtests.test("turning Zoom off restores the full image"):
            display._on_dom_event(event("wheel", deltaY=-1))
            widget_viewer.widgets["zoom"].value = False
            assert widget_viewer.state["roi"] == full


@needs_display
class TestViewerPyglet:
    """The window backend, which needs a display to open one."""

    def test_window(self, window, subtests):
        """The window opens at the image resolution and is visible."""
        viewer = window.window.viewer

        with subtests.test("the texture is the image, not the window"):
            assert window.window.shape == tuple(viewer.parent.shape[:2])
            assert window.window.visible

        with subtests.test("a script gets no widgets to host"):
            assert viewer.widgets == {}

        with subtests.test("the view encodes for save and copy"):
            assert viewer.display.png()[:4] == b"\x89PNG"

        with subtests.test("save and copy export the view, not the whole sensor"):
            H, W = viewer.parent.shape[0], viewer.parent.shape[1]
            viewer.render(np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1)))
            whole = viewer.display.png()
            viewer._zoom(.5, .5, True)
            assert viewer.parse(viewer.state["roi"]).shape != (H, W)
            assert viewer.display.png() != whole, "the export ignored the zoom"
            viewer._reset_roi()

    def test_draws(self, window, camera_small, subtests):
        """Every write reaches the frame the window draws from; none is skipped."""
        viewer = window.window.viewer
        display = viewer.display

        with subtests.test("successive frames each reach the texture"):
            seen = []
            for value in (40, 80, 120, 160):
                viewer.render(np.full(camera_small.shape, value, np.uint8))
                seen.append(int(display._index.max()))
            assert all(b > a for a, b in zip(seen, seen[1:])), seen

        with subtests.test("a single write right after opening is drawn"):
            camera_small.live(activate=False)
            camera_small.live(backend="pyglet")
            camera_small.viewer.render(np.full(camera_small.shape, 200, np.uint8))
            assert camera_small.viewer.display._index.max() > 0

    def test_orientation(self, window, camera_small):
        """The image reaches the framebuffer upright, unmirrored, and letterboxed."""
        import pyglet.gl as gl

        viewer = window.window.viewer
        H, W = camera_small.shape[0], camera_small.shape[1]

        # A marker in one corner only, so a flip or a mirror cannot look the same.
        img = np.zeros((H, W), np.uint8)
        img[: H // 4, : W // 4] = 255
        viewer.state["range"] = [0, 255]
        viewer.render(img)

        def grab(w):
            w.switch_to()
            fw, fh = w.get_framebuffer_size()
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            w._blit(w.cframe)
            gl.glReadBuffer(gl.GL_BACK)
            buf = (gl.GLubyte * (4 * fw * fh))()
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
            gl.glReadPixels(0, 0, fw, fh, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, buf)
            # glReadPixels is bottom-origin; flip to the image convention.
            return np.frombuffer(bytes(buf), np.uint8).reshape(fh, fw, 4)[::-1]

        frame = window.wait(window.submit(grab, window.window), timeout=10)
        fh, fw = frame.shape[:2]
        bright = frame[..., :3].max(axis=-1) > 127

        assert bright[fh // 8, fw // 8], "the marked corner is not top-left on screen"
        assert not bright[fh // 8, -fw // 8], "the image is mirrored left to right"
        assert not bright[-fh // 8, fw // 8], "the image is flipped top to bottom"

    def test_input(self, window, subtests):
        """Mouse and keyboard reach the viewer instead of being consumed."""
        from pyglet.window import key, mouse

        w = window.window
        viewer = w.viewer
        full = list(viewer.state["roi"])
        wait, submit = window.wait, window.submit

        with subtests.test("the wheel zooms toward the cursor"):
            wait(submit(w.on_mouse_scroll, 40, 40, 0, 1))
            assert viewer.state["roi"] != full

        with subtests.test("dragging pans without resizing"):
            zoomed = list(viewer.state["roi"])
            wait(submit(w.on_mouse_press, 40, 40, mouse.LEFT, 0))
            wait(submit(w.on_mouse_drag, 60, 40, 20, 0, mouse.LEFT, 0))
            panned = viewer.state["roi"]
            assert panned != zoomed
            assert panned[2] - panned[0] == pytest.approx(zoomed[2] - zoomed[0])
            wait(submit(w.on_mouse_release, 60, 40, mouse.LEFT, 0))

        with subtests.test("a right-click restores the full image"):
            wait(submit(w.on_mouse_press, 40, 40, mouse.RIGHT, 0))
            assert viewer.state["roi"] == full

        with subtests.test("r restores the full image"):
            wait(submit(w.on_mouse_scroll, 40, 40, 0, 1))
            wait(submit(w.on_key_press, key.R, 0))
            assert viewer.state["roi"] == full

        with subtests.test("c cycles the colormap on the main thread"):
            expected = viewer.state["cmap_options"][1]
            wait(submit(w.on_key_press, key.C, 0))
            viewer.render()
            assert viewer.state["cmap"] == expected

    def test_slm_shortcuts(self, slm_small):
        """A phase display must not be re-ranged out from under its colormap."""
        from pyglet.window import key

        slm_small.live(backend="pyglet")
        try:
            viewer = slm_small.viewer
            thread = viewer.display.thread
            before = list(viewer.state["range"])
            thread.wait(thread.submit(thread.window.on_key_press, key.A, 0))
            viewer.render()
            assert viewer.state["range"] == before
        finally:
            slm_small.live(activate=False)

    def test_reopen(self, camera_small):
        """A viewer can be closed and opened again in the same process."""
        for _ in range(3):
            camera_small.live(backend="pyglet")
            assert camera_small.viewer.display.thread.running
            camera_small.live(activate=False)
            assert camera_small.viewer is None

    def test_close(self, window, camera_small):
        """Closing the window detaches the viewer from its hardware."""
        window.submit(window.window.on_close)
        for _ in range(50):
            if not window.running:
                break
            time.sleep(.1)
        assert not window.running

        camera_small.get_image()
        assert camera_small.viewer is None

    def test_concurrent(self, camera_small, oblong):
        """A window built while another renders must not borrow its context."""
        camera_small.live(backend="pyglet")
        viewer = camera_small.viewer
        errors = []
        frames = [0]
        stop = threading.Event()
        blank = np.zeros(camera_small.shape, camera_small.dtype)

        # render() routes a fault to _print, so nothing propagates out of it.
        viewer._print = errors.append

        def spin():
            while not stop.is_set():
                viewer.render(blank)
                frames[0] += 1

        spinner = threading.Thread(target=spin, daemon=True)
        try:
            spinner.start()
            for _ in range(3):
                oblong.live(backend="pyglet")
                assert oblong.viewer.display.thread.running
                oblong.live(activate=False)
        finally:
            stop.set()
            spinner.join(timeout=10)
            camera_small.live(activate=False)

        assert frames[0] > 0, "the spinner never rendered"
        assert not errors, "{}/{} frames failed: {}".format(
            len(errors), frames[0], errors[:3]
        )
