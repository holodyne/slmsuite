"""
Unit tests for the Camera base class, exercised through SimulatedCamera.
"""
import logging

import pytest
import numpy as np

from slmsuite.hardware.cameras.camera import Camera
from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography.toolbox.phase import zernike

from conftest import driver_classes

# Non-square sensor, so an axis swapped by the orientation transform cannot hide.
WIDE = (200, 100)

# (label, constructor kwargs, whether the transform swaps the image axes).
ORIENTATIONS = (
    ("identity", dict(rot="0"), False),
    ("rot90", dict(rot="90"), True),
    ("rot180", dict(rot="180"), False),
    ("rot270", dict(rot="270"), True),
    ("flip", dict(fliplr=True), False),
    ("rot90_flip", dict(rot="90", fliplr=True), True),
    ("rot180_flip", dict(rot="180", fliplr=True), False),
    ("rot270_flip", dict(rot="270", fliplr=True), True),
)


class TestCamera:
    """Tests for the Camera base class via SimulatedCamera, and for its drivers."""

    def test_init(self, slm, subtests):
        """Constructor conventions: shape, pitch, bitdepth, and the dtype probe."""
        cam = SimulatedCamera(slm=slm, resolution=(640, 480), pitch_um=(5.5, 5.5), bitdepth=8)

        with subtests.test("shape transposes the (width, height) resolution"):
            assert cam.shape == (480, 640)

        with subtests.test("pitch and bitdepth are stored as given"):
            np.testing.assert_allclose(cam.pitch_um, [5.5, 5.5])
            assert cam.bitdepth == 8

        with subtests.test("a rotated camera reports the shape get_image() returns"):
            cam_rot = SimulatedCamera(slm=slm, resolution=WIDE, rot="90")
            assert cam_rot.shape == (200, 100)
            assert cam_rot.get_image().shape == cam_rot.shape

        with subtests.test("the dtype probe runs once and types the camera from hardware"):
            calls = []

            def spy(self, timeout_s):
                calls.append(timeout_s)
                return np.zeros(self._shape, dtype=np.int16)

            original = SimulatedCamera._get_image_hw
            SimulatedCamera._get_image_hw = spy
            try:
                probed = SimulatedCamera(slm=slm, resolution=(256, 256), bitdepth=12)
            finally:
                SimulatedCamera._get_image_hw = original

            assert calls == [1]
            assert probed.dtype == np.dtype(np.int16)

        with subtests.test("a bitdepth beyond 16 widens the dtype"):
            cam24 = SimulatedCamera(slm=slm, resolution=(256, 256), bitdepth=24)
            assert cam24.dtype == np.dtype(np.uint32)

    def test_bitresolution(self, camera, subtests):
        """bitresolution is 2**bitdepth, times the number of frames summed into a capture."""
        with subtests.test("bitresolution is exactly 2**bitdepth"):
            assert camera.bitresolution == 2 ** camera.bitdepth

        with subtests.test("summing N frames multiplies the range by N"):
            camera.averaging = 4
            assert camera.bitresolution == 4 * 2 ** camera.bitdepth

    def test_set_binning(self, slm, caplog, subtests):
        """set_binning() divides the shape, leaves the WOI unbinned, and honors the transform."""
        cam = SimulatedCamera(slm, resolution=WIDE)

        with subtests.test("binning divides the shape"):
            cam.set_binning(2)
            assert cam.shape == (50, 100)

        with subtests.test("the WOI stays in unbinned coordinates"):
            assert cam.woi == (0, 200, 0, 100) == cam.get_woi()

        with subtests.test("binning of 1 restores the full shape"):
            cam.set_binning(1)
            assert cam.shape == (100, 200)

        rotated = SimulatedCamera(slm, resolution=WIDE, rot="90")

        with subtests.test("an asymmetric request is realized in the transformed frame"):
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                rotated.set_binning((2, 4))
            assert tuple(rotated.binning) == (2, 4)
            unrealized = [r for r in caplog.records if "Attempted to set binning" in r.getMessage()]
            assert not unrealized, [r.getMessage() for r in unrealized]

        with subtests.test("the binning property setter round-trips under rotation"):
            rotated.binning = (4, 2)
            assert tuple(rotated.binning) == (4, 2) == rotated.get_binning()

    def test_get_binning(self, slm, subtests):
        """get_binning() reports the binning in transformed coordinates."""
        cam = SimulatedCamera(slm, resolution=WIDE)

        with subtests.test("an unbinned camera reports (1, 1)"):
            assert cam.get_binning() == (1, 1) == cam.binning

        with subtests.test("get_binning() follows set_binning()"):
            cam.set_binning(2)
            assert cam.get_binning() == (2, 2) == cam.binning

        with subtests.test("a 90 degree rotation swaps the binning axes"):
            rotated = SimulatedCamera(slm, resolution=WIDE, rot="90")
            rotated._binning = (2, 4)   # raw sensor axes, bypassing the transform in set_binning()
            assert rotated.get_binning() == (4, 2) == rotated.binning

    def test_set_woi(self, camera, slm, subtests):
        """A window stays on the sensor, sets the shape, and carries into every capture."""
        try:
            camera.set_woi()
        except NotImplementedError:
            pytest.skip("set_woi not implemented for this camera")

        orig_woi = camera.woi
        orig_shape = camera.shape
        (_, w_max, _, h_max) = orig_woi

        # default_shape is (h, w) unless a 90/270 rotation swaps the axes.
        swapped = camera.shape[0] != h_max
        (sw, sh) = (w_max // 8, h_max // 8)

        try:
            for (label, request) in (
                ("full sensor", (0, w_max, 0, h_max)),
                ("centered half", (w_max // 4, w_max // 2, h_max // 4, h_max // 2)),
                ("patch against the far corner", (w_max - sw, sw, h_max - sh, sh)),
                ("wide thin strip", (0, w_max, h_max * 2 // 5, h_max // 5)),
                ("odd offsets, which stress the snapping",
                 (w_max // 10, w_max * 4 // 5, h_max // 10, h_max * 4 // 5)),
            ):
                with subtests.test(label):
                    camera.set_woi(request)
                    (x, w, y, h) = camera.woi

                    # One-sided: hardware is free to snap the window to its own boundaries.
                    assert x >= 0 and y >= 0
                    assert x + w <= w_max and y + h <= h_max
                    assert w > 0 and h > 0
                    assert camera.shape == ((w, h) if swapped else (h, w))
                    assert camera.get_image().shape == camera.shape

            with subtests.test("the window carries into stacked and averaged captures"):
                camera.set_woi((0, w_max // 2, 0, h_max // 2))
                assert camera.get_images(3).shape == (3, *camera.shape)
                assert camera.get_image(averaging=2).shape == camera.shape

            with subtests.test("None restores the full sensor"):
                camera.set_woi(None)
                assert camera.shape == orig_shape
                assert camera.get_image().shape == orig_shape
        finally:
            # Restore, so that a failure here does not shrink every later test's sensor.
            camera.set_woi(orig_woi)

        cam = SimulatedCamera(slm, resolution=WIDE)

        with subtests.test("a window inside the sensor is taken verbatim"):
            cam.set_woi((10, 80, 5, 40))
            assert cam.woi == (10, 80, 5, 40)

        with subtests.test("a scalar or (w, h) request is centered on the sensor"):
            cam.set_woi(50)
            assert cam.shape == (50, 50)
            assert cam.woi == (75, 50, 25, 50)
            cam.set_woi((80, 40))
            assert cam.shape == (40, 80)
            assert cam.woi == (60, 80, 30, 40)

        with subtests.test("a window past the edge is clipped onto the sensor"):
            with pytest.warns(UserWarning, match="was clipped"):
                cam.set_woi((150, 100, 60, 80))
            assert cam.woi == (150, 50, 60, 40)

    def test_get_woi(self, slm, subtests):
        """get_woi() reports transformed, unbinned coordinates that set_woi() accepts back."""
        cam = SimulatedCamera(slm, resolution=WIDE)

        with subtests.test("set_woi(get_woi()) round-trips"):
            cam.set_woi((10, 40, 20, 30))
            cam.set_woi(cam.get_woi())
            assert cam.get_woi() == (10, 40, 20, 30)

        with subtests.test("coordinates stay unbinned while the shape follows the binning"):
            cam.set_woi((0, 120, 0, 80))
            cam.set_binning(2)
            assert cam.get_woi() == (0, 120, 0, 80) == cam.woi
            assert cam.shape == (40, 60)

        for (label, kwargs, swapped) in ORIENTATIONS:
            rotated = SimulatedCamera(slm, resolution=WIDE, **kwargs)
            (full_w, full_h) = (WIDE[1], WIDE[0]) if swapped else WIDE

            with subtests.test(f"{label} windows the sensor in the image frame"):
                assert rotated.get_woi() == (0, full_w, 0, full_h)

                (w, h) = (full_w // 2, full_h // 2)
                rotated.set_woi((full_w // 4, w, full_h // 4, h))
                assert rotated.shape == (h, w)
                assert rotated.get_image().shape == rotated.shape

                rotated.set_woi(None)
                assert rotated.shape == (full_h, full_w)

    def test_woi_survives_a_pickle_round_trip(self, slm):
        """pickle() records the window and the binning, so a reload delivers the same frame."""
        cam = SimulatedCamera(slm, resolution=WIDE, pitch_um=(5, 5))
        cam.set_binning(2)
        cam.set_woi((20, 80, 10, 60))
        expected = (cam.woi, cam.binning, cam.shape)

        restored = SimulatedCamera(slm, resolution=WIDE, pitch_um=(5, 5))
        restored._unpickle(cam.pickle(attributes=True, metadata=False))

        assert (restored.woi, restored.binning, restored.shape) == expected
        assert restored.get_image().shape == cam.shape

    def test_get_ijraw_to_ijcam(self, slm, subtests):
        """The raw-sensor to camera-image affine places the WOI corners and inverts exactly."""
        woi = (20, 80, 10, 60)      # (x0, w, y0, h) in raw sensor pixels
        (binx, biny) = (2, 4)
        (w_bin, h_bin) = (woi[1] // binx, woi[3] // biny)

        # Camera-image pixel that the WOI origin lands on, per orientation.
        origins = {
            "identity":    (0,         0),
            "rot90":       (0,         w_bin - 1),
            "rot180":      (w_bin - 1, h_bin - 1),
            "rot270":      (h_bin - 1, 0),
            "flip":        (w_bin - 1, 0),
            "rot90_flip":  (h_bin - 1, w_bin - 1),
            "rot180_flip": (0,         h_bin - 1),
            "rot270_flip": (0,         0),
        }

        for (label, kwargs, swapped) in ORIENTATIONS:
            cam = SimulatedCamera(slm, resolution=WIDE, **kwargs)
            # Set the raw window and binning directly, to test the affine apart from capture.
            cam._woi = woi
            cam._binning = (binx, biny)

            affine = cam._get_ijraw_to_ijcam()
            origin = np.array([[float(woi[0])], [float(woi[2])]])
            far = origin + np.array([[binx * (w_bin - 1.0)], [biny * (h_bin - 1.0)]])
            (out_h, out_w) = (w_bin, h_bin) if swapped else (h_bin, w_bin)

            with subtests.test(f"{label} maps the WOI origin onto its image corner"):
                np.testing.assert_allclose(
                    (affine @ origin).flatten(), origins[label], atol=1e-9
                )

            with subtests.test(f"{label} maps the far WOI corner onto the opposite image corner"):
                np.testing.assert_allclose(
                    (affine @ far).flatten(),
                    np.subtract((out_w - 1, out_h - 1), origins[label]),
                    atol=1e-9,
                )

            with subtests.test(f"{label} composed with its inverse is the identity"):
                round_trip = cam._get_ijcam_to_ijraw() @ affine
                np.testing.assert_allclose(round_trip.M, np.eye(2), atol=1e-9)
                np.testing.assert_allclose(round_trip.b, 0, atol=1e-9)

    def test_parse_averaging(self, camera, subtests):
        """_parse_averaging() resolves the frame count and rejects nonsense."""
        camera.averaging = 3

        with subtests.test("None falls back to the attribute"):
            assert camera._parse_averaging(None) == 3

        with subtests.test("preserve_none keeps None"):
            assert camera._parse_averaging(None, preserve_none=True) is None

        with subtests.test("False is a single frame"):
            assert camera._parse_averaging(False) == 1

        with subtests.test("a count passes through"):
            assert camera._parse_averaging(5) == 5

        with subtests.test("a negative count raises"):
            with pytest.raises(ValueError, match="averaging must be positive"):
                camera._parse_averaging(-1)

    def test_get_dtype(self, camera, subtests):
        """get_dtype() reports a type wide enough for whatever get_image() sums into it."""
        with subtests.test("a single unbinned frame keeps the hardware dtype"):
            assert camera.get_dtype(averaging=1) == camera.dtype
            assert camera.get_dtype(averaging=False) == camera.dtype
            assert camera.get_dtype(averaging=1, binning=(1, 1)) == camera.dtype

        with subtests.test("averaging=None follows the attribute"):
            camera.averaging = None
            assert camera.get_dtype(averaging=None) == camera.get_dtype(averaging=1)
            camera.averaging = 4
            assert camera.get_dtype(averaging=None) == camera.get_dtype(averaging=4)
            camera.averaging = None

        with subtests.test("the dtype holds the largest sum the settings can produce"):
            for (averaging, binning) in ((1000, (1, 1)), (1, (2, 2)), (4, (2, 2))):
                dtype = camera.get_dtype(averaging=averaging, binning=binning)
                largest = (2 ** camera.bitdepth - 1) * averaging * binning[0] * binning[1]
                assert dtype == float or np.iinfo(dtype).max >= largest

        with subtests.test("hdr returns float whatever else is set"):
            assert camera.get_dtype(hdr=2) == float
            assert camera.get_dtype(averaging=1, binning=(1, 1), hdr=2) == float

        with subtests.test("a negative count raises"):
            with pytest.raises(ValueError, match="averaging must be positive"):
                camera.get_dtype(averaging=-1)

    def test__get_dtype(self, camera, subtests):
        """_get_dtype() adopts the hardware's dtype, or infers one from the bitdepth."""
        (orig_dtype, orig_bitdepth) = (camera.dtype, camera.bitdepth)

        try:
            with subtests.test("supplied test data types the camera without a probe"):
                assert camera._get_dtype(
                    lambda: camera._get_image_hw_tolerant(timeout_s=1)
                ) == orig_dtype

            with subtests.test("a probe result types the camera and is given a timeout"):
                seen = []
                camera._get_image_hw = lambda timeout_s: (
                    seen.append(timeout_s) or np.zeros((4, 4), dtype=np.int16)
                )
                camera.bitdepth = 12
                assert camera._get_dtype() == np.dtype(np.int16)
                assert camera.dtype == np.dtype(np.int16)
                assert seen == [1]

            with subtests.test("a dtype too narrow for the bitdepth warns"):
                camera._get_image_hw = lambda timeout_s: np.zeros((4, 4), dtype=np.uint8)
                with pytest.warns(UserWarning, match="does not conform"):
                    camera._get_dtype()

            for (bitdepth, inferred) in ((8, np.uint8), (12, np.uint16)):
                camera.bitdepth = bitdepth

                with subtests.test(f"a probe that raises leaves {bitdepth} bits in {inferred.__name__}"):
                    def raising(timeout_s):
                        raise RuntimeError("no hardware")

                    camera._get_image_hw = raising
                    assert camera._get_dtype() == np.dtype(inferred)

                with subtests.test(f"an unusable probe leaves {bitdepth} bits in {inferred.__name__}"):
                    camera._get_image_hw = lambda timeout_s: None
                    assert camera._get_dtype() == np.dtype(inferred)
        finally:
            camera.__dict__.pop("_get_image_hw", None)
            (camera.dtype, camera.bitdepth) = (orig_dtype, orig_bitdepth)

    def test_parse_hdr(self, camera, subtests):
        """_parse_hdr() resolves the exposure count and its base."""
        with subtests.test("preserve_none keeps None"):
            assert camera._parse_hdr(None, preserve_none=True) is None

        with subtests.test("False is a single exposure"):
            assert camera._parse_hdr(False) == (1, 0)

        with subtests.test("a scalar count implies a base of two"):
            assert camera._parse_hdr(3) == (3, 2)

        with subtests.test("a tuple sets count and base"):
            assert camera._parse_hdr((4, 3)) == (4, 3)

    def test_crop_to_woi(self, slm, subtests):
        """Software WOI and binning crop and block-sum the raw frame without overflowing it."""
        cam = SimulatedCamera(slm, resolution=(64, 64), bitdepth=8)
        full = cam.get_image()

        with subtests.test("a window returns exactly that slice of the full frame"):
            cam.set_woi((8, 32, 4, 40))
            np.testing.assert_array_equal(cam.get_image(), full[4:44, 8:40])
            cam.set_woi(None)

        cam.set_binning(2)
        expected = full.reshape(32, 2, 32, 2).sum(axis=(1, 3))

        with subtests.test("each binned pixel is the sum of its block"):
            assert cam.shape == (32, 32)
            np.testing.assert_array_equal(cam.get_image(averaging=False), expected)

        with subtests.test("averaging N binned frames sums to N times the block sum"):
            np.testing.assert_array_equal(cam.get_image(averaging=2), 2 * expected)

        with subtests.test("the stacked path bins identically"):
            stack = cam.get_images(3)
            assert stack.shape == (3, *cam.shape)
            for frame in stack:
                np.testing.assert_array_equal(frame, expected)

        with subtests.test("the binned dtype is set by the block sum, not by hdr"):
            cam12 = SimulatedCamera(slm, resolution=(64, 64), bitdepth=12)
            cam12.set_binning(2)
            for hdr in (None, 3):
                cam12.hdr = hdr
                assert cam12.get_images(2).dtype == np.uint16

    def test_get_image(self, camera, slm, subtests):
        """get_image() returns one transformed frame of camera.shape, optionally summed."""
        img = camera.get_image()

        with subtests.test("the frame matches camera.shape and dtype and becomes last_image"):
            assert img.shape == camera.shape
            assert img.dtype == camera.dtype
            assert camera.last_image is img

        with subtests.test("pixel values are signal within the range the bitdepth allows"):
            assert np.all(img >= 0) and np.all(img <= camera.bitresolution - 1)
            assert np.any(img > 0)

        saved_exposure = camera.exposure_s
        # Low exposure keeps pixels off the rail, where N frames sum to exactly N times one.
        camera.set_exposure(saved_exposure * 0.05)

        try:
            with subtests.test("averaging sums N frames rather than meaning them"):
                one = camera.get_image(averaging=1).astype(float)
                unsaturated = one < (camera.bitresolution - 1) * 0.9
                assert np.any(unsaturated), "expected unsaturated pixels at reduced exposure"
                np.testing.assert_allclose(
                    camera.get_image(averaging=2)[unsaturated], 2 * one[unsaturated], rtol=1e-6
                )

            with subtests.test("averaging=False is a single frame"):
                np.testing.assert_array_equal(
                    camera.get_image(averaging=False), camera.get_image(averaging=1)
                )
        finally:
            camera.set_exposure(saved_exposure)

        with subtests.test("transform=False returns the raw sensor shape, HDR included"):
            rotated = SimulatedCamera(slm, resolution=WIDE, rot="90")
            assert rotated.get_image(hdr=False, transform=False).shape == (100, 200)
            assert rotated.get_image(hdr=2, transform=False).shape == (100, 200)
            assert rotated.get_image(hdr=2).shape == rotated.shape == (200, 100)

    def test_get_images(self, camera, subtests):
        """get_images() stacks raw frames, applying neither averaging nor HDR."""
        count = 3
        dtype = np.dtype(camera.get_dtype(averaging=1, hdr=False))

        with subtests.test("the stack is (count, *shape) of the single-frame dtype"):
            imgs = camera.get_images(count)
            assert imgs.shape == (count, *camera.shape)
            assert imgs.dtype == dtype
            np.testing.assert_array_equal(camera.last_image, imgs[-1])

        with subtests.test("a preallocated buffer is filled in place"):
            out = np.empty((count, *camera.shape), dtype=dtype)
            assert camera.get_images(count, out=out) is out
            assert out.dtype == dtype

        with subtests.test("neither averaging nor hdr widens the stack dtype"):
            camera.averaging = 4
            camera.hdr = 3
            assert camera.get_images(count).dtype == dtype

        with subtests.test("binning does widen it, and still fills a buffer in place"):
            camera.set_binning(2)
            binned_dtype = np.dtype(camera.get_dtype(averaging=1, hdr=False))
            assert camera.get_images(count).dtype == binned_dtype
            out = np.empty((count, *camera.shape), dtype=binned_dtype)
            assert camera.get_images(count, out=out) is out

    def test_get_image_hdr_analysis(self, subtests):
        """get_image_hdr_analysis() recovers the base exposure from a stack of doublings."""
        base = np.linspace(0, 200, 100).reshape(10, 10)
        imgs = np.array([np.minimum(base * 2 ** i, 255) for i in range(3)], dtype=np.uint8)

        with subtests.test("the stitch is the base exposure, to within quantization"):
            stitched = Camera.get_image_hdr_analysis(imgs, overexposure_threshold=200)
            assert stitched.shape == (10, 10)
            assert np.all(np.abs(stitched - base) < 1)

        with subtests.test("explicit exposure times match the implied powers of the base"):
            np.testing.assert_allclose(
                Camera.get_image_hdr_analysis(
                    imgs, overexposure_threshold=200, exposure_power=[1.0, 2.0, 4.0]
                ),
                stitched,
            )

        with subtests.test("exposure times that are all zero raise"):
            with pytest.raises(ValueError, match="cannot all be non-positive"):
                Camera.get_image_hdr_analysis(imgs, exposure_power=[0, 0, 0])

    def test_get_image_hdr(self, camera, subtests):
        """get_image_hdr() stitches exposures, and hands back the stack when asked."""
        with subtests.test("the stitch is one float frame of the camera's shape"):
            hdr = camera.get_image_hdr(exposures=2)
            assert hdr.shape == camera.shape
            assert np.issubdtype(hdr.dtype, np.floating)

        with subtests.test("return_raw hands back the stack and its exposure times"):
            (raw, times) = camera.get_image_hdr(exposures=3, return_raw=True)
            assert raw.shape == (3, *camera.shape)
            assert len(times) == 3

    def test_info(self, camera):
        """info() lists the cameras this class can find, empty where unsupported."""
        assert isinstance(camera.info(verbose=False), list)

    def test_selftest(self, camera, slm, subtests):
        """test() drives exposure, capture, averaging, HDR and info; a sensor that windows or
        bins in hardware also gets set_woi and set_binning."""
        assert camera.test() is True

        with subtests.test("a failing test() still restores the capture settings"):
            camera.averaging = 7
            camera.hdr = 3
            camera.flush = lambda *args, **kwargs: 1 / 0

            try:
                with pytest.raises(AssertionError, match="flush"):
                    camera.test()
            finally:
                del camera.flush

            assert (camera.averaging, camera.hdr) == (7, 3)

        for (label, kwargs, binning) in (
            ("software binning", {}, 2),
            ("a rotated non-square sensor", dict(rot="90"), None),
            ("both at once", dict(rot="270", fliplr=True), (2, 4)),
        ):
            with subtests.test(f"test() holds under {label}"):
                cam = SimulatedCamera(slm, resolution=WIDE, **kwargs)
                if binning is not None:
                    cam.set_binning(binning)
                    assert cam._software_binning, "SimulatedCamera must bin in software."
                assert cam.test() is True

    def test_plot(self, camera, mpl_test, subtests):
        """plot() renders the given array and applies titles, limits, labels, and a colorbar."""
        plt = mpl_test
        img = np.zeros(camera.shape, dtype=camera.dtype)

        # _plot() draws into an open figure if there is one, so each case closes its own.
        with subtests.test("the plotted array is the given image, under the given title"):
            ax = camera.plot(image=img, title="MyTitle")
            assert ax.get_images()[0].get_array().shape == camera.shape
            assert ax.get_title() == "MyTitle"
            plt.close("all")

        with subtests.test("image=False renders last_image"):
            camera.get_image()
            stored_shape = camera.last_image.shape
            ax = camera.plot(image=False)
            assert ax.get_images()[0].get_array().shape == stored_shape
            plt.close("all")

        with subtests.test("colorbar drawn only when requested"):
            ax = camera.plot(image=img, cbar=True)
            assert len(ax.get_figure().axes) == 2
            plt.close("all")
            ax = camera.plot(image=img, cbar=False)
            assert len(ax.get_figure().axes) == 1
            plt.close("all")

        with subtests.test("supplied ax is drawn upon without a colorbar"):
            (fig, given) = plt.subplots()
            ax = camera.plot(image=img, ax=given, cbar=True)
            assert ax is given
            assert len(fig.axes) == 1
            plt.close("all")

        with subtests.test("scalar limits scale the view about its center"):
            ax = camera.plot(image=img)
            (xlim, ylim) = (ax.get_xlim(), ax.get_ylim())
            plt.close("all")

            for factor in (1, 0.5):
                ax = camera.plot(image=img, limits=factor)
                for (lim, scaled) in zip((xlim, ylim), (ax.get_xlim(), ax.get_ylim())):
                    center = np.mean(lim)
                    np.testing.assert_allclose(
                        scaled, center + np.subtract(lim, center) * factor
                    )
                plt.close("all")

        with subtests.test("2x2 limits are applied directly"):
            ax = camera.plot(image=img, limits=[[10, 50], [20, 40]])
            np.testing.assert_allclose(ax.get_xlim(), (10, 50))
            np.testing.assert_allclose(ax.get_ylim(), (20, 40))
            plt.close("all")

        with subtests.test("limits of any other shape raise"):
            with pytest.raises(ValueError, match="not recognized"):
                camera.plot(image=img, limits=[1, 2, 3])
            plt.close("all")

        with subtests.test("labels applied only when the image fills the camera"):
            ax = camera.plot(image=img)
            assert ax.get_xlabel() == "Camera $i$ [pix]"
            assert ax.get_ylabel() == "Camera $j$ [pix]"
            plt.close("all")
            ax = camera.plot(image=img[::2, ::2])
            assert ax.get_xlabel() == "" and ax.get_ylabel() == ""
            plt.close("all")

        with subtests.test("the image is rendered with the default colormap and full scaling"):
            ax = camera.plot(image=np.arange(img.size).reshape(img.shape) % 251)
            im = ax.get_images()[0]
            assert im.get_cmap().name == plt.rcParams["image.cmap"]
            np.testing.assert_allclose(im.get_clim(), (0, 250))
            plt.close("all")

    def test_autoexpose(self, camera, subtests):
        """autoexpose() converges on one exposure and pins the image to set_fraction."""
        with subtests.test("the result does not depend on the starting exposure"):
            camera.set_exposure(0.01)
            from_low = camera.autoexpose(verbose=False)
            camera.set_exposure(1)
            assert camera.autoexpose(verbose=False) == pytest.approx(from_low, rel=0.15)

        with subtests.test("set_fraction is the fraction of the range the image reaches"):
            camera.set_exposure(0.01)
            camera.autoexpose(set_fraction=0.3, verbose=False)
            peak = np.max(camera.get_image())
            assert peak == pytest.approx(0.3 * camera.bitresolution, rel=0.2)

    def test_autofocus(self, camera, slm, subtests):
        """autofocus() recovers a known Zernike defocus and leaves the caller's sweep alone."""
        slm.set_source_analytic()

        fs = FourierSLM(camera, slm)
        fs.fourier_calibrate(array_pitch=10)

        defocus = 1
        slm.source["phase_sim"] = zernike(slm, 4, -defocus, use_mask=False)

        with subtests.test("the recovered defocus cancels the applied one"):
            assert camera.autofocus(set_z=slm) == pytest.approx(defocus, rel=0.25)

        with subtests.test("set_z must be a function or an SLM"):
            with pytest.raises(ValueError, match="set_z must be"):
                camera.autofocus(set_z="not_callable")

        for range_z in ([-1.0, -0.5, 0.0, 0.5, 1.0], np.array([-1, 0, 1])):
            with subtests.test(f"a {type(range_z).__name__} range_z is not mutated"):
                before = np.copy(range_z)
                camera.autofocus(set_z=slm, get_z=0.5, range_z=range_z, verbose=False)
                assert np.array_equal(range_z, before)

        with subtests.test("a custom metric is followed to its own peak"):
            stage = {"z": 0.0}
            peak = 0.37     # Off the 0.2-spaced sweep, so only the fit can land on it.

            def set_z(z):
                stage["z"] = z

            def peaked(_image):
                return 1.0 / (1.0 + ((stage["z"] - peak) / 0.3) ** 2)

            found = camera.autofocus(set_z, get_z=0.0, range_z=1.0, metric=peaked, plot=True)
            assert found == pytest.approx(peak, abs=1e-3)
            assert found not in np.linspace(-1, 1, 11), "the fit interpolates between samples"
            assert stage["z"] == pytest.approx(found), "the stage is left at the optimum"

        with subtests.test("a stage that never moves leaves nothing to fit"):
            def jammed(_z):
                raise RuntimeError("stage jammed")

            with pytest.raises(RuntimeError, match="no valid images"):
                camera.autofocus(jammed, get_z=0.0, range_z=1.0)

    @pytest.mark.parametrize(
        "driver", driver_classes(Camera), ids=lambda cls: cls.__module__.rsplit(".", 1)[-1]
    )
    def test_driver_is_concrete(self, driver):
        """Every shipped camera driver implements the whole abstract interface."""
        assert not driver.__abstractmethods__, (
            f"{driver.__module__}.{driver.__name__} leaves "
            f"{sorted(driver.__abstractmethods__)} abstract, so it cannot be instantiated."
        )
