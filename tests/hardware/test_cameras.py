"""
Unit tests for Camera base class using SimulatedCamera.
"""
import pytest
import numpy as np

from slmsuite.hardware.cameras.camera import Camera
from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.holography.toolbox.phase import zernike


class TestCamera:
    """Tests for the Camera base class via SimulatedCamera."""

    def test_selftest(self, camera, subtests):
        """camera.test() covers core properties, dtype, exposure, capture,
        averaging, HDR, WOI, and info."""
        assert camera.test() is True

    def test_init(self, slm, subtests):
        """Verify constructor sets shape, pitch, bitdepth, and resolution convention."""
        cam = SimulatedCamera(
            slm=slm, resolution=(512, 512), pitch_um=(5.5, 5.5), bitdepth=8
        )

        with subtests.test("shape"):
            assert cam.shape == (512, 512)

        with subtests.test("pitch_um"):
            np.testing.assert_allclose(cam.pitch_um, [5.5, 5.5])

        with subtests.test("bitdepth"):
            assert cam.bitdepth == 8

        with subtests.test("bitresolution"):
            assert cam.bitresolution == 256

        with subtests.test("height-width convention"):
            height, width = 480, 640
            cam2 = SimulatedCamera(slm=slm, resolution=(width, height))
            assert cam2.shape == (height, width)

        with subtests.test("defaults"):
            cam3 = SimulatedCamera(slm=slm, resolution=(256, 256))
            assert cam3.exposure_s is not None
            assert cam3.averaging is None or isinstance(cam3.averaging, int)

        with subtests.test("rotation changes shape to match get_image output"):
            cam_rot = SimulatedCamera(
                slm=slm, resolution=(200, 100), rot="90"
            )
            # shape always equals the shape of images returned by get_image().
            # ROT90 of a 100-row × 200-col sensor → 200-row × 100-col output.
            assert cam_rot.shape == (200, 100)
            img = cam_rot.get_image()
            assert img.shape == cam_rot.shape

        cam.close()

    def test_basic_properties(self, camera, subtests):
        """bitresolution equals 2^bitdepth and scales correctly with averaging."""

        with subtests.test("bitresolution equals 2^bitdepth"):
            assert camera.bitresolution == 2 ** camera.bitdepth

        with subtests.test("bitresolution scales with averaging"):
            orig_avg = camera.averaging
            camera.averaging = 4
            assert camera.bitresolution == (2 ** camera.bitdepth) * 4
            camera.averaging = orig_avg

    def test_get_image(self, camera, subtests):
        """get_image() returns correct shape, dtype, last_image pointer, and pixel range."""

        with subtests.test("shape matches camera.shape"):
            img = camera.get_image()
            assert img.shape == camera.shape

        with subtests.test("dtype matches camera.dtype"):
            img = camera.get_image()
            assert img.dtype == camera.dtype

        with subtests.test("last_image is the returned image"):
            img = camera.get_image()
            assert camera.last_image is img

        with subtests.test("pixel values are non-negative"):
            img = camera.get_image()
            assert np.all(img >= 0)

        with subtests.test("pixel values do not exceed bitresolution - 1"):
            img = camera.get_image()
            assert np.all(img <= camera.bitresolution - 1)

        with subtests.test("image contains nonzero signal"):
            img = camera.get_image()
            assert np.any(img > 0)

    def test_averaging_sum(self, camera, subtests):
        """get_image(averaging=N) sums N frames; pixel values scale as N, not 1."""
        saved_exposure = camera.exposure_s
        # Low exposure keeps every pixel well below saturation so each of N
        # identical (noise-free) frames contributes the same value.
        camera.set_exposure(saved_exposure * 0.05)

        try:
            with subtests.test("averaging=2 doubles single-frame pixel values"):
                img1 = camera.get_image(averaging=1)
                img2 = camera.get_image(averaging=2)
                img1_f = img1.astype(float)
                # Only evaluate on pixels that are not saturated in the single frame.
                unsaturated = img1_f < (camera.bitresolution - 1) * 0.9
                assert np.any(unsaturated), "expected unsaturated pixels at reduced exposure"
                np.testing.assert_allclose(
                    img2[unsaturated],
                    2.0 * img1_f[unsaturated],
                    rtol=1e-6,
                )

            with subtests.test("averaging=False is equivalent to averaging=1"):
                img_one = camera.get_image(averaging=1)
                img_false = camera.get_image(averaging=False)
                np.testing.assert_array_equal(img_one, img_false)

        finally:
            camera.set_exposure(saved_exposure)

    def test_get_images(self, camera, subtests):
        """get_images() returns a frame stack with correct shape, dtype, and last_image update."""
        count = 3

        with subtests.test("shape is (count, H, W)"):
            imgs = camera.get_images(count)
            assert imgs.shape == (count, camera.shape[0], camera.shape[1])

        with subtests.test("dtype matches camera.dtype"):
            imgs = camera.get_images(2)
            assert imgs.dtype == camera.dtype

        with subtests.test("last_image is the final captured frame"):
            imgs = camera.get_images(count)
            np.testing.assert_array_equal(camera.last_image, imgs[-1])

        with subtests.test("preallocated out buffer is filled with correct shape and dtype"):
            out = np.empty(
                (count, camera.shape[0], camera.shape[1]),
                dtype=camera.dtype,
            )
            imgs = camera.get_images(count, out=out, transform=False)
            assert imgs.shape == out.shape
            assert imgs.dtype == camera.dtype

    def test_get_dtype(self, camera, subtests):
        """_get_dtype infers dtype from various get_image callables."""
        orig_dtype = camera.dtype
        orig_bitdepth = camera.bitdepth

        # (expected_dtype, fake_dtype_flag, bitdepth)
        #   fake_dtype_flag=False  -> use real capture
        #   fake_dtype_flag=None   -> raise to trigger fallback
        #   fake_dtype_flag=<type> -> return zeros of that type
        cases = [
            (orig_dtype, False, orig_bitdepth),
            (np.dtype(np.uint8), None, 8),
            (np.dtype(np.uint16), None, 12),
            (np.dtype(np.uint8), np.uint8, 8),
            (np.dtype(np.uint16), np.uint16, 12),
        ]

        try:
            for solution_dtype, fake_dtype, bitdepth in cases:
                with subtests.test(f"dtype={solution_dtype}, fake={fake_dtype}, bits={bitdepth}"):
                    def fake_get_image(_fd=fake_dtype, _sd=solution_dtype):
                        if _fd is False:
                            return camera._get_image_hw_tolerant(timeout_s=1)
                        elif _fd is None:
                            raise RuntimeError("Fake error")
                        else:
                            return np.zeros((5, 5), dtype=_sd)

                    camera.bitdepth = bitdepth
                    dtype = camera._get_dtype(fake_get_image)
                    assert dtype is solution_dtype
                    assert dtype is camera.dtype
        finally:
            camera.dtype = orig_dtype
            camera.bitdepth = orig_bitdepth

    def test_parse_averaging(self, camera, subtests):
        """_parse_averaging returns correct values and raises on bad input."""
        orig_averaging = camera.averaging

        try:
            camera.averaging = 1

            with subtests.test("preserve_none"):
                assert camera._parse_averaging(None, preserve_none=True) is None

            with subtests.test("None falls back to self.averaging"):
                assert camera._parse_averaging(None) == camera.averaging

            with subtests.test("False returns 1"):
                assert camera._parse_averaging(False) == 1

            with subtests.test("explicit int"):
                assert camera._parse_averaging(5) == 5

            with subtests.test("negative raises"):
                with pytest.raises(ValueError, match="Cannot have negative averaging"):
                    camera._parse_averaging(-1)
        finally:
            camera.averaging = orig_averaging

    def test_get_dtype(self, camera, subtests):
        """get_dtype() returns the correct effective dtype for various settings."""
        with subtests.test("averaging=1 keeps hardware dtype"):
            assert camera.get_dtype(averaging=1) == camera.dtype

        with subtests.test("averaging=False keeps hardware dtype"):
            assert camera.get_dtype(averaging=False) == camera.dtype

        with subtests.test("high averaging may promote to float"):
            dtype_high = camera.get_dtype(averaging=1000)
            assert dtype_high == camera.dtype or dtype_high == float

        with subtests.test("averaging=None defaults to self.averaging or 1"):
            orig = camera.averaging
            camera.averaging = None
            assert camera.get_dtype(averaging=None) == camera.get_dtype(averaging=1)
            camera.averaging = orig

        with subtests.test("hdr active forces float"):
            assert camera.get_dtype(hdr=2) == float

        with subtests.test("negative averaging raises"):
            with pytest.raises(ValueError, match="averaging must be positive"):
                camera.get_dtype(averaging=-1)

        with subtests.test("software binning widens dtype"):
            # 2x2 binning on uint8: max sum = 4*255 = 1020, needs >8 bits → float or uint16
            dtype_bin = camera.get_dtype(averaging=1, binning=(2, 2))
            max_val = (2 ** camera.bitdepth - 1) * 4
            if dtype_bin != float:
                assert np.iinfo(dtype_bin).max >= max_val, (
                    f"dtype {dtype_bin} cannot hold max binned value {max_val}"
                )

        with subtests.test("binning=(1,1) keeps hardware dtype"):
            assert camera.get_dtype(averaging=1, binning=(1, 1)) == camera.dtype

    def test_parse_hdr(self, camera, subtests):
        """_parse_hdr returns correct tuples for various inputs."""
        with subtests.test("preserve_none"):
            assert camera._parse_hdr(None, preserve_none=True) is None

        with subtests.test("False disables"):
            assert camera._parse_hdr(False) == (1, 0)

        with subtests.test("scalar uses base 2"):
            assert camera._parse_hdr(3) == (3, 2)

        with subtests.test("tuple passthrough"):
            assert camera._parse_hdr((4, 3)) == (4, 3)

    def test_get_image_hdr_analysis(self, subtests):
        """get_image_hdr_analysis produces correct output and validates input."""
        test_img = np.random.rand(10, 10) * 200
        test_imgs = np.array(
            [
                np.minimum(test_img * (2 ** i), 255)
                for i in range(3)
            ],
            dtype=np.uint8
        )

        with subtests.test("basic analysis"):
            result = Camera.get_image_hdr_analysis(
                test_imgs,
                overexposure_threshold=200
            )
            assert isinstance(result, np.ndarray)
            assert result.shape == (10, 10)
            assert result.dtype in (np.float64, np.float32)

            assert np.all(np.abs(result - test_img) < 1)

        with subtests.test("custom exposure_power list"):
            result = Camera.get_image_hdr_analysis(
                test_imgs,
                overexposure_threshold=200,
                exposure_power=[1.0, 2.0, 4.0]
            )
            assert isinstance(result, np.ndarray)
            assert result.shape == (10, 10)

            assert np.all(np.abs(result - test_img) < 1)

        with subtests.test("all-zero exposure_power raises"):
            with pytest.raises(ValueError):
                Camera.get_image_hdr_analysis(test_imgs, exposure_power=[0, 0, 0])

    def test_autoexposure(self, camera, subtests):
        """Autoexposure converges to same result from different starting points."""
        with subtests.test("convergence"):
            camera.set_exposure(0.01)
            result1 = camera.autoexpose(verbose=False)
            camera.set_exposure(1)
            result2 = camera.autoexpose(verbose=False)
            assert pytest.approx(result1, rel=0.15) == result2

        with subtests.test("custom set_fraction"):
            camera.set_exposure(0.01)
            result3 = camera.autoexpose(set_fraction=0.3, verbose=False)
            assert result3 > 0

    def test_autofocus(self, camera, slm, subtests):
        """Autofocus recovers known defocus applied via Zernike."""
        slm = slm
        slm.set_source_analytic()

        fs = FourierSLM(camera, slm)
        fs.fourier_calibrate(array_pitch=10, verbose=False)

        defocus_zernike = 1
        slm.source["phase_sim"] = zernike(slm, 4, -defocus_zernike, use_mask=False)

        with subtests.test("recovers defocus"):
            defocus_opt = camera.autofocus(set_z=slm, verbose=False)
            assert pytest.approx(defocus_opt, rel=0.25) == defocus_zernike

        with subtests.test("set_z validation"):
            with pytest.raises(ValueError, match="set_z must be"):
                camera.autofocus(set_z="not_callable")

    def test_woi(self, camera, subtests):
        """
        WOI (window of interest) test: various sizes and offsets.

        For each candidate WOI the test verifies:
        - ``camera.woi`` is updated after ``set_woi``
        - ``camera.shape`` is consistent with the WOI dimensions
        - ``get_image()`` returns an array whose shape matches ``camera.shape``
        - The WOI stays within sensor bounds
        - The snapped WOI offset + size does not exceed the sensor boundary

        Cameras that do not implement ``set_woi`` are skipped.
        """
        # Skip if set_woi is not implemented.
        try:
            camera.set_woi()
        except NotImplementedError:
            pytest.skip("set_woi not implemented for this camera")

        orig_woi = camera.woi
        orig_shape = camera.shape

        x0, w_max, y0, h_max = orig_woi  # full-sensor WOI after reset

        # Determine normal vs rotated orientation once.
        # default_shape is (height, width) for normal, (width, height) for 90/270 rot.
        normal_orientation = (camera.shape[0] == h_max)

        def expected_shape(w, h):
            """Return numpy (rows, cols) shape for a WOI of pixel dims (w, h)."""
            return (h, w) if normal_orientation else (w, h)

        def check_woi(label, woi_request):
            """Set WOI, capture an image, and assert consistency."""
            with subtests.test(label):
                camera.set_woi(woi_request)
                x, w, y, h = camera.woi

                # WOI must stay inside sensor.
                assert x >= 0, f"OffsetX {x} < 0"
                assert y >= 0, f"OffsetY {y} < 0"
                assert x + w <= w_max, f"x+w={x+w} exceeds sensor width {w_max}"
                assert y + h <= h_max, f"y+h={y+h} exceeds sensor height {h_max}"
                assert w > 0 and h > 0, "WOI dimensions must be positive"

                # camera.shape must be consistent with WOI.
                exp_shape = expected_shape(w, h)
                assert camera.shape == exp_shape, (
                    f"camera.shape {camera.shape} != expected {exp_shape} "
                    f"for woi=({x},{w},{y},{h})"
                )

                # Captured image must match camera.shape.
                img = camera.get_image()
                assert img.shape == camera.shape, (
                    f"get_image() shape {img.shape} != camera.shape {camera.shape}"
                )

        try:
            # Full sensor (explicit)
            check_woi("full sensor", (0, w_max, 0, h_max))

            # Halves
            check_woi("left half",   (0, w_max // 2, 0, h_max))
            check_woi("right half",  (w_max // 2, w_max // 2, 0, h_max))
            check_woi("top half",    (0, w_max, 0, h_max // 2))
            check_woi("bottom half", (0, w_max, h_max // 2, h_max // 2))

            # Quadrant corners
            check_woi("top-left quarter",     (0,          w_max // 2, 0,          h_max // 2))
            check_woi("top-right quarter",    (w_max // 2, w_max // 2, 0,          h_max // 2))
            check_woi("bottom-left quarter",  (0,          w_max // 2, h_max // 2, h_max // 2))
            check_woi("bottom-right quarter", (w_max // 2, w_max // 2, h_max // 2, h_max // 2))

            # Centred half-size patch
            check_woi("centred half", (w_max // 4, w_max // 2, h_max // 4, h_max // 2))

            # Thin strips
            check_woi("wide strip (centre rows)",  (0, w_max, h_max * 3 // 8, h_max // 4))
            check_woi("tall strip (centre cols)",  (w_max * 3 // 8, w_max // 4, 0, h_max))

            # Small patch (~1/8 sensor), offset to several positions
            sw, sh = w_max // 8, h_max // 8
            check_woi("small patch ; near origin",        (0,               sw, 0,               sh))
            check_woi("small patch ; top-right corner",   (w_max - sw,      sw, 0,               sh))
            check_woi("small patch ; bottom-left corner", (0,               sw, h_max - sh,      sh))
            check_woi("small patch ; bottom-right corner",(w_max - sw,      sw, h_max - sh,      sh))
            check_woi("small patch ; centre",             (w_max // 2 - sw // 2, sw,
                                                           h_max // 2 - sh // 2, sh))

            # Asymmetric: very wide but short, and very tall but narrow
            check_woi("wide thin strip",  (0, w_max, h_max * 2 // 5, h_max // 5))
            check_woi("narrow tall strip",(w_max * 2 // 5, w_max // 5, 0, h_max))

            # Non-power-of-two offsets (stress-test snapping arithmetic)
            check_woi("odd offset ; 10% inset",
                      (w_max // 10, w_max * 4 // 5, h_max // 10, h_max * 4 // 5))
            check_woi("odd offset ; 30% inset",
                      (w_max * 3 // 10, w_max * 2 // 5, h_max * 3 // 10, h_max * 2 // 5))

        finally:
            # Always restore original WOI so subsequent tests see the full sensor.
            camera.set_woi(orig_woi)
            assert camera.shape == orig_shape, (
                f"Failed to restore original shape {orig_shape}; got {camera.shape}"
            )

    def test_plot(self, camera, subtests):
        """Camera.plot() renders the correct array shape and applies metadata."""
        import matplotlib.pyplot as plt

        with subtests.test("plotted array shape matches camera shape"):
            img = np.zeros(camera.shape, dtype=camera.dtype)
            ax = camera.plot(image=img, title="Shape Test")
            assert ax.get_images()[0].get_array().shape == camera.shape
            plt.close("all")

        with subtests.test("title is applied"):
            ax = camera.plot(image=np.zeros(camera.shape), title="MyTitle")
            assert ax.get_title() == "MyTitle"
            plt.close("all")

        with subtests.test("last_image rendered when image=False"):
            camera.get_image()
            stored_shape = camera.last_image.shape
            ax = camera.plot(image=False)
            assert ax.get_images()[0].get_array().shape == stored_shape
            plt.close("all")

    # ------------------------------------------------------------------
    # WOI / binning coordinate-system tests
    # ------------------------------------------------------------------

    def test_ijraw_to_ijcam(self, slm, subtests):
        """
        _get_ijraw_to_ijcam() maps raw sensor pixels to camera-image pixels correctly
        for all 8 orientation codes, WOI offsets, and binning factors.
        """
        from slmsuite.holography.analysis import OrientationTransform

        # Sensor: width=200, height=100 (non-square to expose axis-swap bugs).
        W, H = 200, 100
        WOI = (20, 80, 10, 60)   # (x0=20, w=80, y0=10, h=60)  unbinned, untransformed
        BINNING = (2, 4)          # (biny=2, binx=4)

        # WOI = (x0=20, w=80, y0=10, h=60), BINNING = (biny=2, binx=4)
        # → w_bin=20, h_bin=30
        # The push-orientation matrix maps unt=(0,0) to t (the translation vector).
        # Each row below: (label, cam-kwargs, expected-ijcam-of-WOI-origin)
        binx, biny = BINNING[0], BINNING[1]
        w_bin = WOI[1] // binx   # 20
        h_bin = WOI[3] // biny   # 30

        rot_configs = [
            ("identity",   dict(rot="0"),                  (0,         0        )),
            ("rot90",      dict(rot="90"),                 (0,         w_bin - 1)),
            ("rot180",     dict(rot="180"),                (w_bin - 1, h_bin - 1)),
            ("rot270",     dict(rot="270"),                (h_bin - 1, 0        )),
            ("flip",       dict(fliplr=True),              (w_bin - 1, 0        )),
            ("flip_rot90", dict(rot="90",  fliplr=True),  (0,         0        )),
            ("flip_rot180",dict(rot="180", fliplr=True),  (0,         h_bin - 1)),
            ("flip_rot270",dict(rot="270", fliplr=True),  (h_bin - 1, w_bin - 1)),
        ]

        for label, kwargs, expected_origin_cam in rot_configs:
            cam = SimulatedCamera(slm, resolution=(W, H), **kwargs)
            cam.close()

            # Apply WOI and binning directly to the private attributes so we
            # can test the transform without going through hardware I/O.
            cam._woi = WOI
            cam._binning = BINNING

            affine = cam._get_ijraw_to_ijcam()
            inv = cam._get_ijcam_to_ijraw()

            # --- WOI untransformed origin maps to a known corner of the camera image ---
            # The push-orientation transform moves the WOI origin to the corner
            # that becomes (0,0) of the RESULT IMAGE only for IDENTITY/FLIP_ROT90.
            # For other orientations it maps to the translation vector t.
            woi_origin = np.array([[float(WOI[0])], [float(WOI[2])]])   # (x0, y0)
            cam_at_origin = affine @ woi_origin
            with subtests.test(f"{label} : WOI origin → correct corner"):
                np.testing.assert_allclose(
                    cam_at_origin.flatten(),
                    list(expected_origin_cam),
                    atol=1e-9,
                    err_msg=f"label={label} got {cam_at_origin.flatten()} expected {expected_origin_cam}",
                )

            # --- A step of one binned pixel in x maps to a shift of 1 in the
            #     appropriate output axis ---
            binx, biny = BINNING[0], BINNING[1]
            pt_x = np.array([[float(WOI[0] + binx)], [float(WOI[2])]])  # x+1binned-step
            pt_y = np.array([[float(WOI[0])], [float(WOI[2] + biny)]])  # y+1binned-step

            delta_x = (affine @ pt_x - cam_at_origin).flatten()
            delta_y = (affine @ pt_y - cam_at_origin).flatten()

            with subtests.test(f"{label} : binned x-step magnitude = 1"):
                assert abs(np.linalg.norm(delta_x) - 1.0) < 1e-9, \
                    f"label={label} delta_x={delta_x}"

            with subtests.test(f"{label} : binned y-step magnitude = 1"):
                assert abs(np.linalg.norm(delta_y) - 1.0) < 1e-9, \
                    f"label={label} delta_y={delta_y}"

            # --- Round-trip: ijraw → ijcam → ijraw ---
            test_pts = np.array([
                [WOI[0], WOI[2]],
                [WOI[0] + WOI[1] - 1, WOI[2] + WOI[3] - 1],
                [WOI[0] + binx * 3, WOI[2] + biny * 2],
            ], dtype=float).T   # shape (2, N)

            cam_pts = affine @ test_pts
            raw_pts_rt = inv @ cam_pts

            with subtests.test(f"{label} : round-trip"):
                np.testing.assert_allclose(raw_pts_rt, test_pts, atol=1e-9,
                                           err_msg=f"label={label}")

    def test_woi_with_rotation(self, slm, subtests):
        """
        set_woi() + rotation: shape and woi are consistent for a non-square camera.

        WOI coordinates passed to set_woi() are in transformed (rotated/flipped),
        unbinned pixel coordinates — the orientation the user sees.
        """
        W, H = 200, 100  # non-square

        rot_configs = [
            ("identity",   dict(rot="0"),   False),   # (label, kwargs, axes_swapped)
            ("rot90",      dict(rot="90"),  True),
            ("rot180",     dict(rot="180"), False),
            ("rot270",     dict(rot="270"), True),
            ("flip",       dict(fliplr=True), False),
            ("flip_rot90", dict(rot="90", fliplr=True), True),
        ]

        for label, kwargs, axes_swapped in rot_configs:
            cam = SimulatedCamera(slm, resolution=(W, H), **kwargs)
            cam.close()

            # Full-sensor dims in the transformed (user-visible) frame.
            full_W_t = H if axes_swapped else W   # width in transformed frame
            full_H_t = W if axes_swapped else H   # height in transformed frame

            # Request a centred sub-window in the transformed frame (unbinned).
            sub_w, sub_h = full_W_t // 2, full_H_t // 2
            x0_t, y0_t  = full_W_t // 4, full_H_t // 4
            woi_req = (x0_t, sub_w, y0_t, sub_h)

            with subtests.test(f"{label} : set_woi and shape"):
                cam.set_woi(woi_req)

                # camera.shape is the shape of the image returned by get_image().
                # It equals (sub_h, sub_w) in the user-visible (transformed) frame.
                assert cam.shape == (sub_h, sub_w), (
                    f"{label}: cam.shape={cam.shape} expected ({sub_h},{sub_w})"
                )

            with subtests.test(f"{label} : get_image shape matches default_shape"):
                img = cam.get_image()
                assert img.shape == cam.shape, (
                    f"{label}: img.shape={img.shape} != default_shape={cam.shape}"
                )

            with subtests.test(f"{label} : reset to full sensor"):
                cam.set_woi(None)
                # shape equals the full-sensor size in the transformed (user-visible) frame.
                expected_full = cam.transform.transform_shape((H, W))
                assert cam.shape == expected_full, (
                    f"{label}: reset failed, shape={cam.shape} expected {expected_full}"
                )

    def test_woi_clipping(self, slm, subtests):
        """set_woi() clips out-of-bounds coordinates to the sensor boundaries."""
        cam = SimulatedCamera(slm, resolution=(200, 100))
        cam.close()
        W, H = 200, 100

        # WOI that extends past the right and bottom edges.
        with subtests.test("clips right+bottom overflow"):
            cam.set_woi((150, 100, 60, 80))   # x0+w=250>W, y0+h=140>H
            x, w, y, h = cam.woi
            assert x + w <= W, f"x+w={x+w} exceeds W={W}"
            assert y + h <= H, f"y+h={y+h} exceeds H={H}"
            assert w > 0 and h > 0

        # WOI entirely inside sensor.
        with subtests.test("no clipping needed"):
            cam.set_woi((10, 80, 5, 40))
            x, w, y, h = cam.woi
            assert (x, w, y, h) == (10, 80, 5, 40), f"woi changed unexpectedly: {cam.woi}"

        cam.set_woi(None)

    def test_binning_shape(self, slm, subtests):
        """set_binning() halves camera.shape and updates woi correctly."""
        cam = SimulatedCamera(slm, resolution=(200, 100))
        cam.close()

        with subtests.test("shape after 2x2 binning"):
            cam.set_binning(2)
            assert cam.shape == (50, 100), f"shape={cam.shape}"   # (H//2, W//2)

        with subtests.test("binning affects woi property"):
            x, w, y, h = cam.woi
            # Full sensor with 2x2 binning: binned width=100, height=50
            assert (x, w, y, h) == (0, 100, 0, 50), f"woi={cam.woi}"

        with subtests.test("shape reset after binning=1"):
            cam.set_binning(1)
            assert cam.shape == (100, 200), f"shape={cam.shape}"

    def test_woi_and_binning(self, slm, subtests):
        """Combined WOI + binning: shape and get_image() shape are correct."""
        cam = SimulatedCamera(slm, resolution=(200, 100))
        cam.close()

        # set_woi accepts unbinned (full-resolution) transformed coordinates.
        # With no rotation: unbinned WOI (0, 120, 0, 80) covers 120×80 pixels.
        # Then apply 2x2 binning → binned shape = (40, 60).
        cam.set_woi((0, 120, 0, 80))   # unbinned, transformed
        cam.set_binning(2)

        with subtests.test("shape with WOI + binning"):
            assert cam.shape == (40, 60), f"shape={cam.shape}"   # 80//2, 120//2

        with subtests.test("woi in binned coords"):
            x, w, y, h = cam.woi
            assert (x, w, y, h) == (0, 60, 0, 40), f"woi={cam.woi}"

        with subtests.test("get_image shape with WOI + binning"):
            img = cam.get_image()
            assert img.shape == cam.shape, (
                f"img.shape={img.shape} != default_shape={cam.shape}"
            )

        cam.set_binning(1)
        cam.set_woi(None)

    def test_get_images_woi(self, camera, subtests):
        """get_images() returns (N, h, w) matching camera.shape after set_woi."""
        try:
            camera.set_woi()
        except NotImplementedError:
            pytest.skip("set_woi not implemented")

        camera.set_woi((0, camera.shape[1] // 2, 0, camera.shape[0] // 2))
        N = 3

        with subtests.test("stack shape with WOI"):
            imgs = camera.get_images(N)
            assert imgs.shape == (N, *camera.shape), (
                f"imgs.shape={imgs.shape} expected (N={N}, *{camera.shape})"
            )

        with subtests.test("individual images match default_shape"):
            for i, img in enumerate(imgs):
                assert img.shape == camera.shape, (
                    f"imgs[{i}].shape={img.shape} != {camera.shape}"
                )

        camera.set_woi(None)

    def test_averaging_woi(self, camera, subtests):
        """get_image(averaging=N) returns shape matching default_shape after set_woi."""
        try:
            camera.set_woi()
        except NotImplementedError:
            pytest.skip("set_woi not implemented")

        camera.set_woi((0, camera.shape[1] // 2, 0, camera.shape[0] // 2))

        with subtests.test("averaging=2 shape with WOI"):
            img = camera.get_image(averaging=2)
            assert img.shape == camera.shape, (
                f"img.shape={img.shape} != default_shape={camera.shape}"
            )

        with subtests.test("averaging=4 shape with WOI"):
            img = camera.get_image(averaging=4)
            assert img.shape == camera.shape

        camera.set_woi(None)

    def test_woi_none_resets_full_sensor(self, camera, subtests):
        """set_woi(None) restores full sensor shape."""
        try:
            camera.set_woi()
        except NotImplementedError:
            pytest.skip("set_woi not implemented")

        orig_shape = camera.shape

        with subtests.test("shrink then reset"):
            camera.set_woi((0, orig_shape[1] // 3, 0, orig_shape[0] // 3))
            camera.set_woi(None)
            assert camera.shape == orig_shape, (
                f"after set_woi(None): shape={camera.shape} != {orig_shape}"
            )

        with subtests.test("shape after None reset matches get_image"):
            img = camera.get_image()
            assert img.shape == camera.shape

    def test_software_binning_dtype(self, slm, subtests):
        """
        Software binning must not overflow the raw pixel dtype.

        A block-sum of N×M pixels can produce values up to (N*M) * max_pixel, which
        overflows uint8 for any binning > 1×1.  The fix is to promote the accumulation
        dtype before summing; verify that both the averaging=1 (single-frame) path and
        the averaging>1 (multi-frame) path preserve the correct values.
        """
        # Use a camera with software WOI + software binning (base Camera with no hw overrides).
        # SimulatedCamera has hardware binning, so we test software binning by patching flags.
        cam = SimulatedCamera(slm, resolution=(64, 64), bitdepth=8)
        cam.close()

        # Force software binning by overriding the flag (without real hardware).
        cam._software_binning = True
        cam._binning = (2, 2)   # 2×2 → bin_factor = 4, max sum = 255*4 = 1020 > uint8 max

        # Capture a full-sensor image so _hw_image_shape is the full sensor.
        cam._software_woi = True   # also software WOI so image comes from full sensor path

        with subtests.test("averaging=1 dtype does not overflow"):
            img = cam.get_image(averaging=False)
            # Values should not wrap around: a binned sum of 4 pixels is ≥ any individual pixel.
            raw = cam.get_image.__wrapped__(cam) if hasattr(cam.get_image, '__wrapped__') else None
            # The key invariant: dtype is wide enough to represent the summed value.
            max_possible_bin_sum = (2**cam.bitdepth - 1) * cam._binning[0] * cam._binning[1]
            assert img.dtype.itemsize * 8 >= max_possible_bin_sum.bit_length(), (
                f"dtype {img.dtype} cannot hold max bin sum {max_possible_bin_sum}"
            )
            # Shape must match the binned camera shape.
            assert img.shape == cam.shape, f"img.shape={img.shape} cam.shape={cam.shape}"

        with subtests.test("averaging=2 dtype does not overflow"):
            img2 = cam.get_image(averaging=2)
            max_possible = (2**cam.bitdepth - 1) * cam._binning[0] * cam._binning[1] * 2
            assert img2.dtype.itemsize * 8 >= max_possible.bit_length(), (
                f"dtype {img2.dtype} cannot hold max averaged+binned sum {max_possible}"
            )
            assert img2.shape == cam.shape

        with subtests.test("software binning values are additive not truncated"):
            # Create a camera where we can predict the raw pixel values:
            # SimulatedCamera at very high exposure so pixels saturate to bitresolution-1.
            cam2 = SimulatedCamera(slm, resolution=(64, 64), bitdepth=8)
            cam2.close()
            cam2._software_binning = True
            cam2._binning = (2, 2)
            cam2._software_woi = True

            # Get the unbinned raw image by temporarily disabling software binning.
            cam2._software_binning = False
            raw_img = cam2.get_image(averaging=False)
            cam2._software_binning = True

            # Now get the binned image.
            binned_img = cam2.get_image(averaging=False)

            # Each binned pixel should equal the sum of its 2×2 block in the raw image.
            H, W = raw_img.shape
            expected = raw_img.reshape(H//2, 2, W//2, 2).sum(axis=(1, 3))
            np.testing.assert_array_equal(
                binned_img, expected,
                err_msg="Binned pixel values do not equal the raw block sum"
            )

    def test_get_binning(self, slm, subtests):
        """get_binning() returns current binning in transformed coordinates."""
        cam = SimulatedCamera(slm, resolution=(200, 100))
        cam.close()

        with subtests.test("default is (1, 1)"):
            assert cam.get_binning() == (1, 1)

        with subtests.test("matches binning property at default"):
            assert cam.get_binning() == cam.binning

        with subtests.test("matches binning property after set_binning(2)"):
            cam.set_binning(2)
            assert cam.get_binning() == (2, 2)
            assert cam.get_binning() == cam.binning
            cam.set_binning(1)

        with subtests.test("asymmetric _binning=(2, 4) with software binning"):
            cam._software_binning = True
            cam._binning = (2, 4)
            assert cam.get_binning() == (2, 4)
            assert cam.get_binning() == cam.binning
            cam._binning = (1, 1)

        with subtests.test("90-degree rotation swaps binning axes"):
            cam_rot = SimulatedCamera(slm, resolution=(200, 100), rot="90")
            cam_rot.close()
            cam_rot._software_binning = True
            cam_rot._binning = (2, 4)   # untransformed (bx=2, by=4)
            # ROT90 swaps x↔y, so get_binning() returns (by, bx) = (4, 2)
            assert cam_rot.get_binning() == (4, 2)
            assert cam_rot.get_binning() == cam_rot.binning

    def test_get_woi(self, slm, subtests):
        """get_woi() returns the current WOI in the same coordinates as the woi property."""
        cam = SimulatedCamera(slm, resolution=(200, 100))
        cam.close()

        with subtests.test("full sensor at default"):
            assert cam.get_woi() == (0, 200, 0, 100)

        with subtests.test("matches woi property when binning=1"):
            # binning=1 → binned == unbinned, so get_woi() and woi agree
            assert cam.get_woi() == cam.woi

        with subtests.test("set_woi then get_woi round-trips"):
            cam.set_woi((20, 80, 10, 60))
            assert cam.get_woi() == (20, 80, 10, 60)
            cam.set_woi(None)

        with subtests.test("get_woi returns unbinned coords even when binning != 1"):
            cam._software_binning = True
            cam._binning = (2, 2)
            # get_woi returns unbinned (full-sensor: 200×100), not binned (100×50)
            assert cam.get_woi() == (0, 200, 0, 100)
            # woi property returns binned
            assert cam.woi == (0, 100, 0, 50)
            cam._binning = (1, 1)

        with subtests.test("set_woi(get_woi()) is a valid round-trip"):
            cam.set_woi((10, 40, 20, 30))
            cam.set_woi(cam.get_woi())
            assert cam.get_woi() == (10, 40, 20, 30)
            cam.set_woi(None)

        with subtests.test("90-degree rotation: get_woi in transformed frame"):
            cam_rot = SimulatedCamera(slm, resolution=(200, 100), rot="90")
            cam_rot.close()
            # Untransformed sensor: W=200, H=100.  After ROT90: W_t=100, H_t=200.
            # Full-sensor woi in transformed frame: (x=0, w=100, y=0, h=200)
            assert cam_rot.get_woi() == (0, 100, 0, 200)

        with subtests.test("get_woi equals woi property when binning=1"):
            assert cam.get_woi() == cam.woi
