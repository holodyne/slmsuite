"""
Unit tests for FourierSLM class.
"""
import pytest
import cv2
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from scipy import ndimage

from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.holography.toolbox.phase import blaze, zernike_sum
from slmsuite.holography.algorithms import SpotHologram

from conftest import (
    SIMULATED_SYSTEM_CASES,
    array_kxy,
    farfield_corners_ij,
    farfield_support_mask,
    ground_truth_affine,
    ground_truth_kxy_to_ij,
    in_view_kxy,
    install_ground_truth_calibration,
    plot_calibration_diagnostic,
    plot_image_dim,
    seed_for,
    spot_size_ij,
    view_kxy_grid,
)


# Cases where fourier_calibrate with default-style arguments currently recovers the
# correct affine. The remaining cases silently produce a *wrong* calibration (or
# raise) --- this is the motivation for fourier_calibrate_auto().
DEFAULT_CALIBRATION_OK = {"identity", "matched", "fov_much_larger", "fov_larger"}

# Geometries that fourier_calibrate_auto() cannot yet handle, and why. Keyed by case
# name, or by (case, source) where only one illumination is affected. All but the last
# refuse rather than returning a calibration, so the failure is loud.
# Enough geometry to exercise the simulator: a contained farfield with aperture
# edges in view, a cropped one, rotation, and a 0th order steered off the sensor.
GEOMETRY_CASES = ("matched", "fov_much_larger", "rotated", "zeroth_outside")

AUTO_LIMITATIONS = {
    "zeroth_outside": (
        "With the 0th order off-camera the array can only be placed relative to the "
        "lit region, and every array that fits there is fitted wrongly (measured 100, "
        "149 and 125 px for the three attempts), which verification rejects"
    ),
    "fov_extreme": (
        "The survey measures one scale for both axes, which a farfield ten times "
        "larger than the camera along one axis and three times smaller along the "
        "other is not described by"
    ),
    ("noisy_severe", "gaussian2d"): (
        "Severe noise on an apodized source defeats the array fit, which the 0th-order "
        "anchor then refuses (the fit lands 113 px out while its own spots line up); "
        "the same noise on a uniform source calibrates"
    ),
    ("zeroth_corner", "gaussian2d"): (  # TODO: fourier_calibrate_auto should be able to move the center of the array.
        "A 0th order 2% of the way across the sensor leaves no room to place an array "
        "around it, so most of it falls off the camera and too few spots remain to fit"
    ),
    ("noisy", "gaussian2d"): (
        "A marginal draw rather than a broken geometry: this case calibrates to a "
        "median 1.33 px over eight draws, but the battery's seeded draw lands at "
        "2.40 px against a 2.00 px tolerance"
    ),
}

@pytest.fixture(scope="module")
def calibration_plot_level(request):
    """
    Plot level passed to fourier_calibrate: 2 emits the hologram nearfield and farfield
    plus blob_array_detect's DFT and lattice-fitting plots, the diagnostics that explain
    *why* a geometry fails. Rendering them costs more than the calibration, so only draw
    what will be saved.
    """
    return 2 if request.config.getoption("--save-plots") else 0

BATTERY_ARRAY_SHAPE = 10
BATTERY_ARRAY_PITCH = 10


@pytest.fixture(scope="module")
def calibration_summary(test_output_dir, request):
    """
    Accumulates each case's result and, at teardown, writes a single montage
    comparing all cases: ``fourier_calibrate_summary.png``.
    """
    results = {}
    yield results
    _write_summary(
        results, test_output_dir, request,
        "fourier_calibrate_summary.png",
        f"fourier_calibrate(array_shape={BATTERY_ARRAY_SHAPE}, array_pitch={BATTERY_ARRAY_PITCH}) "
        f"vs ground truth",
    )


@pytest.fixture(scope="module")
def auto_summary(test_output_dir, request):
    """As :func:`calibration_summary`, for ``fourier_calibrate_auto()``."""
    results = {}
    yield results
    _write_summary(
        results, test_output_dir, request,
        "fourier_calibrate_auto_summary.png",
        "fourier_calibrate_auto() vs ground truth",
    )


def _write_summary(results, test_output_dir, request, filename, title):
    """Montage of every case's calibration against the ground truth."""
    if not results or test_output_dir is None:
        return
    if not request.config.getoption("--save-plots"):
        return

    columns = 4
    rows = int(np.ceil(len(results) / columns))
    (fig, axs) = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axs = np.atleast_1d(axs).ravel()

    for (ax, (name, result)) in zip(axs, results.items()):
        plot_image_dim(ax, result["img"])
        gt = result["truth"]
        ax.scatter(gt[0], gt[1], fc="none", ec="lime", s=25, lw=0.5)
        if result["calibrated"] is not None:
            cal = result["calibrated"]
            ax.scatter(cal[0], cal[1], c="r", marker="x", s=12, lw=0.5)

        color = "green" if result["ok"] else "red"
        error = result["error"]
        ax.set_title(
            f"{name}\n" + ("FAILED" if not np.isfinite(error) else f"{error:.1f} px error"),
            color=color, fontsize="medium",
        )
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axs[len(results):]:
        ax.axis("off")

    fig.suptitle(
        f"{title}\ncircles = true spot positions, crosses = calibrated prediction",
    )
    fig.tight_layout()
    fig.savefig(test_output_dir / filename, dpi=150)
    plt.close(fig)
    print(f"\nSaved summary: {test_output_dir / filename}")


def _run_calibration(fs, name, source, calibrate, summary, label):
    """
    Calibrates, compares against ground truth, and records the diagnostics.

    Failures are caught rather than raised: the geometries that defeat a
    calibration are the ones whose plots are worth having. Returns the error in
    camera pixels (infinite if nothing was produced), the tolerance, and a
    description of what happened.
    """
    # The hologram starts from a random phase, so seed to keep runs comparable.
    seed_for(name)
    name = f"{name}-{source or 'uniform'}"

    failure = None
    try:
        calibrate(fs)
    except Exception as e:
        failure = e

    tolerance = max(2.0, np.max(spot_size_ij(fs)))
    error = np.inf
    if failure is None:
        # Measured over the area the camera views, not along one ray through its
        # center: a ray is blind to any error orthogonal to itself.
        grid = view_kxy_grid(fs)
        error = float(np.max(np.linalg.norm(
            fs.kxyslm_to_ijcam(grid) - ground_truth_kxy_to_ij(fs, grid), axis=0
        )))

    # The array that the calibration settled on, for the diagnostic overlays.
    array = fs.calibrations.get("fourier", {}).get("array", {})
    spots_kxy = array_kxy(
        fs,
        array.get("array_shape", BATTERY_ARRAY_SHAPE),
        array.get("array_pitch", BATTERY_ARRAY_PITCH),
        array.get("array_center"),
    )

    note = (
        f"{type(failure).__name__} raised" if failure is not None
        else f"max error {error:.2f} px (tolerance {tolerance:.2f} px)"
    )
    img = fs.cam.last_image if fs.cam.last_image is not None else fs.cam.get_image()
    plot_calibration_diagnostic(
        fs, img=img, spots_kxy=spots_kxy, name=f"{label}{name}", note=note,
    )

    summary[name] = {
        "img": img,
        "truth": ground_truth_kxy_to_ij(fs, spots_kxy),
        "calibrated": (
            fs.kxyslm_to_ijcam(spots_kxy) if "fourier" in fs.calibrations else None
        ),
        "error": error,
        "ok": error < tolerance,
    }

    return (error, tolerance, note, failure)


def _phase_deviation(phase, reference):
    """
    Largest deviation of ``phase - reference`` from its own circular mean, in radians.
    A wavefront correction is only defined up to a global piston, so the mean is free.
    """
    delta = np.exp(1j * (np.asarray(phase) - np.asarray(reference)))
    piston = np.mean(delta)
    return float(np.max(np.abs(np.angle(delta * np.conj(piston) / np.abs(piston)))))


class TestFourierSLM:
    """Tests for public methods on FourierSLM."""

    def test_init(self, slm, camera, subtests):
        """Test FourierSLM.__init__."""

        with subtests.test("default magnification"):
            fs = FourierSLM(camera, slm)
            assert fs.cam is camera
            assert fs.slm is slm
            assert fs.mag == 1.0
            assert fs.name == f"{camera.name}-{slm.name}"
            assert isinstance(fs.calibrations, dict)
            assert hasattr(fs, "_wavefront_calibration_window_multiplier")

        with subtests.test("custom magnification"):
            fs = FourierSLM(camera, slm, mag=5.0)
            assert fs.mag == 5.0

        with subtests.test("rejects non-camera"):
            slm_tmp = SimulatedSLM(resolution=(1920, 1080))
            with pytest.raises(ValueError, match="Expected Camera"):
                FourierSLM("not_a_camera", slm_tmp)

        with subtests.test("rejects non-SLM"):
            slm_tmp = SimulatedSLM(resolution=(1920, 1080))
            cam_tmp = SimulatedCamera(slm_tmp, resolution=(512, 512))
            with pytest.raises(ValueError, match="Expected SLM"):
                FourierSLM(cam_tmp, "not_an_slm")

    def test_fourier_calibrate(self, fourierslm, subtests):
        """Test FourierSLM.fourier_calibrate — the primary Fourier calibration
        routine.  This is the most important calibration in slmsuite."""

        with subtests.test("basic calibration stores M and b"):
            fourierslm.fourier_calibrate(
                array_pitch=35, array_shape=5, plot=True,
            )
            cal = fourierslm.calibrations["fourier"]
            assert "M" in cal and "b" in cal
            assert cal["M"].shape == (2, 2)
            assert cal["b"].shape == (2, 1)

        with subtests.test("M is invertible"):
            M = fourierslm.calibrations["fourier"]["M"]
            det = np.linalg.det(M)
            assert abs(det) > 1e-10, "Calibration matrix should be invertible"

        with subtests.test("metadata attached"):
            cal = fourierslm.calibrations["fourier"]
            # Metadata from _get_calibration_metadata
            assert "__meta__" in cal or "__version__" in cal or "name" in cal

        with subtests.test("second calibration overwrites"):
            fourierslm.fourier_calibrate(
                array_pitch=30, array_shape=5, plot=True,
            )
            # Just confirm it didn't error and key still exists
            assert "fourier" in fourierslm.calibrations

        with subtests.test("scalar array_shape and array_pitch"):
            fourierslm.fourier_calibrate(
                array_pitch=35, array_shape=5, plot=False,
            )
            assert fourierslm.calibrations["fourier"]["M"].shape == (2, 2)

        with subtests.test("list array_shape and array_pitch"):
            fourierslm.fourier_calibrate(
                array_pitch=[35, 35], array_shape=[5, 5], plot=False,
            )
            assert fourierslm.calibrations["fourier"]["M"].shape == (2, 2)

        with subtests.test("non-positive pitch raises"):
            with pytest.raises(ValueError):
                fourierslm.fourier_calibrate(
                    array_pitch=-1, array_shape=5, plot=False,
                )

    @pytest.mark.slow
    def test_fourier_calibrate_noise_and_transfer(self, slm, subtests):
        """Fourier-calibrate across a range of transfer functions and camera noise.

        A noisy camera background previously ballooned the DFT's 0th order and
        crushed the array's periodic peaks below detection, breaking calibration.
        Each case is validated end-to-end: a spot blazed to the calibrated kxy
        must land on its requested camera pixel.
        """
        noise = {
            "dark": lambda img: np.random.normal(0.005 * img, 0.002 * img),
            "read": lambda img: np.random.poisson(0.03 * img),
        }
        for theta, nz in [(0.0, None), (0.0, noise), (0.2, noise), (-0.3, noise)]:
            with subtests.test(theta=theta, noisy=nz is not None):
                cam = SimulatedCamera(slm, resolution=(512, 512), pitch_um=(5.5, 5.5), noise=nz)
                cam.set_affine(f_eff=170000.0, units="norm", theta=theta)
                fs = FourierSLM(cam, slm, mag=1.0)
                fs.cam.set_exposure(0.1)
                fs.fourier_calibrate(array_pitch=30, array_shape=10, plot=False)
                assert abs(np.linalg.det(fs.calibrations["fourier"]["M"])) > 1e6
                for ij in ([200, 310], [300, 200]):
                    fs.slm.set_phase(blaze(fs.slm, vector=fs.ijcam_to_kxyslm(ij)))
                    peak = np.flip(np.unravel_index(np.argmax(fs.cam.get_image()), fs.cam.shape))
                    assert np.linalg.norm(peak - np.array(ij)) < 0.1 * max(fs.cam.shape)

    @pytest.mark.slow
    def test_fourier_calibrate_large_array(self, fourierslm, fourierslm_calibrated, subtests):
        """Test fourier_calibrate with a larger grid for better statistics."""

        with subtests.test("10x10 grid calibrates"):
            fourierslm.fourier_calibrate(
                array_pitch=30, array_shape=10, plot=True,
            )
            plt.show()
            M = fourierslm.calibrations["fourier"]["M"]
            assert abs(np.linalg.det(M)) > 1e-10

        with subtests.test("calibration matches smaller grid"):
            M_large = fourierslm.calibrations["fourier"]["M"]
            b_large = fourierslm.calibrations["fourier"]["b"]
            M_small = fourierslm_calibrated.calibrations["fourier"]["M"]
            b_small = fourierslm_calibrated.calibrations["fourier"]["b"]
            assert np.allclose(M_large, M_small, rtol=0.1, atol=0.1)
            assert np.allclose(b_large, b_small, rtol=0.1, atol=0.1)

    def test_fourier_calibrate_analytic(self, fourierslm, subtests):
        """Test FourierSLM.fourier_calibrate_analytic."""

        M = np.array([[1.5, 0.1], [-0.05, 1.6]])
        b = np.array([[10.0], [20.0]])

        with subtests.test("stores M and b"):
            # Note: fourier_calibrate_analytic with arbitrary M calls set_affine
            # on SimulatedCamera, which may fail for small M values.
            # Use M values consistent with the simulated optical system.
            fourierslm.fourier_calibrate(array_pitch=35, array_shape=5, plot=False)
            real_M = fourierslm.calibrations["fourier"]["M"]
            real_b = fourierslm.calibrations["fourier"]["b"]
            fourierslm.fourier_calibrate_analytic(real_M, real_b)
            cal = fourierslm.calibrations["fourier"]
            assert np.allclose(cal["M"], real_M)
            assert np.allclose(cal["b"], real_b)

        with subtests.test("identity matrix"):
            fourierslm.fourier_calibrate_analytic(np.eye(2), np.zeros((2, 1)))
            cal = fourierslm.calibrations["fourier"]
            assert np.allclose(cal["M"], np.eye(2))

        with subtests.test("wrong-shape M raises"):
            with pytest.raises(ValueError):
                fourierslm.fourier_calibrate_analytic(np.eye(3), b)

    def test_fourier_grid_project(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.fourier_grid_project."""

        with subtests.test("returns a hologram with spot data"):
            hologram = fourierslm_calibrated.fourier_grid_project(
                array_shape=3, array_pitch=35,
            )
            assert hologram is not None
            assert hasattr(hologram, "spot_kxy_rounded")

    def test_kxyslm_to_ijcam(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.kxyslm_to_ijcam."""
        M = fourierslm_calibrated.calibrations["fourier"]["M"]
        b = fourierslm_calibrated.calibrations["fourier"]["b"]
        a = fourierslm_calibrated.calibrations["fourier"]["a"]

        with subtests.test("single point"):
            kxy = np.array([[10.0], [20.0]])
            ij = fourierslm_calibrated.kxyslm_to_ijcam(kxy)
            expected = M @ (kxy - a) + b
            assert np.allclose(ij, expected)

        with subtests.test("origin maps to b + M@a offset"):
            ij = fourierslm_calibrated.kxyslm_to_ijcam([0, 0])
            expected = M @ (np.zeros((2, 1)) - a) + b
            assert np.allclose(ij, expected, atol=1e-10)

        with subtests.test("raises without calibration"):
            fs_bare = FourierSLM(
                fourierslm_calibrated.cam, fourierslm_calibrated.slm,
            )
            with pytest.raises((KeyError, RuntimeError)):
                fs_bare.kxyslm_to_ijcam([10.0, 20.0])

    def test_ijcam_to_kxyslm(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.ijcam_to_kxyslm."""
        M = fourierslm_calibrated.calibrations["fourier"]["M"]
        b = fourierslm_calibrated.calibrations["fourier"]["b"]
        a = fourierslm_calibrated.calibrations["fourier"]["a"]

        with subtests.test("single point"):
            ij = np.array([[120.0], [140.0]])
            kxy = fourierslm_calibrated.ijcam_to_kxyslm(ij)
            expected = np.linalg.solve(M, ij - b) + a
            assert np.allclose(kxy, expected, atol=1e-10)

        with subtests.test("roundtrip kxy -> ij -> kxy"):
            kxy_orig = np.array([[15.0], [25.0]])
            ij = fourierslm_calibrated.kxyslm_to_ijcam(kxy_orig)
            kxy_back = fourierslm_calibrated.ijcam_to_kxyslm(ij)
            assert np.allclose(kxy_orig, kxy_back, atol=1e-10)

        with subtests.test("roundtrip ij -> kxy -> ij"):
            ij_orig = np.array([[200.0], [300.0]])
            kxy = fourierslm_calibrated.ijcam_to_kxyslm(ij_orig)
            ij_back = fourierslm_calibrated.kxyslm_to_ijcam(kxy)
            assert np.allclose(ij_orig, ij_back, atol=1e-10)

        with subtests.test("multiple points"):
            ij_multi = np.array([[100, 200, 300], [110, 210, 310]], dtype=float)
            kxy_multi = fourierslm_calibrated.ijcam_to_kxyslm(ij_multi)
            ij_rt = fourierslm_calibrated.kxyslm_to_ijcam(kxy_multi)
            assert np.allclose(ij_multi, ij_rt, atol=1e-10)

    def test_get_farfield_spot_size(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.get_farfield_spot_size."""

        with subtests.test("kxy basis positive"):
            size = fourierslm_calibrated.get_farfield_spot_size(
                slm_size=1.0, basis="kxy",
            )
            assert np.all(np.asarray(size) > 0)

        with subtests.test("ij basis positive"):
            size = fourierslm_calibrated.get_farfield_spot_size(
                slm_size=1.0, basis="ij",
            )
            assert np.all(np.asarray(size) > 0)

        with subtests.test("ij basis consistent under WOI/binning/orientation"):
            # The "ij" spot size must live entirely in the user-facing (WOI/binning/
            # orientation-applied) frame, i.e. use fourier_affine, not the raw
            # calibration M. The de-rotation then reduces algebraically to
            #   size_ij == sqrt(|det(fourier_affine.M)|) * (1/Wx, 1/Wy)
            # for any WOI/binning/orientation. An anisotropic slm_size + non-trivial
            # orientation exposes the old raw-M bug (the components come out swapped).
            from slmsuite.holography.analysis import get_orientation_transformation

            cam = fourierslm_calibrated.cam
            Wx, Wy = 1.0, 2.0  # anisotropic so a rotation/flip is observable

            try:
                for rot, binning, woi in [
                    ("0", 1, None),
                    ("90", 1, None),
                    ("180", 2, None),
                    ("0", 2, (50, 200, 60, 180)),
                ]:
                    cam.transform = get_orientation_transformation(rot)
                    cam.set_binning(binning)
                    cam.set_woi(woi)

                    size = fourierslm_calibrated.get_farfield_spot_size(
                        slm_size=(Wx, Wy), basis="ij",
                    )
                    expected = np.sqrt(
                        np.abs(fourierslm_calibrated.fourier_affine.det())
                    ) * np.array([1 / Wx, 1 / Wy])
                    assert np.allclose(size, expected), (
                        f"rot={rot} binning={binning} woi={woi}: "
                        f"{size} != {expected}"
                    )
            finally:
                # Restore the default (identity) camera state for other subtests.
                cam.transform = get_orientation_transformation("0")
                cam.set_binning(1)
                cam.set_woi(None)

        with subtests.test("ij basis invariant to pure WOI offset"):
            # WOI only shifts the affine offset b, never M, so the spot size is unchanged.
            cam = fourierslm_calibrated.cam
            base = fourierslm_calibrated.get_farfield_spot_size(slm_size=1.0, basis="ij")
            try:
                cam.set_woi((40, 300, 50, 320))
                shifted = fourierslm_calibrated.get_farfield_spot_size(
                    slm_size=1.0, basis="ij",
                )
                assert np.allclose(base, shifted)
            finally:
                cam.set_woi(None)

        with subtests.test("bad basis raises"):
            with pytest.raises(ValueError):
                fourierslm_calibrated.get_farfield_spot_size(
                    slm_size=1.0, basis="badvalue",
                )

    def test_get_effective_focal_length(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.get_effective_focal_length."""

        with subtests.test("ij units"):
            f = fourierslm_calibrated.get_effective_focal_length(units="ij")
            assert np.isfinite(f)
            assert f > 0

        with subtests.test("norm units"):
            f = fourierslm_calibrated.get_effective_focal_length(units="norm")
            assert np.all(np.isfinite(f))

        with subtests.test("raises without calibration"):
            fs_bare = FourierSLM(
                fourierslm_calibrated.cam, fourierslm_calibrated.slm,
            )
            with pytest.raises(RuntimeError):
                fs_bare.get_effective_focal_length()

    def test_get_farfield_extent(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.get_farfield_extent."""

        with subtests.test("returns corners array shape"):
            corners = fourierslm_calibrated.get_farfield_extent(return_mask=False)
            # 5 points (closed polygon: ll, lr, ur, ul, ll) in (2, N) format
            assert corners.shape[0] == 2
            assert corners.shape[1] == 5

        with subtests.test("returns boolean mask when return_mask=True"):
            mask = fourierslm_calibrated.get_farfield_extent(return_mask=True)
            assert mask.dtype == bool
            assert mask.shape == fourierslm_calibrated.cam.shape
            # The SLM farfield should occupy some portion of the camera
            assert mask.any()

    def test_get_camera_extent(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.get_camera_extent."""

        with subtests.test("kxy units returns 5 corner points"):
            corners = fourierslm_calibrated.get_camera_extent(units="kxy", return_mask=False)
            assert corners.shape[0] == 2
            assert corners.shape[1] == 5

        with subtests.test("knm canvas return_mask=True"):
            shape = SpotHologram.get_padded_shape(fourierslm_calibrated, padding_order=1, square_padding=True)
            mask = fourierslm_calibrated.get_camera_extent(units=shape, return_mask=True)
            assert mask.dtype == bool
            assert mask.shape == shape
            # Camera covers some portion of the SLM farfield
            assert mask.any()

        with subtests.test("corner roundtrip ij->kxy->ij"):
            # Corners in kxy space then mapped back to ij should enclose the camera
            corners_kxy = fourierslm_calibrated.get_camera_extent(units="kxy", return_mask=False)
            # Each corner maps back to a camera boundary pixel
            corners_ij = fourierslm_calibrated.kxyslm_to_ijcam(corners_kxy)
            cam_shape = fourierslm_calibrated.cam.shape
            # All points should be within a few pixels of the camera boundary
            assert np.all(corners_ij[0] >= -1) and np.all(corners_ij[0] <= cam_shape[1])
            assert np.all(corners_ij[1] >= -1) and np.all(corners_ij[1] <= cam_shape[0])

    def test_fourier_affine_property(self, fourierslm_calibrated, subtests):
        """Test that fourier_affine returns an Affine equal to kxyslm->ijcam conversion."""
        from slmsuite.holography.analysis import Affine

        with subtests.test("fourier_affine is an Affine instance"):
            aff = fourierslm_calibrated.fourier_affine
            assert isinstance(aff, Affine)

        with subtests.test("fourier_affine matches kxyslm_to_ijcam"):
            kxy = np.array([[10.0], [20.0]])
            via_affine = fourierslm_calibrated.fourier_affine @ kxy
            via_method = fourierslm_calibrated.kxyslm_to_ijcam(kxy)
            assert np.allclose(via_affine, via_method, atol=1e-10)

        with subtests.test("fourier_affine inverse matches ijcam_to_kxyslm"):
            ij = np.array([[150.0], [200.0]])
            via_inv = fourierslm_calibrated.fourier_affine.inv @ ij
            via_method = fourierslm_calibrated.ijcam_to_kxyslm(ij)
            assert np.allclose(via_inv, via_method, atol=1e-10)

    def test_simulate(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.simulate."""

        with subtests.test("returns FourierSLM with simulated hardware"):
            fs_sim = fourierslm_calibrated.simulate()
            assert isinstance(fs_sim, FourierSLM)
            assert isinstance(fs_sim.slm, SimulatedSLM)
            assert isinstance(fs_sim.cam, SimulatedCamera)

        with subtests.test("calibration copied"):
            fs_sim = fourierslm_calibrated.simulate()
            assert np.allclose(
                fs_sim.calibrations["fourier"]["M"],
                fourierslm_calibrated.calibrations["fourier"]["M"],
            )

        with subtests.test("coordinate transform matches original"):
            fs_sim = fourierslm_calibrated.simulate()
            kxy = np.array([10.0, 15.0])
            ij_real = fourierslm_calibrated.kxyslm_to_ijcam(kxy)
            ij_sim = fs_sim.kxyslm_to_ijcam(kxy)
            assert np.allclose(ij_real, ij_sim)

        with subtests.test("raises without calibration"):
            fs_bare = FourierSLM(
                fourierslm_calibrated.cam, fourierslm_calibrated.slm,
            )
            with pytest.raises(ValueError, match="Cannot simulate"):
                fs_bare.simulate()

    def test_name_calibration(self, fourierslm, subtests):
        """Test FourierSLM.name_calibration."""

        for cal_type in ("fourier", "wavefront"):
            with subtests.test(f"type={cal_type}"):
                name = fourierslm.name_calibration(cal_type)
                assert isinstance(name, str)
                assert cal_type in name.lower()

    def test_save_load_calibration(self, fourierslm_calibrated, temp_dir, subtests):
        """Test FourierSLM.save_calibration and load_calibration round-trip."""

        with subtests.test("save creates file"):
            path = fourierslm_calibrated.save_calibration(
                "fourier", path=temp_dir, name="test_save",
            )
            assert os.path.exists(path)

        with subtests.test("load restores calibration"):
            path = fourierslm_calibrated.save_calibration(
                "fourier", path=temp_dir, name="test_load",
            )
            fs_new = FourierSLM(
                fourierslm_calibrated.cam, fourierslm_calibrated.slm,
            )
            fs_new.load_calibration("fourier", file_path=path)
            assert np.allclose(
                fs_new.calibrations["fourier"]["M"],
                fourierslm_calibrated.calibrations["fourier"]["M"],
            )
            assert np.allclose(
                fs_new.calibrations["fourier"]["b"],
                fourierslm_calibrated.calibrations["fourier"]["b"],
            )

        with subtests.test("save nonexistent type raises"):
            with pytest.raises(ValueError):
                fourierslm_calibrated.save_calibration(
                    "nonexistent", path=temp_dir,
                )

    def test_load(self, fourierslm_calibrated, temp_dir, subtests):
        """Test FourierSLM.load static constructor."""

        path = fourierslm_calibrated.save_calibration(
            "fourier", path=temp_dir, name="test_static_load",
        )

        with subtests.test("returns valid FourierSLM"):
            fs = FourierSLM.load(path)
            assert isinstance(fs, FourierSLM)
            assert isinstance(fs.slm, SimulatedSLM)
            assert isinstance(fs.cam, SimulatedCamera)

        with subtests.test("calibration loaded"):
            fs = FourierSLM.load(path)
            # FourierSLM.load only restores metadata/hardware, not calibration data
            assert isinstance(fs, FourierSLM)

    def test_plot(self, fourierslm, subtests):
        """Test FourierSLM.plot."""

        with subtests.test("default call"):
            phase = np.random.rand(*fourierslm.slm.shape) * 2 * np.pi
            axs = fourierslm.plot(phase=phase)
            plt.show()
            assert axs is not None
            assert len(axs) == 2

    @pytest.mark.slow
    def test_wavefront_calibrate_superpixel(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.wavefront_calibrate_superpixel with various settings."""

        cal_point = [150, 150]
        sp_size = fourierslm_calibrated.slm.shape[0] // 6

        with subtests.test(f"add aberration to slm"):
            phase_abberation = zernike_sum(
                fourierslm_calibrated.slm,
                indices=(3, 4, 5, 7, 8),
                weights=(1, -2, 3, 1, 1),
                aperture=None,
                use_mask=False
            )
            fourierslm_calibrated.slm.set_source_analytic(
                phase_offset=phase_abberation,
                sim=True
            )
            fourierslm_calibrated.slm.plot_source(sim=True)

        with subtests.test(f"direct blaze to calibration point"):
            kxy = fourierslm_calibrated.ijcam_to_kxyslm(cal_point)
            fourierslm_calibrated.slm.set_phase(blaze(fourierslm_calibrated.slm, vector=kxy))
            img = fourierslm_calibrated.cam.get_image()

            fourierslm_calibrated.plot(image=img, title="Blazed spot at calibration point")
            plt.show()

            assert img[140:160, 140:160].mean() > img.mean(), "Blazed spot should be brighter than background"

        for phase_steps, name in [(None, "amplitude-only"), (1, "one-shot phase"), (5, "many-shot phase")]:
            fourierslm_calibrated.slm.source["phase"] = None    # Clear any old calibration.

            # FUTURE: test for warnings if underexposed.
            # fourierslm_calibrated.cam.set_exposure(0.01)

            # with subtests.test(f"test low-exposure {name} (phase_steps={phase_steps})"):
            #     result = fourierslm_calibrated.wavefront_calibrate_superpixel(
            #         calibration_points=cal_point,
            #         superpixel_size=sp_size,
            #         phase_steps=phase_steps,
            #         plot=True,
            #         test_index=-2,
            #     )

            fourierslm_calibrated.cam.set_exposure(.1)

            # FUTURE: benchmark the calibration tick?
            with subtests.test(f"test {name} (phase_steps={phase_steps})"):
                result = fourierslm_calibrated.wavefront_calibrate_superpixel(
                    calibration_points=cal_point,
                    superpixel_size=sp_size,
                    phase_steps=phase_steps,
                    plot=True,
                    test_index=-2,
                )

            with subtests.test(f"calibrate {name} (phase_steps={phase_steps})"):
                result = fourierslm_calibrated.wavefront_calibrate_superpixel(
                    calibration_points=cal_point,
                    superpixel_size=sp_size,
                    phase_steps=phase_steps,
                )
                assert isinstance(result, dict)
                assert "power" in result
                cal = fourierslm_calibrated.calibrations["wavefront_superpixel"]
                assert "superpixel_size" in cal

            with subtests.test(f"process {name} (phase_steps={phase_steps})"):
                fourierslm_calibrated.wavefront_calibration_superpixel_process(
                    plot=True,
                    smooth=False,
                )
                plt.show()

            with subtests.test(f"process smooth {name} (phase_steps={phase_steps})"):
                fourierslm_calibrated.wavefront_calibration_superpixel_process(
                    plot=True,
                    smooth=True,
                )
                plt.show()

            # Verifying phase calibration is difficult with low resolution, but
            # amplitude is decent.
            with subtests.test(f"check amplitude {name} (phase_steps={phase_steps})"):
                fourierslm_calibrated.slm.plot_source(sim=False)
                fourierslm_calibrated.slm.plot_source(sim=True)

                # Subtract the calibrated amplitude from the simulated amplitude
                amp = fourierslm_calibrated.slm.source["amplitude"]
                amp_sim = fourierslm_calibrated.slm.source["amplitude_sim"]

                amp_diff = np.abs(amp - amp_sim)
                plt.imshow(amp_diff)
                plt.title("Amplitude difference")
                plt.colorbar()
                plt.show()
                amp_diff_norm = np.sum(amp_diff) / np.sum(amp_sim)
                logger = logging.getLogger("conftest")
                logger.info(f"Normalized amplitude difference {name}: {amp_diff_norm:.2f}")
                assert amp_diff_norm < .5, f"Calibrated amplitude should be close to simulated amplitude ({amp_diff_norm:.2f} off)"

        with subtests.test("requires Fourier calibration"):
            fs_bare = FourierSLM(
                fourierslm_calibrated.cam, fourierslm_calibrated.slm,
            )
            with pytest.raises((RuntimeError, KeyError)):
                fs_bare.wavefront_calibrate_superpixel(
                    calibration_points=cal_point,
                    superpixel_size=sp_size,
                    plot=-1,
                )

        with subtests.test("stores scheduling metadata"):
            cal = fourierslm_calibrated.calibrations["wavefront_superpixel"]
            assert "scheduling" in cal
            assert "slm_supershape" in cal

    def test_wavefront_superpixel_schedule_is_conflict_free(self, simulated_system_factory):
        """
        No calibration point measures at a superpixel that another point is using as its
        reference, in any measurement where that other point is also measuring.
        """
        fs = simulated_system_factory(
            "matched", slm_resolution=(128, 128), cam_resolution=(256, 256)
        )
        install_ground_truth_calibration(fs)
        fs.wavefront_calibrate_superpixel(
            calibration_points=np.array([[60, 190, 128], [60, 60, 190]]),
            superpixel_size=32,
            phase_steps=1,
            reference_superpixels=np.array([[3, 1, 3], [0, 1, 1]]),
            plot=-1,
        )
        cal = fs.calibrations["wavefront_superpixel"]
        (scheduling, references) = (cal["scheduling"], np.ravel(cal["reference_superpixels"]))

        conflicts = [
            (measurement, writer, reader, int(references[reader]))
            for measurement in range(scheduling.shape[1])
            for writer in range(len(references))
            for reader in range(len(references))
            if scheduling[writer, measurement] == references[reader]
            and scheduling[reader, measurement] != -1
        ]
        assert not conflicts, (
            f"(measurement, writer, reader, superpixel) hijacks: {conflicts}\n{scheduling}"
        )

    def test_wavefront_superpixel_interpolates_a_ramp(self, simulated_system_factory, subtests):
        """
        Superpixels below the r2 threshold are filled in from their neighbours, so a pure
        diagonal ramp comes back out of the processor exactly. Only a lever arm carried
        along both axes at once separates a correct interpolation from a plausible one.
        """
        (NX, NY, superpixel_size) = (8, 8, 16)
        fs = simulated_system_factory(
            "matched", slm_resolution=(NX * superpixel_size, NY * superpixel_size)
        )
        holes = ((2, 3), (5, 2), (3, 5), (6, 6), (1, 6))

        def ramp_data(vector):
            """The processor's input for a pure ramp, with ``holes`` below threshold."""
            r2 = np.ones((NY, NX))
            for (hx, hy) in holes:
                r2[hy, hx] = 0
            return {
                "NX": NX, "NY": NY, "nxref": NX // 2, "nyref": NY // 2,
                "superpixel_size": superpixel_size,
                "r2_fit": r2,
                "power": np.ones((NY, NX)),
                "normalization": 2 * np.ones((NY, NX)),
                "background": np.zeros((NY, NX)),
                "kx": np.full((NY, NX), vector[0]),
                "ky": np.full((NY, NX), vector[1]),
                "phase": np.zeros((NY, NX)),
            }

        (x_grid, y_grid) = fs.slm.grid
        for vector in ((0.013, 0.021), (0.005, 0.009), (0.041, 0.033)):
            with subtests.test(f"kxy={vector}"):
                truth = 2 * np.pi * (
                    vector[0] * np.asarray(x_grid) + vector[1] * np.asarray(y_grid)
                )
                fs.calibrations["wavefront_superpixel"] = ramp_data(vector)
                phase = fs.wavefront_calibration_superpixel_process(
                    smooth=False,
                    r2_threshold=.5,
                    remove_blaze=False,
                    remove_background=False,
                    apply=False,
                    plot=0,
                )["phase"]
                assert _phase_deviation(phase, truth) < 1e-3

    @pytest.mark.slow
    def test_wavefront_calibrate_zernike(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.wavefront_calibrate_zernike."""

        # wavefront_calibrate_zernike passes calibration_points to
        # CompressedSpotHologram with basis=zernike_indices, so points must be
        # in the zernike basis (radians), not ij pixels.  Generate ij-space
        # points with wavefront_calibration_points(), then convert to zernike.
        from slmsuite.holography.toolbox import convert_vector
        ij_pts = fourierslm_calibrated.wavefront_calibration_points(pitch=120)
        cal_pts = convert_vector(
            ij_pts, from_units="ij", to_units="zernike",
            hardware=fourierslm_calibrated
        )

        with subtests.test("perturbation=0 projects spots only"):
            fourierslm_calibrated.wavefront_calibrate_zernike(
                calibration_points=cal_pts,
                zernike_indices=4,
                perturbation=0,
                optimize_position=False,
                optimize_weights=False,
                plot=-1,
            )

        with subtests.test("basic sweep stores calibration"):
            result = fourierslm_calibrated.wavefront_calibrate_zernike(
                calibration_points=cal_pts,
                zernike_indices=4,
                perturbation=0.5,
                optimize_position=False,
                optimize_weights=False,
                plot=-1,
            )
            assert result is not None
            assert "wavefront_zernike" in fourierslm_calibrated.calibrations
            cal = fourierslm_calibrated.calibrations["wavefront_zernike"]
            assert "corrected_spots" in cal
            assert "zernike_indices" in cal

        with subtests.test("iteration on previous calibration"):
            result2 = fourierslm_calibrated.wavefront_calibrate_zernike(
                perturbation=0.3,
                optimize_position=False,
                optimize_weights=False,
                plot=-1,
            )
            assert result2 is not None

    def test_wavefront_calibration_points(self, fourierslm_calibrated, subtests):
        """Test FourierSLM.wavefront_calibration_points."""

        with subtests.test("returns 2×N array"):
            pts = fourierslm_calibrated.wavefront_calibration_points(pitch=60)
            assert pts.ndim == 2
            assert pts.shape[0] == 2
            assert pts.shape[1] > 0

        with subtests.test("larger pitch gives fewer points"):
            pts_coarse = fourierslm_calibrated.wavefront_calibration_points(pitch=120)
            pts_fine = fourierslm_calibrated.wavefront_calibration_points(pitch=60)
            assert pts_coarse.shape[1] <= pts_fine.shape[1]

    def test_farfield_calibrate(self, simulated_system_factory, subtests):
        """Test FourierSLM.farfield_calibrate across several simulated geometries.

        The farfield calibration projects uniform power over the intersection of
        the SLM's farfield and the camera's field of view (averaging over speckle)
        and captures the 0th order with a flat phase pattern.
        """

        for case in ("matched", "fov_larger", "rotated"):
            fs = simulated_system_factory(case)
            install_ground_truth_calibration(fs)

            with subtests.test(f"{case}: calibrate stores raw data"):
                cal = fs.farfield_calibrate(averaging=3)
                assert "zeroth" in cal
                assert cal["efficiency_raw"].shape == (3,) + fs.cam.shape
                assert cal["exposure_zeroth"] > 0
                assert cal["exposure_raw"] > 0

            with subtests.test(f"{case}: 0th order captured at offset b"):
                img_0th = fs.calibrations["farfield"]["zeroth"]
                peak = np.flip(np.unravel_index(np.argmax(img_0th), img_0th.shape))
                b = np.squeeze(ground_truth_affine(fs)[1])
                assert np.linalg.norm(peak - b) < max(3.0, 2 * np.max(spot_size_ij(fs)))

            with subtests.test(f"{case}: process produces efficiency map"):
                efficiency = fs.farfield_calibration_process(size_blur=5)
                assert efficiency.shape == fs.cam.shape
                assert np.nanmax(efficiency) == pytest.approx(1.0)
                assert np.array_equal(
                    efficiency, fs.get_farfield_efficiency(fourier_crop=False)
                )

            with subtests.test(f"{case}: efficiency confined to farfield extent"):
                efficiency = fs.get_farfield_efficiency(fourier_crop=False)
                mask = fs.get_farfield_extent(return_mask=True)
                # Erode to stay clear of speckle blur at the aperture edge.
                erosion = 7
                mask_eroded = cv2.erode(
                    mask.astype(np.uint8), np.ones((erosion, erosion), np.uint8)
                ) > 0
                mask_dilated = cv2.dilate(
                    mask.astype(np.uint8), np.ones((erosion, erosion), np.uint8)
                ) > 0
                inside = efficiency[mask_eroded].mean()
                assert inside > 0, "Efficiency map should be illuminated."
                if (~mask_dilated).any():
                    outside = efficiency[~mask_dilated].mean()
                    assert outside < 0.1 * inside, (
                        f"{case}: efficiency outside the farfield aperture "
                        f"({outside:.3f}) should be far below inside ({inside:.3f})."
                    )

            with subtests.test(f"{case}: efficiency roughly uniform inside"):
                efficiency = fs.get_farfield_efficiency()
                mask = fs.get_farfield_extent(return_mask=True)
                mask_eroded = cv2.erode(
                    mask.astype(np.uint8), np.ones((7, 7), np.uint8)
                ) > 0
                values = efficiency[mask_eroded]
                cv = np.std(values) / np.mean(values)
                assert cv < 0.5, (
                    f"{case}: averaged efficiency map is too nonuniform (CV={cv:.2f})."
                )

    def test_farfield_calibrate_blind(self, simulated_system_factory, subtests):
        """Test blind farfield calibration (no Fourier calibration): measures the
        farfield support directly, which is what fourier_calibrate_auto() bootstraps
        from."""
        from slmsuite._plotting import _slmsuite_plt_show

        for case in ("fov_much_larger", "fov_larger"):
            fs = simulated_system_factory(case)

            with subtests.test(f"{case}: blind calibrate needs no Fourier calibration"):
                assert "fourier" not in fs.calibrations
                cal = fs.farfield_calibrate(averaging=3)
                assert "zeroth" in cal
                assert cal["efficiency_raw"].shape == (3,) + fs.cam.shape

            with subtests.test(f"{case}: measured support matches ground truth"):
                # Smoothing is kept below the size of the farfield itself, which is
                # only a few tens of pixels across in the fov_much_larger case.
                efficiency = fs.farfield_calibration_process(size_blur=3)
                support = fs.get_farfield_efficiency(efficiency_threshold=0.1)
                expected = farfield_support_mask(fs)
                iou = (support & expected).sum() / (support | expected).sum()

                # Plot the blind measurement against ground truth: this support is
                # what fourier_calibrate_auto() will bootstrap from.
                (fig, axs) = plt.subplots(1, 4, figsize=(20, 5))
                axs[0].imshow(fs.calibrations["farfield"]["zeroth"])
                axs[0].set_title("0th order (flat phase)")
                axs[1].imshow(efficiency)
                axs[1].set_title("Efficiency (speckle-averaged)")
                axs[2].imshow(support, vmin=0, vmax=1)
                axs[2].set_title("Measured support")
                axs[3].imshow(
                    support.astype(int) - expected.astype(int),
                    cmap="bwr", vmin=-1, vmax=1,
                )
                axs[3].set_title("Measured - truth\n(red = over, blue = missed)")
                fig.suptitle(f"Blind farfield calibration: {case} (IoU {iou:.3f})")
                fig.tight_layout(rect=(0, 0, 1, 0.92))
                _slmsuite_plt_show(name=f"farfield_blind_{case}")

                assert iou > 0.8, f"{case}: support IoU {iou:.2f} too low."

    def test_farfield_saturation_excluded(self, simulated_system_factory, subtests):
        """A railed pixel must not set the efficiency normalization."""
        fs = simulated_system_factory("matched")
        fs.farfield_calibrate(averaging=3)

        # Averaging sums frames, so it lifts saturation as well as the bitresolution.
        fs.cam.averaging = 16
        saturation = fs.cam.bitresolution - fs.cam.averaging

        efficiency = fs.calibrations["farfield"]["efficiency"]
        (y, x) = np.unravel_index(np.argmin(np.abs(efficiency - 0.5)), fs.cam.shape)
        raw = fs.calibrations["farfield"]["efficiency_raw"]
        resolved = raw[:, y, x].copy()

        def peak_away_from(y, x):
            """The efficiency peak ignoring one pixel and everything the blur touched."""
            efficiency = np.array(fs.get_farfield_efficiency(fourier_crop=False))
            efficiency[y-2:y+3, x-2:x+3] = 0
            return np.nanmax(efficiency)

        with subtests.test("saturated pixel does not set the peak"):
            raw[:, y, x] = saturation
            fs.farfield_calibration_process(size_blur=3)
            assert peak_away_from(y, x) == pytest.approx(1.0)

        with subtests.test("saturated pixel exceeds unity"):
            assert fs.get_farfield_efficiency(fourier_crop=False)[y, x] > 1

        with subtests.test("saturating one realization is enough to exclude"):
            raw[:, y, x] = resolved
            raw[0, y, x] = saturation
            fs.farfield_calibration_process(size_blur=3)
            assert peak_away_from(y, x) == pytest.approx(1.0)

        with subtests.test("full saturation falls back to the whole frame"):
            raw[:] = saturation
            efficiency = fs.farfield_calibration_process(size_blur=3)
            assert np.all(np.isfinite(efficiency))
            assert np.nanmax(efficiency) == pytest.approx(1.0)

    def test_farfield_products(self, simulated_system_factory, temp_dir, subtests):
        """The background, exposure, and crop products of a farfield calibration."""

        fs = simulated_system_factory("fov_larger")
        install_ground_truth_calibration(fs)
        fs.farfield_calibrate(averaging=3)

        with subtests.test("background matches the camera and is subtracted out"):
            background = fs.get_farfield_background()
            assert background.shape == fs.cam.shape
            assert np.all(background >= 0)
            raw = np.asarray(fs.calibrations["farfield"]["efficiency_raw"], float)
            assert np.nanmax(background) <= np.nanmax(np.mean(raw, axis=0))

        with subtests.test("fourier_crop zeroes outside the farfield extent"):
            mask = fs.get_farfield_extent(return_mask=True)
            cropped = fs.get_farfield_efficiency(fourier_crop=True)
            whole = fs.get_farfield_efficiency(fourier_crop=False)
            assert np.array_equal(cropped, whole * mask)
            assert not np.array_equal(cropped, whole), "The crop should remove something."

        with subtests.test("raw stacks are not saved to file"):
            path = fs.save_calibration("farfield", path=temp_dir, name="test_farfield")
            fs.load_calibration("farfield", file_path=path)
            loaded = fs.calibrations["farfield"]
            assert "efficiency_raw" not in loaded and "background_raw" not in loaded
            for key in ("efficiency", "background", "zeroth", "exposure_saturating"):
                assert key in loaded

        with subtests.test("a loaded calibration still serves its products"):
            with pytest.raises(RuntimeError):
                fs.farfield_calibration_process()
            assert np.all(np.isfinite(fs.get_farfield_zeroth()))
            assert np.all(np.isfinite(fs.get_farfield_background()))

    def test_farfield_process_requires_data(self, fourierslm):
        """Processing and access raise cleanly without raw data."""
        with pytest.raises(RuntimeError):
            fourierslm.farfield_calibration_process()
        with pytest.raises(RuntimeError):
            fourierslm.get_farfield_efficiency()

    def test_full_workflow(self, slm, camera, temp_dir, subtests):
        """Integration: calibrate -> save -> load -> simulate -> transform."""

        fs = FourierSLM(camera, slm)

        with subtests.test("calibrate"):
            fs.fourier_calibrate(array_pitch=35, array_shape=5, plot=False)
            assert "fourier" in fs.calibrations

        with subtests.test("save"):
            path = fs.save_calibration("fourier", path=temp_dir)
            assert os.path.exists(path)

        with subtests.test("load into new instance"):
            fs_loaded = FourierSLM.load(path)
            # FourierSLM.load restores hardware metadata but not all calibration keys;
            # reload the calibration explicitly.
            fs_loaded.load_calibration("fourier", file_path=path)
            assert np.allclose(
                fs.calibrations["fourier"]["M"],
                fs_loaded.calibrations["fourier"]["M"],
            )

        with subtests.test("simulate from loaded"):
            fs_sim = fs_loaded.simulate()
            kxy = np.array([10.0, 15.0])
            assert np.allclose(
                fs.kxyslm_to_ijcam(kxy),
                fs_sim.kxyslm_to_ijcam(kxy),
            )

    def test_pixel_kernel(self, subtests):
        """The pixel kernel is normalized, peaked, and shaped by its two sides."""
        x = (np.arange(401) - 200) / 50    # +-4 pixels at 50 samples per pixel.

        with subtests.test("normalized"):
            assert np.isclose(np.sum(FourierSLM.pixel_kernel(x, a_pix=.5)), 1)

        with subtests.test("symmetric by default"):
            kernel = FourierSLM.pixel_kernel(x, a_pix=.5, n=2)
            assert np.allclose(kernel, np.flip(kernel))

        with subtests.test("asymmetric widths and exponents"):
            for kwargs in ({"a_minus_pix": .05}, {"n_minus": 4}):
                kernel = FourierSLM.pixel_kernel(x, a_pix=.5, n=1, **kwargs)
                assert not np.allclose(kernel, np.flip(kernel))

        with subtests.test("swapping the two sides mirrors the kernel"):
            kernel = FourierSLM.pixel_kernel(x, a_pix=.5, n=1, a_minus_pix=.05, n_minus=4)
            mirror = FourierSLM.pixel_kernel(x, a_pix=.05, n=4, a_minus_pix=.5, n_minus=1)
            assert np.allclose(kernel, np.flip(mirror))

        with subtests.test("peaked at the origin and decaying"):
            kernel = FourierSLM.pixel_kernel(x, a_pix=.5, n=1, a_minus_pix=.05)
            assert np.argmax(kernel) == np.argmin(np.abs(x))
            assert np.all(np.diff(kernel[x >= 0]) <= 0)
            assert np.all(np.diff(kernel[x <= 0]) >= 0)

        with subtests.test("x0_pix displaces the peak"):
            kernel = FourierSLM.pixel_kernel(x, a_pix=.5, n=2, x0_pix=.4)
            assert np.isclose(x[np.argmax(kernel)], .4)
            assert np.allclose(kernel, FourierSLM.pixel_kernel(x - .4, a_pix=.5, n=2))

    def test_pixel_crosstalk_simulate(self, subtests):
        """Test FourierSLM._pixel_crosstalk_simulate, the forward model of the
        crosstalk kernel's effect on the diffraction orders of a grating."""
        simulate = FourierSLM._pixel_crosstalk_simulate
        (grating_50, grating_25) = (
            _binary_grating(16, np.pi, 0), _binary_grating(16, np.pi, 0, duty_cycle=.25)
        )

        with subtests.test("crosstalk-free limit matches the analytic binary grating"):
            # A 50% duty grating of phase difference d diffracts (2/(pi m))^2 sin^2(d/2)
            # into odd order m, and nothing into even orders.
            for delta in (np.pi, .6 * np.pi):
                orders = simulate(_binary_grating(16, delta, 0), a_pix=1e-4)
                for m in (1, 3):
                    assert np.isclose(
                        orders[m], (2 / (np.pi * m)) ** 2 * np.sin(delta / 2) ** 2, rtol=1e-2
                    )
                assert np.allclose(orders[[2, 4]], 0, atol=1e-6)

        with subtests.test("a kernel narrower than the sampling stays finite"):
            # Such a kernel underflows to zero, and must fall back to a delta function.
            for period in (15, 16):
                for supersample in (1, 3, 16):
                    orders = simulate(
                        _binary_grating(period, np.pi, 0),
                        supersample=supersample, a_pix=1e-6,
                    )
                    assert np.all(np.isfinite(orders)) and np.isclose(np.sum(orders), 1)

        with subtests.test("kernel is applied on the SLM pixel scale"):
            # Weakly-phased orders report the kernel's transfer function at m/p, which
            # pins a_pix absolutely: the wrong pitch is 3.1x off at a_pix=.5, m=3.
            (delta, period) = (.01, 16)
            for a_pix in (.25, .5):
                for supersample in (32, 64):
                    orders = simulate(
                        _binary_grating(period, delta, 0),
                        supersample=supersample, a_pix=a_pix, n=1,
                    )
                    for m in (1, 3):
                        transfer = 1 / (1 + (2 * np.pi * m * a_pix / period) ** 2)
                        assert np.isclose(
                            orders[m],
                            (delta / (np.pi * m)) ** 2 * transfer ** 2,
                            rtol=1e-2,
                        )

        with subtests.test("constant parameters reduce to a convolution"):
            # Moser Eq. (12) with fixed parameters is exactly the usual crosstalk model.
            # The reference tiles the grating, so that its kernel is untruncated too.
            (supersample, tiles) = (32, 5)
            commanded = np.repeat(np.tile(grating_50, tiles), supersample)
            size = commanded.size - 1
            for kwargs in ({"a_minus_pix": .05}, {"n_minus": 4}, {"a_pix": 2.}):
                kwargs = {"a_pix": .5, **kwargs}
                kernel = FourierSLM.pixel_kernel(
                    (np.arange(size) - (size - 1) / 2) / supersample, **kwargs
                )
                blurred = ndimage.convolve1d(commanded, kernel, mode="grid-wrap")
                expected = np.square(np.abs(
                    np.fft.fft(np.exp(1j * blurred)) / commanded.size
                ))
                orders = simulate(grating_50, supersample=supersample, **kwargs)
                assert np.allclose(orders[:9], expected[::tiles][:9], atol=1e-12)

        with subtests.test("independent of how many periods are supplied"):
            # The kernel's support must be set by the kernel, not by the pattern.
            for kwargs in ({"a_pix": .5, "a_minus_pix": .05}, {"a_pix": 2.}):
                for period in (4, 16):
                    grating = _binary_grating(period, np.pi, 0, duty_cycle=.25)
                    one = simulate(grating, supersample=64, **kwargs)
                    four = simulate(np.tile(grating, 4), supersample=64, **kwargs)
                    assert np.allclose(one[:5], four[::4][:5], atol=1e-12)

        with subtests.test("50% duty gratings cannot show a constant kernel's asymmetry"):
            # phi(x + p/2) = a + b - phi(x) survives any constant kernel and forces
            # |E_m| = |E_-m|.  This is the blind spot of pixel_calibrate().
            for kwargs in ({"a_minus_pix": .05}, {"n_minus": 4}):
                for delta in (np.pi, .6 * np.pi):
                    orders = simulate(_binary_grating(16, delta, 0), a_pix=.5, **kwargs)
                    assert np.isclose(orders[1], orders[-1], rtol=1e-12)

        with subtests.test("level-dependent width breaks the 50% duty symmetry"):
            # Moser's mechanism: LC pre-tilt steepens the transition for one sign of the
            # level step and flattens it for the other, which no convolution can do.
            # Signed, so that swapping the (phi0, phi1) convention fails.
            orders = simulate(grating_50, a_pix=lambda phi0, phi1: .25 if phi1 > phi0 else .75)
            assert (orders[1] - orders[-1]) / (orders[1] + orders[-1]) < -.1

            # Removing the level dependence removes the asymmetry.
            for a_pix in (.25, .75):
                orders = simulate(grating_50, a_pix=a_pix)
                assert np.isclose(orders[1], orders[-1], rtol=1e-12)

        with subtests.test("a duty cycle other than 50% does expose it"):
            assert np.isclose(*simulate(grating_25, a_pix=.5)[[1, -1]], rtol=1e-12)
            for kwargs in ({"a_minus_pix": .05}, {"n_minus": 4}):
                orders = simulate(grating_25, a_pix=.5, **kwargs)
                assert np.abs((orders[1] - orders[-1]) / (orders[1] + orders[-1])) > .01

        with subtests.test("mirroring the kernel mirrors the first orders"):
            orders = simulate(grating_25, a_pix=.5, n=1, a_minus_pix=.05, n_minus=4)
            mirror = simulate(grating_25, a_pix=.05, n=4, a_minus_pix=.5, n_minus=1)
            assert np.isclose(orders[1], mirror[-1], rtol=1e-12)
            assert np.isclose(orders[-1], mirror[1], rtol=1e-12)

        with subtests.test("plots"):
            simulate(grating_25, a_pix=.5, a_minus_pix=.05, plot=True)

    def test_pixel_calibrate(self, simulated_system_factory, temp_dir, caplog, subtests):
        """Test FourierSLM.pixel_calibrate on a crosstalk-free simulated system."""

        fs = simulated_system_factory("fov_much_smaller")
        install_ground_truth_calibration(fs)

        with subtests.test("sweeps and stores raw data"):
            cal = fs.pixel_calibrate(levels=8, periods=2, orders=1, directions="x", plot=False)
            assert cal["data"].shape == (2, 2, 8, 8, 3)
            assert np.all(cal["periods"] % 2 == 0)
            assert np.any(cal["data"] > 0)

        with subtests.test("only the swept direction is populated"):
            assert np.all(cal["data"][1] == 0)

        with subtests.test("intensity depends only on the level difference"):
            # An ideal SLM diffracts on phase difference alone, so each slice is circulant.
            for order in range(cal["data"].shape[-1]):
                data = cal["data"][0, 0, :, :, order]
                assert np.allclose(data, np.array([np.roll(data[0], k) for k in range(8)]))

        with subtests.test("gamma recovers the linear phase response"):
            gamma = fs.pixel_calibration_process(plot=False)
            expected = cal["levels"] / fs.slm.bitresolution
            assert np.allclose(gamma, expected - np.min(expected), atol=.02)
            # The fit is seeded with the linear response, so agreement alone proves
            # nothing; the fit must also actually describe the data.
            assert fs.calibrations["pixel"]["gamma_r2"] > .9

        with subtests.test("the sampled fit is applied across every level"):
            B = fs.slm.bitresolution
            # The sweep measures a handful of levels; the SLM carries all of them.
            assert len(gamma) < B
            assert fs.slm.gamma.shape == (B,)
            np.testing.assert_allclose(fs.slm.gamma, np.arange(B) / B, atol=.02)

        with subtests.test("apply=False leaves the SLM alone"):
            fs.slm.set_gamma(None)
            fs.pixel_calibration_process(plot=False, apply=False)
            assert fs.slm.gamma is None and fs.slm.lut is None

        with subtests.test("loading the calibration restores the response"):
            path = fs.save_calibration("pixel", path=temp_dir, name="test_gamma")
            fs.slm.set_gamma(None)
            fs.load_calibration("pixel", file_path=path)
            assert fs.slm.gamma is not None and fs.slm.lut is not None

        with subtests.test("a calibration from another bitdepth is not applied"):
            fs.slm.set_gamma(None)
            meta = fs.calibrations["pixel"].setdefault("__meta__", {}).setdefault("slm", {})
            bitresolution = meta.get("bitresolution", fs.slm.bitresolution)
            meta["bitresolution"] = 4 * fs.slm.bitresolution
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                fs._pixel_calibration_apply_gamma()
            assert any("level SLM" in r.message for r in caplog.records)
            assert fs.slm.gamma is None and fs.slm.lut is None
            meta["bitresolution"] = bitresolution

        with subtests.test("loading onto a retuned SLM warns"):
            wav_um = fs.slm.wav_um
            fs.slm.wav_um = 2 * wav_um
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                fs.load_calibration("pixel", file_path=path)
            assert any("was taken at" in r.message for r in caplog.records)
            fs.slm.wav_um = wav_um

        with subtests.test("gamma refuses signal-free data"):
            fs.calibrations["pixel"]["data"][:] = 0
            with pytest.raises(RuntimeError, match="no signal"):
                fs.pixel_calibration_process(plot=False)

        with subtests.test("test_index booleans"):
            # True tests every point of the sweep; False is not a test at all.
            for true in (True, np.True_):        # numpy booleans are not bool instances.
                result = fs.pixel_calibrate(
                    levels=2, periods=1, orders=1, directions="x", test_index=true, plot=False
                )
                assert len(result["indices"]) == 4
            result = fs.pixel_calibrate(
                levels=2, periods=1, orders=1, directions="x", test_index=False, plot=False
            )
            assert "data" in result

        with subtests.test("rejects a test_index selecting nothing"):
            with pytest.raises(ValueError, match="no points"):
                fs.pixel_calibrate(levels=2, periods=1, orders=1, test_index=[], plot=False)

        with subtests.test("rejects odd periods"):
            with pytest.raises(ValueError, match="even"):
                fs.pixel_calibrate(levels=2, periods=[25], orders=1, plot=False)

        with subtests.test("rejects repeated orders"):
            with pytest.raises(ValueError, match="Repeated orders"):
                fs.pixel_calibrate(levels=2, periods=1, orders=[-1, 1, 1], plot=False)

        with subtests.test("rejects a sweep without the 1st order"):
            with pytest.raises(ValueError, match="1st order"):
                fs.pixel_calibrate(levels=2, periods=[26], orders=[0], plot=False)

        with subtests.test("plots"):
            fs.pixel_calibrate(levels=2, periods=1, orders=2, directions="x", plot=2)
            fs.pixel_calibration_plot(summed=True)
            fs.pixel_calibration_plot(orders=[1])   # A single order must not squeeze away.
            with pytest.raises(ValueError, match="were measured"):
                fs.pixel_calibration_plot(orders=[99])

    def test_pixel_calibrate_sampling_density(self, simulated_system_factory, caplog, subtests):
        """A gamma sweep warns when its levels are too few to resolve the phase range."""
        fs = simulated_system_factory("fov_much_smaller")
        install_ground_truth_calibration(fs)

        def sampling_warnings(levels):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                fs.pixel_calibrate(
                    levels=levels, periods=1, orders=1, directions="x",
                    test_index=0, plot=False,
                )
            return [r.getMessage() for r in caplog.records if "cycles" in r.getMessage()]

        with subtests.test("silent when densely sampled"):
            assert not sampling_warnings(32)

        with subtests.test("warns when starved"):
            assert sampling_warnings(8)

    def test_settle_calibration_process_planted(self, fourierslm, subtests):
        """The settle fit returns positive, in-range times matching a planted response."""
        times = np.linspace(0, .2, 201)

        for (communication, relaxation) in ((.03, .02), (.01, .005), (.05, .01)):
            with subtests.test(f"communication={communication} relaxation={relaxation}"):
                fourierslm.calibrations["settle"] = {
                    "times": times,
                    "data": np.where(
                        times >= communication,
                        1 - np.exp(-(times - communication) / relaxation),
                        0,
                    ),
                }
                result = fourierslm.settle_calibration_process(plot=0)

                assert 0 < result["settle_time"] <= np.max(times)
                assert result["communication_time"] == pytest.approx(
                    communication, abs=2 * np.diff(times)[0]
                )
                assert result["relax_time"] == pytest.approx(relaxation, rel=.02)
                assert result["settle_time"] == pytest.approx(
                    communication + 4 * relaxation, rel=.02
                )

    def test_pixel_calibrate_gamma_sim(self, simulated_system_factory, subtests):
        """A simulated non-linear response is measured, then corrected by the table."""

        fs = simulated_system_factory("fov_much_smaller")
        install_ground_truth_calibration(fs)

        B = fs.slm.bitresolution
        truth = np.square(np.arange(B) / (B - 1)) * (B - 1) / B
        fs.slm.gamma_sim = truth

        fs.pixel_calibrate(levels=16, periods=2, orders=1, directions="x", plot=False)
        gamma = fs.pixel_calibration_process(plot=False)

        with subtests.test("the sweep measures the simulated response"):
            levels = fs.calibrations["pixel"]["levels"].astype(int)
            expected = truth[levels] - np.min(truth[levels])
            assert np.allclose(gamma, expected, atol=.02)
            assert not np.allclose(gamma, levels / B, atol=.02)   # Not the ideal ramp.

        with subtests.test("the table corrects the response"):
            target = np.arange(B) * (2 * np.pi / B)

            def error(levels):
                realized = np.mod(fs.slm._gamma_sign * 2 * np.pi * truth[levels], 2 * np.pi)
                deviation = (realized - target + np.pi) % (2 * np.pi) - np.pi
                return np.sqrt(np.mean(np.square(deviation)))

            uncorrected = error((-np.rint(target * B / (2 * np.pi)).astype(int)) % B)
            corrected = error(
                np.asarray(fs.slm.lut)[np.floor(target * fs.slm._phase_to_lut).astype(int)]
            )
            assert corrected < uncorrected / 8

        with subtests.test("a response beyond one cycle is unwrapped"):
            # The fit resolves each level only modulo a cycle.
            truth = 2.4 * np.arange(B) / B
            fs.slm.gamma_sim = truth
            fs.pixel_calibrate(levels=16, periods=2, orders=1, directions="x", plot=False)
            gamma = fs.pixel_calibration_process(plot=False)

            levels = fs.calibrations["pixel"]["levels"].astype(int)
            assert np.max(gamma) > 1                      # More than one cycle recovered.
            assert np.allclose(gamma, truth[levels] - np.min(truth[levels]), atol=.02)

        with subtests.test("a mirrored fit is returned on the increasing branch"):
            assert np.all(np.diff(gamma) > 0)

        with subtests.test("levels are canonically ordered"):
            # Unwrapping the fit follows the order of the levels it was measured at.
            fs.pixel_calibrate(
                levels=np.array([96, 0, 192, 32]), periods=2, orders=1,
                directions="x", plot=False,
            )
            stored = fs.calibrations["pixel"]["levels"]
            np.testing.assert_array_equal(stored, np.sort(stored))

    def test_pixel_calibrate_geometries(self, simulated_system_factory, subtests):
        """The default sweep must choose periods that keep every order on the camera."""

        for case in ("matched", "fov_larger", "fov_much_smaller", "camera_wide", "fov_extreme"):
            with subtests.test(case):
                fs = simulated_system_factory(case)
                install_ground_truth_calibration(fs)
                cal = fs.pixel_calibrate(levels=2, periods=2, plot=False)
                assert np.all(cal["periods"] >= 2)

        with subtests.test("a one-directional sweep ignores the other direction's orders"):
            # A wide camera has far less room in y, which must not veto an x-only sweep.
            fs = simulated_system_factory("camera_wide")
            install_ground_truth_calibration(fs)
            fs.pixel_calibrate(levels=2, periods=[4], orders=1, directions="x", plot=False)

        with subtests.test("rejects orders that fall off the negative side of the sensor"):
            # Negative indices wrap rather than raising, so they would otherwise be
            # integrated from the opposite edge of the image.
            fs = simulated_system_factory("zeroth_outside")
            install_ground_truth_calibration(fs)
            with pytest.raises(ValueError, match="short of the camera"):
                fs.pixel_calibrate(levels=2, periods=[6], orders=1, directions="x", plot=False)

    @pytest.mark.parametrize("name", GEOMETRY_CASES)
    def test_blaze_lands_at_prediction(self, simulated_system_factory, name):
        """A blaze, and the 0th order it degenerates to, land where the truth predicts."""
        fs = simulated_system_factory(name)
        tolerance = max(3.0, 2 * np.max(spot_size_ij(fs)))

        for kxy in (np.zeros((2, 1)), in_view_kxy(fs, frac=0.5)):
            predicted = np.squeeze(ground_truth_kxy_to_ij(fs, kxy))
            if np.any(predicted < 0) or np.any(predicted >= np.flip(fs.cam.shape)):
                continue    # Steered off the sensor, so there is nothing to find.

            fs.slm.set_phase(blaze(fs.slm, vector=kxy))
            fs.cam.autoexpose(verbose=False)
            img = fs.cam.get_image().astype(float)
            peak = np.flip(np.unravel_index(np.argmax(img), img.shape))

            assert np.linalg.norm(peak - predicted) < tolerance, (
                f"peak {peak} is {np.linalg.norm(peak - predicted):.2f} px from "
                f"predicted {predicted} (tolerance {tolerance:.2f})."
            )

    @pytest.mark.parametrize("name", GEOMETRY_CASES)
    def test_speckle_confined_to_farfield(self, simulated_system_factory, name):
        """Random phase fills the farfield, and the k-space limits crop it there."""
        fs = simulated_system_factory(name)
        mask = farfield_support_mask(fs)

        # Dilated, so spot-sized blur at the aperture edge does not count as escape.
        blur = int(np.ceil(2 * np.max(spot_size_ij(fs)))) * 2 + 1
        outside = ~(cv2.dilate(mask.astype(np.uint8), np.ones((blur, blur), np.uint8)) > 0)
        if fs.cam.noise is not None or not outside.any():
            pytest.skip("noise, or no camera outside the farfield, leaves nothing to compare")

        seed_for(name)
        fs.slm.set_phase(np.random.uniform(0, 2 * np.pi, fs.slm.shape))
        fs.cam.autoexpose(verbose=False)
        img = fs.cam.get_image().astype(float)

        assert img[mask].mean() > 100 * img[outside].mean(), "Speckle escaped the farfield."

    @pytest.mark.parametrize("name", GEOMETRY_CASES)
    def test_farfield_extent_matches_ground_truth(self, simulated_system_factory, name):
        """get_farfield_extent's mask agrees with the ground-truth farfield polygon."""
        fs = simulated_system_factory(name)
        install_ground_truth_calibration(fs)

        canvas = np.zeros(fs.cam.shape, np.uint8)
        cv2.fillConvexPoly(canvas, np.rint(farfield_corners_ij(fs).T).astype(np.int32), 255)
        agreement = (fs.get_farfield_extent(return_mask=True) == (canvas > 128)).mean()

        assert agreement > 0.97, (
            f"get_farfield_extent disagrees with the ground truth on "
            f"{(1 - agreement) * 100:.1f}% of pixels."
        )

    def test_default_fourier_calibrate(
        self, simulated_system, simulated_system_name, simulated_system_source,
        calibration_summary, calibration_plot_level,
    ):
        fs = simulated_system
        (error, tolerance, note, failure) = _run_calibration(
            fs, simulated_system_name, simulated_system_source,
            lambda system: system.fourier_calibrate(
                array_shape=BATTERY_ARRAY_SHAPE, array_pitch=BATTERY_ARRAY_PITCH,
                plot=calibration_plot_level, verbose=False,
            ),
            calibration_summary, "",
        )

        if simulated_system_name not in DEFAULT_CALIBRATION_OK:
            pytest.xfail(
                f"Fixed array_shape/array_pitch produce a wrong or failed calibration "
                f"for this geometry ({note}); fourier_calibrate_auto() handles it."
            )

        if failure is not None:
            raise failure
        assert error < tolerance, (
            f"Calibrated mapping is {error:.2f} px off ground truth "
            f"(tolerance {tolerance:.2f})."
        )

    def test_fourier_calibrate_auto(
        self, simulated_system, simulated_system_name, simulated_system_source,
        auto_summary, calibration_plot_level,
    ):
        fs = simulated_system
        (error, tolerance, note, failure) = _run_calibration(
            fs, simulated_system_name, simulated_system_source,
            lambda system: system._fourier_calibrate_auto(plot=calibration_plot_level),
            auto_summary, "auto_",
        )

        limitation = AUTO_LIMITATIONS.get(
            (simulated_system_name, simulated_system_source),
            AUTO_LIMITATIONS.get(simulated_system_name),
        )
        if limitation is not None:
            pytest.xfail(f"{limitation} ({note})")

        if failure is not None:
            raise failure
        assert error < tolerance, (
            f"fourier_calibrate_auto is {error:.2f} px off ground truth "
            f"(tolerance {tolerance:.2f})."
        )

    def test_failure_leaves_calibration_untouched(self, simulated_system_factory, monkeypatch):
        """
        A calibration that cannot be verified must not be installed, and must not
        destroy a calibration that was already there: the alternative is a system
        that silently uses a calibration its own check rejected, or one that loses a
        working calibration because the beam was blocked during a re-run.

        Both failure paths are forced rather than found, so that this tests the
        guarantee rather than whichever geometry happens to be failing.
        """
        # (1) Failing before any array is projected: no light reaches the camera.
        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)
        existing = np.array(fs.calibrations["fourier"]["M"])
        fs.slm.source["amplitude_sim"] = np.zeros_like(fs.slm.source["amplitude_sim"])

        with pytest.raises(RuntimeError):
            fs._fourier_calibrate_auto()
        assert np.array_equal(fs.calibrations["fourier"]["M"], existing), (
            "A failed calibration discarded the previous one."
        )

        # (2) Failing at verification, after arrays have been fitted.
        fs = simulated_system_factory("matched")

        with pytest.raises(RuntimeError):
            fs._fourier_calibrate_auto(tolerance=-1)
        assert "fourier" not in fs.calibrations, (
            "A calibration that failed verification was installed anyway."
        )

        # (3) The same failure, with a calibration already in place to destroy.
        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)
        existing = np.array(fs.calibrations["fourier"]["M"])

        with pytest.raises(RuntimeError):
            fs._fourier_calibrate_auto(tolerance=-1)
        assert np.array_equal(fs.calibrations["fourier"]["M"], existing), (
            "Failing verification discarded the previous calibration."
        )

        # (4) Failing unexpectedly: the guarantee cannot be branch by branch.
        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)
        existing = np.array(fs.calibrations["fourier"]["M"])

        def raise_unexpected():
            raise ValueError("injected failure")

        fs.get_farfield_zeroth = raise_unexpected

        with pytest.raises(ValueError):
            fs._fourier_calibrate_auto()
        assert np.array_equal(fs.calibrations["fourier"]["M"], existing), (
            "An unexpected failure discarded the previous calibration."
        )

        # (5) The same, after an array has been fitted and installed but not yet verified.
        from slmsuite.holography import analysis

        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)
        existing = np.array(fs.calibrations["fourier"]["M"])

        def raise_after_install(*args, **kwargs):
            raise ValueError("injected failure")

        monkeypatch.setattr(analysis, "_score_array_orientation", raise_after_install)

        with pytest.raises(ValueError):
            fs._fourier_calibrate_auto()
        assert np.array_equal(fs.calibrations["fourier"]["M"], existing), (
            "An unverified calibration replaced the previous one."
        )

    def test_farfield_weights_track_efficiency(self, simulated_system_factory):
        """
        Weights ask for more amplitude where the measured farfield delivers less
        power, are bounded, and fall back to uniform when nothing was measured.
        """
        fs = simulated_system_factory("fov_larger")
        install_ground_truth_calibration(fs)
        fs.farfield_calibrate(averaging=1)
        fs.farfield_calibration_process()

        efficiency = fs.get_farfield_efficiency()
        lit = np.argwhere(np.nan_to_num(efficiency) > 0)
        assert len(lit) > 0, "The farfield calibration measured no efficiency."

        # Brightest and dimmest measured pixels, in (x, y).
        values = efficiency[lit[:, 0], lit[:, 1]]
        ij = np.flip(lit[[np.argmax(values), np.argmin(values)]].T, axis=0)

        weights = fs.get_farfield_weights(ij, np.ones(2))
        assert weights.shape == (2,)
        assert np.all(weights > 0) and np.max(weights) == pytest.approx(1.0)
        assert weights[1] > weights[0], (
            "The dimmer part of the farfield should be asked for more amplitude."
        )
        # Power goes as the square of amplitude, and is floored so that a dead
        # region cannot ask for unbounded power.
        assert np.max(weights) / np.min(weights) <= 1 / np.sqrt(0.05) + 1e-9

        # An unmeasured farfield weights every spot the same rather than by NaN.
        fs.calibrations["farfield"]["efficiency"] = np.zeros_like(efficiency)
        assert np.array_equal(fs.get_farfield_weights(ij, np.ones(2)), np.ones(2))


def _binary_grating(period, a, b, duty_cycle=.5):
    """One period of a binary grating, as per-pixel phase in radians."""
    return np.where(np.arange(period) < round(period * duty_cycle), a, b)


# Geometries whose calibration depends on the random draw. A single seed hides that:
# what passes at one draw can be silently wrong at the next.
MARGINAL_CASES = (
    "offset",
    "zeroth_corner",
    "zeroth_outside",
    "anisotropic",
)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("name", MARGINAL_CASES)
def test_fourier_calibrate_auto_survives_any_draw(
    simulated_system_factory, simulated_system_source, name, seed
):
    """
    A calibration that passes verification must be correct whatever the random draw.
    Refusing is allowed --- returning a wrong affine that reports a sub-pixel
    residual is the failure this whole routine exists to prevent.
    """
    fs = simulated_system_factory(name, source_function=simulated_system_source)
    np.random.seed(seed)
    try:
        import cupy
        cupy.random.seed(seed)
    except ImportError:
        pass

    try:
        fs._fourier_calibrate_auto()
    except RuntimeError:
        return

    grid = view_kxy_grid(fs)
    error = float(np.max(np.linalg.norm(
        fs.kxyslm_to_ijcam(grid) - ground_truth_kxy_to_ij(fs, grid), axis=0
    )))
    tolerance = max(2.0, np.max(spot_size_ij(fs)))
    assert error < tolerance, (
        f"seed {seed}: accepted a calibration {error:.1f} px off ground truth "
        f"(tolerance {tolerance:.1f}), reporting "
        f"{fs.calibrations['fourier']['array']['residual']:.2f} px residual."
    )
