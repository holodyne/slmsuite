"""
Pytest configuration and fixtures for slmsuite tests.

The fixtures in this file support testing with any SLM or Camera subclass.
By default, SimulatedSLM and SimulatedCamera are used for fast, hardware-free testing.

To test with real hardware, set environment variables:
    SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.thorlabs.ThorlabsSLM
    SLMSUITE_TEST_SLM_ARGS='{"monitor_id": 1}'
    SLMSUITE_TEST_CAMERA_CLASS=slmsuite.hardware.cameras.thorlabs.ThorlabsCamera
    SLMSUITE_TEST_CAMERA_ARGS='{"serial": "12345"}'

Automatic Features:
- All tests automatically log to tests/output/YYYYMMDD_HHMMSS/pytest.log
- Matplotlib plots automatically saved to tests/output/YYYYMMDD_HHMMSS/
- Random seed generated per session (logged for reproducibility)
- slmsuite package logging: INFO level
- External packages logging: WARNING level and above only
"""
import importlib
import json
import logging
import os
import sys
import tempfile
import zlib
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Test modules import the simulated-system builders below from this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Globals

_TEST_RUN_OUTPUT_DIR = None


def get_test_run_output_dir():
    """Helper function to get current test run output directory."""
    return _TEST_RUN_OUTPUT_DIR


@pytest.fixture(scope="session")
def test_output_dir():
    """Directory that this test run's plots and logs are saved to."""
    return get_test_run_output_dir()


try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False


@pytest.fixture(scope="session")
def has_cupy():
    """Check if CuPy is available for GPU tests."""
    return HAS_CUPY


@pytest.fixture(scope="session")
def random_seed():
    """
    Generate and return a random seed for the test session.

    The seed is logged and can be used to reproduce test runs.
    Also sets numpy's global random seed for reproducibility.

    Returns
    -------
    int
        Random seed value for this test session
    """
    import random
    seed = random.randint(0, 2**32 - 1)

    # Set numpy's random seed
    np.random.seed(seed)

    # Set CuPy's random seed if available
    if HAS_CUPY:
        cp.random.seed(seed)

    # Log the seed for reproducibility
    logger = logging.getLogger("conftest")
    logger.info(f"Random seed for this session: {seed}")
    print(f"\nRandom seed for this session: {seed}")

    return seed


# Fixtures for SLM and Camera instances, with dynamic configuration via environment variables.

from slmsuite._plotting import _slmsuite_plt_show
from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.holography.algorithms import SpotHologram
from slmsuite.holography.toolbox import format_2vectors, phase

_TEST_SMALL_RESOLUTION = (128, 128)


def _get_class_from_string(class_path):
    """
    Import and return a class from a module path string.

    Parameters
    ----------
    class_path : str
        Full path to class, e.g., 'slmsuite.hardware.slms.simulated.SimulatedSLM'

    Returns
    -------
    class
        The imported class
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@pytest.fixture
def slm_class():
    """
    Return the SLM class to use for testing.

    By default returns SimulatedSLM. Can be overridden via environment variable:
    SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.santec.Santec

    Returns
    -------
    class
        SLM subclass to instantiate
    """
    class_path = os.environ.get('SLMSUITE_TEST_SLM_CLASS', None)
    if class_path:
        return _get_class_from_string(class_path)
    return SimulatedSLM


@pytest.fixture
def slm_kwargs():
    """
    Return keyword arguments for SLM instantiation.

    By default returns arguments for SimulatedSLM. Can be overridden via:
    SLMSUITE_TEST_SLM_ARGS='{"bitdepth": 12}'

    Returns
    -------
    dict
        Keyword arguments for SLM constructor
    """
    args_json = os.environ.get('SLMSUITE_TEST_SLM_ARGS', None)
    if args_json:
        return json.loads(args_json)

    # Default args for SimulatedSLM
    return {
        'resolution': (1920, 1080),
        'pitch_um': (8.0, 8.0),
        'bitdepth': 8,
        'wav_um': 0.78
    }


@pytest.fixture
def slm(slm_class, slm_kwargs):
    """
    Fixture providing an SLM instance for testing.
    """
    slm_instance = slm_class(**slm_kwargs)
    yield slm_instance

    # Cleanup
    try:
        slm_instance.close()
    except Exception:
        pass


@pytest.fixture
def slm_small(slm_kwargs):
    """
    Fixture providing an SLM instance for testing.
    """
    kwargs = slm_kwargs.copy()
    kwargs['resolution'] = _TEST_SMALL_RESOLUTION
    slm_instance = SimulatedSLM(**kwargs)
    yield slm_instance

    # Cleanup
    try:
        slm_instance.close()
    except Exception:
        pass


@pytest.fixture
def camera_class():
    """
    Return the Camera class to use for testing.

    By default returns SimulatedCamera. Can be overridden via:
    SLMSUITE_TEST_CAMERA_CLASS=slmsuite.hardware.cameras.alliedvision.AlliedVision

    Returns
    -------
    class
        Camera subclass to instantiate
    """
    class_path = os.environ.get('SLMSUITE_TEST_CAMERA_CLASS', None)
    if class_path:
        return _get_class_from_string(class_path)
    return SimulatedCamera


@pytest.fixture
def camera_kwargs(slm):
    """
    Return keyword arguments for Camera instantiation.

    By default returns arguments for SimulatedCamera. Can be overridden via:
    SLMSUITE_TEST_CAMERA_ARGS='{"serial": "12345", "bitdepth": 8}'

    Parameters
    ----------
    slm : SLM
        SLM instance (required for SimulatedCamera)

    Returns
    -------
    dict
        Keyword arguments for Camera constructor
    """
    args_json = os.environ.get('SLMSUITE_TEST_CAMERA_ARGS', None)
    if args_json:
        return json.loads(args_json)

    # Default args for SimulatedCamera
    return {
        'slm': slm,
        'resolution': (512, 512),
        'pitch_um': (5.5, 5.5),
        'bitdepth': 8
    }


@pytest.fixture
def camera(camera_class, camera_kwargs):
    """
    Fixture providing a Camera instance for testing.

    By default returns SimulatedCamera, but can be configured to return any Camera subclass
    via environment variables SLMSUITE_TEST_CAMERA_CLASS and SLMSUITE_TEST_CAMERA_ARGS.
    """
    cam = camera_class(**camera_kwargs)
    yield cam

    # Cleanup
    try:
        cam.close()
    except Exception:
        pass


@pytest.fixture
def camera_small(slm_small, camera_kwargs):
    """
    Fixture providing a Camera instance for testing.
    """
    kwargs = camera_kwargs.copy()
    kwargs["resolution"] = _TEST_SMALL_RESOLUTION
    kwargs["slm"] = slm_small
    cam = SimulatedCamera(**kwargs)
    yield cam

    # Cleanup
    try:
        cam.close()
    except Exception:
        pass

@pytest.fixture
def fourierslm(camera, slm):
    """
    Fixture providing a FourierSLM instance for testing.
    """
    camera.set_exposure(0.1)  # Don't overexpose the fourier calibration.
    fs = FourierSLM(camera, slm, mag=1.0)
    yield fs

    # Cleanup
    try:
        fs.close()
    except Exception:
        pass

@pytest.fixture
def fourierslm_calibrated(fourierslm):
    """FourierSLM with a completed Fourier calibration."""
    fourierslm.fourier_calibrate(array_pitch=30, array_shape=10, plot=True)
    return fourierslm


# Simulated SLM-camera systems spanning geometry corner cases: camera FOV much
# larger/smaller than the SLM farfield, rotation, shear, offset 0th order, noise,
# apertures, anisotropy, source illumination and wavefront aberration.
#
# The ground truth is stored in the hardware itself (``cam.M`` / ``cam.b`` for
# affine-placed cameras; derived from the shapes for the identity case) and is
# retrieved with ``ground_truth_affine()``, so calibration routines can be validated
# without a chicken-and-egg dependence on the calibration under test.
#
# The key geometry parameter is ``ratio``: the width of the SLM's accessible
# farfield (the k-space square +/- wav/2*pitch) divided by the width of the camera's
# field of view. Below one the camera sees the whole farfield cropped by its
# aperture; above one it sees only an interior patch.

def seed_for(name):
    """
    Seed every generator a case can draw from, so that it behaves the same however
    the rest of the suite has left the random state. Holograms initialize from
    cupy's generator when cupy is installed, so seeding numpy alone leaves them
    free-running.
    """
    seed = zlib.crc32(name.encode())
    np.random.seed(seed)
    try:
        import cupy
        cupy.random.seed(seed)
    except ImportError:
        pass


SIMULATED_SYSTEM_DEFAULTS = dict(
    slm_resolution=(128, 128),      # (width, height)
    slm_pitch_um=(8.0, 8.0),
    cam_resolution=(128, 128),      # (width, height)
    cam_pitch_um=(5.5, 5.5),
    wav_um=0.78,
    slm_bitdepth=8,
    cam_bitdepth=8,
    ratio=None,         # farfield width / camera width. None -> identity (no affine).
    theta=0.0,          # camera rotation (radians, ccw)
    shear_angle=0.0,    # camera shear (radians); float or (x, y)
    offset_frac=None,   # fractional (x, y) position of the 0th order on the camera.
                        # None -> camera center. May lie outside [0, 1].
    noise=None,         # SimulatedCamera noise dict
    source_function=None,   # set_source_analytic() profile. None -> uniform illumination.
    aberration=None,    # (ANSI indices, weights) for zernike_sum(), in radians.
)

# A real source underfills its SLM, so every case is run both ways.
SIMULATED_SYSTEM_SOURCES = (None, "gaussian2d")

# Camera noise model shared by noisy cases: exposure-dependent dark background plus
# exposure-independent Poisson readout noise (fractions of the dynamic range),
# scaled by ``severity``.
def _default_noise(severity=1):
    return {
        "dark": lambda img: np.random.normal(
            0.005 * severity * img, 0.002 * severity * img
        ),
        "read": lambda img: np.random.poisson(0.03 * severity * img),
    }


SIMULATED_SYSTEM_CASES = {
    # Direct knm sampling (no affine interpolation): the camera crops the central
    # (cam_resolution) window of the SLM-shaped k-space grid, one pixel per knm cell.
    # Effectively a "camera FOV much smaller than farfield" case (ratio ~ N*pitch/wav).
    "identity": dict(),
    # Farfield exactly fills the camera.
    "matched": dict(ratio=1.0),
    # Camera FOV much larger than the farfield: the whole farfield occupies the
    # central ~1/25 of the camera area.
    "fov_much_larger": dict(ratio=0.2),
    # Camera FOV moderately larger than the farfield.
    "fov_larger": dict(ratio=0.5),
    # Camera FOV moderately smaller than the farfield.
    "fov_smaller": dict(ratio=2.0),
    # Camera FOV much smaller than the farfield: sees a small interior patch,
    # no aperture edges visible. Diffraction-limited spots are several pixels wide.
    "fov_much_smaller": dict(ratio=6.0),
    # Rotated camera.
    "rotated": dict(ratio=1.5, theta=np.radians(20)),
    # Sheared (non-orthogonal) axes.
    "sheared": dict(ratio=1.2, shear_angle=(np.radians(8), np.radians(-5))),
    # 0th order far off-center (near a camera corner).
    "offset": dict(ratio=1.0, offset_frac=(0.28, 0.35)),
    # 0th order entirely outside the camera FOV (common when the 0th order is
    # deliberately steered off-camera). The camera sees an off-axis k-space stripe.
    "zeroth_outside": dict(ratio=1.5, offset_frac=(-0.3, 0.5)),
    # Camera noise on top of a moderately-smaller FOV.
    "noisy": dict(ratio=1.5, noise="default"),
    # Anisotropic everything: rectangular SLM and camera, anisotropic pixel pitches,
    # and different farfield/camera ratio per axis.
    "anisotropic": dict(
        ratio=(1.5, 0.7),
        slm_resolution=(160, 96),
        cam_resolution=(192, 128),
        cam_pitch_um=(5.5, 4.5),
    ),
    # An odd number of mirrors in the train inverts the image's parity, so the
    # affine has a negative determinant and no rotation can reproduce it.
    "mirrored": dict(ratio=(-1.0, 1.0)),
    # Noise, more of it, and noise on a farfield that does not fill the camera.
    "noisy_severe": dict(ratio=1.5, noise=3),
    # 0th order in the very corner of the sensor.
    "zeroth_corner": dict(ratio=1.2, offset_frac=(0.02, 0.02)),
    # A camera far from square, so that x and y cannot be confused for each other.
    "camera_wide": dict(ratio=1.0, cam_resolution=(256, 96)),
    # Non-square SLM pixels, which make the farfield itself non-square: the one
    # thing that stops the two k-space axes from sharing a scale.
    "pitch_anisotropic": dict(ratio=1.0, slm_pitch_um=(8.0, 12.0)),
    # One axis of the farfield far larger than the camera, the other far smaller.
    "fov_extreme": dict(ratio=(0.3, 3.0)),
    # Aberrated wavefronts, well beyond the Marechal limit: the source phase is not
    # flat, so spots are broadened and distorted rather than diffraction-limited.
    # Weights are radians against ANSI Zernike polynomials of unit peak-to-valley.
    "defocus": dict(ratio=1.0, aberration=((4,), (3.0,))),
    "coma": dict(ratio=1.5, theta=np.radians(15), aberration=((7, 8), (3.0, -2.4))),
}


def f_eff_from_ratio(ratio, cam_width_px, cam_pitch_um, slm_pitch_um, wav_um):
    """
    Effective focal length (``"norm"`` units, wavelengths) such that the SLM farfield
    width equals ``ratio`` times the camera width along the corresponding axis.

    The farfield spans ``wav_um / slm_pitch_um`` in normalized ``kxy`` units and an
    effective focal length ``f_eff`` maps ``kxy`` to ``f_eff * wav_um / cam_pitch_um``
    camera pixels.
    """
    return ratio * cam_width_px * cam_pitch_um * slm_pitch_um / wav_um**2


def build_simulated_system(name, **overrides):
    """
    Constructs a :class:`FourierSLM` (:class:`SimulatedCamera` +
    :class:`SimulatedSLM`) for a named case in :data:`SIMULATED_SYSTEM_CASES`
    (with optional config ``overrides``).
    Retrieve the ground-truth placement with :func:`ground_truth_affine`.
    """
    config = dict(SIMULATED_SYSTEM_DEFAULTS)
    config.update(SIMULATED_SYSTEM_CASES[name])
    config.update(overrides)

    noise = config["noise"]
    if noise is not None:
        noise = _default_noise(1 if noise == "default" else noise)

    slm = SimulatedSLM(
        resolution=config["slm_resolution"],
        pitch_um=config["slm_pitch_um"],
        bitdepth=config["slm_bitdepth"],
        wav_um=config["wav_um"],
    )
    if config["source_function"] is not None:
        slm.set_source_analytic(config["source_function"], units="frac", sim=True)
    if config["aberration"] is not None:
        (indices, weights) = config["aberration"]
        slm.source["phase_sim"] = slm.source["phase_sim"] + phase.zernike_sum(
            slm, indices, weights
        )
    cam = SimulatedCamera(
        slm,
        resolution=config["cam_resolution"],
        pitch_um=config["cam_pitch_um"],
        bitdepth=config["cam_bitdepth"],
        noise=noise,
    )

    if config["ratio"] is not None:
        ratio = np.broadcast_to(np.squeeze(config["ratio"]), (2,))
        cam_res = np.squeeze(config["cam_resolution"])
        cam_pitch = np.squeeze(config["cam_pitch_um"])
        slm_pitch = np.squeeze(config["slm_pitch_um"])
        f_eff = [
            f_eff_from_ratio(ratio[a], cam_res[a], cam_pitch[a], slm_pitch[a], config["wav_um"])
            for a in range(2)
        ]

        offset = None  # build_affine defaults to the camera center.
        if config["offset_frac"] is not None:
            offset = np.squeeze(config["offset_frac"]) * cam_res

        M, b = cam.build_affine(
            f_eff,
            units="norm",
            theta=config["theta"],
            shear_angle=config["shear_angle"],
            offset=offset,
        )
        cam.set_affine(M, b)

    return FourierSLM(cam, slm)


# --- Ground-truth geometry helpers ---
# These operate on the FourierSLM's hardware directly; nothing below stores state.

def ground_truth_affine(fs):
    """
    The ground-truth affine ``ij = M @ kxy + b`` of a system built by
    :func:`build_simulated_system`, as ``(M, b)`` with shapes ``(2, 2)`` and
    ``(2, 1)``. For affine-placed cameras this is the :class:`SimulatedCamera`'s
    own placement (``cam.M`` / ``cam.b``). For the identity (no-affine) case, the
    camera crops the central window of the SLM-shaped knm grid, equivalent to
    ``M = diag(N * pitch)`` with ``b`` at the camera center.
    """
    if getattr(fs.cam, "M", None) is not None:
        return (np.array(fs.cam.M, dtype=float), format_2vectors(fs.cam.b).astype(float))
    else:
        (Ny, Nx) = fs.slm.shape
        pitch = np.squeeze(fs.slm.pitch)  # Normalized (x, y) pitch.
        M = np.diag([Nx * pitch[0], Ny * pitch[1]]).astype(float)
        b = format_2vectors(np.flip(fs.cam.shape) / 2).astype(float)
        return (M, b)


def ground_truth_kxy_to_ij(fs, kxy):
    """Ground-truth version of :meth:`FourierSLM.kxyslm_to_ijcam`."""
    (M, b) = ground_truth_affine(fs)
    return M @ format_2vectors(kxy) + b


def ground_truth_ij_to_kxy(fs, ij):
    """Ground-truth version of :meth:`FourierSLM.ijcam_to_kxyslm`."""
    (M, b) = ground_truth_affine(fs)
    return np.linalg.solve(M, format_2vectors(ij) - b)


def install_ground_truth_calibration(fs):
    """
    Installs the ground-truth affine as the Fourier calibration. Bypasses
    :meth:`FourierSLM.fourier_calibrate_analytic` because that method also sets
    the camera affine when absent, which would switch the identity case's camera
    out of its direct-sampling mode.
    """
    (M, b) = ground_truth_affine(fs)
    fs.calibrations["fourier"] = {
        "M": M,
        "b": b,
        "a": format_2vectors([0, 0]).astype(float),
    }
    fs.calibrations["fourier"].update(fs._get_calibration_metadata())
    return fs.calibrations["fourier"]


def farfield_extent_kxy(fs):
    """Farfield half-widths ``(kx_max, ky_max)`` in normalized units."""
    return 1 / (2 * np.squeeze(fs.slm.pitch))


def farfield_corners_ij(fs):
    """Corners of the farfield square on the camera, shape ``(2, 4)``."""
    (kx, ky) = farfield_extent_kxy(fs)
    corners = np.array([[-kx, kx, kx, -kx], [-ky, -ky, ky, ky]])
    return ground_truth_kxy_to_ij(fs, corners)


def camera_corners_kxy(fs):
    """Corners of the camera FOV in kxy space, shape ``(2, 4)``."""
    (h, w) = fs.cam.shape
    corners = np.array([[0, w, w, 0], [0, 0, h, h]], dtype=float)
    return ground_truth_ij_to_kxy(fs, corners)


def view_bounds_kxy(fs):
    """Bounds of the intersection of the SLM's farfield with the camera's FOV."""
    ff = farfield_extent_kxy(fs)
    cam = camera_corners_kxy(fs)
    lo = np.maximum(cam.min(axis=1), -ff)
    hi = np.minimum(cam.max(axis=1), ff)
    if np.any(lo >= hi):
        raise RuntimeError(f"'{fs.name}': camera does not view the farfield.")
    return (lo, hi)


def in_view_kxy(fs, frac=0.5):
    """
    A kxy vector inside the intersection of the farfield and the camera FOV,
    suitable as a test blaze target. ``frac`` scales from the center of the
    intersection's bounding box (0) towards its edge (1).
    """
    (lo, hi) = view_bounds_kxy(fs)
    return format_2vectors((lo + hi) / 2 + frac * (hi - lo) / 2 * np.array([0.5, -0.3]))


def view_kxy_grid(fs, count=3, frac=0.8):
    """
    Grid of kxy vectors spanning what the camera views, for measuring a calibration
    over the area it is used on. Points along a single ray cannot see an error
    orthogonal to that ray, which is most of the ways an affine can be wrong.
    """
    (lo, hi) = view_bounds_kxy(fs)
    (x, y) = np.meshgrid(*[
        (lo[axis] + hi[axis]) / 2
        + frac * (hi[axis] - lo[axis]) / 2 * np.linspace(-1, 1, count)
        for axis in range(2)
    ])
    return np.vstack((x.ravel(), y.ravel()))


def spot_size_ij(fs):
    """Approximate diffraction-limited spot size (x, y) in camera pixels."""
    (M, _) = ground_truth_affine(fs)
    (Ny, Nx) = fs.slm.shape
    pitch = np.squeeze(fs.slm.pitch)
    spot_kxy = np.array([1 / (Nx * pitch[0]), 1 / (Ny * pitch[1])])
    return np.abs(M) @ spot_kxy


def farfield_support_mask(fs):
    """
    Ground-truth support of the farfield on the camera: the region of the camera
    that falls inside the SLM's accessible k-space square.
    """
    (h, w) = fs.cam.shape
    (M, b) = ground_truth_affine(fs)
    ij = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=0).reshape(2, -1)
    kxy = np.linalg.solve(M, ij - b)
    ff = farfield_extent_kxy(fs)
    mask = (np.abs(kxy[0]) < ff[0]) & (np.abs(kxy[1]) < ff[1])
    return mask.reshape(h, w)


def array_kxy(fs, array_shape, array_pitch, array_center=None):
    """
    The ``"kxy"`` positions of the spots of a calibration array specified in the
    ``"knm"`` basis, following the canvas convention of
    :meth:`FourierSLM.fourier_grid_project()`. Note that the projected array omits
    its last two spots as a parity check, so those positions carry no light.
    """
    shape = SpotHologram.get_padded_shape(fs, padding_order=1, square_padding=True)
    step = np.broadcast_to(np.squeeze(array_pitch), (2,)) / (
        np.flip(np.squeeze(shape)) * np.squeeze(fs.slm.pitch)
    )
    array_shape = np.broadcast_to(np.squeeze(array_shape), (2,)).astype(int)

    (gx, gy) = np.meshgrid(
        (np.arange(array_shape[0]) - (array_shape[0] - 1) / 2) * step[0],
        (np.arange(array_shape[1]) - (array_shape[1] - 1) / 2) * step[1],
    )
    kxy = np.vstack((gx.ravel(), gy.ravel()))

    if array_center is not None:
        kxy = kxy + format_2vectors(np.squeeze(array_center) * step / np.squeeze(array_pitch))

    return kxy


def _plot_polygon(ax, corners, **kwargs):
    """Plot a closed polygon from ``(2, N)`` corners."""
    closed = np.hstack((corners, corners[:, [0]]))
    ax.plot(closed[0], closed[1], **kwargs)


def plot_image_dim(ax, img, **kwargs):
    """
    ``imshow`` with a compressive norm, so that dim spots
    (which calibration images are full of) remain visible next to bright ones.
    """
    return ax.imshow(
        img,
        norm=matplotlib.colors.PowerNorm(0.4, vmin=0, vmax=max(1, np.max(img))),
        **kwargs,
    )


def plot_calibration_diagnostic(fs, img=None, spots_kxy=None, name="", note=""):
    """
    Three panels against the ground truth: the array image overlaid with the true
    and calibrated spot positions, the error field between them (uniform for an
    offset error, swirling for a rotation, diverging for a scale), and the geometry
    that the calibration had to work with.
    """
    if img is None:
        img = fs.cam.last_image
    calibrated = "fourier" in fs.calibrations

    (fig, axs) = plt.subplots(1, 3, figsize=(18, 6))

    # 1) The image, with ground-truth and calibrated spot positions.
    plot_image_dim(axs[0], img)
    _plot_polygon(axs[0], farfield_corners_ij(fs), c="orange", ls="--", lw=1, label="farfield")
    if spots_kxy is not None:
        gt = ground_truth_kxy_to_ij(fs, spots_kxy)
        axs[0].scatter(
            gt[0], gt[1], fc="none", ec="lime", s=60, lw=0.75, label="truth",
        )
        if calibrated:
            cal = fs.kxyslm_to_ijcam(spots_kxy)
            axs[0].scatter(cal[0], cal[1], c="r", marker="x", s=25, lw=0.75, label="calibrated")
    axs[0].set_title("Array image")
    axs[0].legend(loc="upper right", fontsize="x-small")

    # 2) The calibration error field over the camera's view of the farfield.
    corners = camera_corners_kxy(fs)
    ff = farfield_extent_kxy(fs)
    lo = np.maximum(corners.min(axis=1), -ff)
    hi = np.minimum(corners.max(axis=1), ff)
    (gx, gy) = np.meshgrid(np.linspace(lo[0], hi[0], 7), np.linspace(lo[1], hi[1], 7))
    probe = np.vstack((gx.ravel(), gy.ravel()))
    gt = ground_truth_kxy_to_ij(fs, probe)

    if calibrated:
        cal = fs.kxyslm_to_ijcam(probe)
        error = np.linalg.norm(cal - gt, axis=0)
        axs[1].quiver(
            gt[0], gt[1], cal[0] - gt[0], cal[1] - gt[1],
            angles="xy", scale_units="xy", scale=1, width=0.004,
        )
        axs[1].scatter(gt[0], gt[1], s=4, c="lime")
        axs[1].set_title(
            f"Error: truth $\\rightarrow$ calibrated\n"
            f"median {np.median(error):.1f} px, max {np.max(error):.1f} px"
        )
    else:
        axs[1].scatter(gt[0], gt[1], s=4, c="lime")
        axs[1].set_title("Error (no calibration produced)")
    axs[1].invert_yaxis()
    axs[1].set_aspect("equal")

    # 3) Geometry: farfield support, camera FOV, array extent.
    # vmin/vmax are explicit: a mask that is entirely support would otherwise
    # autoscale to a degenerate range and render black.
    axs[2].imshow(farfield_support_mask(fs), cmap="Greys_r", vmin=0, vmax=1)
    (h, w) = fs.cam.shape
    _plot_polygon(
        axs[2], np.array([[0, w, w, 0], [0, 0, h, h]], dtype=float),
        c="c", lw=1.5, label="camera",
    )
    _plot_polygon(axs[2], farfield_corners_ij(fs), c="orange", ls="--", lw=1, label="farfield")
    if spots_kxy is not None:
        gt = ground_truth_kxy_to_ij(fs, spots_kxy)
        axs[2].scatter(gt[0], gt[1], fc="none", ec="lime", s=15, lw=0.5, label="array")
    axs[2].set_title("Geometry (white = support, black = cropped)")
    axs[2].set_facecolor("0.6")     # Distinguish "off camera" from "support".
    axs[2].legend(loc="upper right", fontsize="x-small")
    axs[2].set_aspect("equal")
    # Show everything, including geometry outside the camera.
    margin = 0.1 * max(w, h)
    stack = np.hstack((farfield_corners_ij(fs), np.array([[0, w], [0, h]], dtype=float)))
    axs[2].set_xlim(stack[0].min() - margin, stack[0].max() + margin)
    axs[2].set_ylim(stack[1].max() + margin, stack[1].min() - margin)

    for ax in axs:
        ax.set_xlabel("Camera $x$ [pix]")
        ax.set_ylabel("Camera $y$ [pix]")

    fig.suptitle(f"{name}{': ' if note else ''}{note}")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _slmsuite_plt_show(name=f"diagnostic_{name}")




def _close_quietly(fs):
    try:
        fs.close()
    except Exception:
        pass




@pytest.fixture(params=list(SIMULATED_SYSTEM_CASES.keys()))
def simulated_system_name(request):
    """Name of the simulated-system case (parameterized over every case)."""
    return request.param


@pytest.fixture(params=SIMULATED_SYSTEM_SOURCES, ids=lambda s: s or "uniform")
def simulated_system_source(request):
    """SLM illumination of the simulated-system case (parameterized over every source)."""
    return request.param


@pytest.fixture
def simulated_system(simulated_system_name, simulated_system_source):
    """
    Parameterized fixture running a test over every simulated-system case.
    Yields a plain ``FourierSLM`` (SimulatedCamera + SimulatedSLM); retrieve the
    ground truth with ``simulated_systems.ground_truth_affine(fs)`` and friends.
    """
    seed_for(simulated_system_name)
    fs = build_simulated_system(simulated_system_name, source_function=simulated_system_source)
    fs.cam.set_exposure(0.1)  # Don't overexpose by default.
    yield fs
    _close_quietly(fs)


@pytest.fixture
def simulated_system_factory():
    """
    Factory fixture for specific named cases:
    ``fs = simulated_system_factory("rotated")``.
    Accepts config overrides as keyword arguments.
    """
    built = []

    def factory(name, **overrides):
        seed_for(name)
        fs = build_simulated_system(name, **overrides)
        fs.cam.set_exposure(0.1)
        built.append(fs)
        return fs

    yield factory

    for fs in built:
        _close_quietly(fs)


# Matplotlib configuration (saving of plots)

@pytest.fixture(scope="session", autouse=True)
def configure_matplotlib_for_testing(request):
    """
    Configure matplotlib for testing environment.

    - Use Agg backend (non-interactive) to prevent blocking
    - Replace plt.show() to save figures with descriptive names
    - Save to current test run's timestamped directory

    TODO: determine when plt.show() is called outside of
    slmsuite's `slmsuite_plt_show()`.
    """
    # Check if we should save plots (default: True)
    save_plots = request.config.getoption("--save-plots")

    # Use non-interactive backend for testing
    matplotlib.use("Agg")
    plt.ioff()  # Disable interactive mode

    # Store original plt.show
    original_show = plt.show

    if save_plots:
        # Track figure count per test
        test_fig_counts = {}

        def custom_show(name=None, *args, **kwargs):
            """
            Replacement for plt.show() that saves figures with descriptive names.

            Format: {module}_{class}_{function}[_{name}]_fig{N}.png
            Saved to: tests/output/{timestamp}/
            """
            # Get output directory for this test run
            output_dir = get_test_run_output_dir()
            if output_dir is None:
                print("Warning: Test run output directory not initialized")
                plt.close('all')
                return

            # Get current test info from pytest environment variable
            test_name = os.environ.get('PYTEST_CURRENT_TEST', '')

            if not test_name:
                # Fallback if called outside test context
                key = f"unknown_{name}" if name else "unknown"
                filename = output_dir / f"{key}_fig{len(test_fig_counts)}.png"
                figs = [plt.figure(n) for n in plt.get_fignums()]
                for fig in figs:
                    fig.savefig(filename, dpi=150, bbox_inches='tight')
            else:
                # Parse test path: "tests/holography/test_algorithms.py::TestHologram::test_gs_converges (call)"
                test_path = test_name.split(' ')[0]  # Remove "(call)" part

                # Extract components
                parts = []
                if '::' in test_path:
                    file_and_rest = test_path.split('::')
                    # Get module name from file path
                    module = file_and_rest[0].split('/')[-1].replace('.py', '')
                    parts.append(module)
                    # Add class and function if present
                    parts.extend(file_and_rest[1:])
                else:
                    parts.append('unknown')

                # Append plot-site name from _slmsuite_plt_show if provided
                if name:
                    parts.append(name)

                # Build filename key; each unique key has its own counter
                key = '_'.join(parts)

                # Save all open figures
                figs = [plt.figure(n) for n in plt.get_fignums()]
                for fig in figs:
                    test_fig_counts[key] = test_fig_counts.get(key, 0) + 1
                    filename = output_dir / f"{key}_fig{test_fig_counts[key]}.png"
                    fig.savefig(filename, dpi=150, bbox_inches='tight')
                    # Print relative path
                    rel_path = filename.relative_to(Path("tests/output"))
                    print(f"Saved plot: tests/output/{rel_path}")

            # Close figures to free memory
            plt.close('all')

        # Replace plt.show and configure slmsuite's internal handler
        plt.show = custom_show
        import slmsuite
        slmsuite.configure_plotting(custom_show)
    else:
        # If plots disabled, just close figures silently
        def no_show(_name=None, *_args, **_kwargs):
            plt.close('all')
        plt.show = no_show
        import slmsuite
        slmsuite.configure_plotting(no_show)

    yield

    # Restore original plt.show and slmsuite handler
    plt.show = original_show
    slmsuite.configure_plotting("show")


@pytest.fixture
def mpl_test(request):
    """
    Per-test fixture for matplotlib tests.

    Provides automatic figure cleanup and easy access to plt.
    """
    # Clear any existing figures before test
    plt.close('all')

    yield plt

    # Cleanup after test
    plt.close('all')


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--save-plots",
        action="store_true",
        default=False,
        help="Save matplotlib plots to tests/output/{timestamp}/"
    )

# Logging and final configuration

@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(autouse=True)
def test_logger(request):
    """
    Provides a test-specific logger with proper naming.

    This fixture is automatically used for every test (autouse=True).
    Tests can access the logger via request.node.test_logger if needed.

    Logger name format: {module}.{class}.{function}
    Example: test_algorithms.TestHologram.test_gs_converges
    """
    # Build logger name from test node
    parts = []
    if request.module:
        module_name = request.module.__name__.split('.')[-1]
        parts.append(module_name)
    if request.cls:
        parts.append(request.cls.__name__)
    if request.function:
        parts.append(request.function.__name__)

    logger_name = ".".join(parts)
    logger = logging.getLogger(logger_name)

    # Store logger in request for access by tests if needed
    request.node.test_logger = logger

    # Log test start
    logger.info("=== START ===")

    yield logger

    # Log test result
    if hasattr(request.node, 'rep_call'):
        outcome = request.node.rep_call.outcome
        logger.info(f"=== {outcome.upper()} ===")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for logging."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU-marked tests when CuPy is not available."""
    if not HAS_CUPY:
        skip_gpu = pytest.mark.skip(reason="CuPy not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


def pytest_configure(config):
    """Configure pytest with dynamic log file path and custom settings."""
    # Create output directory with timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path("tests/output")
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store for later use by fixtures
    global _TEST_RUN_OUTPUT_DIR
    _TEST_RUN_OUTPUT_DIR = output_dir

    # Create/update 'latest' symlink for convenience
    latest_link = output_base / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(timestamp, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Symlinks may fail on Windows without developer mode
        pass

    # Set log file path dynamically
    log_file = output_dir / "pytest.log"
    config.option.log_file = str(log_file)

    # Save benchmark results to the output directory
    config.option.benchmark_storage = str(output_dir)
    config.option.benchmark_autosave = True

    # Configure logging: suppress all external packages to WARNING level
    # Only allow INFO and above from slmsuite package
    logging.captureWarnings(True)

    # Set all loggers to WARNING by default (external packages)
    logging.getLogger().setLevel(logging.WARNING)

    # Explicitly set common external packages to WARNING
    for package in ['matplotlib', 'PIL', 'numpy', 'cupy', 'h5py']:
        logging.getLogger(package).setLevel(logging.WARNING)

    # Capture everything, let handler set level
    logging.getLogger('slmsuite').setLevel(logging.DEBUG)

    print(f"\nTest output directory: {output_dir}")


def pytest_sessionfinish(session, exitstatus):
    """Print summary message at end of test session."""
    output_dir = get_test_run_output_dir()
    if output_dir:
        print(f"\nTest output saved to: {output_dir}")