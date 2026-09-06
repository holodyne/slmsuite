"""
Hardware-in-the-loop validation of :meth:`FourierSLM.simulate()`.

Skipped unless ``SLMSUITE_TEST_HARDWARE=1``. Everything else in ``tests/`` runs against
simulated hardware; this module opens a real camera and SLM and asserts that the
simulation cloned from them reproduces what they actually do.

Configuration (environment variables, all optional except the gate):

===============================  =============================================
``SLMSUITE_TEST_HARDWARE``       ``1`` to run this module at all.
``SLMSUITE_TEST_CAMERA_SERIAL``  Camera serial. Default ``22562470``.
``SLMSUITE_TEST_CAMERA_PITCH``   Camera pixel pitch, um. Default ``2.74``.
``SLMSUITE_TEST_CAMERA_BITS``    Camera bitdepth. Default ``12``.
``SLMSUITE_TEST_EXPOSURE``       Camera exposure, s. Default ``0.002``.
``SLMSUITE_TEST_PLM_MODEL``      PLM model. Default ``p67``.
``SLMSUITE_TEST_PLM_DISPLAYS``   Comma-separated display numbers. Default ``1,2``.
``SLMSUITE_TEST_PLM_INDEX``      Which of the opened PLMs is the device under test.
                                 Default ``1``, i.e. ``slm2``. The rest are blanked.
``SLMSUITE_TEST_WAV_UM``         Wavelength, um. Default ``0.488``.
``SLMSUITE_TEST_CAL_DIR``        Directory holding the calibration h5 files. Default is
                                 the ``20260810 - First Closed Loop Tests`` experiment
                                 directory.
``SLMSUITE_TEST_PIXEL_CAL``      Pixel calibration filename.
``SLMSUITE_TEST_WAVEFRONT_CAL``  Wavefront calibration filename.
===============================  =============================================

Run with::

    SLMSUITE_TEST_HARDWARE=1 pytest tests/hardware/test_simulate_hardware.py -v -s
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from slmsuite._plotting import _slmsuite_plt_show
from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.cameraslms import FourierSLM
from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.holography.toolbox import convert_vector
from slmsuite.holography.toolbox.phase import blaze

from conftest import plot_image_dim


pytestmark = pytest.mark.skipif(
    os.environ.get("SLMSUITE_TEST_HARDWARE") != "1",
    reason="Set SLMSUITE_TEST_HARDWARE=1 to run against the camera and SLM.",
)

DEFAULT_CAL_DIR = os.path.join(
    os.path.expanduser("~"),
    "Documents", "Experiments", "20260810 - First Closed Loop Tests",
)

#: Blaze magnitudes probed along each axis, in normalized ``kxy``, applied in both
#: directions. Small enough that the order stays on the sensor, and large enough that it
#: clears its own width so that :func:`order_position` can separate it from where it
#: started.
PROBE_AMPLITUDES = (0.002, 0.004)

#: Boxcar side length used to suppress stuck pixels before locating a spot. Wide compared
#: to a hot pixel, narrow compared to the spot.
SMOOTH_PX = 15

#: How far the simulated spot may sit from the measured one, in camera pixels. Locating
#: the order by differencing (see :func:`order_position`) takes the stuck pixels and the
#: edge ghost out of the measurement, so this can be near a pixel; the headroom that
#: remains is for the residual aberration that broadens the measured spot but not the
#: simulated one.
POSITION_TOLERANCE_PX = 3.0

#: Order-separating blaze folded into the SLM's source phase, in normalized frequency.
#: Without it the corrected order sits on top of the undiffracted light.
SOURCE_BLAZE_FREQ = (0.25, 0)

#: Side length of the window cropped around the order once it has been located. The full
#: sensor is 16 megapixels, which the simulation would have to interpolate in its
#: entirety for every frame; the probed spots move by a couple hundred pixels, so a
#: window this size contains all of them. Cropping also exercises the fact that the
#: Fourier calibration lives in raw sensor pixels while images are delivered in the WOI.
WOI_PX = 1024


def _env(name, default, cast=str):
    return cast(os.environ.get(name, default))


def grab(cam):
    """
    One frame, flushed first.

    The camera holds a frame in flight, so a grab straight after writing the SLM
    returns the *previous* pattern. Every measurement here writes and then reads, so
    every read must flush.
    """
    cam.flush()
    return cam.get_image()


def warm_up(fs):
    """
    Write a pattern and throw the result away.

    The first pattern written to a PLM after it is opened takes far longer to reach the
    mirrors than :attr:`~slmsuite.hardware.slms.slm.SLM.settle_time_s` allows for, so
    the first frame measured in a session can show the display as it was *before* that
    write. The result is not obviously wrong -- it is a bright, plausible image of the
    undiffracted light -- so it silently poisons whatever is measured from it. Paying
    that cost up front, on a measurement nobody reads, keeps it out of the data.
    """
    fs.slm.set_phase(blaze(fs.slm, np.array([max(PROBE_AMPLITUDES), 0.0])), settle=True)
    grab(fs.cam)
    fs.slm.set_phase(None, settle=True)
    grab(fs.cam)


def autoexpose(cam, target=0.5, tries=10, bounds=(2e-5, 0.05), half=48):
    """
    Nudge the exposure until the brightest pixel *of the spot* sits near ``target`` of
    full scale.

    Neither obvious quantity works on its own here. The raw frame maximum never moves,
    because the sensor has stuck pixels that read full scale at every exposure.
    :func:`find_spot`'s boxcar suppresses those, but its smoothed peak is far below the
    true peak of a spot a few pixels wide, so driving *that* to half scale saturates the
    spot. So: locate the spot with the boxcar, then measure the raw peak in a box around
    it, where the stuck pixels elsewhere on the sensor cannot reach.
    """
    full = cam.bitresolution - 1
    for _ in range(tries):
        image = grab(cam)
        (position, _) = find_spot(image)

        (x, y) = (int(position[0]), int(position[1]))
        box = image[max(0, y - half):y + half, max(0, x - half):x + half]
        peak = float(np.max(box)) if box.size else 0.0

        if peak <= 0:
            break
        ratio = target * full / peak
        if 0.7 < ratio < 1.4:
            break
        new = float(np.clip(cam.exposure_s * ratio, *bounds))
        if new == cam.exposure_s:
            break
        cam.set_exposure(new)
    return cam.exposure_s


def find_spot(image, smooth_px=SMOOTH_PX, half=48, near=None, search=None):
    """
    Locate the beam, tolerating stuck pixels and competing blobs.

    ``argmax`` on a raw frame finds a stuck pixel, not the beam. A boxcar suppresses
    those by ``smooth_px ** 2`` while leaving a real spot nearly untouched; the peak of
    the smoothed frame is then refined by a centroid on the raw one. Pass ``near`` and
    ``search`` when the answer is roughly known, so that an off-axis ghost cannot win a
    global comparison.

    Returns
    -------
    (numpy.ndarray, float)
        ``(x, y)`` in camera pixels, and the background-subtracted smoothed peak.
    """
    import cv2

    work = np.asarray(image, dtype=np.float32)

    origin = np.zeros(2, dtype=int)
    if near is not None and search is not None:
        s = int(search) // 2
        y0, y1 = int(max(0, near[1] - s)), int(min(work.shape[0], near[1] + s))
        x0, x1 = int(max(0, near[0] - s)), int(min(work.shape[1], near[0] + s))
        work = work[y0:y1, x0:x1]
        origin = np.array([x0, y0])

    work = work - np.median(work)
    smoothed = cv2.blur(work, (smooth_px, smooth_px))

    xy = np.array(np.unravel_index(np.argmax(smoothed), smoothed.shape))[::-1]
    brightness = float(smoothed[xy[1], xy[0]])

    y0, y1 = max(0, xy[1] - half), min(work.shape[0], xy[1] + half)
    x0, x1 = max(0, xy[0] - half), min(work.shape[1], xy[0] + half)
    box = np.clip(work[y0:y1, x0:x1], 0, None)

    total = box.sum()
    if total <= 0:
        return xy + origin, brightness

    (yy, xx) = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
    centroid = np.array([(xx * box).sum() / total, (yy * box).sum() / total])

    return centroid + origin, brightness


def flat_frame(fs):
    """A frame with a flat phase on the SLM, as the reference every probe differs from."""
    fs.slm.set_phase(None, settle=True)
    return grab(fs.cam).astype(np.float32)


def order_position(fs, kxy, flat):
    """
    Where the modulated order sits when the SLM is blazed by ``kxy``.

    A brightest-spot search cannot be trusted for this, and neither can tracking the
    spot from probe to probe. The undiffracted light is a fixed feature of comparable
    brightness, so a global search picks whichever happens to win; and a tracking window
    wide enough to follow the order between probes is also wide enough to capture that
    fixed feature. Both failure modes are silent, and both yield an affine fit to light
    that never moved.

    The order is instead identified by what it *does*. Only the order responds to the
    blaze, so it is the strongest residue of the difference between this frame and the
    flat one: positive where it went, negative where it came from.

    Parameters
    ----------
    fs : ~slmsuite.hardware.cameraslms.FourierSLM
        The system to drive.
    kxy : array_like
        Blaze to apply, in normalized ``kxy``. Zero returns the flat position.
    flat : numpy.ndarray
        The reference from :func:`flat_frame`.

    Returns
    -------
    numpy.ndarray
        ``(x, y)`` in camera pixels.
    """
    kxy = np.squeeze(kxy)

    if not np.any(kxy):
        # The flat position is where a blaze takes light *away* from.
        (position, departed) = find_spot(flat - _blazed_frame(fs, [max(PROBE_AMPLITUDES), 0]))
        return position

    (position, arrived) = find_spot(_blazed_frame(fs, kxy) - flat)

    assert arrived > 0, (
        f"Blazing by {kxy} put no light anywhere on the sensor. The order is being "
        "extinguished rather than steered, or it has left the field of view."
    )

    return position


def _blazed_frame(fs, kxy):
    """A frame with the SLM blazed by ``kxy``."""
    fs.slm.set_phase(blaze(fs.slm, np.squeeze(np.asarray(kxy, dtype=float))), settle=True)
    frame = grab(fs.cam).astype(np.float32)
    fs.slm.set_phase(None, settle=True)
    return frame


@pytest.fixture(scope="module")
def hardware():
    """The opened camera and the SLM under test."""
    from slmsuite.hardware.cameras.flir import FLIR
    from slmsuite.hardware.slms.texasinstruments import PLM

    cam = FLIR(
        serial=_env("SLMSUITE_TEST_CAMERA_SERIAL", "22562470"),
        pitch_um=_env("SLMSUITE_TEST_CAMERA_PITCH", "2.74", float),
        bitdepth=_env("SLMSUITE_TEST_CAMERA_BITS", "12", int),
    )
    # A FLIR keeps its WOI across sessions, so a window left behind by an earlier run
    # would silently crop the order out of view.
    cam.set_woi(None)
    cam.set_exposure(_env("SLMSUITE_TEST_EXPOSURE", "0.002", float))

    displays = [int(d) for d in _env("SLMSUITE_TEST_PLM_DISPLAYS", "1,2").split(",")]
    slms = PLM.open_all(
        _env("SLMSUITE_TEST_PLM_MODEL", "p67"),
        display_numbers=displays,
        wav_um=_env("SLMSUITE_TEST_WAV_UM", "0.488", float),
        settle_time_s=0.05,
    )
    slms = slms if isinstance(slms, (list, tuple)) else [slms]

    # Deliberately no teardown: powering an EVM down forces a slow rediscovery that
    # this rig does not reliably come back from, and process exit closes the devices.
    yield (cam, slms, slms[_env("SLMSUITE_TEST_PLM_INDEX", "1", int)])


@pytest.fixture(scope="module")
def fs(hardware):
    """
    The device under test, calibrated as the experiment calibrates it: the measured
    phase response from the pixel calibration and the measured source from the
    wavefront calibration, then a Fourier calibration measured by steering one spot.
    """
    (cam, slms, slm) = hardware

    cal_dir = _env("SLMSUITE_TEST_CAL_DIR", DEFAULT_CAL_DIR)
    pixel = _env("SLMSUITE_TEST_PIXEL_CAL", "slm2_pixel_calibration_1-5V.h5")
    wavefront = _env(
        "SLMSUITE_TEST_WAVEFRONT_CAL",
        "slm2_superpixel_calibration_offset=0-25_wav=488.h5",
    )

    fs = FourierSLM(cam, slm)

    fs.load_calibration("pixel", os.path.join(cal_dir, pixel))
    fs.pixel_calibration_process(plot=False)

    fs.load_calibration("wavefront", os.path.join(cal_dir, wavefront))
    fs.wavefront_calibration_superpixel_process(r2_threshold=0.5, smooth=True, plot=False)

    # Separate the corrected order from the undiffracted light. This rides in
    # source["phase"], so the simulation inherits it as part of the wavefront.
    slm.source["phase"] = slm.source["phase"] + blaze(
        slm,
        convert_vector(SOURCE_BLAZE_FREQ, from_units="freq", to_units="norm", hardware=fs),
    )

    # Blank everything upstream so only the device under test modulates.
    for other in slms:
        if other is not slm:
            other.set_phase(None)
    slm.set_phase(None)

    warm_up(fs)
    autoexpose(cam)
    fs.steering_residual_px = _fourier_calibrate_by_steering(fs)

    # Crop to the order. The Fourier calibration is stored in raw sensor pixels, so it
    # survives this; the clone has to replay the WOI to land in the same frame.
    spot = order_position(fs, [0, 0], flat_frame(fs))
    fs.cam.set_woi((
        int(spot[0]) - WOI_PX // 2, WOI_PX,
        int(spot[1]) - WOI_PX // 2, WOI_PX,
    ))
    autoexpose(cam)

    return fs


def _fourier_calibrate_by_steering(fs):
    """
    Fourier-calibrate by steering a single spot and least-squares fitting
    ``ij = M kxy + b``.

    Deliberately not
    :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate`, whose spot
    array needs every spot on the sensor, unsaturated, and brighter than everything
    else. On a sensor with stuck pixels and stray light, one steered spot measures the
    same affine far more robustly.
    """
    flat = flat_frame(fs)

    (kxy_list, ij_list) = ([np.zeros(2)], [order_position(fs, [0, 0], flat)])
    for axis in (0, 1):
        for sign in (-1, 1):
            for amplitude in PROBE_AMPLITUDES:
                kxy = np.zeros(2)
                kxy[axis] = sign * amplitude
                kxy_list.append(kxy)
                ij_list.append(order_position(fs, kxy, flat))

    K = np.column_stack([np.array(kxy_list), np.ones(len(kxy_list))])
    IJ = np.array(ij_list)
    (solution, *_) = np.linalg.lstsq(K, IJ, rcond=None)
    M = solution[:2].T

    # A small residual alone proves nothing: if the SLM never modulated, every point
    # sits on top of every other and the degenerate fit through them is perfect. Demand
    # that the spot actually swept, and that the affine it implies is invertible.
    swept = (IJ.max(axis=0) - IJ.min(axis=0)).min()
    assert swept > 10 * POSITION_TOLERANCE_PX, (
        f"The order moved only {swept:.1f} px across the whole sweep. The SLM is not "
        "modulating, or the order is outside the camera's field of view."
    )
    assert np.linalg.cond(M) < 100, (
        f"The measured affine is nearly singular (condition number "
        f"{np.linalg.cond(M):.3g}); the two axes were not independently resolved."
    )

    # How well an affine describes this rig at all. A simulation built from this affine
    # cannot land its spots closer to the measured ones than this, so it bounds what the
    # comparison below can ask for.
    residual = np.sqrt(((IJ - K @ solution) ** 2).sum(axis=1)).max()
    assert residual < 0.1 * swept, (
        f"The Fourier affine does not fit its own measurement ({residual:.2f} px max "
        f"residual over a {swept:.0f} px sweep), so there is nothing meaningful to "
        "compare the simulation against."
    )

    fs.fourier_calibrate_analytic(M, solution[2])

    return residual


@pytest.fixture(scope="module")
def fs_sim(fs):
    """The simulation cloned from the calibrated hardware, matched to its counts."""
    fs.slm.set_phase(None, settle=True)
    reference = grab(fs.cam)

    # A true dark frame would need the beam blocked, so the frame's median stands in for
    # the sensor's pedestal: it is subtracted from the count-matching target and
    # reinstated as the simulation's dark term.
    background = np.full(fs.cam.shape, np.median(reference), dtype=float)

    fs_sim = fs.simulate(reference=reference, background=background)
    (fs_sim.reference, fs_sim.background) = (reference, background)

    return fs_sim


class TestSimulatedHardware:
    def test_clone_is_simulated(self, fs, fs_sim, subtests):
        with subtests.test("hardware is simulated"):
            assert isinstance(fs_sim.slm, SimulatedSLM)
            assert isinstance(fs_sim.cam, SimulatedCamera)
            assert fs_sim.slm is not fs.slm and fs_sim.cam is not fs.cam

        with subtests.test("same class"):
            assert type(fs_sim) is type(fs)

        with subtests.test("camera framing"):
            # The clone is the delivered image, so it carries no window of its own.
            assert fs_sim.cam.shape == fs.cam.shape
            assert fs_sim.cam.woi == (0, fs.cam.shape[1], 0, fs.cam.shape[0])
            assert fs_sim.cam.bitresolution == fs.cam.bitresolution
            assert np.allclose(fs_sim.cam.pitch_um, fs.cam.pitch_um)
            assert fs_sim.cam.exposure_s == fs.cam.exposure_s

        with subtests.test("SLM geometry"):
            assert fs_sim.slm.shape == fs.slm.shape
            assert fs_sim.slm.bitdepth == fs.slm.bitdepth
            assert fs_sim.slm.wav_um == fs.slm.wav_um
            assert np.allclose(fs_sim.slm.pitch_um, fs.slm.pitch_um)

    def test_phase_response_is_cloned(self, fs, fs_sim, subtests, test_logger):
        """
        A PLM realizes a handful of non-uniform phase states. The clone must both
        quantize through and realize the same table, or its far-field is not the
        hardware's.
        """
        with subtests.test("quantization table"):
            assert (fs.slm.gamma is None) == (fs_sim.slm.gamma is None)
            if fs.slm.gamma is not None:
                assert np.allclose(_host(fs_sim.slm.gamma), _host(fs.slm.gamma))

        with subtests.test("realized response"):
            assert fs_sim.slm.gamma_sim is not None, (
                "The hardware SLM has a measured phase response, but the clone "
                "simulates an ideal linear one."
            )
            assert np.allclose(fs_sim.slm.gamma_sim, _host(fs.slm.gamma))

        with subtests.test("same grayscale levels are written"):
            phase = blaze(fs.slm, (0.003, 0.002))
            fs.slm.set_phase(phase)
            fs_sim.slm.set_phase(phase)

            levels = np.unique(_host(fs_sim.slm.display))
            test_logger.info("Clone writes %s distinct levels: %s", len(levels), levels)
            assert len(levels) <= fs.slm.bitresolution
            assert levels.max() < fs.slm.bitresolution

    def test_source_is_cloned(self, fs, fs_sim, subtests):
        with subtests.test("measured source becomes the simulated truth"):
            assert np.allclose(
                fs_sim.slm.source["amplitude_sim"], _host(fs.slm._get_source_amplitude())
            )
            assert np.allclose(
                fs_sim.slm.source["phase_sim"], -_host(fs.slm._get_source_phase())
            )

        with subtests.test("aperture"):
            (spec, spec_sim) = (fs.slm.aperture.spec, fs_sim.slm.aperture.spec)
            if isinstance(spec, str):
                assert spec_sim == spec
            else:
                assert np.allclose(spec_sim, spec)

            (center, center_sim) = (fs.slm.aperture.center, fs_sim.slm.aperture.center)
            assert (center is None) == (center_sim is None)
            if center is not None:
                assert np.allclose(center_sim, center)

        with subtests.test("editing the clone does not touch the hardware"):
            before = _host(fs.slm.source["amplitude"]).copy()
            fs_sim.slm.source["amplitude_sim"] *= 0
            assert np.array_equal(_host(fs.slm.source["amplitude"]), before)
            fs_sim.slm.source["amplitude_sim"] = np.array(before)

    def test_steered_spots_land_together(self, fs, fs_sim, test_logger):
        """
        The comparison that matters: blaze to the same targets on hardware and in
        simulation, and check that the order lands in the same place on both.
        """
        results = []

        # Both systems are measured the same way: against their own flat frame.
        (flat_hw, flat_sim) = (flat_frame(fs), flat_frame(fs_sim))

        for axis in (0, 1):
            for sign in (-1, 1):
                for amplitude in PROBE_AMPLITUDES:
                    kxy = np.zeros(2)
                    kxy[axis] = sign * amplitude

                    hw = order_position(fs, kxy, flat_hw)
                    sim = order_position(fs_sim, kxy, flat_sim)

                    results.append((kxy, hw, sim, np.linalg.norm(hw - sim)))

        test_logger.info(
            "Fourier calibration residual: %.2f px", fs.steering_residual_px,
        )
        for (kxy, hw, sim, error) in results:
            test_logger.info(
                "kxy %s: hardware %s, simulation %s, %.2f px apart",
                np.round(kxy, 4), np.round(hw, 1), np.round(sim, 1), error,
            )

        # The clone can only be as faithful as the affine it was built from, so the
        # affine's own fit error sets the floor on what can be asked of it.
        tolerance = POSITION_TOLERANCE_PX + 2 * fs.steering_residual_px
        errors = np.array([r[-1] for r in results])
        assert errors.max() < tolerance, (
            f"Simulated spots land up to {errors.max():.1f} px from the measured ones, "
            f"beyond the {tolerance:.1f} px the Fourier calibration "
            f"({fs.steering_residual_px:.1f} px residual) can account for."
        )

    def test_counts_are_matched(self, fs, fs_sim, test_logger, subtests):
        """
        The simulated camera must sit on the hardware's count scale, or its exposure is
        not the hardware's exposure and nothing about saturation transfers.

        What is matched is the *collected* signal, before the readout clips it, and that
        match is exact. The two nonetheless do not *deliver* equal counts, and the
        difference is physical rather than a defect of the matching: the measured spot
        carries broad wings, and the rig scatters light that a single-transform far-field
        does not reproduce, so the same total energy is spread over far more pixels on
        hardware. Concentrated into a sharper simulated spot, it reaches a higher peak
        and can clip where the hardware does not. Both quantities are logged below.
        """
        fs.slm.set_phase(None, settle=True)
        fs_sim.slm.set_phase(None)

        signal = float(np.sum(fs_sim.reference - fs_sim.background))

        # Signal only, as match_counts measured it: the fitted dark term adds a pedestal
        # over every pixel, which on a sensor this large is a sizeable fraction of it.
        noise = fs_sim.cam.noise
        try:
            fs_sim.cam.noise = None
            rendered = float(fs_sim.cam._get_image_hw(0, quantize=False).sum())
        finally:
            fs_sim.cam.noise = noise

        test_logger.info(
            "collected signal -- reference %.4g, simulation %.4g (ratio %.4f)",
            signal, rendered, rendered / signal,
        )

        with subtests.test("gain matches the collected signal"):
            assert np.isclose(rendered, signal, rtol=1e-3)

        hw = grab(fs.cam).astype(float)
        sim = fs_sim.cam.get_image().astype(float)
        saturation = fs.cam.bitresolution - 1

        test_logger.info(
            "delivered counts -- hardware: signal %.3g, peak %.0f of %d | "
            "simulation: signal %.3g, peak %.0f of %d",
            (hw - np.median(hw)).sum(), hw.max(), saturation,
            (sim - np.median(sim)).sum(), sim.max(), saturation,
        )

        with subtests.test("the simulation is usable at the hardware's exposure"):
            for (name, img) in (("hardware", hw), ("simulation", sim)):
                assert img.max() > 0.1 * saturation, (
                    f"The {name} frame peaks at {img.max():g} of {saturation}, so this "
                    "exposure does not expose it."
                )
                assert np.mean(img >= saturation) < 0.01, (
                    f"{np.mean(img >= saturation):.1%} of the {name} frame is "
                    "saturated."
                )

    def test_plot_comparison(self, fs, fs_sim, test_logger):
        """Saves a hardware-vs-simulation figure for eyeball comparison."""
        kxy = (0.003, 0.002)
        phase = blaze(fs.slm, kxy)

        fs.slm.set_phase(phase, settle=True)
        hw = grab(fs.cam)
        fs_sim.slm.set_phase(phase)
        sim = fs_sim.cam.get_image()

        (fig, axs) = plt.subplots(1, 2, figsize=(14, 6))
        for (ax, img, title) in ((axs[0], hw, "Hardware"), (axs[1], sim, "Simulation")):
            plot_image_dim(ax, img)
            (spot, _) = find_spot(img)
            ax.scatter(spot[0], spot[1], fc="none", ec="lime", s=80, lw=1)
            ax.set_title(f"{title}: max {img.max():g}, spot {np.round(spot, 1)}")
            ax.set_xlabel("Camera $x$ [pix]")
            ax.set_ylabel("Camera $y$ [pix]")

        fig.suptitle(f"blaze at kxy = {kxy}")
        fig.tight_layout()
        _slmsuite_plt_show(name="simulate_hardware_comparison")

        test_logger.info("Saved the hardware-vs-simulation comparison.")


from slmsuite.misc.xp import as_numpy as _host
