"""
Datastructures, methods, and calibrations for an SLM monitored by a camera.
"""

import os
import copy
import matplotlib.pyplot as plt
from slmsuite._plotting import _slmsuite_plt_show
import numpy as np
import warnings

from slmsuite import __version__
from slmsuite._logging import _Loggable
from slmsuite.holography.analysis.files import load_h5, save_h5, generate_path, latest_path

from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.slms.simulated import SimulatedSLM

# Import calibrations (separated into different files for readability).
from slmsuite.hardware.cameraslms._fourier import _FourierCalibration
from slmsuite.hardware.cameraslms._farfield import _FarfieldCalibration
from slmsuite.hardware.cameraslms._pixel import _PixelCalibration
from slmsuite.hardware.cameraslms._settle import _SettleCalibration
from slmsuite.hardware.cameraslms._wavefront import _WavefrontCalibration


def _to_numpy(array):
    """Host copy of ``array``, which may live in GPU memory. Passes ``None`` through."""
    if array is None or not hasattr(array, "get"):
        return array
    return array.get()


class CameraSLM(_Loggable):
    """
    Base class for an SLM with camera feedback.

    Attributes
    ----------
    cam : ~slmsuite.hardware.cameras.camera.Camera
        Instance of :class:`~slmsuite.hardware.cameras.camera.Camera`
        which interfaces with a camera. This camera is
        used to provide closed-loop feedback to an SLM for calibration and holography.
    slm : ~slmsuite.hardware.slms.slm.SLM
        Instance of :class:`~slmsuite.hardware.slms.slm.SLM`
        which interfaces with a phase display.
    name : str
        Stores ``cam.name + '-' + slm.name``.
    mag : float
        Magnification of the camera relative to an experiment plane. For instance,
        ``mag = 10`` could refer to the use of a 10x objective (with appropriate
        imaging lensing) between the experiment plane and the camera.
        In this case, the images apparent on the camera are 10x larger than the true
        objects at the experiment plane.
    """
    _pickle = ["name", "cam", "slm", "mag"]
    _pickle_data = []

    def __init__(self, cam=None, slm=None, mag=1):
        """
        Initialize an SLM linked to a camera, with given magnification between the
        camera and experiment planes.

        Parameters
        ----------
        cam : ~slmsuite.hardware.cameras.camera.Camera OR (int, int) OR None
            Instance of :class:`~slmsuite.hardware.cameras.camera.Camera`
            which interfaces with a camera. This camera is
            used to provide closed-loop feedback to an SLM for calibration and holography.
            If a shape ``(int, int)`` is passed and ``slm=None``,
            then a simulated system is constructed with the desired resolution.
            If ``None``, then the shape defaults to ``(512, 512)``.
        slm : ~slmsuite.hardware.slms.slm.SLM OR None
            Instance of :class:`~slmsuite.hardware.slms.slm.SLM`
            which interfaces with a phase display.
        mag : float
            Magnification of the camera relative to an experiment plane. For instance,
            ``mag = 10`` could refer to the use of a 10x objective (with appropriate
            imaging lensing) between the experiment plane and the camera.
            In this case, the images apparent on the camera are ten times larger than
            the true objects at the experiment plane.

            Note
            ~~~~
            This magnification is currently isotropic. In the future, anisotropy between
            the camera and experiment planes could be implemented.
        """
        # First, handle the case where we want to quickly construct a simulated system.
        if cam is None:
            cam = (512, 512)

        if isinstance(cam, (list, tuple)):
            if slm is not None:
                raise ValueError("When a shape is passed for cam, slm must be None.")
            slm = SimulatedSLM(resolution=cam, pitch_um=8)
            slm.set_source_analytic(sim=True)
            slm.set_source_analytic(sim=False)
            cam = SimulatedCamera(slm=slm, pitch_um=8)

        # Now actually parse the cameras.
        if not hasattr(cam, "get_image"):
            raise ValueError(f"Expected Camera to be passed as cam. Found {type(cam)}")
        self.cam = cam

        if not hasattr(slm, "set_phase"):
            raise ValueError(f"Expected SLM to be passed as slm. Found {type(slm)}")
        self.slm = slm

        self.name = self.cam.name + "-" + self.slm.name
        self.mag = float(mag)

        self.calibrations = {}

        # Initialize logger.
        _Loggable.__init__(self)
        self.log_state()

    def plot(
        self,
        phase=None,
        image=None,
        slm_limits=None,
        cam_limits=None,
        title="",
        axs=None,
        cbar=True,
        **kwargs
    ):
        """
        Plots the provided phase and image for the child hardware on a pair of subplot axes.

        Parameters
        ----------
        phase : ndarray OR None
            Phase to be plotted.
            If ``None``, grabs the last written :attr:`phase` from the SLM.

            Important
            ---------
            Writes this ``phase`` to the SLM if ``image`` is ``None``.
        image : ndarray OR None
            Image to be plotted. If ``None``, grabs an image from the camera.
        slm_limits, cam_limits : None OR float OR [[float, float], [float, float]]
            Scales the limits by a given factor or uses the passed limits directly.
        title : str
            Super title for the axes.
        axs : (matplotlib.pyplot.axis, matplotlib.pyplot.axis) OR None
            Axes to plot upon.
        cbar : bool
            Also plot a colorbar.
        **kwargs
            Passed to :meth:`set_phase()`

        Returns
        -------
        (matplotlib.pyplot.axis, matplotlib.pyplot.axis)
            Axes of the plotted phase and image.
        """
        if image is None and phase is not None and np.shape(phase) == self.slm.shape:
            self.slm.set_phase(phase, **kwargs)


        should_show = False
        if axs is None:
            if len(plt.get_fignums()) > 0:
                fig = plt.gcf()
            else:
                fig = plt.figure(figsize=(20,8))
                should_show = True
            axs = (fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2))
        else:
            fig = None
            if len(axs) != 2:
                raise ValueError(f"Expected axs to be a tuple of two axes. Found length {len(axs)} tuple.")

        self.slm.plot(phase=phase, limits=slm_limits, title="", ax=axs[0], cbar=cbar)
        self.cam.plot(image=image, limits=cam_limits, title="", ax=axs[1], cbar=cbar)

        if fig is not None:
            fig.suptitle(title)
        plt.tight_layout()

        if should_show:
            _slmsuite_plt_show(name="fourierslm_plot")

        return axs


class NearfieldSLM(CameraSLM):
    """
    **(NotImplemented)** Class for an SLM which is not nearly in the Fourier domain of a camera.

    Parameters
    ----------
    mag : number OR None
        Magnification between the plane where the SLM image is created
        and the camera sensor plane.
    """

    def __init__(self, *args, **kwargs):
        """See :meth:`CameraSLM.__init__`."""
        super().__init__(*args, **kwargs)


# Make full class including all calibrations (separated into different files for readability).
class FourierSLM(
    CameraSLM,
    _FourierCalibration,
    _FarfieldCalibration,
    _PixelCalibration,
    _SettleCalibration,
    _WavefrontCalibration,
):
    r"""
    Class for an SLM and camera separated by a Fourier transform.
    This class includes methods for system calibration.

    Attributes
    ----------
    calibrations : dict
        "fourier" : dict
            The affine transformation that maps between
            the k-space of the SLM (kxy) and the pixel-space of the camera (ij).

            See :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate()`.

            This data is critical for much of :mod:`slmsuite`'s functionality.
        "farfield" : dict
            Raw data measuring the diffraction efficiency over the farfield
            (including aperture cropping) and the 0th order scatter.

            See
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.farfield_calibrate()`.
            Usable data is produced by running
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.farfield_calibration_process()`.
        "wavefront_superpixel" : dict
            Raw data for correcting aberrations in the optical system (``phase``) and
            measuring the optical amplitude distribution incident on the SLM (``amp``),
            measured by interfering pairs of superpixels.

            See
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate_superpixel()`.
            Usable data is produced by running
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_superpixel_process()`,
            which writes the result onto
            :attr:`~slmsuite.hardware.slms.slm.SLM.source`.

            This data is critical for crisp holography.

            Note
            ~~~~
            Calibrations saved before this key was split in two are named ``"wavefront"``;
            :meth:`wavefront_calibration_superpixel_process()` reads either.
        "wavefront_zernike" : dict
            Raw data for the same correction, measured instead by optimizing Zernike
            coefficients against a metric.

            See
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate_zernike()`.
        "pixel" : dict
            Raw data for measuring the crosstalk and :math:`V_\pi` of sections of the
            SLM via measurements on the diffractive orders of binary gratings.

            See
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.pixel_calibrate()`.
            Usable data is produced by running
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.pixel_calibration_process()`.

            **This data is currently unused; exploring
            computationally-efficient ways to apply the crosstalk without oversampling.**
        "settle" : dict
            Raw data for determining the temporal system response of the SLM.

            See
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.settle_calibrate()`.
            Usable data is produced by running
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.settle_calibration_process()`.

            This data informs the user's choice of `settle_time_s`, the time to wait to
            acquire data after a pattern is displayed. This is, of course, a tradeoff
            between measurement speed and measurement precision.
    """
    _pickle = ["name", "cam", "slm", "mag", "_wavefront_calibration_window_multiplier"]
    _pickle_data = ["calibrations"]

    def __init__(self, *args, **kwargs):
        r"""See :meth:`CameraSLM.__init__`."""
        super().__init__(*args, **kwargs)

        # Size of the calibration point window relative to the spot radius.
        self._wavefront_calibration_window_multiplier = 4

    def simulate(self, reference=None, background=None, settle=False, source=None):
        r"""
        Clones the hardware-based experiment into a simulation, such that the same
        algorithm can be run against either and the results compared.

        The clone replicates the geometry, the calibrations, and the physical
        characteristics of the hardware:

        -  The camera's placement in the SLM's :math:`k`-space, from the Fourier
           calibration, along with its pixel pitch, dynamic range, exposure, averaging,
           and HDR settings. The simulated sensor is built as the image the hardware
           *delivers*, so the camera's WOI, binning, and orientation are inherited
           through :attr:`fourier_affine`. Thus, 
           :meth:`~slmsuite.hardware.cameraslms.FourierSLM.kxyslm_to_ijcam` agrees
           between the hardware and the clone.
        -  The SLM's measured phase response (:attr:`~slmsuite.hardware.slms.slm.SLM.gamma`,
           from the pixel calibration) is installed as both the clone's quantization
           table *and* the response it actually realizes
           (:attr:`~slmsuite.hardware.slms.simulated.SimulatedSLM.gamma_sim`), so a
           coarsely-quantized or nonlinear SLM is simulated as such.
        -  The SLM's :attr:`~slmsuite.hardware.slms.slm.SLM.aperture` and its measured
           source illumination and wavefront (from the wavefront calibration). The
           measured wavefront correction becomes the simulated aberration, so applying
           the correction cancels it, exactly as on hardware.

        The clone's :attr:`~slmsuite.hardware.slms.slm.SLM.source` is deep-copied, so
        modifying it (e.g. injecting a known aberration into ``"phase_sim"`` to test
        what a calibration recovers) does not disturb the hardware.

        Note
        ~~~~
        Since simulation mode needs the Fourier relationship between the SLM and
        camera, the :class:`~slmsuite.hardware.cameraslms.FourierSLM` should be
        Fourier-calibrated prior to cloning for simulation.

        Caution
        ~~~~~~~
        The clone is an *ideally corrected* version of the experiment. Its simulated
        aberration is exactly the measured correction, so applying that correction
        leaves a diffraction-limited spot, whereas the hardware retains whatever the
        calibration failed to measure. The clone also models only the single-transform
        far-field of the SLM: scatter, stray light, and non-affine distortion in the
        optical train are absent. Expect the simulated spot to be tighter and cleaner
        than the measured one, and expect the two to agree on where light lands rather
        than on exactly how much of it arrives.

        Parameters
        ----------
        reference : array_like OR None
            An image from the hardware camera, taken at the current exposure with the
            current pattern on the SLM. If provided, the simulated camera's gain is set
            such that its images total the same number of counts
            (see :meth:`~slmsuite.hardware.cameras.simulated.SimulatedCamera.match_counts`).
            Without this, simulated counts are arbitrary, as the simulated far-field is
            normalized to unit power.
        background : array_like OR None
            A blank (no signal) image from the hardware camera at the same exposure. If
            provided, the simulated camera's noise is fit to it (see
            :meth:`~slmsuite.hardware.cameras.simulated.SimulatedCamera.set_noise_from_background`)
            and its total is subtracted from ``reference``.
        settle : bool
            Whether to also clone the SLM's
            :attr:`~slmsuite.hardware.slms.slm.SLM.settle_time_s`. Defaults to ``False``,
            as waiting for a simulated SLM to stabilize only slows the simulation down.
        source : dict OR None
            Overrides for the clone's :attr:`~slmsuite.hardware.slms.slm.SLM.source`,
            applied on top of the copy taken from the hardware. Use this to set a
            simulated truth (``"amplitude_sim"``, ``"phase_sim"``) other than the
            measured one.

        Returns
        -------
        FourierSLM
            A :class:`~slmsuite.hardware.cameraslms.FourierSLM` (of the same class as
            ``self``) with simulated hardware.
        """
        # Make sure we have a Fourier calibration.
        if not "fourier" in self.calibrations:
            raise ValueError("Cannot simulate() a FourierSLM without a Fourier calibration.")

        slm_sim = self._simulate_slm(settle=settle, source=source)
        cam_sim = self._simulate_cam(slm_sim)

        # Combine the two and pass FourierSLM attributes from hardware. type(self) so
        # that a subclass clones into its own class and keeps its methods; a subclass
        # whose constructor does not accept (cam, slm, mag) must override simulate().
        fs_sim = type(self)(cam_sim, slm_sim, mag=self.mag)

        fs_sim.calibrations = copy.deepcopy(self.calibrations)
        fs_sim._wavefront_calibration_window_multiplier = self._wavefront_calibration_window_multiplier

        # The simulated sensor *is* the hardware's delivered image, so its raw pixels are
        # the hardware's "ij" pixels. Restate the Fourier calibration in that frame,
        # which is what makes kxyslm_to_ijcam() agree between the two.
        fs_sim.calibrations["fourier"].update(self.fourier_affine.to_dict())

        # Radiometry: put the simulated camera on the hardware's count scale. This must
        # happen last, as it depends on the exposure, the WOI, and the displayed phase.
        if reference is not None:
            cam_sim.match_counts(reference, background=background)
        if background is not None:
            cam_sim.set_noise_from_background(background)

        return fs_sim

    def _simulate_slm(self, settle=False, source=None):
        """
        The :class:`~slmsuite.hardware.slms.simulated.SimulatedSLM` half of
        :meth:`simulate()`. See that method for the parameters.
        """
        # The measured phase response, which the clone both quantizes through and
        # realizes. numpy: SimulatedSLM.gamma_sim cannot read GPU memory.
        gamma = _to_numpy(self.slm.gamma)

        # The simulated truth. The aperture-masked source is what the hardware actually
        # uses, so it is what the simulation should reproduce; note the sign, as the
        # measured phase is the *correction* for the aberration we want to simulate.
        source_sim = copy.deepcopy(self.slm.source)
        if "amplitude_sim" not in source_sim:
            source_sim["amplitude_sim"] = _to_numpy(self.slm._get_source_amplitude())
            source_sim["phase_sim"] = -_to_numpy(self.slm._get_source_phase())
        if source is not None:
            source_sim.update(source)

        slm_sim = SimulatedSLM(
            np.flip(self.slm.shape),
            source=source_sim,
            gamma_sim=gamma,
            bitdepth=self.slm.bitdepth,
            name=self.slm.name+"_sim",
            wav_um=self.slm.wav_um,
            wav_design_um=self.slm.wav_design_um,
            pitch_um=self.slm.pitch_um,
            settle_time_s=self.slm.settle_time_s if settle else 0,
            gpu=self.slm.xp is not np,
        )

        # Quantize through the same lookup table that the hardware does.
        if gamma is not None:
            slm_sim.set_gamma(gamma)

        # The aperture defines the working grid, the Zernike scaling, and the region of
        # the source in use. Both SLMs share a grid (same shape, pitch, and wavelength),
        # so the aperture transfers as-is.
        slm_sim.set_aperture(self.slm.aperture)

        # Defaults for set_phase().
        slm_sim.phase_correct = self.slm.phase_correct
        slm_sim.settle = self.slm.settle and settle

        # Display what the hardware is displaying. The hardware's `phase` is already
        # corrected, so the correction must not be applied a second time.
        slm_sim.set_phase(self.slm.phase, phase_correct=False, settle=False)

        return slm_sim

    def _simulate_cam(self, slm_sim):
        """
        The :class:`~slmsuite.hardware.cameras.simulated.SimulatedCamera` half of
        :meth:`simulate()`, imaging ``slm_sim``.
        """
        # The stored Fourier calibration maps kxy onto *raw* sensor pixels, while
        # `fourier_affine` maps it onto the delivered image, folding in the camera's WOI,
        # binning, and orientation. Building the simulated sensor as the delivered image
        # inherits all three: it needs no WOI, no binning, and no transform of its own,
        # and it renders only the pixels the hardware actually returns.
        affine = self.fourier_affine

        # Binning sums pixels, so a binned frame ranges beyond the raw bitdepth. The
        # clone reads that frame out directly, so the widened range is its bitdepth.
        # (`bitresolution` is `2**bitdepth` times the averaging and software binning.)
        bitdepth = int(np.log2(self.cam.bitresolution // (self.cam.averaging or 1)))

        cam_sim = SimulatedCamera(
            slm_sim,
            resolution=np.flip(self.cam.shape),
            M=copy.copy(affine.M),
            b=copy.copy(affine.b),
            bitdepth=bitdepth,
            averaging=self.cam.averaging,
            hdr=self.cam.hdr,
            pitch_um=self.cam.pitch_um,
            exposure_bounds_s=self.cam.exposure_bounds_s,
            name=self.cam.name+"_sim",
        )
        cam_sim.set_exposure(self.cam.exposure_s)

        # Cloning a simulation: carry the simulated detector characteristics too, so
        # that the clone reproduces the original exactly.
        if isinstance(self.cam, SimulatedCamera):
            cam_sim.gain = self.cam.gain
            cam_sim.noise = copy.copy(self.cam.noise)
            cam_sim._aperture = copy.copy(self.cam._aperture)

        return cam_sim

    @classmethod
    def _load_class(cls, meta):
        """
        The class which :meth:`load()` should rebuild as.

        An explicit subclass on the call (``MyFourierSLM.load(...)``) wins. Otherwise the
        class recorded when the file was written is looked up among the subclasses which
        have been imported, mirroring the ``type(self)`` that :meth:`simulate()` keeps:
        a system saved as a subclass reloads with that subclass's methods.

        Falls back to :class:`FourierSLM` if the name cannot be resolved, which happens
        when the module defining it has not been imported. The result still carries every
        calibration; it just lacks the subclass's methods, so say so rather than let it
        pass as the class the user asked for.
        """
        if cls is not FourierSLM:
            return cls

        name = meta.get("__class__", None)
        if name is None or name == FourierSLM.__name__:
            return cls

        def descendants(base):
            for sub in base.__subclasses__():
                yield sub
                yield from descendants(sub)

        matches = {sub for sub in descendants(FourierSLM) if sub.__name__ == name}

        if len(matches) == 1:
            return matches.pop()

        warnings.warn(
            f"File was saved from '{name}', which is "
            + ("ambiguous among the imported subclasses"
               if matches else "not among the imported subclasses")
            + f"; rebuilding as {FourierSLM.__name__}. Import the module defining "
            f"'{name}' before loading, or call '{name}.load()' directly, to keep its "
            "methods."
        )
        return cls

    @classmethod
    def load(cls, file_path : str):
        """
        Rebuilds a system as a simulation from the metadata in an :mod:`slmsuite` file,
        without the hardware present. Both a calibration written by
        :meth:`save_calibration()` and a pickle written by
        :meth:`~slmsuite._pickling._Picklable.save()` carry this metadata; a pickle
        saved with ``attributes=True`` also carries the SLM's measured
        :attr:`~slmsuite.hardware.slms.slm.SLM.source`, its measured phase response,
        :attr:`~slmsuite.hardware.slms.slm.SLM.aperture`, and displayed
        :attr:`~slmsuite.hardware.slms.slm.SLM.phase`, along with every calibration, and
        so reconstructs the most.

        A pickle of a system which was already simulated round trips exactly:
        the simulated phase response
        (:attr:`~slmsuite.hardware.slms.simulated.SimulatedSLM.gamma_sim`), the
        camera's placement, gain, pixel efficiency, and any
        :meth:`~slmsuite.hardware.cameras.simulated.SimulatedCamera.set_noise_from_background()`
        noise all return with it. Saving the result of :meth:`simulate()` and
        loading it back is therefore the way to keep a simulated system across
        sessions.

        If the file supplies a Fourier calibration, the simulated camera is placed by
        it, so the result images the same region of :math:`k`-space that the hardware
        did. Otherwise the camera samples the SLM's far-field directly and only the
        shapes are meaningful.

        The subclass is kept, as :meth:`simulate()` keeps it: a file written from a
        subclass of :class:`FourierSLM` reloads as that subclass, provided the module
        defining it has been imported (or that its :meth:`load()` is called directly).
        A subclass whose constructor does not accept ``(cam, slm, mag)`` must override
        this method, exactly as it must override :meth:`simulate()`.

        Note
        ~~~~
        The camera's orientation transform and binning are not recorded in the
        metadata, so neither is restored; a window of interest is folded into the
        camera's placement. A :attr:`~slmsuite.hardware.cameras.simulated.SimulatedCamera.noise`
        of hand-written callables cannot be written to an ``.h5`` and is lost. To clone a
        system that is actually connected, with its phase response, aperture, and noise,
        use :meth:`simulate()` instead.

        Parameters
        ----------
        file_path : str
            Path to the ``.h5`` file to read.

        Returns
        -------
        FourierSLM
            A :class:`~slmsuite.hardware.cameraslms.FourierSLM` object with simulated
            hardware.
        """
        # Read in the file.
        data = load_h5(file_path)

        # Check to see if it has the information we need.
        if not "__meta__" in data:
            raise ValueError(
                f"Cannot interpret file {file_path} without field '__meta__'. "
            )
        for field in ("cam", "slm"):
            if not field in data["__meta__"]:
                raise ValueError(
                    f"Cannot interpret file {file_path} without metadata field '{field}'. "
                )
        cam_data = data["__meta__"]["cam"]
        slm_data = data["__meta__"]["slm"]

        # Create the SLM and Camera objects. Every calibration is wavelength specific,
        # so the wavelengths matter as much as the shapes.
        slm = SimulatedSLM(
            resolution=np.flip(slm_data["shape"]),
            pitch_um=slm_data["pitch_um"],
            bitdepth=slm_data["bitdepth"],
            wav_um=slm_data["wav_um"],
            wav_design_um=slm_data["wav_design_um"],
            source=slm_data.get("source", None),
            name=slm_data["name"],
        )
        cam = SimulatedCamera(
            slm=slm,
            resolution=np.flip(cam_data["shape"]),
            bitdepth=cam_data["bitdepth"],
            pitch_um=cam_data["pitch_um"],
            averaging=cam_data.get("averaging", None),
            hdr=cam_data.get("hdr", None),
            name=cam_data["name"],
        )
        calibrations = data["__meta__"].get("calibrations", {})

        # A calibration file carries its payload at the top level rather than under
        # "__meta__"; a Fourier one is the payload that places the camera.
        if "M" in data and "b" in data and "fourier" not in calibrations:
            calibrations["fourier"] = {
                key: value for (key, value) in data.items() if key != "__meta__"
            }

        # A Fourier calibration places the camera below, and corrects for a window of
        # interest which a camera's own pickled affine does not record. Let it win, and
        # do not place the camera twice: each placement rebuilds the padded far-field.
        if "fourier" in calibrations:
            cam_data = {k: v for (k, v) in cam_data.items() if k not in ("M", "b")}

        # Restore the exposure and the simulated detector characteristics.
        cam._unpickle(cam_data)

        fs = cls._load_class(data["__meta__"])(cam, slm, mag=data["__meta__"]["mag"])
        fs.name = data["__meta__"]["name"]

        fs.calibrations = calibrations
        if "_wavefront_calibration_window_multiplier" in data["__meta__"]:
            fs._wavefront_calibration_window_multiplier = (
                data["__meta__"]["_wavefront_calibration_window_multiplier"]
            )

        # The phase response is not pickled on the SLM; it is rebuilt from the pixel
        # calibration that measured it. Before the SLM restores its own state, which
        # re-displays the stored phase through this lookup table.
        if "pixel" in fs.calibrations:
            fs._pixel_calibration_apply_gamma()

        # Restore what the SLM's constructor does not take: the display defaults, the
        # aperture, and the displayed phase.
        slm._unpickle(slm_data)

        if "fourier" in fs.calibrations:
            fs._load_place_camera(cam_data)

        return fs

    def _load_place_camera(self, cam_data):
        """
        Places the simulated camera of :meth:`load()` using the Fourier calibration that
        was read with it. That calibration maps onto *raw* sensor pixels while the
        rebuilt camera is the delivered image, so a window of interest has to be
        subtracted off; binning and orientation are not recorded and cannot be.
        """
        M = np.array(self.calibrations["fourier"]["M"], dtype=float)
        b = np.array(self.calibrations["fourier"]["b"], dtype=float).reshape(2, 1)

        woi = cam_data.get("woi", None)
        if woi is not None:
            (height, width) = self.cam.shape
            if (woi[1], woi[3]) != (width, height):
                self.logger.warning(
                    "The saved camera was binned or reoriented (window %s delivering "
                    "shape %s), which the metadata does not record; the simulated "
                    "camera is placed as though it were neither.",
                    tuple(woi), (height, width),
                )
            b = b - np.array([[woi[0]], [woi[2]]], dtype=float)

        self.calibrations["fourier"]["M"] = M
        self.calibrations["fourier"]["b"] = b
        self.cam.set_affine(M, b)

    ### Automatic Calibration ###

    def _calibrate(self):
        """
        **(Not Implemented)**
        Attempts to autonomously calibrate the system.
        Conducts any missing calibrations. Also looks for saved calibration files under
        default filenames and loads them if they are found.

        See
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate()`,
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.settle_calibrate()`,
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.pixel_calibrate()`, and
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate_superpixel()`.
        """
        def calibration_detected(calibration_type):
            self.logger.info("%s calibration...", calibration_type.replace("_", " ").capitalize())
            if calibration_type in self.calibrations:
                self.logger.info("Found calibration from %s.", self.calibrations[calibration_type]["__time__"])
                return True
            else:
                try:
                    self.load_calibration(calibration_type)
                    self.logger.info("Loaded calibration from %s.", self.calibrations[calibration_type]["__time__"])
                    return True
                except FileNotFoundError:
                    return False
                except Exception as e:
                    self.logger.warning("Unable to load '%s' calibration: %s", calibration_type, e)
                    return False

        # Fourier
        if not calibration_detected("fourier"):
            self.fourier_calibrate()

        if not calibration_detected("settle"):
            self.settle_calibrate()

        if not calibration_detected("pixel"):
            self.pixel_calibrate()

        if not calibration_detected("wavefront_superpixel"):
            self.wavefront_calibrate_superpixel()

        self.logger.info("Fourier calibration (final)...")
        self.fourier_calibrate()

    ### Calibration Helpers ###

    def name_calibration(self, calibration_type):
        """
        Creates ``"{self.name}-{calibration_type}-calibration"``.

        Parameters
        ----------
        calibration_type : str
            The type of calibration to save. See :attr:`calibrations` for supported
            options.

        Returns
        -------
        name : str
            The generated name.
        """
        return f"{self.name}-{calibration_type}-calibration"

    def write_calibration(self, calibration_type, path, name):
        "Backwards-compatibility alias for :meth:`save_calibration()`."
        warnings.warn(
            "The backwards-compatible alias FourierSLM.write_calibration will be depreciated "
            "in favor of FourierSLM.save_calibration in a future release."
        )
        self.save_calibration(calibration_type, path, name)

    # Raw image stacks dominate the file size and only serve reprocessing in-session.
    _CALIBRATION_UNSAVED = ("efficiency_raw", "background_raw")

    def save_calibration(self, calibration_type, path=".", name=None):
        """
        Saves the calibration to a file like ``"path/name_id.h5"``.
        Raw image stacks are omitted; the processed results are saved.

        Parameters
        ----------
        calibration_type : str
            The type of calibration to save. See :attr:`calibrations` for supported
            options. Works for any key of :attr:`calibrations`.
        path : str
            Path to directory to save in. Default is current directory.
        name : str OR None
            Name of the save file. If ``None``, will use :meth:`name_calibration`.

        Returns
        -------
        str
            The file path that the calibration was saved to.
        """
        if not calibration_type in self.calibrations:
            raise ValueError(
                f"Could not find calibration '{calibration_type}' in calibrations. Options:\n"
                + str(list(self.calibrations.keys()))
            )

        if name is None:
            name = self.name_calibration(calibration_type)
        file_path = generate_path(path, name, extension="h5")
        save_h5(file_path, {
            key: value
            for (key, value) in self.calibrations[calibration_type].items()
            if key not in self._CALIBRATION_UNSAVED
        })

        self.logger.info("Saved '%s' calibration to '%s'.", calibration_type, file_path)

        return file_path

    def read_calibration(self, calibration_type, file_path=None):
        "Backwards-compatibility alias for :meth:`load_calibration()`."
        warnings.warn(
            "The backwards-compatible alias FourierSLM.read_calibration will be depreciated "
            "in favor of FourierSLM.load_calibration in a future release."
        )
        self.load_calibration(calibration_type, file_path)

    def load_calibration(self, calibration_type, file_path=None):
        """
        Loads the calibration from a file.

        Parameters
        ----------
        calibration_type : str
            The type of calibration to load. See :attr:`calibrations` for supported
            options.
        file_path : str OR None
            Full path to the calibration file. If ``None``, will
            search the current directory for a file with a name like
            the one returned by :meth:`name_calibration`.

        Returns
        -------
        str
            The file path that the calibration was loaded from.

        Raises
        ------
        FileNotFoundError
            If a file is not found.
        """
        if file_path is None:
            path = os.path.abspath(".")

            if len(calibration_type) > 4 and calibration_type[-3:] == ".h5":
                file_path = calibration_type
                split = file_path.split("-")
                if len(split) > 3 and "calibration_" in split[-1]:
                    calibration_type = split[-2]
                else:
                    raise ValueError(
                        f"Could not parse calibration type from '{file_path}'."
                    )
            else:
                name = self.name_calibration(calibration_type)
                file_path = latest_path(path, name, extension="h5")

            if file_path is None:
                raise FileNotFoundError(
                    "Unable to find a calibration file like\n{}"
                    "".format(os.path.join(path, name))
                )

        self.calibrations[calibration_type] = cal = load_h5(file_path)
        self.logger.info("Loaded '%s' calibration from '%s'.", calibration_type, file_path)
        cal_ver = "an unknown version" if not "__version__" in cal else cal["__version__"]

        if cal_ver != __version__:
            self.logger.warning(
                "You are using slmsuite %s, but the calibration in '%s' was created in %s.",
                __version__, file_path, cal_ver,
            )

        # Every calibration is wavelength specific, so flag a retuned source.
        cal_wav_um = cal.get("__meta__", {}).get("slm", {}).get("wav_um", None)
        if cal_wav_um is not None and not np.isclose(cal_wav_um, self.slm.wav_um):
            self.logger.warning(
                "The '%s' calibration was taken at %s um, but this SLM is set to %s um.",
                calibration_type, cal_wav_um, self.slm.wav_um,
            )

        # Restore the measured phase response, as the SLM applies it on every write.
        if calibration_type == "pixel":
            self._pixel_calibration_apply_gamma()

        return file_path

    def _get_calibration_metadata(self):
        return self.pickle(attributes=False, metadata=True)      # Pickle without heavy data.

FourierSLM.fourier_calibration_build.__doc__ = SimulatedCamera.build_affine.__doc__
