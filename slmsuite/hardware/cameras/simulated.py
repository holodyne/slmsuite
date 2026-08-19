"""
Simulated camera to image the simulated SLM.
"""

import numpy as np
import warnings

try:
    import cupy as cp
    from cupyx.scipy.ndimage import map_coordinates
except Exception:
    cp = np
    from scipy.ndimage import map_coordinates

import matplotlib.pyplot as plt

from slmsuite.hardware.cameras.camera import Camera
from slmsuite.holography.algorithms import Hologram
from slmsuite.holography import toolbox


class SimulatedCamera(Camera):
    """
    Simulated camera.

    Outputs simulated images (i.e., the far-field of an SLM interpolated to
    camera pixels based on the camera's location and orientation.
    Serves as a future testbed for simulation of other imaging artifacts, including non-affine
    aberrations (e.g. pincushion distortion) and imaging readout noise.

    Note
    ~~~~
    For fastest simulation, initialize :class:`SimulatedCamera` with a
    :class:`~slmsuite.hardware.slms.simulated.SimulatedSLM` *only*. Simulated camera images
    will directly sample the (quickly) computed SLM far-field (``"knm"``) via a one-to-one
    mapping instead of interpolating the SLM's far-field intensity at
    each camera pixel location (i.e. ``"knm"``->``"ij"`` basis change),
    which may also require additional padding (computed automatically upon initialization) for
    sufficient resolution.

    Attributes
    ----------
    grid : (numpy.ndarray, numpy.ndarray)
        Pixel column/row number (``x_grid``, ``y_grid``) in the ``"ij"`` basis used for
        far-field interpolation.
    shape_padded : (int, int)
        Size of the FFT computational space required to faithfully reproduce the far-field at
        full camera resolution.
    noise : dict
        Dictionary of single-argument functions (returning the normalized noise amplitude
        for any normalized input pixel amplitude) to simulate various noise sources. Currently,
        ``'dark'`` and ``'read'``, representing exposure-dependent dark current/background noise
        and exposure-independent readout noise, respectively, are the only accepted keys.

        Example
        ~~~~~~~
        The following code adds a Gaussian background with 50% mean and 5% standard
        deviation (relative to the dynamic range at the default ``self.exposure_s = 1``) and
        a Poisson readout noise (independent of ``self.exposure_s``) with an average value
        of 20% of the camera's dynamic range.

        .. code-block:: python

            self.noise = {
                'dark': lambda img: np.random.normal(0.5*img, 0.05*img),
                'read': lambda img: np.random.poisson(0.2*img)
            }

        Note
        ~~~~
        Callables cannot be written to an ``.h5``, so a noise assigned this way does not
        survive :meth:`~slmsuite._pickling._Picklable.save()`. Noise set through
        :meth:`set_noise_from_background()` does, as that model is two scalars.

    """
    _pickle = Camera._pickle + ["gain", "M", "b", "_noise_spec"]
    _pickle_data = Camera._pickle_data + ["_aperture"]

    def __init__(
        self, slm, resolution=None, M=None, b=None, noise=None, pitch_um=None, gain=1, **kwargs
    ):
        """
        Initialize simulated camera.

        Parameters
        ----------
        slm : ~slmsuite.hardware.slms.simulated.SimulatedSLM
            Simulated SLM creating the image.
        resolution : (int, int)
            See :attr:`resolution`. If ``None``, defaults to the resolution of ``slm``.
        M, b : array_like
            Passed to :meth:`set_affine()`. Can be set later, but the camera cannot be
            used until then.
        noise : dict
            See :attr:`noise`.
        pitch_um : (float, float) OR None
            Pixel pitch in microns. If ``None``, certain calibrations and conversions
            are not available (e.g. :meth:`build_affine()` for certain units).
        gain : float
            Gain to emulate physical cameras while keeping the same values for exposure time.
        **kwargs
            See :meth:`.Camera.__init__` for permissible options.
        """

        # Store a reference to the SLM: we need this to compute the far-field camera images.
        self._slm = slm

        # Backing store for the `noise` property, set before anything can read it.
        self._noise = self._noise_spec = None

        # Don't interpolate (slower) by default unless required.
        self._interpolate = False

        if resolution is None:
            resolution = slm.shape[::-1]
        elif any([r != s for r, s in zip(resolution, slm.shape[::-1])]):
            self._interpolate = True

        # dtype and other parameters are set here in init.
        super().__init__(resolution, pitch_um=pitch_um, **kwargs)

        # Digital gain emulates exposure
        self.gain = gain

        # Add user-defined noise dictionary
        self.noise = noise

        # Hidden: efficiency of each camera pixel; an image of shape ``_shape`` or None.
        self._aperture = None

        # Placement in the SLM's k-space; None until set_affine() interpolates.
        self.M = self.b = None

        # Compute the camera pixel grid in `basis` units (currently "ij")
        self.grid = np.meshgrid(
            np.arange(resolution[0]),
            np.arange(resolution[1]),
        )

        # Defaults to alignment with the SLM grid.
        self.set_affine(M, b)

    def close(self):
        pass

    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, noise):
        # A dictionary of callables cannot be written to an .h5, so only the scalars
        # recorded by _set_noise() survive a save(). Disown them for any other noise,
        # rather than let them silently reappear on the reloaded camera.
        self._noise_spec = None
        self._noise = noise

    def _unpickle(self, data):
        """
        Restores pickled state data not restored by constructor. See
        :meth:`~slmsuite._pickling._Picklable._unpickle`. 
        """
        super()._unpickle(data)

        self.gain = data.get("gain", 1)
        self._aperture = data.get("_aperture", None)

        # Noise handled separately to avoid callable in h5
        spec = data.get("_noise_spec", None)
        if spec is not None:
            self._set_noise(spec["dark"], spec["read"])

        # The pickled affine is already stated in this camera's delivered frame, unlike
        # a Fourier calibration's, which FourierSLM.load() must correct for a WOI.
        (M, b) = (data.get("M", None), data.get("b", None))
        if M is not None and b is not None:
            self.set_affine(M, b)

    def set_affine(self, M=None, b=None, **kwargs):
        """
        Set the camera's placement in the SLM's k-space. ``M`` and/or ``b``, if provided,
        are used to transform the :class:`SimulatedCamera`'s ``"ij"`` grid to a ``"knm"`` grid
        for interpolation against the :class:`~slmsuite.hardware.slms.simulated.SimulatedSLM`'s
        ``"knm"`` grid. Keyword arguments, if provided, are passed to :meth:`.build_affine()`
        to build ``M`` and ``b``.

        Parameters
        ----------
        M : array_like
            2 x 2 affine transform matrix to convert between SLM's :math:`k`-space and the
            simulated camera's pixel basis (``"ij"``). If ``None``, defaults to the
            identity matrix.
        b : array_like
            Lateral displacement (in pixels) of the camera center from the SLM's
            optical axis. If ``None``, defaults to ``(0,0)`` offset.
        **kwargs : dict, optional
            Various orientation parameters passed to :meth:`.build_affine()`
            to build ``M`` and ``b``, if not provided. See options documented in this
            method. ``f_eff`` is a required keyword.
        """

        # If kwargs are passed instead of M and b, use these to build M, b
        if M is None or b is None:
            f_eff = kwargs.pop("f_eff", None)
            if f_eff is not None:
                M, b = self.build_affine(f_eff, **kwargs)

        self._interpolate = not (M is None or b is None)
        self.M = self.b = None
        self.grid = np.meshgrid(
            np.arange(self._shape[1]),
            np.arange(self._shape[0]),
        )
        self.shape_padded = self._slm.shape
        self._supersample = (1, 1)
        self._knm_cam_super = None
        self._pixel_area = 1.0

        if self._interpolate:
            self.M = M
            self.b = b

            # Affine transform the camera grid ("ij"->"kxy")
            self.grid = toolbox.transform_grid(self, M, b, direction="rev")

            # Fourier space must be sufficiently padded to resolve the camera pixels.
            dkxy = np.sqrt(
                (self.grid[0][:2, :2] - self.grid[0][0, 0]) ** 2 +
                (self.grid[1][:2, :2] - self.grid[1][0, 0]) ** 2
            )
            dkxy_min = dkxy.ravel()[1:].min()

            self.shape_padded = Hologram.get_padded_shape(self._slm, precision=dkxy_min)

            # Convert kxy -> knm (0,0 at corner): 1/dx -> Npx
            self.knm_cam = cp.array(
                [
                    self.shape_padded[0] * self._slm.pitch[1] * self.grid[1] + self.shape_padded[0] / 2,
                    self.shape_padded[1] * self._slm.pitch[0] * self.grid[0] + self.shape_padded[1] / 2,
                ]
            )


            if (
                cp.amax(cp.abs(self.knm_cam[0] - self.shape_padded[0]/2)) > self.shape_padded[0]/2 or
                cp.amax(cp.abs(self.knm_cam[1] - self.shape_padded[1]/2)) > self.shape_padded[1]/2
            ):
                self.logger.warning(
                    "Camera extends beyond the accessible SLM k-space;"
                    " some pixels may not be targetable."
                )

            # Real pixels integrate over their footprint; sampling only their centers
            # would alias away spots smaller than a k-space cell. Columns of dknm are
            # the knm steps along ij.
            dknm = np.flip(np.linalg.inv(M), axis=0) * np.array(
                [[self.shape_padded[0] * self._slm.pitch[1]],
                 [self.shape_padded[1] * self._slm.pitch[0]]]
            )
            self._supersample = tuple(
                max(1, int(np.ceil(np.sqrt(np.sum(step ** 2))))) for step in np.flip(dknm.T, axis=0)
            )
            self._pixel_area = np.abs(np.linalg.det(dknm))

            if self._supersample != (1, 1):
                (sv, su) = self._supersample
                dknm = cp.array(dknm)
                offset = (
                    dknm[:, 1, None, None] * (((cp.arange(sv) + 0.5) / sv - 0.5)[None, :, None])
                    + dknm[:, 0, None, None] * (((cp.arange(su) + 0.5) / su - 0.5)[None, None, :])
                )
                self._knm_cam_super = cp.reshape(
                    self.knm_cam[:, :, None, :, None] + offset[:, None, :, None, :],
                    (2, self._shape[0] * sv, self._shape[1] * su),
                ).astype(cp.float32)

        display = self._slm.display
        if hasattr(display, "get"):
            display = display.get()
        phase = self._slm._display2phase(display)

        # Suppress power of 2 warning from Hologram.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=Warning)

            self._hologram = Hologram(
                self.shape_padded,
                amp=self._slm.source["amplitude_sim"],
                phase=phase - phase.min() + self._slm.source["phase_sim"],
                slm_shape=self._slm,
            )

    def build_affine(
            self,
            f_eff,
            units="norm",
            theta=0,
            shear_angle=0,
            offset=None,
        ):
        """
        Builds an affine transform defining the SLM to camera transformation as
        detailed in :meth:`~slmsuite.hardware.cameraslms.FourierSLM.kxyslm_to_ijcam`.

        Parameters
        ----------
        f_eff : float OR (float, float)
            Effective focal length of the
            optical train separating the Fourier-domain SLM from the camera. If a ``float`` is
            provided, ``f_eff`` is isotropic; otherwise, ``f_eff`` is defined along the SLM's
            :math:`x` and :math:`y` axes.
        units : str {"norm", "ij", "m", "cm", "mm", "um", "nm"}
            Units for the focal length ``f_eff``.

            -  ``"norm"``
                Normalized focal length in wavelengths according to the SLM's
                :attr:`~slmsuite.hardware.slms.slm.SLM.wav_um`.
                This is the default unit.
            -  ``"ij"``
                Focal length in units of camera pixels.
            -  ``"m"``, ``"cm"``, ``"mm"``, ``"um"``, ``"nm"``
                Focal length in metric units.

        theta : float
            Rotation angle (in radians, ccw) of the camera relative to the SLM orientation.
            Defaults to zero (i.e., aligned with the SLM).
        shear_angle : float OR (float, float)
            Shearing angles (in radians) along the SLM's :math:`x` and :math:`y` axes.
            If a ``float`` is provided, shear is applied isotropically.
            Defaults to zero (i.e., no shear).
        offset : (float, float) OR None
            Lateral displacement (in pixel units) of the SLM's optical axis
            from the camera's origin. If ``None``, defaults to be centered on the center
            of the camera.

        Returns
        -------
        numpy.ndarray
            Affine matrix :math:`M`. Shape ``(2, 2)``.
        numpy.ndarray
            Affine vector :math:`b`. Shape ``(1, 2)``.
        """
        if offset is None:
            offset = np.flip(self.shape) / 2

        return toolbox.build_affine(
            f_eff,
            units=units,
            theta=theta,
            shear_angle=shear_angle,
            offset=offset,
            cam_pitch_um=self.pitch_um,
            wav_um=self._slm.wav_um,
        )

    @staticmethod
    def info(verbose=True):
        """See :meth:`.Camera.info`. Returns a list with a single simulated camera entry."""
        info_list = ["SimulatedCamera"]
        if verbose:
            print(info_list)
        return info_list

    def flush(self, timeout_s=1):
        """
        See :meth:`.Camera.flush`.
        """
        pass

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        return self.exposure_s

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        self._exposure_s = exposure_s

    # Future: use WOI with Zoom FFT?

    def _get_image_hw(self, timeout_s, quantize=True):
        """
        See :meth:`.Camera._get_image_hw`. Computes and samples the affine-transformed SLM far-field.

        Parameters
        ----------
        timeout_s : float
            Unused; a simulated frame is always ready.
        quantize : bool
            Whether to apply the readout: clipping at saturation and casting to
            :attr:`dtype`. :meth:`match_counts` passes ``False`` to measure the
            collected signal before the readout discards the part of it below one count.

        Returns
        -------
        numpy.ndarray
            Array of shape :attr:`shape`
        """
        if not hasattr(self, "_hologram"):
            raise RuntimeError(
                "Cannot display SimulatedCamera before affine transformation is defined."
            )

        # Update phase; calculate the far-field (keep on GPU if using cupy for follow-on interp)
        # FUTURE: in the case where sim is being used inside a GS loop, there could be
        # something clever here to use the existing Hologram's data.

        # Analog phase
        # self._hologram.reset_phase(self._slm.phase + self._slm.source["phase_sim"])

        # Quantized phase
        self._hologram.amp = cp.array(self._slm.source["amplitude_sim"], dtype=self._hologram.dtype)
        display = cp.asarray(self._slm.display)
        phase = self._slm._display2phase(display, dtype=self._hologram.dtype)
        phase = phase - phase.min() + cp.asarray(
            self._slm.source["phase_sim"], dtype=self._hologram.dtype
        )

        self._hologram.reset_phase(phase)

        ff = self._hologram.get_farfield(get=False)
        intensity = cp.abs(ff) ** 2

        # Use map_coordinates for fastest interpolation
        # Note: by default, map_coordinates sets pixels outside the SLM k-space to 0 as desired
        if self._interpolate:
            # Each pixel collects the mean over its footprint times the footprint's area;
            # partially outside the k-space aperture, subsamples there return zero.
            if self._knm_cam_super is not None:
                (sv, su) = self._supersample
                img = map_coordinates(intensity, self._knm_cam_super, order=0)
                img = img.reshape(self._shape[0], sv, self._shape[1], su).mean(axis=(1, 3))
            else:
                img = map_coordinates(intensity, self.knm_cam, order=0)
            img = img * self._pixel_area
        else:
            img = toolbox.unpad(intensity, self._shape)
        if cp != np:
            img = img.get()

        # Efficiency of each camera pixel (apertures in the optical train, vignetting).
        if self._aperture is not None:
            img = img * self._aperture

        img = img * (self.exposure_s * self.gain)

        frame_bitresolution = 2 ** self.bitdepth

        # Basic noise sources.
        if self.noise is not None:
            for key in self.noise.keys():
                if key == 'dark':
                    # Background/dark current - exposure dependent
                    dark = self.noise['dark'](np.ones_like(img) * frame_bitresolution) * self.exposure_s
                    img = img + dark
                elif key == 'read':
                    # Readout noise - exposure independent
                    read = self.noise['read'](np.ones_like(img) * frame_bitresolution)
                    img = img + read
                else:
                    raise RuntimeError('Unknown noise source %s specified!'%(key))

        if not quantize:
            return img

        # Truncate to valid readout range [0, 2**bitdepth - 1]
        img = np.clip(img, 0, frame_bitresolution - 1)

        return img.astype(self.dtype)

    def match_counts(self, reference, background=None):
        """
        Sets :attr:`gain` such that the frame this camera renders *right now* totals the
        same counts as ``reference``. Use this to put a simulated camera on the same
        radiometric scale as the hardware camera it stands in for: display the same
        phase on the simulated SLM, then pass a hardware image taken at the same
        exposure. Without this, the far-field is normalized to unit total power and the
        simulated counts are arbitrary.

        Note
        ~~~~
        Call this after :attr:`exposure_s` is set, since exposure scales the counts.

        Parameters
        ----------
        reference : array_like
            An image from the camera being emulated.
        background : array_like OR None
            A blank (no signal) image from the same camera at the same exposure, whose
            total is subtracted from ``reference`` so that only signal is matched.

        Returns
        -------
        float
            The new :attr:`gain`.
        """
        target = float(np.sum(np.asarray(reference, dtype=float)))
        if background is not None:
            target -= float(np.sum(np.asarray(background, dtype=float)))

        if not target > 0:
            raise ValueError(
                "Expected reference to carry positive signal above background; "
                f"found a total of {target} counts."
            )

        # Render at unit gain without noise: the total is then the gain-independent,
        # unquantized signal that `gain` scales.
        # `_noise_spec` rides along, as setting `noise` disowns it.
        (gain, noise, spec) = (self.gain, self.noise, self._noise_spec)
        try:
            (self.gain, self.noise) = (1.0, None)
            # Deliver the frame as get_image() would, so that a windowed or binned
            # camera is matched over the pixels it actually reads out.
            frame = self._get_image_hw(0, quantize=False)
            total = float(np.sum(self.transform(self._crop_to_woi(frame))))
        finally:
            (self.gain, self.noise) = (gain, noise)
            self._noise_spec = spec

        if not total > 0:
            raise ValueError(
                "The simulated camera collects no light, so its gain cannot be matched "
                "to a reference. Is the far-field within the camera's field of view?"
            )

        self.gain = target / total

        return self.gain

    def set_noise_from_background(self, background, exposure_s=None):
        """
        Sets :attr:`noise` from a blank (no signal) image of the camera being emulated.
        The mean of ``background`` is attributed to the exposure-dependent ``'dark'``
        term and its standard deviation to the exposure-independent ``'read'`` term.

        Note
        ~~~~
        This is the Gaussian background plus Gaussian read term that
        :class:`SimulatedCamera` models. It is *not* photon shot noise, which scales
        with the signal and is not simulated here.

        Parameters
        ----------
        background : array_like
            A blank image from the camera being emulated.
        exposure_s : float OR None
            The exposure ``background`` was taken at. Defaults to :attr:`exposure_s`.

        Returns
        -------
        dict
            The new :attr:`noise`.
        """
        background = np.asarray(background, dtype=float)

        if exposure_s is None:
            exposure_s = self.exposure_s
        if not exposure_s > 0:
            raise ValueError(f"Expected a positive exposure; found {exposure_s}.")

        bitresolution = 2 ** self.bitdepth

        # 'dark' is scaled by exposure_s downstream, so divide it out here.
        dark = float(np.mean(background)) / (bitresolution * exposure_s)
        read = float(np.std(background)) / bitresolution

        return self._set_noise(dark, read)

    def _set_noise(self, dark, read):
        """
        Sets :attr:`noise` to the Gaussian background/read model from its two scalars,
        recording them in :attr:`_noise_spec` so that the model survives a
        :meth:`~slmsuite._pickling._Picklable.save()`. The callables themselves cannot
        be written to an ``.h5``; these two floats can.
        """
        (dark, read) = (float(dark), float(read))

        self.noise = {
            "dark": lambda img, dark=dark: dark * img,
            "read": lambda img, read=read: np.random.normal(0, read * img),
        }
        self._noise_spec = {"dark": dark, "read": read}

        return self.noise
