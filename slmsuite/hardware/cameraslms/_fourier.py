import matplotlib.pyplot as plt
import numpy as np
import cv2

from slmsuite.holography import analysis, toolbox
from slmsuite.holography.algorithms import Hologram, SpotHologram
from slmsuite.holography.analysis import Affine
from slmsuite.holography.toolbox import format_2vectors, format_shape, format_vectors
from slmsuite.misc.math import REAL_TYPES


class _FourierCalibration(object):
    """
    Hidden superclass with Fourier calibration methods
    (SLM angle-space to camera-space conversion).
    """

    ### Fourier Calibration ###

    def fourier_calibrate(
        self,
        array_shape=None,
        array_pitch=None,
        array_center=None,
        plot=False,
        autofocus=False,
        autoexpose=False,
        **kwargs,
    ):
        """
        Project and fit a SLM computational Fourier space ``"knm"`` grid onto
        camera pixel space ``"ij"`` for affine fitting.
        An array produced by
        :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
        is projected for analysis by
        :meth:`~slmsuite.holography.analysis.blob_array_detect()`.
        These arguments are in ``"knm"`` space because:

        - The ``"ij"`` space has not yet been calibrated.
        - The ``"kxy"`` space can lead to non-integer ``array_pitch`` in
          ``"knm"``-space. This is not ideal (see Tip).

        Tip
        ~~~
        For best results, ``array_pitch`` should be integer data. Otherwise non-uniform
        rounding to the SLM's computational :math:`k`-space (``"knm"``-space) can result
        in non-uniform pitch and a bad fit. The user is warned if non-integer data is given.

        Parameters
        ----------
        array_shape, array_pitch
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
            **in the** ``"knm"`` **basis.**
            If either ``array_shape`` or ``array_pitch`` is ``None``,
            a series of calibrations are performed (Farfield and Fourier with different parameters) 
            to find suitable values and autonomously calibrate the SLM.
        array_center
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
            **in the** ``"knm"`` **basis.**  ``array_center`` is not passed directly, and is
            processed as being relative to the center of ``"knm"`` space, the position
            of the 0th order. If ``None`` the array is centered.
        plot : bool OR int
            Enables debug plots:

            - 0 is no plots,
            - 1 is only the final fit plot, unless there is an error,
            - 2 is all plots.
        autofocus : bool OR dict
            Brings the calibration grid into focus by adjusting
            the focus term of the SLM's wavefront calibration.
            If a dictionary is passed, the dictionary is passed to
            :meth:`~slmsuite.hardware.cameras.camera.Camera.autofocus()`.
        autoexpose : bool OR dict
            Whether or not to automatically set the camera exposure on the projected
            array. If a dictionary is passed, it is passed to
            :meth:`~slmsuite.hardware.cameras.camera.Camera.autoexpose()`.
        **kwargs : dict
            Passed to :meth:`.fourier_grid_project()`, which passes them to
            :meth:`~slmsuite.holography.algorithms.SpotHologram.optimize()`.

        Returns
        -------
        dict
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["fourier"]`
        """
        if array_shape is None or array_pitch is None:
            return self._fourier_calibrate_meta(
                plot=plot,
                autofocus=autofocus,
                autoexpose=autoexpose,
                **kwargs
            )
        else:
            return self._fourier_calibrate_single(
                array_shape=array_shape,
                array_pitch=array_pitch,
                array_center=array_center,
                plot=plot,
                autofocus=autofocus,
                autoexpose=autoexpose,
                **kwargs
            )

    def _fourier_calibrate_single(
        self,
        array_shape,
        array_pitch,
        array_center=None,
        plot=0,
        autofocus=False,
        autoexpose=False,
        orientation=None,
        method="autocorrelation",
        **kwargs
    ):
        """Helper function for Fourier calibration."""
        # Parse variables
        if isinstance(array_shape, REAL_TYPES):
            array_shape = [int(array_shape), int(array_shape)]
        if isinstance(array_pitch, REAL_TYPES):
            array_pitch = [array_pitch, array_pitch]
        if np.any(np.array(array_pitch) <= 0):
            raise ValueError("array_pitch must be positive.")

        # Make and project a GS hologram across a normal grid of kvecs
        try:
            hologram = self.fourier_grid_project(
                array_shape=array_shape, array_pitch=array_pitch, array_center=array_center, **kwargs
            )
        except Exception as e:
            # The exception is the report; the caller decides whether it is fatal.
            self.logger.info(
                "fourier_calibrate failed during array holography. Try the following:\n"
                "- Reducing the array_pitch or array_shape,\n"
                "- Checking SLM parameters."
            )
            raise e

        # The rounding of the values might cause the center to shift from the desired
        # value. To compensate for this, we find the true written center.
        # The first two points are ignored for balance against the parity check omission
        # of the last two points.
        array_center = np.mean(hologram.spot_kxy_rounded[:, 2:], axis=1)

        if plot > 1:
            hologram.plot_farfield()
            hologram.plot_nearfield()

        self.cam.flush()

        # Optional step -- expose and focus the spots.
        if autoexpose or isinstance(autoexpose, dict):
            self.cam.autoexpose(**(autoexpose if isinstance(autoexpose, dict) else {}))

        if autofocus or isinstance(autofocus, dict):
            self.cam.autofocus(
                plot=plot,
                **{
                    "set_z": self.slm,
                    **(autofocus if isinstance(autofocus, dict) else {}),
                },
            )

        img = self.cam.get_image()

        # Get orientation of projected array
        try:
            orientation = analysis.blob_array_detect(
                img, array_shape, orientation=orientation, method=method, plot=plot
            )
        except Exception as e:
            self.logger.error("fourier_calibrate failed during array detection and fitting.")
            raise e

        a = format_2vectors(array_center)
        M = np.array(orientation["M"])
        b = format_2vectors(orientation["b"])

        # blob_array_detect returns the calibration from ij to the space of the array, so
        # as a last step we must convert from the array to (centered) knm space, and then
        # one step further to kxy space. This is done by a simple scaling.
        scaling = (
            self.slm.pitch
            * np.flip(np.squeeze(hologram.shape))
            / np.squeeze(array_pitch)
        )

        M = np.array([
            [M[0, 0] * scaling[0], M[0, 1] * scaling[1]],
            [M[1, 0] * scaling[0], M[1, 1] * scaling[1]],
        ])

        kxyslm_to_ijcam = Affine(M, b, a)
        # kxyslm -> ijcam -> ijraw
        kxyslm_to_ijraw = self.cam._get_ijcam_to_ijraw() @ kxyslm_to_ijcam
        self.calibrations["fourier"] = kxyslm_to_ijraw.to_dict()
        self.calibrations["fourier"]["meta"] = {
            "array_shape" : array_shape,
            "array_pitch" : array_pitch,
            "array_center" : array_center,
            "autoexpose" : autoexpose,
            "autofocus" : autofocus,
        }
        self.calibrations["fourier"].update(self._get_calibration_metadata())

        return self.calibrations["fourier"]

    ### Automatic ("meta") Fourier Calibration ###

    # Fraction of peak efficiency which counts as farfield the SLM can address.
    _FOURIER_CAL_META_EFF_THRESH = 0.1

    # Dynamic range to aim a projected array at; above one blooms sub-pixel spots.
    _FOURIER_CAL_META_OVEREXPOSE = 8.0

    def _fourier_calibrate_meta(
        self,
        tolerance=None,
        max_attempts=6,
        plot=False,
        **kwargs,
    ):
        """
        Fourier calibration without user-chosen array parameters.

        Parameters
        ----------
        tolerance : float OR None
            Residual in camera pixels which the calibration must reach to be accepted.
            Defaults to the larger of two pixels and the spot size.
        max_attempts : int
            Arrays to try before giving up, each smaller and coarser than the last.
        plot, **kwargs
            Passed to :meth:`_fourier_calibrate_single()`.

        Returns
        -------
        dict
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations` ``["fourier"]``.
        """
        # Held back until a new calibration verifies, else a failed run uncalibrates.
        previous = self.calibrations.pop("fourier", None)
        best = None

        try:

            # (0) Survey the farfield from its calibration, measuring one if absent.
            if "efficiency" not in self.calibrations.get("farfield", {}):
                self.farfield_calibrate()
            support = self.get_farfield_efficiency(
                fourier_crop=False,
                efficiency_threshold=self._FOURIER_CAL_META_EFF_THRESH
            )

            zeroth = self.get_farfield_zeroth()
            zeroth_seen = np.any(zeroth > 1)

            if not np.any(support):
                raise RuntimeError(
                    "The farfield survey found no light on the camera, so there is nothing "
                    "to calibrate against. Check the source and the camera exposure."
                )

            aperture = np.mean(
                1 / (np.flip(np.squeeze(self.slm.shape)) * np.squeeze(self.slm.pitch))
            )
            scale = np.sqrt(np.count_nonzero(support) * np.prod(self.slm.pitch))

            # The lit area gives the scale exactly, unless the camera crops the farfield,
            # where it only bounds it from below. A flat phase then paints the aperture's
            # own uncropped spot, as long as the camera resolves it.
            width = np.count_nonzero(zeroth > 0.5 * np.nanmax(zeroth))
            if zeroth_seen and width > 1 and (
                np.any(support[[0, -1], :]) or np.any(support[:, [0, -1]])
            ):
                scale = max(scale, np.sqrt(width) / aperture)

            spot = max(1.0, scale * aperture)
            if tolerance is None:
                tolerance = max(2.0, spot)

            shape = SpotHologram.get_padded_shape(self, padding_order=1, square_padding=True)
            cell = 1 / (np.flip(np.squeeze(shape)) * np.squeeze(self.slm.pitch))
            step = scale * np.mean(cell)                # Camera pixels per knm cell.
            (y, x) = np.nonzero(support)
            extent = np.array([np.ptp(x) + 1.0, np.ptp(y) + 1.0])

            # (1) Project an initial array with the goal of seeing a lattice,
            # even if the full grid is not in view.
            pitch = int(np.clip(np.rint(
                min(max(12.0, 8 * spot), max(np.min(extent) / 8, 5.0 * spot)) / step
            ), 1, 64))

            # Without the 0th order in view, fill the whole farfield rather than its extent.
            count = 1.5 * np.max(extent) / (pitch * step) if zeroth_seen else np.inf

            for retry in range(3):
                try:
                    array_shape = np.clip(count, 4, max(4, 0.9 * np.min(shape) / pitch))
                    # Exposed on the array, not on the 0th order, which outshines it
                    # wherever the zeroth exceeds what one spot of the array will hold.
                    window = self.get_farfield_efficiency(
                        fourier_crop=False,
                        efficiency_threshold=self._FOURIER_CAL_META_EFF_THRESH,
                        zeroth_threshold=array_shape ** 2,
                    )
                    self._fourier_calibrate_single(
                        array_shape=array_shape, array_pitch=pitch,
                        autoexpose={
                            "set_fraction": self._FOURIER_CAL_META_OVEREXPOSE,
                            "window": window if np.any(window) else None,
                            "verbose": False,
                        },
                        plot=plot, verbose=False,
                    )
                    break
                except RuntimeError:
                    if retry == 2:
                        raise
                    (count, pitch) = (count / 2, 2 * pitch)

            # The only exposure measured. Every later array follows it by spot count, as
            # the same light divided between more spots leaves each of them dimmer.
            exposure_per_spot = self.cam.get_exposure() / (array_shape ** 2)

            M = self.fourier_affine.M
            offset = (
                format_2vectors(np.flip(np.unravel_index(np.argmax(zeroth), zeroth.shape)))
                if zeroth_seen else self.kxyslm_to_ijcam([0, 0])
            )

            # (2) Now use the lattice information to position the array better.
            corners = np.linalg.solve(M, np.vstack((x, y)).astype(float) - offset)
            margin = np.abs(np.linalg.inv(M)) @ np.full(2, 2.0 + spot)
            low = corners.min(axis=1) + margin
            high = corners.max(axis=1) - margin
            center = np.rint(0.5 * (low + high) / cell)

            # Attempt zero is the array already projected above: if it found the center as
            # well as the lattice, a second array has nothing left to add.
            for attempt in range(int(max_attempts) + 1):
                if attempt == 0:
                    design = {
                        "array_shape": (int(array_shape), int(array_shape)),
                        "array_pitch": (pitch, pitch),
                        "array_center": (0.0, 0.0),
                    }
                    exposure = self.cam.get_exposure()
                else:
                    # One pitch serves both axes, so an axes-swapped affine designs the
                    # same array. A non-integer shape would be rounded up into an array
                    # which no longer straddles the center it is designed around.
                    span = np.maximum(high - low, 0) * (0.9 ** attempt)
                    grid = np.full(2, int(max(1, np.max(np.rint(
                        max(6.0, 4 * spot) * 1.5 ** (attempt - 1)
                        / (np.linalg.norm(M, axis=0) * cell)
                    )))))
                    limit = 2 * (0.48 * np.flip(np.squeeze(shape)) - np.abs(center)) / grid
                    design = {
                        "array_shape": tuple(
                            int(np.rint(np.clip(
                                np.minimum(span / (grid * cell), limit)[axis], 3, 13
                            )))
                            for axis in range(2)
                        ),
                        "array_pitch": tuple(int(p) for p in grid),
                        "array_center": tuple(float(c) for c in center),
                    }

                    # From scratch: last attempt's calibration would place this array.
                    self.calibrations.pop("fourier", None)
                    exposure = exposure_per_spot * np.prod(design["array_shape"])
                    try:
                        self.cam.set_exposure(exposure)
                        self._fourier_calibrate_single(
                            orientation={
                                "M": M @ np.diag(grid * cell),
                                "b": M @ format_2vectors(center * cell) + offset,
                            },
                            plot=plot,
                            **design,
                            **kwargs,
                        )
                    except Exception as e:
                        self.logger.info(
                            "fourier_calibrate_meta attempt %d failed: %s", attempt, e
                        )
                        continue

                # Verify at a normal exposure, where the gaps between spots survive the bloom.
                self.cam.set_exposure(exposure / self._FOURIER_CAL_META_OVEREXPOSE)
                self.cam.flush()
                img = analysis.image_remove_field(self.cam.get_image(), deviations=None)
                middle = np.array(design["array_center"])
                pitch_kxy = np.array(design["array_pitch"]) * cell
                lattice = self.fourier_affine.M @ np.diag(pitch_kxy)
                found = analysis._score_array_orientation(
                    img,
                    lattice,
                    self.kxyslm_to_ijcam(middle * cell),
                    design["array_shape"],
                    1 + 2 * int(np.clip(0.25 * np.min(np.linalg.norm(lattice, axis=0)), 1, 7)),
                    threshold=0.05,
                )

                error = np.inf
                if found is not None:
                    (code, lit, dark, residual) = found

                    # Relabeling pivots about the array, so b moves with M to hold it fixed.
                    relabel = analysis.OrientationTransform.from_code(code).M()
                    old = self.calibrations["fourier"]["M"]
                    new = old @ np.diag(pitch_kxy) @ relabel @ np.diag(1 / pitch_kxy)
                    self.calibrations["fourier"]["b"] += (old - new) @ (
                        format_2vectors(middle * cell) - self.calibrations["fourier"]["a"]
                    )
                    self.calibrations["fourier"]["M"] = new

                    # The 0th order was measured rather than inferred, so a calibration
                    # which does not map it back to where it was seen is wrong however
                    # well its spots line up.
                    anchored = not zeroth_seen or np.linalg.norm(
                        self.kxyslm_to_ijcam([0, 0]) - offset
                    ) <= 2 * tolerance

                    # A whole-lattice shift lights all but one row, so demand more than that.
                    if (
                        anchored
                        and lit >= 1 - 0.5 / (np.prod(design["array_shape"]) - 2)
                        and dark <= 0.5
                    ):
                        error = residual

                self.logger.info(
                    "fourier_calibrate_meta attempt %d: %.2f px residual.", attempt, error
                )

                if best is None or error < best[0]:
                    best = (error, dict(self.calibrations["fourier"]))
                if error <= tolerance:
                    break

            if best is None or best[0] > tolerance:
                self.calibrations.pop("fourier", None)
                raise RuntimeError(
                    f"fourier_calibrate_meta left a "
                    f"{np.inf if best is None else best[0]:.2f} px residual against a "
                    f"{tolerance:.2f} px tolerance. Inspect the farfield with plot=2: "
                    f"the array may be too dim, or the camera may see too little of "
                    f"the farfield to calibrate."
                )

            # The loop may have ended on a worse attempt than the best.
            self.calibrations["fourier"] = best[1]
            self.calibrations["fourier"]["meta"].update({
                "support": support, "residual": best[0],
            })

            return self.calibrations["fourier"]
        except BaseException:
            # An unverified calibration is worse than none: discard whatever is half done.
            self.calibrations.pop("fourier", None)
            raise
        finally:
            # Any failure leaves the previous calibration in place, not destroyed.
            if previous is not None and "fourier" not in self.calibrations:
                self.calibrations["fourier"] = previous


    ### Fourier Calibration Helpers ###

    def fourier_grid_project(
        self, array_shape=10, array_pitch=10, array_center=None, spot_amp=None, **kwargs
    ):
        """
        Projects a Fourier space grid ``"knm"`` onto pixel space ``"ij"``.
        The chosen computational :math:`k`-space ``"knm"`` uses a computational shape generated by
        :meth:`~slmsuite.holography.algorithms.SpotHologram.get_padded_shape()`
        corresponding to the smallest square shape with power-of-two sidelength that is
        larger than the SLM's shape.

        Parameters
        ----------
        array_shape, array_pitch
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
            **in the** ``"knm"`` **basis.**
        array_center
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
            **in the** ``"knm"`` **basis.**  ``array_center`` is not passed directly, and is
            processed as being relative to the center of ``"knm"`` space, the position
            of the 0th order.
        spot_amp : array_like OR None
            Relative amplitude to ask of each spot, which
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.get_farfield_weights()`
            can set from measured efficiency so that the spots arrive equally bright.
        **kwargs
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.optimize()`.

        Returns
        -------
        ~slmsuite.holography.algorithms.SpotHologram
            Optimized hologram.
        """
        # Check that the pitch is an integer.
        if not np.all(np.isclose(array_pitch, np.rint(array_pitch))):
            self.logger.warning("array_pitch is non-integer")

        # Make the spot array
        shape = SpotHologram.get_padded_shape(self, padding_order=1, square_padding=True)
        hologram = SpotHologram.make_rectangular_array(
            shape,
            array_shape=array_shape,
            array_pitch=array_pitch,
            array_center=(
                None
                if array_center is None
                else (
                    format_2vectors(array_center) +
                    format_2vectors((shape[1] / 2.0, shape[0] / 2.0))
                )
            ),
            basis="knm",
            orientation_check=True,
            spot_amp=spot_amp,
            cameraslm=self,
            name="Fourier Grid",
        )

        # Default optimize settings.
        kwargs.setdefault("maxiter", 10)
        kwargs.setdefault("name", "Fourier Grid")

        # Warn the user in case they mistyped a default argument or something.
        for key in kwargs.keys():
            if key not in [
                "method", "maxiter", "verbose", "callback", "feedback",
                "stat_groups", "name", "fixed_phase", "raw_stats", "blur_ij",
            ]:
                self.logger.warning(
                    "Unexpected argument '%s' passed to fourier_grid_project(). "
                    "This may be ignored.", key
                )

        # Optimize and project the hologram
        hologram.optimize(**kwargs)

        self.slm.set_phase(hologram.get_phase(), settle=True)

        return hologram

    def fourier_calibrate_analytic(self, M, b):
        r"""
        Sets the Fourier calibration to a user-supplied affine transformation.

        ``M`` and ``b`` define the mapping :math:`\vec{y} = M\vec{x} + \vec{b}`
        from SLM Fourier space (``"kxy"``) to raw camera sensor coordinates
        (before WOI offset, binning, or orientation transform).
        When none of those transforms are active, raw coordinates equal image coordinates.

        See :meth:`fourier_calibration_build` to construct ``M`` and ``b`` analytically
        from a known focal length.

        Parameters
        ----------
        M : array_like
            2×2 affine matrix mapping kxy → raw camera pixels.
        b : array_like
            Length-2 translation vector (raw camera pixel coordinates).

        Returns
        -------
        dict
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations` ``["fourier"]``
        """
        # Parse arguments.
        M = np.squeeze(M)
        if np.any(M.shape != (2,2)):
            raise ValueError("Expected a 2x2 matrix for M.")
        a = format_2vectors([0,0])
        b = format_2vectors(b)

        self.calibrations["fourier"] = {
            "M": M,
            "b": b,
            "a": a
        }
        self.calibrations["fourier"].update(self._get_calibration_metadata())

        # Set the camera's virtual calibration if it is not already set.
        if hasattr(self.cam, "set_affine") and not hasattr(self.cam, "M"):
            self.cam.set_affine(M, b)

        return self.calibrations["fourier"]

    def fourier_calibration_build(
        self,
        f_eff,
        units="norm",
        theta=0,
        shear_angle=0,
        offset=None,
    ):
        """
        Builds analytic ``M`` and ``b`` from a known focal length, suitable for passing to
        :meth:`fourier_calibrate_analytic`.
        Delegates to :meth:`~slmsuite.holography.toolbox.build_affine`,
        defaulting ``offset`` to the camera center.

        Parameters
        ----------
        f_eff : float
            Effective focal length in ``units``.
        units : str
            Length units for ``f_eff``.
        theta : float
            Rotation angle in radians.
        shear_angle : float
            Shear angle in radians.
        offset : array_like or None
            Camera-space offset of the optical axis. Defaults to the camera center.

        Returns
        -------
        (M, b) : tuple of numpy.ndarray
            Affine parameters suitable for :meth:`fourier_calibrate_analytic`.
        """
        if offset is None:
            offset = np.flip(self.cam.shape) / 2
        return toolbox.build_affine(
            f_eff,
            units=units,
            theta=theta,
            shear_angle=shear_angle,
            offset=offset,
            cam_pitch_um=self.cam.pitch_um,
            wav_um=self.slm.wav_um,
        )

    ### Fourier Calibration User Results ###

    def _kxyslm_to_ijcam_depth(self, kxy_depth):
        """Helper function for handling depth conversion."""
        f_eff = np.mean(self.get_effective_focal_length("norm"))
        if self.cam.pitch_um is None:
            cam_pitch_um = np.nan
        else:
            cam_pitch_um = np.mean(self.cam.pitch_um)
        return kxy_depth * (self.slm.wav_um * f_eff * f_eff / cam_pitch_um)

    def _ijcam_to_kxyslm_depth(self, ij_depth):
        """Helper function for handling depth conversion."""
        f_eff = np.mean(self.get_effective_focal_length("norm"))
        if self.cam.pitch_um is None:
            cam_pitch_um = np.nan
        else:
            cam_pitch_um = np.mean(self.cam.pitch_um)
        return ij_depth * (cam_pitch_um / (self.slm.wav_um * f_eff * f_eff))

    def kxyslm_to_ijcam(self, kxy):
        r"""
        Converts SLM Fourier space (``"kxy"``) to camera pixel space (``"ij"``).
        For blaze vectors :math:`\vec{x}` and camera pixel indices :math:`\vec{y}`
        (with binning and WOI applied), computes:

        .. math:: \vec{y} = M \cdot \vec{x} + \vec{b}

        where :math:`M` and :math:`\vec{b}` are computed from stored calibrations.

        Important
        ~~~~~~~~~

        If the vectors are three-dimensional, the third depth dimension is treated according to:

        .. math:: y_z = \frac{f_\text{eff}^2}{\pi}x_z

        where :math:`y_z` is the normalized depth of the spot relative to the focal plane and
        :math:`x_z` is equivalent to focal power, equivalent to
        the quadratic term of a simple thin :meth:`~slmsuite.holography.toolbox.phase.lens()`.
        The constant of proportionality makes use of the normalized effective focal length
        :math:`f_\text{eff}` of the imaging system between the SLM and camera.
        This information is encoded in the Fourier calibration, and revealed by
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.get_effective_focal_length()`.

        Important
        ~~~~~~~~~
        When a WOI or binning is applied to a camera, this conversion helper function uses the
        WOI/binned coordinate system, i.e. the coordinate system of the returned image.
        After all, the user is working with the returned image.
        Internally, the stored calibration is kept in terms of the **full**,
        **untransformed** camera coordinates.

        Parameters
        ----------
        kxy : array_like
            2D or 3D vector(s) in SLM Fourier space.
            Cleaned with :meth:`~slmsuite.holography.toolbox.format_vectors()`.

        Returns
        -------
        ij : numpy.ndarray
            2D or 3D vector(s) in camera pixel coordinates.

        Raises
        ------
        RuntimeError
            If the Fourier calibration does not exist.
        """
        self._check_fourier_calibration_stale()

        kxy = format_vectors(kxy, handle_dimension="pass")
        ij = self.fourier_affine * kxy[:2, :]

        # Handle z if needed.
        if kxy.shape[0] == 3:
            return np.vstack((ij, self._kxyslm_to_ijcam_depth(kxy[[2], :])))
        else:
            return ij

    def ijcam_to_kxyslm(self, ij):
        r"""
        Converts camera pixel space (``"ij"``) to SLM Fourier space (``"kxy"``).
        For camera pixel indices :math:`\vec{y}` (with binning and WOI applied)
        and blaze vectors :math:`\vec{x}`, computes:

        .. math:: \vec{x} = M^{-1} \cdot (\vec{y} - \vec{b}) + \vec{a}

        where :math:`M` and :math:`\vec{b}` are computed from stored calibrations.

        Important
        ~~~~~~~~~

        If the vectors are three-dimensional, the third depth dimension is treated according to:

        .. math:: x_z = \frac{1}{f} = \frac{1}{f_\text{eff}^2}\frac{\Delta_{xy} y_z}{\lambda}

        where :math:`x_z`, equivalent to normalized focal power, is the focal term
        needed to focus a spot at :math:`y_z` pixel depth.
        Here, :math:`\frac{\Delta_{xy} y_z}{\lambda}` is the same depth in normalized units.
        Importantly, this is depth relative to the plane of the camera, which might
        differ from the relative depth in an experimental plane.
        Focal power is equivalent to
        the quadratic term of a simple thin :meth:`~slmsuite.holography.toolbox.phase.lens()`.
        The constant of proportionality makes use of the normalized effective focal length
        :math:`f_\text{eff}` of the imaging system between the SLM and camera.
        This information is encoded in the Fourier calibration, and revealed by
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.get_effective_focal_length()`.

        Important
        ~~~~~~~~~
        When a WOI or binning is applied to a camera, this conversion helper function uses the
        WOI/binned coordinate system, i.e. the coordinate system of the returned image.
        After all, the user is working with the returned image.
        Internally, the stored calibration is kept in terms of the **full**,
        **untransformed** camera coordinates.

        Parameters
        ----------
        ij : array_like
            2D or 3D vector(s) in camera pixel coordinates (WOI/binning/orientation applied).
            Cleaned with :meth:`~slmsuite.holography.toolbox.format_vectors()`.

        Returns
        -------
        kxy : numpy.ndarray
            2D or 3D vector(s) in SLM Fourier space.

        Raises
        ------
        RuntimeError
            If the Fourier calibration does not exist.
        """
        self._check_fourier_calibration_stale()

        ij = format_vectors(ij, handle_dimension="pass")
        kxy = self.fourier_affine.inv * ij[:2, :]

        # Handle z if needed.
        if ij.shape[0] == 3:
            return np.vstack((kxy, self._ijcam_to_kxyslm_depth(ij[[2], :])))
        else:
            return kxy

    def _check_fourier_calibration_stale(self):
        """
        Raises :exc:`RuntimeError` if no Fourier calibration exists.
        Warns if the wavefront calibration is newer than the Fourier calibration.
        """
        if "fourier" not in self.calibrations:
            raise RuntimeError("Fourier calibration must exist to be used.")

        try:
            if "wavefront_superpixel" in self.calibrations and "fourier" in self.calibrations:
                if (
                    self.calibrations["wavefront_superpixel"]["__timestamp__"] >
                    self.calibrations["fourier"]["__timestamp__"]
                ):
                    self.logger.warning(
                        "The wavefront calibration is newer (%s) than the Fourier "
                        "calibration (%s). The Fourier calibration may be stale.",
                        self.calibrations["wavefront_superpixel"]["__time__"],
                        self.calibrations["fourier"]["__time__"],
                    )
        except Exception:
            pass

    def _get_kxyslm_to_ijraw(self):
        """
        Gets the raw affine transformation from the Fourier calibration.

        Returns
        -------
        affine : Affine
        """
        return Affine(
            self.calibrations["fourier"]["M"],
            self.calibrations["fourier"]["b"],
            self.calibrations["fourier"]["a"],
        )

    @property
    def fourier_affine(self):
        """
        Affine transformation from SLM Fourier space (``"kxy"``) to camera pixel space (``"ij"``),
        accounting for the camera's current WOI, binning, and orientation.

        Returns
        -------
        affine : :class:`~slmsuite.holography.analysis.Affine`

        Raises
        ------
        RuntimeError
            If the Fourier calibration does not exist.
        """
        self._check_fourier_calibration_stale()

        return self.cam._get_ijraw_to_ijcam() @ self._get_kxyslm_to_ijraw()

    # Conversion functions to extract useful information from the Fourier calibration.

    def get_farfield_spot_size(self, slm_size=None, basis="kxy"):
        """
        Calculates the size of a spot produced by blazed patch of size ``slm_size`` on the SLM.
        If this patch is the size of the SLM, then we will find in the farfield (camera)
        domain, the size of a diffraction-limited spot for a fully-illuminated surface.
        As the ``slm_size`` of the patch on the SLM decreases, the diffraction limited
        spot size in the farfield domain will of course increase. This calculation
        is accomplished using the calibration produced by
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate()`
        and stored in
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["fourier"]`.

        Parameters
        ----------
        slm_size : (float, float) OR int OR float OR None
            Size of patch on the SLM in normalized units.
            A scalar is interpreted as the width and height of a square.
            If ``None``, defaults to the normalized SLM size.
        basis : {"kxy", "ij"}
            Basis of the returned size;
            ``"kxy"`` for SLM :math:`k`-space, ``"ij"`` for camera size.

        Returns
        -------
        (float, float)
            Size in x and y of the spot in the desired ``basis``.

        Raises
        ------
        ValueError
            If the basis argument was malformed.
        """
        # Default to effective SLM aperture size (based on amplitude profile if measured)
        if slm_size is None:
            psf_kxy = self.slm.get_spot_radius_kxy()
            slm_size = (1 / psf_kxy, 1 / psf_kxy)
        # Float input -> square region
        elif isinstance(slm_size, REAL_TYPES):
            slm_size = (slm_size, slm_size)

        if basis == "kxy":
            return (1 / slm_size[0], 1 / slm_size[1])
        elif basis == "ij":
            # Use the WOI/binning/orientation-aware affine
            # so the de-rotation lives in the same coordinate frame as kxyslm_to_ijcam below.
            M = self.fourier_affine.M
            # Compensate for spot rotation s.t. spot size is along camera axes
            size_kxy = np.linalg.inv(M / np.sqrt(np.abs(np.linalg.det(M)))) @ np.array(
                (1 / slm_size[0], 1 / slm_size[1])
            )
            return np.abs(self.kxyslm_to_ijcam([0, 0]) - self.kxyslm_to_ijcam(size_kxy)).flatten()
        else:
            raise ValueError('Unrecognized basis "{}".'.format(basis))

    def get_effective_focal_length(self, units="norm"):
        """
        Uses the Fourier calibration to estimate the scalar effective focal length of the
        optical train separating the Fourier-domain SLM from the camera.
        This currently assumes an isotropic imaging train without cylindrical optics.

        Tip
        ~~~
        This effective focal length between the SLM and camera is potentially different
        from the effective focal length between the SLM and experiment.

        Parameters
        ----------
        units : str {"ij", "norm", "m", "cm", "mm", "um", "nm"}
            Units for the focal length.

            -  ``"ij"``
                Focal length in units of camera pixels.

            -  ``"norm"``
                Normalized focal length in wavelengths.

            -  ``"m"``, ``"cm"``, ``"mm"``, ``"um"``, ``"nm"``
                Focal length in metric units.

        Returns
        -------
        f_eff : float
            Effective focal length.
        """
        if "fourier" not in self.calibrations:
            raise RuntimeError("Fourier calibration must exist to be used.")

        # Gather f_eff in pix/rad. This is WOI-invariant (WOI only shifts the affine
        # offset, not M) and binning-invariant in metric units (the 1/bin factor in
        # fourier_affine cancels the *bin factor in cam.pitch_um).
        f_eff = np.sqrt(np.abs(self.fourier_affine.det()))

        # Gather other conversions.
        if units != "ij" and self.cam.pitch_um is None:
            self.logger.warning("cam.pitch_um must be set to use units '%s'", units)
            return np.nan

        # Convert.
        if units == "ij":
            pass
        elif units == "norm":
            f_eff *= np.array(self.cam.pitch_um) / self.slm.wav_um
        elif units in toolbox.LENGTH_FACTORS.keys():
            f_eff *= np.array(self.cam.pitch_um) / toolbox.LENGTH_FACTORS[units]
        else:
            raise ValueError(f"Unit '{units}' not recognized as a length.")

        return f_eff

    # Helper functions to plot the rectangles of the camera and SLM farfield onto each other.

    def get_farfield_extent(self, return_mask=False):
        """
        Find the extent of the SLM's farfield **in the coordinates of the camera**:
        either the corners in ``ij`` units
        or as a boolean mask of the camera's shape.

        Parameters
        ----------
        return_mask : bool
            If ``False``, returns a ``(2, 5)`` array of the farfield corner coordinates
            in camera pixel space (closed polygon with camera origin repeated).
            If ``True``, returns a boolean mask of shape ``cam.shape`` that is ``True``
            where the SLM farfield falls on the camera.

        Returns
        -------
        numpy.ndarray
        """
        ll = [0, 0]
        lr = [1, 0]
        ur = [1, 1]
        ul = [0, 1]

        corners_knm = toolbox.format_2vectors(
            np.vstack((ll, lr, ur, ul, ll)).T
        )
        corners_ij = self.kxyslm_to_ijcam(
            toolbox.convert_vector(
                corners_knm,
                from_units="knm",
                to_units="kxy",
                hardware=self,
                shape=(1,1),
            )
        )

        if not return_mask:
            return corners_ij
        else:
            # Fill the shadow of the camera on the canvas.
            canvas = np.zeros(self.cam.shape, dtype=np.uint8)
            pts = np.rint(corners_ij.T).astype(np.int32)  # (N, 2) required by cv2
            cv2.fillConvexPoly(canvas, pts, 255, cv2.LINE_4)

            return canvas > 128

    def get_camera_extent(self, units="kxy", return_mask=False):
        """
        Find the extent of the camera **in the coordinates of the farfield**:
        either the corners in the specified farfield coordinate system
        or as a boolean mask in ``knm`` space.

        Parameters
        ----------
        units : str OR (int, int) OR Hologram
            Target coordinate system.  A string (e.g. ``"kxy"``, ``"knm"``) returns corners
            in that basis.  A shape tuple or :class:`~slmsuite.holography.algorithms.Hologram`
            returns corners in ``"knm"`` space for that grid (and supports mask output).
        return_mask : bool
            If ``False``, returns a ``(2, 5)`` array of camera corner coordinates in ``units``
            (closed polygon with camera origin repeated).
            If ``True``, requires ``units`` to be a shape; returns a boolean mask of that
            shape that is ``True`` where the camera falls on the farfield.

        Returns
        -------
        numpy.ndarray
        """
        cam_shape = self.cam.shape

        ll = [0, 0]
        lr = [cam_shape[1] - 1, 0]
        ur = [cam_shape[1] - 1, cam_shape[0] - 1]
        ul = [0, cam_shape[0] - 1]

        corners_kxy = self.ijcam_to_kxyslm(
            toolbox.format_2vectors(np.vstack((ll, lr, ur, ul, ll)).T)
        )

        if isinstance(units, str):
            if return_mask:
                raise ValueError("return_mask must be False if units is a string.")
            return toolbox.convert_vector(
                corners_kxy,
                from_units="kxy",
                to_units=units,
                hardware=self,
            )
        else:   # Is the shape of the knm space
            if isinstance(units, Hologram):
                units = units.shape

            units = format_shape(units)

            corners_knm = toolbox.convert_vector(
                corners_kxy,
                from_units="kxy",
                to_units="knm",
                hardware=self,
                shape=units,
            )

            if not return_mask:
                return corners_knm
            else:
                # Fill the shadow of the camera on the knm canvas.
                canvas = np.zeros(units, dtype=np.uint8)
                pts = np.rint(corners_knm.T).astype(np.int32)  # (N, 2) required by cv2
                cv2.fillConvexPoly(canvas, pts, 255, cv2.LINE_4)

                return canvas > 128
