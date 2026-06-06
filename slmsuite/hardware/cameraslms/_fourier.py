import matplotlib.pyplot as plt
import numpy as np
import warnings
import cv2

from slmsuite.holography import analysis
from slmsuite.holography import toolbox
from slmsuite.holography.algorithms import SpotHologram
from slmsuite.holography.toolbox import format_2vectors, format_vectors, format_shape
from slmsuite.misc.math import INTEGER_TYPES, REAL_TYPES

from slmsuite.hardware.cameras.simulated import SimulatedCamera

class _FourierCalibration(object):
    """
    Hidden superclass with Fourier calibration methods
    (SLM angle-space to camera-space conversion).
    """

    ### Fourier Calibration ###

    def fourier_calibrate(
        self,
        array_shape=10,
        array_pitch=10,
        array_center=None,
        plot=False,
        autofocus=False,
        autoexposure=False,
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
        array_center
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
            **in the** ``"knm"`` **basis.**  ``array_center`` is not passed directly, and is
            processed as being relative to the center of ``"knm"`` space, the position
            of the 0th order.
        plot : bool OR int
            Enables debug plots:

            - 0 is no plots,
            - 1 is only the final fit plot, unless there is an error,
            - 2 is all plots.
        autofocus : bool OR dict
            Whether or not to autofocus the camera.
            If a dictionary is passed, autofocus is performed,
            and the dictionary is passed to
            :meth:`~slmsuite.hardware.cameras.camera.Camera.autofocus()`.
        autoexposure : bool OR dict
            Whether or not to automatically set the camera exposure.
            If a dictionary is passed, autoexposure is performed,
            and the dictionary is passed to
            :meth:`~slmsuite.hardware.cameras.camera.Camera.autoexposure()`.
        **kwargs : dict
            Passed to :meth:`.fourier_grid_project()`, which passes them to
            :meth:`~slmsuite.holography.algorithms.SpotHologram.optimize()`.

        Returns
        -------
        dict
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["fourier"]`
        """
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
            warnings.warn(
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

        # Optional step -- autofocus and autoexpose the spots
        if autofocus or isinstance(autofocus, dict):
            # Pre-expose
            if autoexposure or isinstance(autoexposure, dict):
                if isinstance(autoexposure, dict):
                    self.cam.autoexposure(**autoexposure)
                else:
                    self.cam.autoexposure()

            # Focus
            if isinstance(autofocus, dict):
                self.cam.autofocus(plot=plot, **autofocus)
            else:
                self.cam.autofocus(plot=plot)

        # Post-expose
        if autoexposure or isinstance(autoexposure, dict):
            if isinstance(autoexposure, dict):
                self.cam.autoexposure(**autoexposure)
            else:
                self.cam.autoexposure()

        img = self.cam.get_image()

        # Get orientation of projected array
        try:
            orientation = analysis.blob_array_detect(img, array_shape, plot=plot)
        except Exception as e:
            warnings.warn("fourier_calibrate failed during array detection and fitting.")
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

        self.calibrations["fourier"] = {
            "M": M,
            "b": b,
            "a": a
        }
        self.calibrations["fourier"].update(self._get_calibration_metadata())

        return self.calibrations["fourier"]

    ### Fourier Calibration Helpers ###

    def fourier_grid_project(self, array_shape=10, array_pitch=10, array_center=None, **kwargs):
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
        **kwargs
            Passed to :meth:`~slmsuite.holography.algorithms.SpotHologram.optimize()`.

        Returns
        -------
        ~slmsuite.holography.algorithms.SpotHologram
            Optimized hologram.
        """
        # Check that the pitch is an integer.
        if not np.all(np.isclose(array_pitch, np.rint(array_pitch))):
            warnings.warn("array_pitch is non-integer")

        # Make the spot array
        shape = SpotHologram.get_padded_shape(self, padding_order=1, square_padding=True)
        hologram = SpotHologram.make_rectangular_array(
            shape,
            array_shape=array_shape,
            array_pitch=array_pitch,
            array_center=None
            if array_center is None
            else (
                format_2vectors(array_center) +
                format_2vectors((shape[1] / 2.0, shape[0] / 2.0))
            ),
            basis="knm",
            orientation_check=True,
            cameraslm=self,
        )

        # Default optimize settings.
        if "maxiter" not in kwargs:
            kwargs["maxiter"] = 10

        # Warn the user in case they mistyped a default argument or something.
        for key in kwargs.keys():
            if key not in [
                "method", "maxiter", "verbose", "callback", "feedback",
                "stat_groups", "name", "fixed_phase", "raw_stats", "blur_ij",
            ]:
                warnings.warn(
                    f"Unexpected argument '{key}' passed to fourier_grid_project(). "
                    "This may be ignored."
                )

        # Optimize and project the hologram
        hologram.optimize(**kwargs)

        self.slm.set_phase(hologram.get_phase(), settle=True)

        return hologram

    def fourier_calibrate_analytic(self, M, b):
        """
        Sets the Fourier calibration to a user-selected affine transformation.

        See :meth:`fourier_calibration_build()` to generate this transformation from a
        known or measured focal length.

        Parameters
        ----------
        M : numpy.ndarray
            Affine matrix :math:`M`. Shape ``(2, 2)``.
        b : numpy.ndarray
            Affine vector :math:`b`. Shape ``(2, 1)``.

        Returns
        -------
        dict
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["fourier"]`
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
        META: This docstring will be overwritten by ``SimulatedCamera.build_affine``'s
        after this class.
        """
        if offset is None:
            offset = np.flip(self.cam.shape) / 2
        return SimulatedCamera._build_affine(
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
        For blaze vectors :math:`\vec{x}` and camera pixel indices :math:`\vec{y}`, computes:

        .. math:: \vec{y} = M \cdot (\vec{x} - \vec{a}) + \vec{b}

        where :math:`M`, :math:`\vec{b}`, and :math:`\vec{a}` are stored in
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations` ``["fourier"]``.

        If the vectors are three-dimensional, the third depth dimension is treated according to:

        .. math:: y_z = \frac{f_\text{eff}^2}{\pi}x_z

        where :math:`y_z` is the normalized depth of the spot relative to the focal plane and
        :math:`x_z` is equivalent to focal power, equivalent to
        the quadratic term of a simple thin :meth:`~slmsuite.holography.toolbox.phase.lens()`.
        The constant of proportionality makes use of the normalized effective focal length
        :math:`f_\text{eff}` of the imaging system between the SLM and camera.
        This information is encoded in the Fourier calibration, and revealed by
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.get_effective_focal_length()`.

        Parameters
        ----------
        kxy : array_like
            Vector or array of vectors to convert. Can be 2D or 3D.
            Cleaned with :meth:`~slmsuite.holography.toolbox.format_vectors()`.

        Returns
        -------
        ij : numpy.ndarray
            Vector or array of vectors in camera spatial coordinates. Can be 2D or 3D.

        Raises
        ------
        RuntimeError
            If the fourier plane calibration does not exist.
        """
        if not "fourier" in self.calibrations:
            raise RuntimeError("Fourier calibration must exist to be used.")

        self._check_fourier_calibration_stale()

        kxy = format_vectors(kxy, handle_dimension="pass")

        # Apply the xy transformation.
        ij = np.matmul(
            self.calibrations["fourier"]["M"],
            kxy[:2, :] - self.calibrations["fourier"]["a"]
        ) + self.calibrations["fourier"]["b"]

        # Handle z if needed.
        if kxy.shape[0] == 3:
            return np.vstack((ij, self._kxyslm_to_ijcam_depth(kxy[[2], :])))
        else:
            return ij

    def ijcam_to_kxyslm(self, ij):
        r"""
        Converts camera pixel space (``"ij"``) to SLM Fourier space (``"kxy"``).
        For camera pixel indices :math:`\vec{y}` and blaze vectors :math:`\vec{x}`, computes:

        .. math:: \vec{x} = M^{-1} \cdot (\vec{y} - \vec{b}) + \vec{a}

        where :math:`M`, :math:`\vec{b}`, and :math:`\vec{a}` are stored in
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["fourier"]`.

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

        Parameters
        ----------
        ij : array_like
            Vector or array of vectors to convert. Can be 2D or 3D.
            Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.

        Returns
        -------
        kxy : numpy.ndarray
            Vector or array of vectors in slm angular coordinates. Can be 2D or 3D.

        Raises
        ------
        RuntimeError
            If the fourier plane calibration does not exist.
        """
        if not "fourier" in self.calibrations:
            raise RuntimeError("Fourier calibration must exist to be used.")

        self._check_fourier_calibration_stale()

        ij = format_vectors(ij, handle_dimension="pass")

        # Apply the xy transformation.
        kxy = np.matmul(
            np.linalg.inv(self.calibrations["fourier"]["M"]),
            ij[:2, :] - self.calibrations["fourier"]["b"]
        ) + self.calibrations["fourier"]["a"]

        # Handle z if needed.
        if ij.shape[0] == 3:
            return np.vstack((kxy, self._ijcam_to_kxyslm_depth(ij[[2], :])))
        else:
            return kxy

    def _check_fourier_calibration_stale(self):
        """
        Checks if the wavefront calibration is newer than the Fourier calibration.

        Warns if this is true. Does nothing if either calibration is not present or
        if another error occurs.
        """
        try:
            if "wavefront_superpixel" in self.calibrations and "fourier" in self.calibrations:
                if (
                    self.calibrations["wavefront_superpixel"]["__timestamp__"] >
                    self.calibrations["fourier"]["__timestamp__"]
                ):
                    warnings.warn(
                        f"The wavefront calibration is newer "
                        f"({self.calibrations['wavefront_superpixel']['__time__']}) "
                        f"than the Fourier calibration "
                        f"({self.calibrations['fourier']['__time__']}). "
                        "The Fourier calibration may be stale."
                    )
        except:
            pass

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
            M = self.calibrations["fourier"]["M"]
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
        if not "fourier" in self.calibrations:
            raise RuntimeError("Fourier calibration must exist to be used.")

        # Gather f_eff in pix/rad.
        f_eff = np.sqrt(np.abs(np.linalg.det(self.calibrations["fourier"]["M"])))

        # Gather other conversions.
        if units != "ij" and self.cam.pitch_um is None:
            warnings.warn(f"cam.pitch_um must be set to use units '{units}'")
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

    def get_farfield_extent(self, return_corners=True):
        """
        Find the extent of the SLM's farfield in the plane of the camera.

        Parameters
        ----------
        return_corners : bool
            If ``True``, returns the coordinates of the corners of the farfield.
            If ``False``, return a boolean image with the SLM's farfield.

        Returns
        -------
        """
        cam_shape = self.cam.shape

        ll = [0, 0]
        lr = [1, 0]
        ur = [1, 1]
        ul = [0, 1]

        corners_knm = toolbox.format_2vectors(
            np.vstack((ll, lr, ur, ul, ll)).T
        )
        corners_ij = toolbox.convert_vector(
            corners_knm,
            from_units="knm",
            to_units="ij",
            hardware=self.cameraslm.slm,
            shape=(1,1),
        )

        if return_corners:
            return corners_ij
        else:
            # Fill the shadow of the camera on the canvas.
            canvas = np.zeros(self.cam.shape, dtype=np.uint8)
            pts = np.rint(corners_ij).astype(np.int32)
            cv2.fillConvexPoly(canvas, pts, 255, cv2.LINE_4)

            return canvas > 128

    def get_camera_extent(self, units="kxy", return_corners=True):
        """
        Find the extent of the camera in the basis of various farfield units.

        Parameters
        ----------
        units : str OR (int, int) OR Hologram

        return_corners : bool
            If ``True``, returns the coordinates of the corners of the camera.
            If ``False``, and a shape was passed to units, returns

        Returns
        -------
        """
        cam_shape = self.cam.shape

        ll = [0, 0]
        lr = [cam_shape[1] - 1, 0]
        ur = [cam_shape[1] - 1, cam_shape[0] - 1]
        ul = [0, cam_shape[0] - 1]

        corners_ij = toolbox.format_2vectors(
            np.vstack((ll, lr, ur, ul, ll)).T
        )

        if isinstance(units, str):
            return toolbox.convert_vector(
                corners_ij,
                from_units="ij",
                to_units=basis,
                hardware=self.cameraslm.slm,
            )
        else:
            units = format_shape(units)

            corners_knm = toolbox.convert_vector(
                corners_ij,
                from_units="ij",
                to_units="knm",
                hardware=self.cameraslm.slm,
                shape=units,
            )

            if return_corners:
                return corners_knm
            else:
                # Fill the shadow of the camera on the canvas.
                canvas = np.zeros(units, dtype=np.uint8)
                pts = np.rint(corners_knm).astype(np.int32)
                cv2.fillConvexPoly(canvas, pts, 255, cv2.LINE_4)

                return canvas > 128
