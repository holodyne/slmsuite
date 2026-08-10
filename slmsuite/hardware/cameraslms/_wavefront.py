import matplotlib.pyplot as plt
from slmsuite._plotting import _slmsuite_plt_show
import numpy as np
import warnings

from slmsuite.holography import analysis
from slmsuite.holography import toolbox
from slmsuite.holography.toolbox import format_2vectors, fit_3pt, convert_vector

from slmsuite.hardware.cameraslms._wavefront_superpixel import _WavefrontCalibrationSuperpixel
from slmsuite.hardware.cameraslms._wavefront_zernike import _WavefrontCalibrationZernike

from slmsuite.misc.math import INTEGER_TYPES, REAL_TYPES

class _WavefrontCalibration(
    _WavefrontCalibrationSuperpixel,
    _WavefrontCalibrationZernike,
):
    """
    Hidden superclass with wavefront calibration methods
    (measure SLM wavefront phase [and amplitude]).
    """
    ### Wavefront Calibration Entrypoint ###

    def wavefront_calibrate(
        self,
        *args,
        method=None,
        **kwargs,
    ):
        """
        Backwards-compatible method to switch between
        the superpixel :meth:`wavefront_calibrate_superpixel`
        and Zernike :meth:`wavefront_calibrate_zernike`
        implementations of wavefront calibration.

        Important
        ~~~~~~~~~
        Wavefront calibration will generally shift spot centers slightly, making a
        previous Fourier calibration "stale". It is recommended to perform Fourier
        calibration after wavefront calibration.
        """
        if method is None:
            method = "superpixel"

        if method == "superpixel":
            if "interference_point" in kwargs:
                warnings.warn(
                    "The 'interference_point' argument is deprecated. "
                    "Use 'calibration_points' instead."
                )
                kwargs["calibration_points"] = kwargs.pop("interference_point")

            if "calibration_point" in kwargs:
                warnings.warn(
                    "The 'calibration_point' argument is deprecated. "
                    "Use 'calibration_points' instead."
                )
                kwargs["calibration_points"] = kwargs.pop("calibration_point")

            return self.wavefront_calibrate_superpixel(*args, **kwargs)
        elif method == "zernike":
            return self.wavefront_calibrate_zernike(*args, **kwargs)
        else:
            raise ValueError(f"Wavefront calibration method '{method}' not recognized.")

    ### Wavefront Calibration Common Helper ###

    def _wavefront_calibration_points_parse(self, calibration_points, **kwargs):
        # Parse calibration_points.
        if calibration_points is None or isinstance(calibration_points, INTEGER_TYPES):
            if isinstance(calibration_points, INTEGER_TYPES):
                if calibration_points <= 0:
                    raise ValueError("If an integer, 'calibration_points' must be positive.")
            # If None, then use the built-in generator.
            calibration_points_ = self.wavefront_calibration_points(**kwargs)
            if calibration_points is None:
                num_points = calibration_points_.shape[1]
            else:
                num_points = min(calibration_points_.shape[1], calibration_points)
            calibration_points = calibration_points_[:, :num_points]

        calibration_points = np.rint(format_2vectors(calibration_points)).astype(int)

        # Error check that the calibration points are within the camera's field of view.
        # If pitch is passed to kwargs, then camera should accommodate points within pitch/2 of the edge.
        pitch = kwargs.get("pitch", 0)

        outside_fov_mask = (
            (calibration_points[0,:] < pitch/2) +
            (calibration_points[1,:] < pitch/2) +
            (calibration_points[0,:] > self.cam.shape[1] - pitch/2) +
            (calibration_points[1,:] > self.cam.shape[0] - pitch/2)
        ) > 0

        if np.any(outside_fov_mask):
            raise ValueError(
                f"Calibration points must be within the camera's field of view. "
                f"Found = {calibration_points[:, outside_fov_mask]} which are outside "
                f"the camera shape {self.cam.shape} and desired pitch={pitch}."
            )

        return calibration_points

    def wavefront_calibration_points(
        self,
        pitch,
        field_exclusion=None,
        field_point=(0,0),
        field_point_units="kxy",
        avoid_points=None,
        avoid_mirrors=True,
        avoid_nyquist=True,
        plot=0,
    ):
        """
        Generates a grid of points to perform wavefront calibration at.

        Parameters
        ----------
        pitch : float OR (float, float)
            The grid of points in the camera plane must have pixel pitch
            greater than this value.
        field_exclusion : float OR None
            Remove all points within ``field_exclusion`` of a ``field_point``.
            Set to zero if no removal is desired.
            If ``None``, defaults to ``pitch``.
        field_point : (float, float)
            Position in the camera domain where the field (pixels not included in superpixels)
            is blazed toward in order to reduce light in the camera's field. The suggested
            approach is to set this outside the field of view of the camera and make
            sure that other diffraction orders are far from the ``calibration_points``.
            Defaults to no blaze (``(0,0)`` in ``"kxy"`` units).
        field_point_units : str
            A unit compatible with
            :meth:`~slmsuite.holography.toolbox.convert_vector()`.
            Defaults to ``"kxy"``.

            Tip
            ~~~
            Setting one coordinate of ``field_point`` to zero is suggested
            to minimize higher order diffraction.
        avoid_points : numpy.ndarray
            Additional points to avoid in the same manner as avoiding the ``field_point``
            and diffractive orders (with the same radius ``field_exclusion``).
            This can, for instance, omit the points outside the camera's field of view,
            points around known stray reflections, or unusual topology.
        avoid_mirrors : bool
            When a 1st order calibration beam is sourced from a
            weak superpixel in the SLM domain, the -1st order of a different
            calibration beam can act as a strong noise source if
            it is sourced from a strong central superpixel.
            If ``True``, this flag aligns the -1st orders to be between
            the 1st orders of the grid of calibration points.
        avoid_nyquist : bool
            If ``True``, omits points that are outside the first Nyquist zone.
        plot : int OR bool
            If ``>= 1``, plots the chosen points against the avoided ones.

        Returns
        -------
        numpy.ndarray
            List of points of shape ``(2, N)`` to calibrate at in the ``"ij"`` basis.

        Raises
        ------
        AssertionError
            If the fourier plane calibration does not exist.
        """
        # Parse field_point.
        field_point = toolbox.convert_vector(
            format_2vectors(field_point),
            from_units=field_point_units,
            to_units="ij",
            hardware=self
        )
        field_point = np.rint(format_2vectors(field_point)).astype(int)

        # Parse field_exclusion.
        if field_exclusion is None:
            field_exclusion = pitch
        if not np.isscalar(field_exclusion):
            field_exclusion = np.mean(field_exclusion)

        # Gather other information.
        zeroth_order = np.rint(self.kxyslm_to_ijcam([0, 0])).astype(int)

        # Generate the initial grid.
        plane = format_2vectors(self.cam.shape[::-1])
        if not np.isscalar(pitch):
            pitch = format_2vectors(pitch)
        grid = np.ceil(plane / pitch - .5)
        spacing = np.floor(plane / (grid + (.5 if avoid_mirrors else 0))).astype(int)
        if avoid_mirrors:
            base_point = spacing * (np.remainder(zeroth_order / spacing - .5, 1) + .25)
        else:
            base_point = spacing / 2

        # In ij coordinates.
        calibration_points = fit_3pt(
            base_point,
            (spacing[0,0], 0),
            (0, spacing[1,0]),
            np.squeeze(grid).astype(int),
            x1=None,
            x2=None
        )

        if avoid_nyquist:
            # knm basis with shape (1,1) maps the first Nyquist zone to 0 -> 1.
            calibration_points_knm = convert_vector(
                calibration_points,
                from_units="ij",
                to_units="knm",
                hardware=self,
                shape=[1,1]
            )

            outside_first_nyquist_zone = (
                (calibration_points_knm[0] < 0) +
                (calibration_points_knm[1] < 0) +
                (calibration_points_knm[0] > 1) +
                (calibration_points_knm[1] > 1)
            ) > 0
            calibration_points = np.delete(calibration_points, outside_first_nyquist_zone, axis=1)

        # Sort by proximity to the center, avoiding the 0th order.
        distance = np.sum(np.square(calibration_points - zeroth_order), axis=0)
        I = np.argsort(distance)
        calibration_points = calibration_points[:, I]

        # Prune points within field_exclusion from a given order (-2, ..., 2).
        dorder = field_point - zeroth_order
        order_points = np.hstack([zeroth_order + dorder * i for i in range(-2, 3)])

        if avoid_points is None:
            avoid_points = order_points
        else:
            avoid_points = np.hstack((format_2vectors(avoid_points), order_points))

        for i in range(avoid_points.shape[1]):
            point = avoid_points[:, [i]]
            distance = np.sum(np.square(calibration_points - point), axis=0)
            calibration_points = np.delete(
                calibration_points,
                distance < field_exclusion*field_exclusion,
                axis=1
            )

            # Plot bad points.
            if plot >= 1: plt.scatter(point[0], point[1], c="r")

        if plot >= 1:
            # Points
            plt.scatter(
                calibration_points[0,:],
                calibration_points[1,:],
                c=np.arange(calibration_points.shape[1]),
                cmap="Blues"
            )

            # Mirrors
            plt.scatter(
                2*zeroth_order[0,0] - calibration_points[0,:],
                2*zeroth_order[1,0] - calibration_points[1,:],
                c=np.arange(calibration_points.shape[1]),
                marker=".",
                cmap="Reds"
            )

            # Future: Plot SLM FoV?

            plt.xlim([0, self.cam.shape[1]])
            plt.ylim([self.cam.shape[0], 0])
            _slmsuite_plt_show(name="wavefront_calibration_points")

        return calibration_points
