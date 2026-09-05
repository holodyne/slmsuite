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

            # A patch that is a large fraction of the addressable area leaves nowhere to
            # put a point, once the pitch/2 margins at the sensor edge and at the edge of
            # the first Nyquist zone plus the field exclusion are all honored. 
            if calibration_points.shape[1] == 0:
                zone = np.ptp(self.get_farfield_extent(), axis=1)
                raise ValueError(
                    f"No calibration points fit at pitch={kwargs.get('pitch')} (the width "
                    f"of one calibration patch). Each point needs its whole patch inside "
                    f"both the camera {tuple(self.cam.shape[::-1])} and the SLM's first "
                    f"Nyquist zone, which spans only {tuple(np.rint(zone).astype(int))} "
                    f"px of it, and outside the field exclusion around the zeroth order. "
                    f"Shrink the patch -- for wavefront diversity, lower 'site_width_ij' "
                    f"or 'sites_per_point'."
                )

        calibration_points = np.rint(format_2vectors(calibration_points)).astype(int)

        # Error check that the calibration points are within the camera's field of view.
        # If pitch is passed to kwargs, then camera should accommodate points within pitch/2 of the edge.
        pitch = kwargs.get("pitch", 0)
        # Per-axis pitches must compare per axis; a raw (2, 1) would broadcast the
        # comparison into a (2, N) truth table and mix the axes together.
        (pitch_x, pitch_y) = (
            (pitch, pitch) if np.isscalar(pitch) else np.ravel(pitch)[:2]
        )

        outside_fov_mask = (
            (calibration_points[0,:] < pitch_x/2) +
            (calibration_points[1,:] < pitch_y/2) +
            (calibration_points[0,:] > self.cam.shape[1] - pitch_x/2) +
            (calibration_points[1,:] > self.cam.shape[0] - pitch_y/2)
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
        # A calibration point owns a patch of width ``pitch`` centered on it, so centers
        # live in ``[pitch/2, plane - pitch/2]``
        margin = pitch / 2.

        # Bounds for the point *centers*, in both domains that can clip a patch: the
        # sensor, and the SLM's first Nyquist zone. Intersecting the two *before* choosing
        # the lattice keeps the points evenly spread, symmetrically, over what is actually
        # addressable. 
        (low, high) = (np.broadcast_to(margin, (2, 1)).astype(float).copy(), plane - margin)

        if avoid_nyquist:
            # Every center in these bounds owns a whole ``pitch``-wide patch inside the
            # first Nyquist zone, which is what the calibration pattern actually tiles.
            zone_ij = self.get_farfield_extent(inscribe=True, margin=margin)
            low = np.maximum(low, np.min(zone_ij, axis=1, keepdims=True))
            high = np.minimum(high, np.max(zone_ij, axis=1, keepdims=True))

        usable = np.maximum(high - low, 0)

        # Points land on integer pixels, so the realized spacing is a whole number and has
        # to be rounded *up* to clear a fractional pitch. Size the lattice against that
        # integer step throughout.
        step = np.ceil(pitch)

        # ``floor(usable/step) + 1`` points fit at >= ``step`` spacing; the count must round
        # *down*, or the realized spacing falls below the ``pitch`` this method documents.
        # ``avoid_mirrors`` takes one fewer per axis, reserving half a spacing to slide the
        # lattice off the mirror lattice below.
        reserve = .5 if avoid_mirrors else 1.
        grid = np.maximum(np.floor(usable / step + reserve), 1)
        spacing = np.maximum(
            np.floor(usable / np.maximum(grid - reserve, .5)), step
        ).astype(int)

        # Room left over to slide the whole lattice within the usable span.
        slack = np.maximum(usable - (grid - 1) * spacing, 0)

        if avoid_mirrors:
            # A point's mirror (-1) order lands at ``2*zeroth_order - point``, so the
            # mirror lattice shares ``spacing`` at offset ``2*zeroth_order - base_point``.
            # Mirrors coincide with points iff ``base_point == zeroth_order`` modulo
            # *half* a spacing, so a quarter-spacing offset (taken mod half a spacing)
            # puts them maximally between points and fits in the slack reserved above.
            # Clamp anyway for a degenerate single-point axis, where no neighbor pins
            # ``spacing`` down.
            base_point = low + np.minimum(
                np.remainder(zeroth_order + spacing / 4. - low, spacing / 2.), slack
            )
        else:
            # Center the lattice in the usable span, so the points sit symmetrically about
            # the addressable region rather than piling against one edge of it.
            base_point = low + slack / 2.

        # Points are rounded to integer pixels downstream, so round the lattice origin up
        # here: a half-integer origin rounds to a spacing one pixel short of ``pitch``.
        base_point = np.ceil(base_point)

        # In ij coordinates.
        calibration_points = fit_3pt(
            base_point,
            (spacing[0,0], 0),
            (0, spacing[1,0]),
            np.squeeze(grid).astype(int),
            x1=None,
            x2=None
        )

        # Prune against the same ``[low, high]`` the lattice was laid out in, which already
        # holds both bounding domains. Two things can push a point back out of it: the
        # ``ceil`` on the origin above, and ``avoid_mirrors`` placing that origin only a
        # quarter spacing in. 
        rounded = np.rint(calibration_points)
        calibration_points = calibration_points[
            :, np.all((rounded >= low) & (rounded <= high), axis=0)
        ]

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
