import copy
import matplotlib.pyplot as plt
from slmsuite._plotting import _slmsuite_plt_show
import numpy as np
from scipy import optimize
from scipy.spatial import Delaunay
from tqdm.auto import tqdm
import warnings

from slmsuite.holography import analysis
from slmsuite.holography import toolbox
from slmsuite.holography.algorithms import CompressedSpotHologram
from slmsuite.holography.toolbox import format_vectors, smallest_distance, convert_vector
from slmsuite.holography.toolbox.phase import _zernike_indices_parse, zernike, zernike_sum

class _WavefrontCalibrationZernike(object):
    """
    Hidden superclass with Zernike wavefront calibration methods
    (project and analyze Zernike modes).
    """
    ### Zernike Wavefront Calibration ###

    def wavefront_calibrate_zernike(
        self,
        calibration_points=None,
        zernike_indices=9,
        perturbation=1,
        callback=None,
        metric=None,
        global_correction=False,
        optimize_focus=True,
        optimize_position=True,
        optimize_weights=True,
        plot=0,
    ):
        r"""
        Perform wavefront calibration by iteratively scanning and subtracting Zernike
        coefficients.

        Parameters
        ----------
        calibration_points : (float, float) OR numpy.ndarray OR int OR None
            Position(s) in the camera domain where interference occurs.
            A passed array should be a standard ``(D, N)`` matrix,
            where ``D`` is the dimension of the Zernike space and ``N`` is the number of points.
            If ``int``, fills the camera field of view with roughly this number of calibration
            points.
            If ``None``, defaults to 100 points, unless a calibration is already saved
            in :attr:`calibrations` under the ``"wavefront_zernike"`` key, in which case
            the ``"corrected_spots"`` from the calibration are used as the baseline.
            This allows the user to iterate on previous calibrations. Note that
            ``zernike_indices`` is also overwritten in this case.

            Important
            ~~~~~~~~~
            These coordinates must be in the ``"zernike"`` basis. Use
            :meth:`~slmsuite.holography.toolbox.convert_vector()` to convert between 2 or
            3 dimensional coordinates to their Zernike counterparts.
        zernike_indices : int OR list of int OR None
            Which Zernike polynomials to calibrate against, defined by ANSI indices. Of shape ``(D,)``.

            Tip
            ~~~
            Use :meth:`~slmsuite.holography.toolbox.phase.zernike_convert_index()`
            to convert to ANSI from radial indexing.

            Important
            ~~~~~~~~~
            If ``None`` is passed, the assumed Zernike basis depends on the
            dimensionality of the provided spots:

            -   If ``D == 2``, then the basis is assumed to be ``[2,1]``
                corresponding to the :math:`x = Z_2 = Z_1^1`
                and :math:`y = Z_1 = Z_1^{-1}` tilt terms.

            -   If ``D == 3``, then the basis is assumed to be ``[2,1,4]``
                corresponding to the previous, with the addition of the
                :math:`Z_4 = Z_2^0` focus term.

            -   If ``D > 3``, then the basis is assumed to be ``[2,1,4,3,5,6...,D]``.
                The piston term (Zernike index 0) is ignored as this constant phase is
                not relevant.
        perturbation : list of float OR float OR None
            Perturbation in radians to iteratively multiply with each of the
            :math:`\pm 1`-normalized Zernike terms.
            If ``float``, tests 11 points in a range of plus to minus this value in radians.
            Defaults to a range of :math:`\pm 1` radians.
            If ``0`` or ``None``, the starting spots are projected and the function returns before optimizing.
        callback : None OR function
            Measure the system to determine the level of aberration. Expected to return
            a list of floats of length ``N`` corresponding to the chosen metric evaluated
            on all the spots. The optimizer will *minimize* the figure of merit.
            This data is fit using a parabola, and the x-offset of the
            parabola is interpreted as the minimum aberration.
        metric : None OR function
            If ``callback`` is ``None``, then the camera is used to measure the system.
            This parameter allows the user to impart a custom figure of merit upon the
            measured camera data. ``metric`` is required to accept a stack of ``N`` images
            consisting of the regions about each of the ``N`` target spots. It is expected to
            return a list of length ``N`` corresponding to the chosen metric evaluated
            on all the images.
            If ``None``, :meth:`._wavefront_calibrate_zernike_default_metric()`
            is used, which is just a wrapper for
            :meth:`~slmsuite.holography.analysis.image_areas()`, a measurement of spot
            size. The optimizer will *minimize* the figure of merit.
        global_correction : bool
            If ``True``, the optimized Zernike coefficients are averaged and applied to the entire SLM.
            This can be useful for the first step of calibration to remove large global aberration terms
            while avoiding noise and uncertainty on individual spots.
            When ``optimize_position`` is ``True``, the ``fit_affine`` flag of
            :meth:`~slmsuite.holography.algorithms.SpotHologram.refine_offset()` is used to extract the global shift.
        optimize_focus : bool
            If ``False``, does not optimize the focus term (ansi index 4). Useful in
            cases where the ``callback`` method is insensitive to :math:`z`-translation
            (e.g. Stark effect of an atom) or in cases where the :math:`z` axis should
            be unchanged.
        optimize_position : bool
            If ``False``, does not optimize the position (tilt, ansi indices 1 and 2) terms.
            Note these are corrected via the :meth:`.SpotHologram.refine_offset()` step
            (gated by this flag), not the per-term Zernike sweep, which always excludes
            piston and tilt.
        optimize_weights : bool OR int
            If ``True``,  optimizes the WGS weights of the hologram one time
            at the beginning of the calibration. Defaults to 10 iterations.
            If integer, then uses this number as the number of iterations to optimize the weights.
            Must be at least 1.
        plot : int or bool
            Whether to provide visual feedback, options are:

            - ``-1`` : No plots or tqdm prints.
            - ``0``, ``False`` : No plots, but tqdm prints.
            - ``1``, ``True`` : Plots on fits and essentials.
            - ``2`` : Plots on everything.

        Returns
        -------
        dict
            The contents of
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["wavefront"]`.

        Raises
        ------
        RuntimeError
            If the Fourier plane calibration does not exist.
        ValueError
            If various points are out of range.
        """
        # Helper function to sweep the amplitude of a Zernike over a pattern.
        def sweep_term(sweep, term, pattern, callback, desc=None):
            result = None
            sweep = np.ravel(sweep)
            N = len(sweep)
            M = None

            iterable = list(enumerate(sweep))
            if plot >= 0:
                iterable = tqdm(iterable, desc=desc, position=0, leave=False)

            for i, x in iterable:
                phase = pattern + x * term
                self.slm.set_phase(phase, settle=True, phase_correct=False)
                this_result = np.array(callback())

                if result is None:
                    M = len(this_result)    # Number of points to measure at.
                    result = np.full((N, M), np.nan, dtype=this_result.dtype)

                if len(this_result) != M:
                    raise RuntimeError()
                else:
                    result[i, :] = this_result

            return result

        # Helper function to fit a parabola to the result of the sweep.
        def fit_term(sweep, result, term, status):
            ddy = np.diff(result, n=2, axis=0)
            a0 = .5 * np.mean(ddy, axis=0) / np.square(np.mean(np.diff(sweep)))
            if True or np.mean(a0) >= 0:    # Determine whether the system has a + or - x^2 term. For now, we force +.
                c0 = np.min(result, axis=0)
                x0 = sweep[np.argmin(result, axis=0)]
            else:
                c0 = np.max(result, axis=0)
                x0 = sweep[np.argmax(result, axis=0)]

            def parabola(x, x0, a, c):
                return c + a * np.square(x - x0)

            g = np.zeros(result.shape[1])
            x = np.zeros(result.shape[1])
            dx = np.zeros(result.shape[1])

            for i in range(result.shape[1]):
                guess = (x0[i], a0[i], c0[i])
                try:
                    popt, pcov = optimize.curve_fit(
                        parabola,
                        sweep,
                        result[:, i],
                        ftol=1e-5,
                        p0=guess,
                        bounds=(
                            [-np.inf, 0, -np.inf],
                            [np.inf, np.inf, np.inf]
                        )
                    )
                    perr = np.sqrt(np.diag(pcov))   # Single sigma error, which can be multiplied later.
                except Exception:
                    popt = guess
                    perr = np.zeros_like(guess)

                g[i] = guess[0]
                x[i] = popt[0]
                dx[i] = perr[0]

            x = np.clip(x, np.min(sweep), np.max(sweep))
            railed = np.sum(np.logical_or(x == np.min(sweep), x == np.max(sweep))) / float(len(x))

            if plot > 0:
                result -= np.min(result, axis=0, keepdims=True)
                result /= np.max(result, axis=0, keepdims=True)
                plt.imshow(
                    result,
                    interpolation="none",
                    extent=[-.5, result.shape[1]-.5, np.max(sweep), np.min(sweep)]
                )
                cbar = plt.colorbar()
                plt.scatter(
                    np.arange(result.shape[1]),
                    g,
                    c="r",
                    marker='x',
                    alpha=.25,
                )
                plt.errorbar(
                    np.arange(result.shape[1]),
                    x,
                    yerr=dx,
                    c="r",
                    marker='.',
                    linestyle='none'
                )
                plt.gca().set_aspect("auto")
                plt.title("Zernike $Z_{" + str(term) + "}$")
                plt.xlabel("Calibration Point [#]")
                plt.ylabel("Perturbation [rad]")
                plt.xlim(-.5, result.shape[1]-.5)
                plt.ylim(np.max(sweep), np.min(sweep))
                cbar.ax.set_ylabel("Figure of Merit [norm]") #, rotation=270)
                _slmsuite_plt_show(name=f"wavefront_calibrate_zernike_{term}")

            return x, dx, railed

        # Parse calibration_points and zernike_indices
        calibration_points_ij = None
        metric_stats = []
        position_stats = []
        weights = None
        spot_integration_width_ij = None

        if calibration_points is None:
            if "wavefront_zernike" in self.calibrations:
                dat = self.calibrations["wavefront_zernike"]
                calibration_points = np.copy(dat["corrected_spots"])
                calibration_points_ij = np.copy(dat["calibration_points_ij"])
                spot_integration_width_ij = np.copy(dat["spot_integration_width_ij"])

                if zernike_indices is None:
                    zernike_indices = np.copy(dat["zernike_indices"])
                else:
                    if np.isscalar(zernike_indices) and zernike_indices < calibration_points.shape[0]:
                            zernike_indices = calibration_points.shape[0]

                    zernike_indices = _zernike_indices_parse(
                        zernike_indices,
                        calibration_points.shape[0],
                        smaller_okay=True
                    )

                    stored_zi = np.copy(dat["zernike_indices"])

                    if len(zernike_indices) >= len(stored_zi):
                        if np.all(zernike_indices[:len(stored_zi)] == stored_zi):
                            pass # Extend zernike indices.
                        else:
                            raise ValueError(
                                f"Requested indices {zernike_indices} "
                                f"is not compatible with stored indices {stored_zi}."
                            )
                    else:
                        raise ValueError(
                            f"Requested indices {zernike_indices} "
                            f"is not compatible with stored indices {stored_zi}."
                        )

                if "metric_stats" in dat:
                    metric_stats = list(copy.copy(dat["metric_stats"]))
                else:
                    metric_stats = []

                if "position_stats" in dat:
                    position_stats = list(copy.copy(dat["position_stats"]))
                else:
                    position_stats = []

                if "weights" in dat:
                    weights = dat["weights"]
                else:
                    weights = None
            else:
                calibration_points = 100

        if np.isscalar(calibration_points):
            pitch = np.sqrt(np.prod(self.cam.shape) / calibration_points)
            calibration_points = self.wavefront_calibration_points(pitch, plot=True)
            # wavefront_calibration_points returns "ij"; convert to "zernike" basis.
            calibration_points = convert_vector(
                calibration_points, from_units="ij", to_units="zernike", hardware=self
            )

        calibration_points = format_vectors(np.copy(calibration_points), handle_dimension="pass")
        zernike_indices = _zernike_indices_parse(zernike_indices, calibration_points.shape[0], smaller_okay=True)
        dp = len(zernike_indices) - calibration_points.shape[0]
        if dp:  # Pad with zeros if the points don't have certain terms.
            calibration_points = np.pad(calibration_points, ((0,dp), (0,0)))

        initial_points = calibration_points.copy()

        # Build hologram
        if calibration_points.shape[1] > 1:
            hologram = CompressedSpotHologram(
                spot_vectors=calibration_points,
                basis=zernike_indices,
                cameraslm=self,
            )

            if not (weights is None):
                hologram.set_weights(weights)

            if calibration_points_ij is None:
                calibration_points_ij = hologram.spot_ij
            else:
                hologram.spot_ij = calibration_points_ij
        else:
            raise ValueError(
                "Zernike wavefront calibration requires at least 2 calibration points; "
                f"got {calibration_points.shape[1]}. Increase 'calibration_points'."
            )

        max_window_size = smallest_distance(calibration_points_ij)  # Size were windows graze each other.
        max_spot_integration_width_ij = int(2 * np.ceil(np.min((.5*max_window_size, 51)) / 2) + 1)
        if spot_integration_width_ij is None:
            spot_integration_width_ij = max_spot_integration_width_ij
        else:
            spot_integration_width_ij = min(int(spot_integration_width_ij), max_spot_integration_width_ij)
        hologram.spot_integration_width_ij = spot_integration_width_ij

        # Parse callback.
        if callback is None:
            def default_callback():
                # self.cam.flush()
                img = self.cam.get_image()

                images = analysis.take(img, calibration_points_ij, spot_integration_width_ij, clip=True).astype(float)
                images = analysis.image_remove_field(images)
                images[np.isnan(images)] = 0
                # Normalize out the per-measurement laser power; guard a dark frame so we
                # don't divide by zero and poison the metric/fit with nan/inf.
                total = np.sum(images)
                if total != 0:
                    images = images / total

                if metric is None:
                    return self._wavefront_calibrate_zernike_default_metric(images)
                else:
                    return metric(images)

            callback = default_callback

        # Tick function.
        def tick():
            if hologram is None:
                pattern = zernike_sum(
                    self.slm,
                    zernike_indices,
                    calibration_points,
                    use_mask=False
                )
            else:
                # Reoptimize the hologram at each step.
                hologram.spot_zernike = calibration_points

                hologram.optimize(
                    "GS",
                    maxiter=3,
                    verbose=0,
                    # raw_stats=True,
                )
                pattern = hologram.get_phase()

            return pattern

        # Parse perturbation
        if perturbation is None:
            perturbation = 1

        hologram.optimize(
            "GS", maxiter=3, verbose=0,
            # raw_stats=True,
            stat_groups=["computational_spot",],
        )

        if optimize_weights:
            if isinstance(optimize_weights, bool):
                maxiter = 10
            else:
                maxiter = int(optimize_weights)
                if maxiter < 1:
                    raise ValueError("optimize_weights must be True, False, or a positive integer.")

            hologram.optimize(
                "WGS-Kim",
                feedback="experimental_spot",
                maxiter=maxiter,
                verbose=True,
                name="optimize_weights",
                stat_groups=["computational_spot", "experimental_spot",],
            )
            if "wavefront_zernike" in self.calibrations:
                self.calibrations["wavefront_zernike"]["weights"] = hologram.get_weights()

        no_perturbation = (
            perturbation is None or
            (np.isscalar(perturbation) and perturbation <= 0) or
            (not np.isscalar(perturbation) and len(perturbation) == 0)
        )

        # If no perturbation, just project the initial spots and return.
        if no_perturbation:
            self.slm.set_phase(tick(), settle=True, phase_correct=False)
            # self.slm.set_phase(hologram.get_phase(), settle=True, phase_correct=False)

            self.cam.flush()
            img = self.cam.get_image()

            if plot:
                take = analysis.take(
                    img,
                    hologram.spot_ij,
                    hologram.spot_integration_width_ij,
                    centered=True,
                    integrate=False,
                )
                max = np.max(take)

                if max >= self.cam.bitresolution-1:
                    self.logger.warning("Image is overexposed.")
                elif max > .5*self.cam.bitresolution:
                    self.logger.warning(
                        "Image might become overexposed during optimization (%s/%s).",
                        max, self.cam.bitresolution-1,
                    )

                self.cam.plot(img, title="Zernike Calibration Status")

                if plot >= 2:

                    plt.figure(figsize=(12, 12))
                    # plt.imshow(tiled)
                    analysis.take_plot(take, separate_axes=False)
                    plt.title("Zernike Calibration Status (Zoom)")
                    _slmsuite_plt_show(name="wavefront_calibrate_zernike")

            return hologram

        # Parse perturbation, maybe returning if perturbation is negative.
        if np.isscalar(perturbation):
            perturbation = np.linspace(-perturbation, perturbation, 11, endpoint=True)
        else:
            perturbation = np.ravel(perturbation)

        # Refine hologram position.
        if optimize_position:
            self.slm.set_phase(tick())
            hologram.refine_offset(img=None, basis="kxy", force_affine=global_correction, plot=plot)

        # Calibration loop.
        result = None
        self.cam.flush()
        for j, i in enumerate(zernike_indices):
            # Ignore the piston and tilt terms, maybe also the focus too.
            if i in [0, 2, 1] or (i == 4 and not optimize_focus):
                continue

            # Generate hologram and record current stats.
            pattern = tick()
            self.slm.set_phase(pattern, settle=True, phase_correct=False)
            metric_stats.append(callback())

            # Determine which Zernike polynomial we are testing.
            term = zernike(self.slm, i, use_mask=False)

            # Test the polynomial. This returns a (N, S) array,
            # where N is the number of spots and S is the number of sweep points.
            result = sweep_term(perturbation, term, pattern, callback, f"Z_{i}")

            # Analyze the results by fitting each to a parabola.
            correction, correction_error, railed = fit_term(perturbation, result, i, calibration_points[j, :])

            # Apply the correction to the spots (globally if desired).
            if global_correction:
                correction = np.mean(correction)
            calibration_points[j, :] += correction

        # Record final stats.
        pattern = tick()
        self.slm.set_phase(pattern, settle=True, phase_correct=False)
        metric_stats.append(callback())
        # position_stats.append(calibration_points)

        self.calibrations["wavefront_zernike"] = {
            "initial_points": initial_points,
            "zernike_indices": zernike_indices,
            "corrected_spots": calibration_points,
            "last_result": result,
            "calibration_points_ij" : calibration_points_ij,
            "spot_integration_width_ij" : spot_integration_width_ij,
            "metric_stats" : metric_stats,
            # "position_stats" : position_stats,
            "weights" : hologram.get_weights(),
        }
        self.calibrations["wavefront_zernike"].update(self._get_calibration_metadata())

        # return hologram

        return self.calibrations["wavefront_zernike"]

    def _wavefront_calibrate_zernike_plot_raw(self, calibration_points=None, index=0):
        dat = self.calibrations["wavefront_zernike"]

        if calibration_points is None:
            calibration_points = np.copy(dat["corrected_spots"])
        calibration_points_ij = np.copy(dat["calibration_points_ij"])
        zernike_indices = np.copy(dat["zernike_indices"])

        aberration = calibration_points[index, :]

        lim = np.max(np.abs(aberration))

        plt.scatter(
            calibration_points_ij[0, :],
            calibration_points_ij[1, :],
            c=aberration,
            cmap="seismic"
        )
        plt.gca().invert_yaxis()
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("Aberration Correction [rad]") #, rotation=270)
        plt.clim(-lim, lim)
        plt.title(f"Zernike $Z_{zernike_indices[index]}$")
        _slmsuite_plt_show(name="wavefront_calibrate_zernike_plot_raw")

    @staticmethod
    def _wavefront_calibrate_zernike_default_metric(images):
        """
        Calculates the spot areas of all the spots in the stack of ``images``.
        Spot area (determinant of the variances) is here a metric of spot aberration,
        where a spot with smaller and tighter area is better.
        """
        variances = analysis.image_variances(images)
        return analysis.image_areas(variances)

    def wavefront_calibrate_zernike_smooth(
        self,
        smoothing=0.25,
        smoothing_xy=0.25,
        smoothing_z=None,
        plot=False,
    ):
        """
        For a 2D array of Zernike-corrected spots, produces a smoothed version of the
        spot coordinates in aberration space by averaging the coordinates of neighbors.
        This is useful for noise reduction.

        Parameters
        ----------
        smoothing : float
            Smoothing factor for higher order terms.
            This weights the original spot coordinates with the average of the
            neighbors. Should be between 0 and 1. Zero retains the original coordinates.
            One fully replaces it with the neighbor average.
        smoothing_xy : float
            Behaves similarly to ``smoothing`` for tip tilt terms. Instead of averaging the full
            coordinate (which would result in all spots eventually converging), the
            error between the current and expected xy position is averaged. The expected
            xy position from an affine Fourier calibration does not account for barrel
            and pincushion distortion, shifts from higher order Zernike terms, or other effects.
            This correction can help mitigate those issues.
        smoothing_z : float OR None
            Not yet implemented. Would behave similarly to ``smoothing_xy`` for the
            focus term. If ``None``, focus would be treated the same as the higher order
            terms.
        plot : bool
            Whether to enable debug plots.
        """
        # Parse inputs.
        if smoothing < 0 or smoothing > 1:
            raise ValueError("Smoothing factor must be between 0 and 1.")
        if smoothing_xy < 0 or smoothing_xy > 1:
            raise ValueError("Smoothing factor must be between 0 and 1.")

        # Build triangulation.
        indices = self.calibrations["wavefront_zernike"]["zernike_indices"]
        I = np.arange(len(indices))
        to_smooth = I[indices > 2]
        x_smooth = I[indices == 2]
        y_smooth = I[indices == 1]
        if smoothing_z is not None:
            raise RuntimeError("Zernike z-smoothing not yet implemented.")

            # if smoothing_z < 0 or smoothing_z > 1:
            #     raise ValueError("Smoothing factor must be between 0 and 1.")
            # z_smooth = I[indices == 4]

        vectors = self.calibrations["wavefront_zernike"]["corrected_spots"]
        final = np.zeros_like(vectors)

        points_ij = self.calibrations["wavefront_zernike"]["calibration_points_ij"]
        base_xy = convert_vector(
            points_ij,
            from_units="ij",
            to_units="zernike",
            hardware=self,
        )

        # Build triangulation (cache in future?).
        points = points_ij[:2, :].T
        tri = Delaunay(points)

        edges = np.array([(i, j) for t in tri.simplices for i, j in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]])
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        lens = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        max_len = 1.5 * np.median(lens)

        simplices = np.array([
            t for t in tri.simplices
            if all(np.linalg.norm(points[[t[i]]]-points[[t[j]]]) <= max_len
                for i, j in [(0,1),(1,2),(2,0)])
        ])

        # Average spot coordinates.
        if plot:
            plt.scatter(*points_ij[:2], c="r", zorder=10)

        for i in range(points_ij.shape[1]):
            neighbors = set()

            for simplex in simplices:
                if i in simplex:
                    neighbors.update(simplex)

            neighbors.discard(i)

            if plot:
                for n in neighbors:
                    plt.plot(
                        [points_ij[0, n], points_ij[0, i]],
                        [points_ij[1, n], points_ij[1, i]],
                        c="k",
                        linewidth=1,
                    )

            # Handle XY terms.
            final[x_smooth, i] = (1-smoothing_xy) * (vectors[x_smooth, i] - base_xy[0, i]) + base_xy[0, i]
            final[y_smooth, i] = (1-smoothing_xy) * (vectors[y_smooth, i] - base_xy[1, i]) + base_xy[1, i]

            for n in neighbors:
                final[x_smooth, i] += smoothing_xy * (vectors[x_smooth, n] - base_xy[0, n]) / len(neighbors)
                final[y_smooth, i] += smoothing_xy * (vectors[y_smooth, n] - base_xy[1, n]) / len(neighbors)

            # Handle higher order terms.
            final[to_smooth, i] = (1-smoothing) * vectors[to_smooth, i]

            for n in neighbors:
                final[to_smooth, i] += smoothing * vectors[to_smooth, n] / len(neighbors)

        if plot:
            plt.gca().invert_yaxis()
            plt.title("Nearest Neighbor Smoothing")

        return final

    def _wavefront_calibrate_zernike_apply(
        self,
        vector,
        from_units="norm",
    ):
        """
        Apply a Zernike-basis wavefront correction to the SLM.

        .. warning::
            Not yet implemented. Expected as part of version 0.5.0.

        Parameters
        ----------
        vector : array_like
            Zernike coefficients describing the wavefront correction.
        from_units : str
            Units of ``vector``. Defaults to ``"norm"``.
        """
        raise NotImplementedError("Expected as a part of 0.5.0")

        if from_units == "knm":
            warnings.warn(
                "'knm' requires shape information, which here defaults to the SLM shape. "
                "This may give unexpected results."
            )

        pass
