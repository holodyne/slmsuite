import matplotlib.pyplot as plt
from slmsuite._plotting import _slmsuite_plt_show
import numpy as np
from tqdm.auto import tqdm

from slmsuite.holography import analysis
from slmsuite.holography import toolbox
from slmsuite.holography.toolbox.phase import binary


class _PixelCalibration(object):
    """
    Hidden superclass with pixel calibration methods
    (gamma and crosstalk correction).
    """
    ### Pixel Crosstalk and Gamma Calibration ###

    # Phase range, in cycles, that a gamma sweep ought to resolve.
    _PIXEL_CAL_EXPECTED_CYCLES = 4

    def pixel_calibrate(
        self,
        levels=32,
        periods=2,
        orders=2,
        directions="xy",
        test_index=None,
        window=None,
        field_period=10,
        autoexpose=False,
        plot=0,
    ):
        r"""
        Measure the phase response (gamma) and pixel crosstalk (blurring) of the SLM.

        Physical SLMs do not produce perfectly sharp and discrete blocks of a desired
        phase at each pixel. Rather, the realized phase might deviate from the desired
        phase (error) and be blurred between pixels (crosstalk).

        We adopt a literature approach to calibrating both phenomena by `measuring the
        system response of binary gratings <https://doi.org/10.1364/OE.20.022334>`_
        at multiple levels (bitlevels of the SLM), periods (periods of the gratings),
        and orders (deflection orders of the response).
        In the future, we intend to fit the measured data to `an upgraded asymmetric
        model of phase crosstalk <https://doi.org/10.1364/OE.27.025046>`_, and then
        apply the model to beam propagation during holographic optimization. A better
        understanding of the system error can lead to holograms that take this error
        into account.

        Note
        ~~~~
        Most often, the user will not need to change parameters for this calibration
        when using it for gamma calibration.
        Pixel crosstalk calibration processing and mitigation is still **experimental**.

        Note
        ~~~~
        This algorithm does not operate at the level of individual pixels, but
        rather on aggregate statistics over a region of pixels.
        Right now, this calibration is done for one region (which defaults to the full
        SLM). In the future, we might want to calibrate many regions across the SLM to
        measure `spatially varying phase response <https://doi.org/10.1364/OE.21.016086>`_

        Note
        ~~~~
        A Fourier calibration must be loaded.

        Caution
        ~~~~~~~
        Data is internally acquired without wavefront calibration applied
        (``.set_phase(..., phase_correct=False)`` is used).
        If the uncalibrated SLM produces too defocussed of a spot,
        then this measurement may not be ideal. On the flip side, a
        too-focussed spot might increase error by integrating over fewer camera pixels.

        Parameters
        ----------
        levels : int OR array_like of int
            Which bitlevels to test, out of the :math:`2^B` levels available for a
            :math:`B`-bit SLM. Note that runtime scales with :math:`\mathcal{O}(L^2)`
            where :math:`L` is the number of bitlevels.
            If an integer is passed, the integer is rounded up to the next largest power of
            two, and this number of bitlevels are sampled.
            Also truncates to the bitresolution of the SLM if necessary (warns the user).
            Defaults to 32 levels.
            The response is resolved only up to half a cycle per sample interval, so too
            few levels alias the phase response.
        periods : int OR array_like of int
            List of periods (in pixels) of the binary gratings that we will apply.
            Must be even integers.
            If a single ``int`` is provided, then a list containing the given number of
            periods is chosen, based upon the field of view of the camera.
        orders : int OR array_like of int
            Orders (..., -1st, 0th, 1st, ...) of the binary gratings to measure data at.
            If scalar :math:`o` is provided,
            measures orders between :math:`-o`th and :math:`o`th order, inclusive.
            When the 0th order is off the camera, orders deflecting further off it are
            dropped if :math:`o` is scalar, and raise otherwise.
        directions : str OR array_like of int
            Directions to apply the binary grating in.
            Can be any combination of "x" and "y".
            Can also be specified with integers, where 0 corresponds to "x" and 1 corresponds to "y".
            If ``None``, defaults to "xy".
        test_index : bool OR int OR list OR None
            Project the grating for only a subset of points in the full sweep, for testing
            purposes. Return the results of the test instead of storing a calibration.
            Indices are taken modulo the length of the sweep.
            ``True`` tests every index; ``None`` (default) or ``False`` runs the full sweep.
        window
            If not ``None``, the pixel calibration is only done over the region of the SLM
            defined by ``window``.
            Passed to :meth:`~slmsuite.holography.toolbox.window_slice()`.
            See :meth:`~slmsuite.holography.toolbox.window_slice()` for various options.
        field_period : int
            If ``window`` is not ``None``, then the field is deflected away in an
            orthogonal direction with a grating of the given period.
        autoexpose : bool OR float
            If ``True``, then the camera exposure is automatically
            adjusted at the start of the sweep, or adjusted to not overexpose the given test indices.
            ``autoexpose=True`` and ``test_index=True`` is recommended to autoexpose the sweep.
        plot : int OR bool
            If ``0``, then no plots are made.
            If ``>= 1``, then the camera image is plotted at the first measurement,
            or for every test index measurement, if test indices are given.
            If ``>= 2``, then the order integration masks are additionally plotted.
        """
        # Parse levels by forcing range and datatype.
        if np.isscalar(levels):
            if levels < 1:
                levels = 1
            levels = int(2 ** (np.ceil(np.log2(levels))))

            if levels > self.slm.bitresolution:
                self.logger.warning(
                    "Requested %s levels are more than the bitresolution. Truncating to %s.",
                    levels, self.slm.bitresolution,
                )
                levels = self.slm.bitresolution

            levels = np.arange(levels) * (self.slm.bitresolution / levels)
        levels = np.asarray(levels)
        valid = (levels >= 0) & (levels < self.slm.bitresolution)
        if not np.all(valid):
            self.logger.warning(
                "Omitting requested levels %s, outside the valid range [0, %s).",
                levels[~valid], self.slm.bitresolution,
            )
            levels = levels[valid]
        levels = levels.astype(self.slm.display.dtype)
        levels = np.unique(levels)  # Processing reads the response in level order.
        N = len(levels)

        if N == 0:
            raise ValueError("No valid levels specified.")

        # The fit is unwrapped, resolving at most half a cycle between sampled levels.
        cycles = (N - 1) / 2
        if cycles < self._PIXEL_CAL_EXPECTED_CYCLES / self.slm.phase_scaling:
            self.logger.warning(
                "%s levels resolve a phase range of only %.1f cycles; an SLM with a "
                "mis-set phase table can span more. Sample more levels.", N, cycles,
            )

        # Parse directions.
        if directions is None:
            directions = "xy"
        directions_ = directions
        directions = []
        if isinstance(directions_, str):
            if "x" in directions_ or "X" in directions_:
                directions.append(0)
            if "y" in directions_ or "Y" in directions_:
                directions.append(1)
        else:
            if 0 in directions_:
                directions.append(0)
            if 1 in directions_:
                directions.append(1)
        D = len(directions)

        if D == 0:
            raise ValueError(f"No valid directions in {directions_}.")

        # How far the camera reaches either side of the 0th order along the swept axes,
        # in units where the edge of k-space is unity.
        camera_extent_kspace = 2 * self.get_camera_extent(units="freq")[directions, :]
        reach = np.array([
            np.min(np.max(camera_extent_kspace, axis=1)),
            np.min(-np.min(camera_extent_kspace, axis=1)),
        ])

        # Parse orders by forcing integer.
        orders_given = not np.isscalar(orders)
        if not orders_given:
            orders = int(orders)
            orders = np.arange(-orders, orders+1)
        orders = np.rint(orders).astype(int)

        if len(np.unique(orders)) != len(orders):
            raise ValueError(f"Repeated orders in {orders}")

        # Gratings deflect either way, so a 0th order off the camera forfeits one side.
        if np.any(reach <= 0):
            keep = orders > 0 if reach[0] > 0 else orders < 0

            if not np.all(keep):
                if orders_given:
                    raise ValueError(
                        f"The 0th order is off the camera, so orders {orders[~keep]} deflect "
                        f"further off it. Only the {'+' if reach[0] > 0 else '-'} side is visible."
                    )
                self.logger.warning(
                    "The 0th order is off the camera. Omitting orders %s.", orders[~keep]
                )
                orders = orders[keep]

        M = len(orders)

        if M == 0:
            raise ValueError("No orders land on the camera.")

        if not (1 in orders or -1 in orders):
            raise ValueError("1st order must be included.")

        # Parse periods. Long enough that the largest order stays inside the far edge of
        # the camera, short enough that the smallest reaches the near edge at all.
        period_min = np.max([2 * abs(o) / reach[0 if o > 0 else 1] for o in orders if o != 0])
        period_max = (
            2 * np.min(np.abs(orders[orders != 0])) / -np.min(reach)
            if np.any(reach < 0) else np.inf
        )

        if period_min > period_max:
            raise ValueError(
                f"No period puts orders {orders} on the camera: they need at least "
                f"{period_min:.1f} pixels to stay inside its far edge, but at most "
                f"{period_max:.1f} to clear the 0th order which has fallen off it."
            )

        # Double for margin, without overshooting back off the camera.
        min_period = 2 * int(np.ceil(min(2 * period_min, period_max) / 2))
        if min_period > period_max:
            min_period -= 2
        min_period = max(2, min_period)

        # Force positive even integer.
        if np.isscalar(periods):
            periods = min_period + 2 * np.arange(periods)

        periods = np.rint(periods).astype(int)  # Force integer.
        P = len(periods)

        # Error check periods.
        if np.any(periods % 2 != 0):
            raise ValueError(f"Periods {periods} must be even integers.")

        if np.any(periods > period_max):
            raise ValueError(
                f"Periods {periods[periods > period_max]} deflect the orders short of "
                f"the camera, which the 0th order has fallen off. "
                f"Periods must be at most {period_max:.1f}."
            )

        if len(np.unique(periods)) != len(periods):
            raise ValueError(f"Repeated periods in {periods}")

        if np.any(periods <= 0):
            raise ValueError(f"Periods {periods} must be positive.")

        # Error check window size vs periods.
        if window is not None:
            (_, w, _, h) = toolbox.window_extent(window)
            if np.any(periods > w // 2) or np.any(periods > h // 2):
                raise ValueError(f"Periods {periods} must be at most half of the window size ({w}, {h}).")

        # Figure out the shape of the stored data. We store for two directions even if only one is measured.
        shape = (2, P, N, N, M)
        length = D * P * N * N
        data = np.zeros(shape)

        # Make all of the x-pointing vectors, then all of the y-pointing vectors.
        vectors_freq = np.zeros((2, 2*P))
        vectors_freq[0, :P] = vectors_freq[1, P:] = np.reciprocal(periods.astype(float))
        vectors_kxy = toolbox.convert_vector(
            vectors_freq,
            from_units="freq",
            to_units="norm",
            hardware=self
        )

        # Make the y-pointing field vector, then the x-pointing field vector.
        field_freq = np.zeros((2, 2))
        field_freq[1, 0] = field_freq[0, 1] = 1 / float(field_period)
        field_kxy = toolbox.convert_vector(
            field_freq,
            from_units="freq",
            to_units="norm",
            hardware=self
        )
        (field_hi, field_lo) = np.array(
            [self.slm.bitresolution / 2, 0]
        ).astype(self.slm.display.dtype)

        field_ij = toolbox.convert_vector(
            field_freq,
            from_units="freq",
            to_units="ij",
            hardware=self
        )

        # Figure out where the orders will appear on the camera.
        vectors_ij = self.kxyslm_to_ijcam(vectors_kxy)
        center = self.kxyslm_to_ijcam((0,0))

        dorder = vectors_ij - center
        dfield = field_ij - center
        order_ij = []

        for i in range(2*P):
            order_ij.append(center + orders * dorder[:, [i]])

        integration_size = max(1, int(np.floor(np.min([
            np.min(np.max(np.abs(dorder), axis=0)),
            np.min(np.max(np.abs(dfield), axis=0))
        ]))))

        if integration_size < 3:
            self.logger.warning(
                "Orders are only %s camera pixels apart; the integration windows are too "
                "small to measure them reliably. Consider a longer period or focal length.",
                integration_size,
            )

        # Error check that the windows of the swept directions fit on the camera.
        half = integration_size // 2
        for i in directions:
            for j in range(P):
                if (
                    np.any(order_ij[j + P*i] - half < 0) or
                    np.any(order_ij[j + P*i] + half >= np.flip(self.cam.shape)[:, np.newaxis])
                ):
                    raise ValueError(
                        f"Some orders miss the camera. "
                        f"Try adjusting the periods or reducing the orders."
                    )

        if plot >= 2:
            canvas = np.zeros(self.cam.shape)
            for i in directions:
                for j in range(P):
                    canvas += analysis.take(
                        images=canvas,
                        vectors=order_ij[j + P*i],
                        size=integration_size,
                        return_mask=True,
                    )
            self.cam.plot(canvas, title="Order integration mask")
            _slmsuite_plt_show(name="pixel_calibrate_masks")

        # ``True`` scans the whole sweep; ``False`` is not a test at all.
        if np.ndim(test_index) == 0 and np.asarray(test_index).dtype == bool:
            test_index = np.arange(length) if test_index else None

        show_tqdm = test_index is None

        if test_index is not None:
            # Force a sorted list of unique in-range integers.
            test_index = np.rint(np.atleast_1d(test_index)).astype(int)
            test_index = sorted(set(np.mod(test_index, length)))

            if len(test_index) == 0:
                raise ValueError("test_index selected no points of the sweep.")

            # Otherwise a full-sweep test would plot thousands of figures.
            if len(test_index) > 8: plot = 0

            results = []
            autoexposure_results = []

        if show_tqdm: iterations = tqdm(range(length))

        # Big sweep.
        index = 0
        for i in directions:                                    # Direction (x,y)
            prange = np.arange(P) + i*P
            for j in range(P):                                  # Period

                for k in range(N):                              # Gray level selection.
                    for l in range(N):
                        # If we're testing, then only execute the test indices.
                        # (Ignore everything else.)
                        if test_index is not None and index not in test_index:
                            index += 1
                            continue

                        current_index = index

                        # (1a) Make the pattern that we are going to project.
                        if window is None:
                            phase = binary(
                                self.slm,
                                vector=vectors_kxy[:, prange[j]],
                                a=levels[k],
                                b=levels[l]
                            )
                        else:
                            # In windowed mode, blaze the field away from the 0th order,
                            # in the direction perpendicular to the target.
                            phase = binary(
                                grid=self.slm,
                                vector=field_kxy[:, i],
                                a=field_hi,
                                b=field_lo
                            )
                            toolbox.imprint(
                                phase,
                                window=window,
                                function=binary,
                                grid=self.slm,
                                vector=vectors_kxy[:, prange[j]],
                                a=levels[k],
                                b=levels[l]
                            )

                        # (1b) We're writing integers, so this goes directly to the SLM,
                        # bypassing phase2gray.
                        self.slm.set_phase(phase, phase_correct=False, settle=True)

                        # (2a) If we need to autoexpose, then do it now that the pattern is on the SLM.
                        if autoexpose:
                            mask = analysis.take(
                                images=np.zeros(self.cam.shape),
                                vectors=order_ij[prange[j]],
                                size=integration_size,
                                return_mask=True,
                            )
                            self.cam.autoexpose(
                                set_fraction=autoexpose,
                                window=mask,
                                verbose=True,
                            )

                            if test_index is None:
                                # Only autoexpose for the first test index, if autoexpose is enabled.
                                autoexpose = False
                            else:
                                autoexposure_results.append(self.cam.get_exposure())

                        # (2b) Integrate over the order regions to get the data for this point.
                        regions = analysis.take(
                            images=self.cam.get_image(),
                            vectors=order_ij[prange[j]],
                            size=integration_size,
                            integrate=False,
                        ).astype(float)

                        data[i,j,k,l,:] = np.sum(regions, axis=(1,2))

                        # (3a) Update the current index of the sweep, and maybe update the progress bar.
                        if show_tqdm: iterations.update()
                        index += 1

                        # (3b) Maybe plot the results for this point.
                        if plot >= 1:
                            self.cam.plot(
                                title=(
                                    f"Pixel Calibrate index {index} "
                                    f"at direction {('x', 'y')[i]}, period {periods[j]}, "
                                    f"levels {levels[k]}, {levels[l]}"
                                )
                            )
                            _slmsuite_plt_show(name="pixel_calibrate_result")

                            # Turn plotting off after the first test index.
                            if test_index is None:
                                plot = 0

                        # (3c) Handle test index results collection and maybe autoexpose adjustment.
                        if test_index is not None:
                            results.append(data[i,j,k,l,:].copy())

                            if current_index == test_index[-1]:
                                if autoexpose:
                                    exposure = np.min(autoexposure_results)
                                    self.cam.set_exposure(exposure)

                                    return {
                                        "indices" : test_index,
                                        "results" : autoexposure_results,
                                    }
                                else:
                                    return {
                                        "indices" : test_index,
                                        "results" : results,
                                    }

        if show_tqdm: iterations.close()

        # Assemble the return dictionary.
        self.calibrations["pixel"] = {
            "levels" : levels,
            "periods" : periods,
            "orders" : orders,
            "directions" : directions,
            "vectors_kxy" : vectors_kxy,
            "order_ij" : order_ij,
            "data": data
        }
        self.calibrations["pixel"].update(self._get_calibration_metadata())

        return self.calibrations["pixel"]

    def _pixel_calibration_process_get_summed(self, orders=None, transpose=False):
        """
        Helper function to get summed data for gamma calibration.

        Parameters
        ----------
        orders : int OR array_like of int
            Which orders to sum over. If an integer is passed, then defaults to the corresponding positive and negative order.
        transpose : bool
            If ``True``, then the summed data is averaged with its transpose across the diagonal. This can help reduce noise by enforcing symmetry.
        """
        cal = self.calibrations["pixel"]
        orders_ = cal["orders"]
        data = cal["data"]

        # Parse orders.
        if orders is None:
            orders = [-1, 1]
        if np.isscalar(orders):
            orders = int(orders)
            orders = [-orders, orders]

        orders = [o for o in orders_ if (o in orders)]
        mask = [(o in orders) for o in orders_]

        # Actually do the sum.
        data_summed = np.sum(data[:,:,:,:,mask], axis=(0,1,4))

        # Average across diagonals to reduce noise.
        if transpose:
            data_summed = (data_summed + np.transpose(data_summed)) / 2

        return data_summed

    def pixel_calibration_plot(self, summed=False, orders=None):
        """
        Plot the pixel calibration data as a series of square heatmaps.
        The :math:`x` and :math:`y` axes of the heatmaps correspond to
        the levels :math:`a` and :math:`b` of the binary grating.

        Parameters
        ----------
        summed : bool
            If ``False``, then the raw data for each order and period is plotted.
            If ``True``, then the data is summed across orders and periods.
            This is the data that is used for gamma calibration.
        orders : int OR array_like of int
            If ``summed=False``, then which orders to plot.
            If ``summed=True``, then which orders to sum over.
            If an integer is passed, then defaults to the corresponding positive and negative order.
        """
        cal = self.calibrations["pixel"]
        periods = cal["periods"]
        orders_ = cal["orders"]
        levels = cal["levels"]
        leveli = np.arange(len(levels))
        data = cal["data"]

        # Parse orders.
        if orders is None:
            if summed:
                orders = [-1, 1]
            else:
                orders = orders_
        if np.isscalar(orders):
            orders = int(orders)
            orders = [-orders, orders]
        orders =  [o for o in orders_ if o in orders]

        if len(orders) == 0:
            raise ValueError(f"None of the requested orders were measured; have {list(orders_)}.")

        if not summed:
            # The requested orders index into the measured orders, not into 0, 1, 2, ...
            index = [list(orders_).index(o) for o in orders]
            M = len(orders)
            cmin = np.min(data)
            cmax = np.max(data)

            for i in cal["directions"]:
                for j, period in enumerate(periods):
                    fig, axs = plt.subplots(1, M, figsize=(5*M, 5), squeeze=False)
                    for o, order in enumerate(orders):
                        im = axs[0, o].imshow(data[i,j,:,:,index[o]])
                        im.set_clim(cmin, cmax)

                        axs[0, o].set_xlabel("Level b")
                        axs[0, o].set_ylabel("Level a")
                        axs[0, o].set_xticks(leveli)
                        axs[0, o].set_yticks(leveli)

                        axs[0, o].set_title(f"Order ${order:+d}$")

                    fig.suptitle(f"${'xy'[i]}$-grating, {period} pixel period")
                    _slmsuite_plt_show(name="pixel_calibration_plot")
        else:
            data_summed = self._pixel_calibration_process_get_summed(orders=orders)
            cmin = np.min(data_summed)
            cmax = np.max(data_summed)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            im = ax.imshow(data_summed)
            im.set_clim(cmin, cmax)

            ax.set_xlabel("Level b")
            ax.set_ylabel("Level a")
            ax.set_xticks(leveli)
            ax.set_yticks(leveli)

            ax.set_title(f"Summed Orders")

            _slmsuite_plt_show(name="pixel_calibration_plot")

    def pixel_calibration_process(self, plot=0, apply=True):
        r"""
        Process the pixel calibration data to extract the phase response curve (gamma).

        Parameters
        ----------
        plot : int OR bool
            If ``>= 1``, then the extracted gamma curve is plotted, as well as the fit of the
            model to the data.
        apply : bool
            If ``True``, load the result onto the SLM with
            :meth:`~slmsuite.hardware.slms.slm.SLM.set_gamma`.

        Returns
        -------
        numpy.ndarray
            Phase response at the sampled ``levels``, in units of :math:`2\pi`,
            on the increasing branch with its minimum at zero.
        """
        # Construct x data.
        cal = self.calibrations["pixel"]
        levels = cal["levels"].astype(int)
        leveli = np.arange(len(levels))
        xy = [l.ravel().astype(int) for l in np.meshgrid(leveli, leveli, indexing="ij")]

        # Construct y data.
        data_summed = self._pixel_calibration_process_get_summed(orders=[-1, +1])
        data_ravel = data_summed.ravel()

        if np.ptp(data_ravel) == 0:
            raise RuntimeError(
                "The 1st orders carry no signal, so gamma cannot be fit. "
                "Check the exposure and that the orders land on the camera."
            )

        # Construct the model.
        def model(_, a, c, *gamma):
            gamma = np.array(gamma)
            dphase = gamma[xy[0].astype(int)] - gamma[xy[1].astype(int)]
            intensity = np.sin(np.pi * dphase) ** 2
            return a * intensity + c

        # Make a guess for the model.
        c_guess = np.min(data_summed)
        a_guess = np.max(data_summed) - c_guess
        gamma_guess = levels / self.slm.bitresolution
        guess = [a_guess, c_guess, *gamma_guess]

        # Now run the fit.
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(model, None, data_ravel, p0=guess)

        # The model resolves each level only modulo a cycle, and mirrors freely.
        gamma = np.unwrap(popt[2:], period=1)
        if np.corrcoef(gamma, leveli)[0, 1] < 0:
            gamma = -gamma
        gamma -= np.min(gamma)

        # Get rsquared of the fit.
        residuals = data_ravel - model(None, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((data_ravel - np.mean(data_ravel))**2)
        r_squared = 1 - (ss_res / ss_tot)

        if not r_squared >= 0.9:
            self.logger.warning("Low R^2 value of %.3f for gamma fit. Fit may be inaccurate.", r_squared)

        if plot >= 2:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            data_fit = model(None, *popt).reshape(data_summed.shape)
            data_resid = data_summed - data_fit
            M = np.max(np.abs(data_resid))
            axs[0].imshow(data_summed)
            axs[1].imshow(data_fit)
            axs[2].imshow(data_resid, cmap="bwr", vmin=-M, vmax=M)

            for ax, title in zip(axs, ["Data", "Fit", "Residuals"]):
                ax.set_title(title)
                if title == "Data":
                    ax.set_ylabel("SLM Level $a$")
                ax.set_xlabel("SLM Level $b$")

            fig.suptitle("Diffraction Order Intensity versus Binary Grating Levels ($a$, $b$)")
            plt.tight_layout()

            _slmsuite_plt_show(name="pixel_calibration_process_residuals")
        if plot >= 1:
            fig, ax = plt.subplots(1, 1)
            ax.plot(levels, gamma, "o-", label="calibrated")
            ax.set_title(f"Pixel Calibration Gamma (R^2: {r_squared:.3f})")
            ax.set_xlabel("SLM Level $i$")
            ax.set_ylabel("SLM Response")
            tick_labels = [0, .5, 1]
            ax.set_yticks(tick_labels)
            ax.set_yticklabels([rf"${int(y*2)}\pi$" for y in tick_labels])
            _slmsuite_plt_show(name="pixel_calibration_process_fit")


        self.calibrations["pixel"]["gamma"] = gamma
        self.calibrations["pixel"]["gamma_r2"] = r_squared

        if apply:
            self._pixel_calibration_apply_gamma()

        return gamma

    def _pixel_calibration_apply_gamma(self):
        """
        Loads the fitted phase response onto the SLM, spreading the sampled levels across
        all of them. Does nothing if the calibration has not been processed.
        """
        cal = self.calibrations["pixel"]
        if "gamma" not in cal or "levels" not in cal:
            return

        # Levels are meaningless across a change of bitdepth.
        bitresolution = self.slm.bitresolution
        measured = cal.get("__meta__", {}).get("slm", {}).get("bitresolution", bitresolution)
        if measured != bitresolution:
            self.logger.warning(
                "The calibration was taken on a %s level SLM, but this one has %s. "
                "Not applying its gamma.",
                measured, bitresolution,
            )
            return

        self.slm.set_gamma(
            self.slm.interpolate_gamma(cal["gamma"], cal["levels"])
        )

    @staticmethod
    def pixel_kernel(x, a_pix=.1, n=1, a_minus_pix=None, n_minus=None, x0_pix=0):
        r"""
        Normalized crosstalk kernel, evaluated at positions ``x`` in units of SLM pixels.
        This is Eq. (9) of `Moser et al. <https://doi.org/10.1364/OE.27.025046>`_,
        generalized to a per-side exponent.

        .. math:: K(x) =    \left\{
                                \begin{array}{ll}
                                    \exp\left(-\left|\frac{x-x_0}{\alpha_+}\right|^{n_+}\right), & x \ge x_0, \\
                                    \exp\left(-\left|\frac{x-x_0}{\alpha_-}\right|^{n_-}\right), & x < x_0.
                                \end{array}
                            \right.

        Parameters
        ----------
        x : array_like
            Positions in pixels, on a uniform grid: the kernel is normalized by its sum.
        a_pix, n : float
            Width :math:`\alpha_+` and exponent :math:`n_+` for :math:`x \ge x_0`.
            ``n=1`` gives an exponential, ``n=2`` a Gaussian.
        a_minus_pix, n_minus : float OR None
            Width :math:`\alpha_-` and exponent :math:`n_-` for :math:`x < x_0`.
            Each defaults to its counterpart, making the kernel symmetric.
        x0_pix : float
            Displacement :math:`x_0` of the peak, in pixels.
        """
        # Parse minus parameters by defaulting to plus parameters.
        if a_minus_pix is None:
            a_minus_pix = a_pix
        if n_minus is None:
            n_minus = n

        x = x - x0_pix

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            kernel = np.where(
                x >= 0,
                np.exp(-np.power(np.abs(x / a_pix), n)),
                np.exp(-np.power(np.abs(x / a_minus_pix), n_minus)),
            )
            total = np.sum(kernel)

        # A kernel narrower than the sampling underflows; that limit is a delta.
        if not total > 0:
            kernel = np.asarray(np.abs(x) == np.min(np.abs(x)), dtype=float)
            total = np.sum(kernel)

        return kernel / total

    # Kernel support in pixels; crosstalk is sub-pixel, so truncating here is negligible.
    _pixel_kernel_reach = 32

    @classmethod
    def _pixel_crosstalk_simulate(cls, phase, supersample=16, plot=0, **kwargs):
        r"""
        Diffraction orders of one period of a pixelated phase pattern subject to pixel
        crosstalk, after `Moser et al. <https://doi.org/10.1364/OE.27.025046>`_.

        Each pixel edge :math:`\phi_0 \rightarrow \phi_1` realizes the transition
        :math:`\phi_0 + (\phi_1 - \phi_0)K(x)` for the integrated :meth:`pixel_kernel`
        :math:`K` (Eq. 10), summed over the periodic pattern (Eq. 12). Constant
        parameters reduce this to convolving the supersampled phase with the kernel.

        Caution
        ~~~~~~~
        A binary grating of 50% duty cycle obeys :math:`\phi(x + p/2) = a + b - \phi(x)`,
        which any *constant* kernel preserves, forcing :math:`|E_m| = |E_{-m}|`. The
        order asymmetry Moser et al. measure on exactly these gratings therefore needs a
        level-dependent ``a_pix``, not merely an asymmetric kernel. A duty cycle other
        than 50% exposes a constant kernel's asymmetry instead.

        Parameters
        ----------
        phase : array_like of float
            One period of the commanded phase in radians, one value per SLM pixel.
        supersample : int
            Subpixel samples per SLM pixel.
        plot : int OR bool
            Whether to plot the blurred phase and the resulting orders.
        **kwargs
            Passed to :meth:`pixel_kernel`. Each may be a scalar or a
            ``callable(phi0, phi1)`` returning the value for that edge. A constant
            ``x0_pix`` merely translates the pattern, so it acts only when level-dependent.

        Returns
        -------
        numpy.ndarray
            Power in each of the ``phase.size * supersample`` diffraction orders, with
            order :math:`m` at index ``m`` (:mod:`numpy` FFT ordering). Sums to unity.
        """
        phase = np.ravel(np.asarray(phase, dtype=float))
        supersample = int(supersample)
        (P, N) = (phase.size, phase.size * supersample)

        x = np.arange(N) / supersample
        reach = max(cls._pixel_kernel_reach, P)
        offset = (np.arange(reach * supersample) - reach * supersample // 2) / supersample
        replicas = P * np.arange(-(reach // P) - 1, reach // P + 2)

        # Sum the transition of every pixel edge, and of its periodic neighbors.
        blurred = np.zeros(N)
        for j in np.flatnonzero(np.roll(phase, -1) != phase):
            (phi0, phi1) = (phase[j], phase[(j + 1) % P])
            transition = np.cumsum(cls.pixel_kernel(
                offset,
                **{k: v(phi0, phi1) if callable(v) else v for (k, v) in kwargs.items()},
            ))
            for replica in replicas:
                blurred += (phi1 - phi0) * np.interp(
                    x - (j + 1 + replica), offset, transition, left=0, right=1
                )

        orders = np.square(np.abs(np.fft.fft(np.exp(1j * blurred)) / N))

        if plot >= 1:
            (fig, axs) = plt.subplots(1, 2, figsize=(10, 4))

            # A normalized kernel preserves the mean; anchor there to compare levels.
            axs[0].plot(x, np.repeat(phase, supersample), label="commanded")
            axs[0].plot(x, blurred + np.mean(phase) - np.mean(blurred), label="blurred")
            axs[0].set_xlabel("SLM pixel")
            axs[0].set_ylabel("Phase [rad]")
            axs[0].legend(fontsize="x-small")

            m = np.arange(-((P - 1) // 2), P // 2 + 1)
            axs[1].stem(m, orders[m])
            axs[1].set_xlabel("Diffraction order")
            axs[1].set_ylabel("Power")

            fig.tight_layout()
            _slmsuite_plt_show(name="pixel_crosstalk_simulate")

        return orders
