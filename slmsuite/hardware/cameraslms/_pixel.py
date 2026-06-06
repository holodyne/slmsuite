import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from tqdm.auto import tqdm
import warnings

from slmsuite.holography import analysis
from slmsuite.holography import toolbox
from slmsuite.holography.toolbox.phase import binary

class _PixelCalibration(object):
    """
    Hidden superclass with pixel calibration methods
    (gamma and crosstalk correction).
    """
    ### Pixel Crosstalk and Gamma Calibration ###

    def pixel_calibrate(
        self,
        levels=32,
        periods=2,
        orders=2,
        directions="xy",
        window=None,
        field_period=10,
        test_index=None,
        autoexpose=False,
        plot=True,
    ):
        r"""
        Measure the pixel crosstalk (blurring) and phase response (gamma) of the SLM.

        Physical SLMs do not produce perfectly sharp and discrete blocks of a desired
        phase at each pixel. Rather, the realized phase might deviate from the desired
        phase (error) and be blurred between pixels (crosstalk).

        We adopt a literature approach to calibrating both phenomena by `measuring the
        system response of binary gratings <https://doi.org/10.1364/OE.20.022334>`_.
        In the future, we intend to fit the measured data to `an upgraded asymmetric
        model of phase crosstalk <https://doi.org/10.1364/OE.27.025046>`_, and then
        apply the model to beam propagation during holographic optimization. A better
        understanding of the system error can lead to holograms that take this error
        into account.

        Note that this algorithm does not operate at the level of individual pixels, but
        rather on aggregate statistics over a region of pixels.
        Right now, this calibration is done for one region (which defaults to the full
        SLM). In the future, we might want to calibrate many regions across the SLM to
        measure `spatially varying phase response <https://doi.org/10.1364/OE.21.016086>`_

        Note
        ~~~~
        A Fourier calibration must be loaded.

        Caution
        ~~~~~~~
        Data must be acquired without wavefront calibration applied.
        If the uncalibrated SLM produces too defocussed of a spot,
        then this measurement may not be ideal. On the flip side, a
        too-focussed spot might increase error by integrating over fewer camera pixels.

        Parameters
        ----------
        levels : int OR array_like of int
            Which bitlevels to test, out of the :math:`2^B` levels available for a
            :math:`B`-bit SLM. Note that runtime scales with :math:`\mathcal{O}(B^2)`.
            If an integer is passed, the integer is rounded up to the next largest power of
            two, and this number of bitlevels are sampled.
            Also truncates to the bitresolution of the SLM if necessary.
            Defaults to 32 levels.
        periods : int OR array_like of int
            List of periods (in pixels) of the binary gratings that we will apply.
            Must be even integers.
            If a single ``int`` is provided, then a list containing the given number of
            periods is chosen, based upon the field of view of the camera.
        orders : int OR array_like of int
            Orders (..., -1st, 0th, 1st, ...) of the binary gratings to measure data at.
            If scalar :math:`o` is provided,
            measures orders between :math:`-o`th and :math:`o`th order, inclusive.
        directions : str OR array_like of int
            Directions to apply the binary grating in.
            Can be any combination of "x" and "y".
            Can also be specified with integers, where 0 corresponds to "x" and 1 corresponds to "y".
            If ``None``, defaults to "xy".
        window
            If not ``None``, the pixel calibration is only done over the region of the SLM
            defined by ``window``.
            Passed to :meth:`~slmsuite.holography.toolbox.window_slice()`.
            See :meth:`~slmsuite.holography.toolbox.window_slice()` for various options.
        field_period : int
            If ``window`` is not ``None``, then the field is deflected away in an
            orthogonal direction with a grating of the given period.
        test_index : bool OR int OR list OR None
            Project the grating for only a subset of points in the full sweep, for testing purposes.
            This test index or test indices is executed modulo the length of the sweep.
            If ``True``, then the test indices are set to scan over the full sweep.
            Defaults to ``None``, which means that no test index is used and the full sweep is executed.
            ``False`` is treated the same as ``None``.
        autoexpose : bool OR float
            If ``True``, then the camera exposure is automatically
            adjusted at the start of the sweep, or adjusted to not overexpose the given test indices.
            ``autoexpose=True`` and ``test_index=True`` is recommended to autoexpose the sweep.
        plot : bool OR int
            If ``False``, then no plots are made.
            If ``True`` or greater than or equal to ``1``, then the camera image is plotted at the first measurement,
            or for every test index measurement, if test indices are given.
            If greater than or equal to ``2``, then the order integration masks are additionally plotted.
        """
        # Parse levels by forcing range and datatype.
        if np.isscalar(levels):
            if levels < 1:
                levels = 1
            levels = int(2 ** (np.ceil(np.log2(levels))))

            if levels > self.slm.bitresolution:
                warnings.warn(
                    f"Requested {levels} levels are more than the "
                    f"bitresolution. Truncating to {self.slm.bitresolution}."
                )
                levels = self.slm.bitresolution

            levels = np.arange(levels) * (self.slm.bitresolution / levels)
        levels = np.mod(levels, self.slm.bitresolution).astype(self.slm.display.dtype)
        N = len(levels)

        # Parse orders by forcing integer.
        if np.isscalar(orders):
            orders = int(orders)
            orders = np.arange(-orders, orders+1)
        orders = np.rint(orders).astype(int)
        M = len(orders)

        if not 1 in orders:
            raise ValueError("1st order must be included.")

        # Find the minimum period based on the FoV size.
        camera_extent_knm = toolbox.convert_vector(
            np.array([
                (0,0),    # top left
                (self.cam.width - 1, 0),    # top right
                (0, self.cam.height - 1),    # bottom left
                (self.cam.width - 1, self.cam.height - 1),    # bottom right
            ]).T,
            from_units="ij",
            to_units="knm",
            shape=(1,1),
            hardware=self,
        )
        camera_extent_kspace = (camera_extent_knm - toolbox.format_2vectors((.5, .5))) * 2
        max_order = np.max(np.abs(orders))
        min_periods = max_order / np.max(np.abs(camera_extent_kspace), axis=1)
        min_period = max(2, 8 * int(np.floor(np.max(min_periods)/2)))

        # Parse periods by forcing positive even integer.
        if np.isscalar(periods):
            periods = min_period + 2 * np.arange(periods)

        periods = np.rint(periods).astype(int)  # Force integer.
        P = len(periods)

        # Error check periods.
        if np.any(periods % 2 != 0):
            raise ValueError(f"Periods {periods} must be even integers.")

        if len(np.unique(periods)) != len(periods):
            raise ValueError(f"Repeated periods in {periods}")

        if np.any(periods <= 0):
            raise ValueError(f"Periods {periods} must be positive.")

        # Parse directions.
        if directions is None:
            directions = "xy"
        directions_ = directions
        directions = []
        if "x" in directions_ or "X" in directions_ or 0 in directions_:
            directions.append(0)
        if "y" in directions_ or "Y" in directions_ or 1 in directions_:
            directions.append(1)
        D = len(directions)

        # Error check window size vs periods.
        if window is not None:
            (_, w, _, h) = toolbox.window_extent(window)
            if np.any(periods > w // 2) or np.any(periods > h // 2):
                raise ValueError(f"Periods {periods} must be at least half of the window size ({w}, {h}).")

        # Figure out our shape. We store for two directions even if only one is measured.
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
        field_freq[0, 0] = field_freq[1, 1] = 1 / float(field_period)
        field_kxy = toolbox.convert_vector(
            field_freq,
            from_units="freq",
            to_units="norm",
            hardware=self
        )
        field_values = np.array([self.slm.bitresolution / 2, 0]).astype(self.slm.display.dtype)
        field_hi, field_lo = field_values

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

        integration_size = int(np.ceil(np.min([
            np.min(np.max(np.abs(dorder), axis=0)),
            np.min(np.max(np.abs(dfield), axis=0))
        ])))

        canvas = np.zeros(self.cam.shape)

        # Figure out if we exceed the camera's field of view, and if so, raise an error.
        try:
            for i in range(2*P):
                mask = analysis.take(
                    images=canvas,
                    vectors=order_ij[i],
                    size=integration_size,
                )
        except IndexError:
            raise ValueError(
                f"Some orders are too far from the center to be captured by the camera. "
                f"Try increasing the period or decreasing the order."
            )

        if plot >= 2:
            for i in directions:
                for j in range(P):
                    mask = analysis.take(
                        images=canvas,
                        vectors=order_ij[j + P*i],
                        size=integration_size,
                        return_mask=True,
                    )
                    canvas += mask
            self.cam.plot(canvas, title="Order integration mask")
            plt.show()

        show_tqdm = True

        if test_index is not None:
            # Make test_index a list if we only got a single integer.
            if np.isscalar(test_index):
                test_index = [test_index]

            # Make sure test_index is a sorted list of unique integers between 0 and length-1.
            test_index = np.rint(test_index).astype(int)
            test_index = np.mod(test_index, length)
            test_index = sorted(list(set(test_index)))

            show_tqdm = False
            results = []
            autoexposure_results = []

        # if True: iterations = tqdm(range(P*(N-1)*N))
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
                        if test_index is not None and index in test_index:
                            index += 1
                            continue

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
                            self.cam.auto_expose(
                                set_fraction=autoexpose,
                                window=mask,
                                verbose=True,
                            )

                            if test_index is None:
                                # Only autoexpose for the first test index, if autoexposure is enabled.
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
                        if plot:
                            self.cam.plot(
                                title=(
                                    f"Pixel Calibrate index {index} "
                                    f"at direction {["x", "y"][i]}, period {periods[j]}, "
                                    f"levels {levels[k]}, {levels[l]}"
                                )
                            )
                            plt.show()

                            # Turn plotting off after the first test index.
                            if test_index is None:
                                plot = False

                        # (3c) Handle test index results collection and maybe autoexposure adjustment.
                        if test_index is not None:
                            results.append(data[i,j,k,l,:].copy())

                            if index == test_index[-1]:
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
        periods = cal["periods"]
        orders_ = cal["orders"]
        levels = cal["levels"]
        leveli = np.arange(len(levels))
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

        if not summed:
            cmin = np.min(data)
            cmax = np.max(data)
            for i, direction in enumerate(["x", "y"]):
                for j, period in enumerate(periods[[0]]):
                    M = len(orders)
                    fig, axs = plt.subplots(1, M, figsize=(5*M, 5))
                    for o, order in enumerate(orders):
                        im = axs[o].imshow(data[i,j,:,:,o])
                        im.set_clim(cmin, cmax)

                        axs[o].set_xlabel("Level b")
                        axs[o].set_ylabel("Level a")
                        axs[o].set_xticks(leveli)
                        axs[o].set_yticks(leveli)

                        axs[o].set_title(f"Order ${order:+d}$")

                    fig.suptitle(f"${direction}$-grating, {period} pixel period")
                    plt.show()
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

            plt.show()

    def pixel_calibration_process_gamma(self, plot=False):
        """
        Process the pixel calibration data to extract the phase response curve (gamma).

        Parameters
        ----------
        plot : bool
            If ``True``, then the extracted gamma curve is plotted, as well as the fit of the
            model to the data.
        """
        # Construct x data.
        cal = self.calibrations["pixel"]
        levels = cal["levels"].astype(int)
        leveli = np.arange(len(levels))
        xy = [l.ravel().astype(int) for l in np.meshgrid(leveli, leveli, indexing="ij")]

        # Construct y data.
        data_summed = self._pixel_calibration_process_get_summed(orders=[-1, +1])
        data_ravel = data_summed.ravel()

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

        # Extract the gamma values from the fit parameters, and normalize so that the minimum is 0.
        gamma = popt[2:]
        gamma -= np.min(gamma)

        # Get rsquared of the fit.
        residuals = data_ravel - model(None, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((data_ravel - np.mean(data_ravel))**2)
        r_squared = 1 - (ss_res / ss_tot)

        if r_squared < 0.9:
            warnings.warn(f"Low R^2 value of {r_squared:.3f} for gamma fit. Fit may be inaccurate.")

        if plot:
            plt.plot(levels, gamma, "o-", label="calibrated")
            plt.title(f"Gamma fit R^2: {r_squared:.3f}")
            plt.show()

            fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            data_fit = model(None, *popt).reshape(data_summed.shape)
            data_resid = data_summed - data_fit
            M = np.max(np.abs(data_resid))
            axs[0].imshow(data_summed)
            axs[1].imshow(data_fit)
            axs[2].imshow(data_resid, cmap="bwr", vmin=-M, vmax=M)
            plt.show()

        # TODO: apply data to SLM.

        self.calibrations["pixel"]["gamma"] = gamma

        return gamma

    @staticmethod
    def pixel_kernel(x, a_pix=.1, n=1, a_minus_pix=None, n_minus=None):
        r"""
        Blurring kernel

        .. math:: K(x) =    \left\{
                                \begin{array}{ll}
                                    \exp\left(-\left|\frac{x}{\alpha_+}\right|^{n_+}\right), & x \ge 0, \\
                                    \exp\left(-\left|\frac{x}{\alpha_-}\right|^{n_-}\right), & x < 0.
                                \end{array}
                            \right.
        """
        # Parse minus parameters by defaulting to plus parameters.
        if a_minus_pix is None:
            a_minus_pix = a_pix
        if n_minus is None:
            n_minus = n

        # Create and normalize the kernel.
        kernel = np.where(
            x >= 0,
            np.exp(-np.power(np.abs(x / a_pix), n)),
            np.exp(-np.power(np.abs(x / a_minus_pix), n_minus)),
        )
        kernel /= np.sum(kernel)

        return kernel

    def _pixel_calibrate_simulate(self, period=16, supersample=16, **kwargs):
        N = int(period * supersample)

        x = np.linspace(-period, period, N)
        x -= np.mean(x)

        y = np.zeros_like(x)
        y[x < 0] = 1

        blaze = np.linspace(0, 2, N)
        # blaze -= np.mean(blaze)

        plt.plot(x, y)
        plt.plot(x, blaze)

        x2 = np.linspace(-2, 2, 4*supersample)
        x2 -= np.mean(x2)
        K = self.pixel_kernel(x2, **kwargs)

        y = ndimage.convolve1d(y, K, mode="wrap")

        plt.plot(x, y)
        plt.show()

        kx = np.arange(float(N)) #/ supersample
        kx -= np.mean(kx)

        Y = np.fft.fftshift(np.fft.fft(np.exp(1j * np.pi * y)))

        Y2 = np.fft.fftshift(np.fft.fft(np.exp(1j * np.pi * blaze)))

        plt.hlines(0, np.min(kx), np.max(kx))
        plt.scatter(kx, np.square(np.abs(Y)))
        plt.scatter(kx, np.square(np.abs(Y2)))
        # plt.xlim(-10, 10)
        plt.show()
