import time
import matplotlib.pyplot as plt
from slmsuite._plotting import _slmsuite_plt_show
import numpy as np
from tqdm.auto import tqdm
from scipy import optimize

from slmsuite.holography import analysis
from slmsuite.holography import toolbox

class _SettleCalibration(object):
    """
    Hidden superclass with settle calibration methods
    (time for the measurement to stabilize).
    """
    ### Settle Time Calibration ###

    def settle_calibrate(
        self, vector=(.005, .005), size=None, times=None, settle_time_s=1
    ):
        """
        Approximates the :math:`1/e` settle time of the SLM.
        This is done by successively removing and applying a blaze to the SLM,
        measuring the intensity at the first order spot versus time delay.

        **(This feature is experimental.)**

        Parameters
        ----------
        vector : array_like
            Point to measure settle time at via a simple blaze in the ``"kxy"`` basis.
        size : int
            Size in pixels of the integration region in the ``"ij"`` basis.
            If ``None``, sets to sixteen times the approximate size of a diffraction-limited spot.
        times : array_like OR None OR int
            List of times to sweep over in search of the :math:`1/e` settle time.
            If ``None``, defaults to 21 points over one second.
            If an integer, defaults to that given number of points over one second.
        settle_time_s : float OR None
            Time between measurements to allow the SLM to re-settle. If ``None``, uses the
            current default in the SLM.
        """
        # Parse vector.
        point = self.kxyslm_to_ijcam(vector)
        blaze = toolbox.phase.blaze(grid=self.slm, vector=vector)

        # Parse size.
        if size is None:
            size = 16 * toolbox.convert_radius(
                self.slm.get_spot_radius_kxy(),
                to_units="ij",
                hardware=self
            )
        size = int(size)

        # Parse times.
        if times is None:
            times = 21
        if np.isscalar(times):
            times = np.linspace(0, 1, int(times), endpoint=True)
        times = np.ravel(times)

        # Parse settle_time_s.
        if settle_time_s is None:
            settle_time_s = self.slm.settle_time_s
        settle_time_s = float(settle_time_s)

        results = []

        verbose = True
        iterations = times
        if verbose:
            iterations = tqdm(times)

        # Collect data
        for t in iterations:
            self.cam.flush()

            # Reset the pattern and wait for it to settle
            self.slm.set_phase(None, settle=False, phase_correct=False)
            time.sleep(settle_time_s)

            # Turn on the pattern and wait for time t
            self.slm.set_phase(blaze, settle=False, phase_correct=False)
            time.sleep(t)

            image = self.cam.get_image()
            results.append(analysis.take(image, point, size, centered=True, integrate=True))

        self.calibrations["settle"] = {
            "times" : times,
            "data" : np.array(results)
        }
        self.calibrations["settle"].update(self._get_calibration_metadata())

        self.settle_calibration_process(plot=False)

        return self.calibrations["settle"]

    def settle_calibration_process(self, plot=True):
        """
        Fits an exponential to the measured data to
        approximate the :math:`1/e` settle time of the SLM.

        Parameters
        ----------
        plot : bool
            Whether to show a debug plot with the exponential fit.

        Returns
        -------
        dict
            The settle time and communication time measured.
        """
        times = self.calibrations["settle"]["times"]
        results = self.calibrations["settle"]["data"]

        # Function to interpolate
        def exponential_jump(x, x0, a, b, c):
            return (c - a*np.exp(-(x-x0) / b)) * np.heaviside(x - x0, 0)

        guess = (np.max(times)/2, np.max(results), np.max(times), np.max(results))

        # Fit the date with the function
        params, _ = optimize.curve_fit(
            exponential_jump,
            times,
            results,
            p0=guess,
            maxfev=10000
        )
        x0, a, b, c = params
        print(params)

        relax_time = b
        com_time = x0
        settle_time = com_time + relax_time*4

        # Evaluate the fitting function in the interval
        x_interp = np.linspace(min(times), max(times), 100)
        g_interp = exponential_jump(x_interp, *guess)
        y_interp = exponential_jump(x_interp, *params)

        if plot:
            title = (
                f"Communication time: {int((1e3*com_time))} ms\n"
                f"$1/e$ Relaxation time: {int((1e3*relax_time))} ms\n"
                f"Suggested $1/e^4$ Settle time: {int((1e3*settle_time))} ms"
            )
            # plt.plot(x_interp, g_interp, "--", linewidth=1, color='g', alpha=.5, label='interpolation')
            plt.plot(x_interp, y_interp, "--", linewidth=2, color='red', label='interpolation')
            plt.plot(times, results, "k.", markersize=7, label='capta')
            plt.xlabel("Time [sec]")
            plt.ylabel("Signal [a.u.]")
            plt.title(title)
            _slmsuite_plt_show(name="settle_calibration_process")

        # Update dictionary with results. FUTURE: Return error bars?
        processed = {
            "settle_time" : settle_time,
            "relax_time" : relax_time,
            "communication_time" : com_time
        }
        self.calibrations["settle"].update(processed)

        return processed
