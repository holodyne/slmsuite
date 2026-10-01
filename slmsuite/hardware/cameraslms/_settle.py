import time
import matplotlib.pyplot as plt
from slmsuite._plotting import _slmsuite_plt_show
import numpy as np
from slmsuite import tqdm
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
        self, 
        vector=(.005, .005), 
        size=None, 
        times=None, 
        settle_time_s=None, 
        autoexpose=True,
        plot=0
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
        autoexpose : bool OR dict
            Whether or not to automatically set the camera exposure on the projected
            array. If a dictionary is passed, it is passed to
            :meth:`~slmsuite.hardware.cameras.camera.Camera.autoexpose()`.
        plot : int OR bool
            If ``>= 1``, shows a debug plot with the exponential fit.
            If ``< 0``, also suppresses the progress bar.
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

        # Create mask.
        mask = analysis.take(
            self.cam.shape, point, size, centered=True, clip=True, return_mask=True
        )

        # Optional step -- expose
        self.slm.set_phase(blaze, phase_correct=False, settle=settle_time_s)
        
        if autoexpose or isinstance(autoexpose, dict):
            self.cam.autoexpose(
                window=mask,
                **(autoexpose if isinstance(autoexpose, dict) else {})
            )
        exposure_s = self.cam.get_exposure()

        if plot >= 2:
            fig, ax = plt.subplots()
            self.cam.plot(ax=ax, title="Target (phase=blaze)")
            ax.contour(mask, levels=[0.5], colors="r")

            _slmsuite_plt_show("settle_calibrate_target")
            
        self.slm.set_phase(None, phase_correct=False, settle=settle_time_s)

        if plot >= 2:
            fig, ax = plt.subplots()
            self.cam.plot(ax=ax, title="Zeroth Order (phase=None)")
            ax.contour(mask, levels=[0.5], colors="r")

            _slmsuite_plt_show("settle_calibrate_none")
            

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

        iterations = tqdm(times) if plot >= 0 else times

        # Collect data
        self.cam.flush()
        
        for t in iterations:

            # Reset the pattern and wait for it to settle
            self.slm.set_phase(None, settle=False, phase_correct=False)
            time.sleep(settle_time_s)

            # Turn on the pattern and wait for time t
            self.slm.set_phase(blaze, settle=False, phase_correct=False)
            time.sleep(t)

            image = self.cam.get_image()
            results.append(float(np.nansum(analysis.take(
                image, point, size, centered=True, clip=True
            ))))

        self.calibrations["settle"] = {
            "times" : times,
            "data" : np.array(results),
            "exposure_s" : exposure_s,
        }
        self.calibrations["settle"].update(self._get_calibration_metadata())

        self.settle_calibration_process(plot=plot)

        return self.calibrations["settle"]

    def settle_calibration_process(self, plot=0):
        """
        Fits an exponential to the measured data to
        approximate the :math:`1/e` settle time of the SLM.

        Parameters
        ----------
        plot : int OR bool
            Whether to show a debug plot with the exponential fit.

        Returns
        -------
        dict
            The settle time and communication time measured.
        """
        times = self.calibrations["settle"]["times"]
        results = np.ravel(self.calibrations["settle"]["data"])

        # Estimate the step location and pre-step footer from the data.
        plateau = np.median(results[results > .5 * np.max(results)])
        on = np.flatnonzero(results > .05 * plateau)
        if len(on) == 0:
            raise RuntimeError("Settle calibration measured no signal.")
        x0_guess = times[on[0] - 1] if on[0] > 0 else times[0]
        before = times < x0_guess
        footer_guess = np.median(results[before]) if np.any(before) else 0

        # Function to interpolate: flat footer before the step,
        # exponential relaxation to the plateau after.
        def exponential(x, x0, a, b, c):
            return np.where(x < x0, (c-a), c - a*np.exp(-(x-x0) / b))

        # Fit the data with the function
        params, _ = optimize.curve_fit(
            exponential,
            times,
            results,
            p0=(x0_guess, plateau - footer_guess, np.ptp(times)/10, plateau),
            bounds=(
                (np.min(times), 0, 1e-6, 0),
                (np.max(times), np.inf, np.ptp(times), np.inf)
            ),
            maxfev=10000
        )
        x0, a, b, c = params
        self.logger.debug("settle fit params: %s", params)

        relax_time = b
        com_time = x0
        settle_time = com_time + relax_time*4

        if settle_time < 0 or settle_time > np.max(times):
            self.logger.warning(
                "Fitted settle time %.3g s is outside the swept range [0, %.3g] s; "
                "the response is likely unresolved at this sampling.",
                settle_time, np.max(times)
            )

        if plot >= 1:
            # Evaluate the fitting function in the interval
            x_interp = np.linspace(min(times), max(times), 100)
            y_interp = exponential(x_interp, *params)

            title = (
                f"Communication time: {int((1e3*com_time))} ms\n"
                f"$1/e$ Relaxation time: {int((1e3*relax_time))} ms\n"
                f"Suggested $1/e^4$ Settle time: {int((1e3*settle_time))} ms"
            )
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
