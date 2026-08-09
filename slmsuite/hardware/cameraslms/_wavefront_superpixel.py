import cv2
import matplotlib.pyplot as plt
from slmsuite._plotting import _slmsuite_plt_show
import numpy as np
from scipy import optimize
from tqdm.auto import tqdm

from slmsuite import __version__
from slmsuite.holography import analysis
from slmsuite.holography import toolbox
from slmsuite.holography.toolbox import imprint, format_2vectors, smallest_distance
from slmsuite.holography.toolbox.phase import blaze
from slmsuite.holography.analysis import image_remove_blaze, image_remove_vortices, image_reduce_wraps
from slmsuite.holography.analysis.fitfunctions import cos, _sinc2d_nomod
from slmsuite.holography.analysis.fitfunctions import _sinc2d_centered_taylor as sinc2d_centered
from slmsuite.misc.math import INTEGER_TYPES, REAL_TYPES

def _blaze_offset(grid, vector, offset=0):
    return blaze(grid=grid, vector=vector) + offset


class _WavefrontCalibrationSuperpixel(object):
    """
    Hidden superclass with superpixel wavefront calibration methods
    (interfere superpixel modes for relative phase measurement).
    """
    ### Superpixel Wavefront Calibration ###

    def wavefront_calibrate_superpixel(
        self,
        calibration_points=None,
        superpixel_size=50,
        reference_superpixels=None,
        exclude_superpixels=(0, 0),
        test_index=None,
        field_point=(0,0),
        field_point_units="kxy",
        phase_steps=1,
        fresh_calibration=True,
        measure_background=False,
        corrected_amplitude=False,
        plot=0,
    ):
        """
        Perform wavefront calibration by
        `iteratively interfering superpixel patches on the SLM
        <https://doi.org/10.1038/nphoton.2010.85>`_.
        This procedure measures the wavefront phase and amplitude.

        Interference occurs at a given ``calibration_points`` in the camera's imaging plane.
        It is at each point where the computed correction is ideal; the further away
        from each point, the less ideal the correction is.
        Correction at many points over the plane permits a better understanding of the
        aberration and greater possibility of compensation.

        Sets :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations["wavefront"]`.
        Run :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibration_superpixel_process`
        afterwards to produce the usable calibration which can be written to the SLM.

        Note
        ~~~~
        A Fourier calibration must be loaded.

        Tip
        ~~~
        If *only amplitude calibration* is desired,
        use ``phase_steps=None`` to omit the more time-consuming phase calibration.

        Tip
        ~~~
        Use ``phase_steps=1`` for faster calibration. This fits the phase fringes of an
        image rather than scanning the fringes over a single camera pixel over many
        ``phase_steps``. This is usually optimal except in cases with excessive noise.

        Parameters
        ----------
        calibration_points : (float, float) OR numpy.ndarray OR None OR int
            Position(s) in the camera domain where interference occurs.
            For multiple positions, this must be of shape ``(2, N)``.
            This is naturally in the ``"ij"`` basis.
            If ``None``, densely fills the camera field of view with calibration points.
            If ``int``, then crops the dense filling to the given number of points,
            prioritizing points near the center of the camera.
        superpixel_size : int
            The width and height in pixels of each SLM superpixel.
            If this is not a divisor of both dimensions in the SLM's :attr:`shape`,
            then superpixels at the edge of the SLM may be cropped and give undefined results.
            Currently, superpixels are forced to be square, and this value must be a scalar.
        reference_superpixels : (int,int) OR numpy.ndarray of int OR None
            The coordinate(s) of the superpixel(s) to reference from.
            For multiple positions, this must be of shape ``(2, N)``.
            Defaults to the center of the SLM if ``None``. If multiple calibration
            points are requested when ``None``, then the references are clustered at the center.
        exclude_superpixels : (int, int) OR numpy.ndarray OR None
            If in ``(nx, ny)`` form, optionally exclude superpixels from the margin,
            That is, the ``nx`` superpixels are omitted from the left and right sides
            of the SLM, with the same for ``ny``. As power is
            typically concentrated in the center of the SLM, this function is useful for
            excluding points that are known to be blocked (e.g. with an iris or other
            pupil), or for quickly testing calibration at the most relevant points.
            Otherwise, if exclude_superpixels is an image with the same dimension as the
            superpixeled SLM, this image is interpreted as a denylist.
            Defaults to ``None``, where no superpixels are excluded.
        test_index : int OR None
            If ``int``, then tests the scheduled calibration corresponding to this index.
            Defaults to ``None``, which runs the full wavefront calibration instead of
            testing at only one index.

            Note
            ~~~~
            Scheduling is relative to the reference superpixel(s), and a change to the
            reference may also shift the positions of the test superpixels for the given index.
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
        phase_steps : int OR None
            The number of interference phases to measure.
            If ``phase_steps`` is 1 (the default), does a one-shot fit on the interference
            pattern to improve speed.

            Tip
            ~~~
            If ``phase_steps`` is ``None``, phase is not measured and only amplitude is measured.
        fresh_calibration : bool
            If ``True``, the calibration is performed without an existing calibration
            (any old global calibration is wiped from the :class:`SLM` and :class:`CameraSLM`).
            If ``False``, the calibration is performed on top of any existing
            calibration. This is useful to determine the quality of a previous
            calibration, as a new calibration should yield zero phase correction needed
            if the previous was perfect. The old calibration will be stored in the
            :attr:`calibrations` under ``"previous_phase_correction"``,
            so keep in mind that this (uncompressed) image will take up significantly
            more memory when saved.
        measure_background : bool
            Whether to measure the background at each point.
        corrected_amplitude : bool
            If ``False``, the power in the uncorrected target beam
            is used as the source of amplitude measurement.
            If ``True``, adds another step to measure the power of the corrected target beam.
        plot : int or bool
            Whether to provide visual feedback, options are:

            - ``-1`` : No plots or tqdm prints.
            - ``0``, ``False`` : No plots, but tqdm prints.
            - ``1``, ``True`` : Plots on fits and essentials.
            - ``2`` : Plots on everything.
            - ``3`` : ``test_index`` not ``None`` only: returns image frames
              to make a movie from the phase measurement (not for general use).

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
        # Parse the superpixel size and derived quantities.
        superpixel_size = int(superpixel_size)

        slm_supershape = tuple(
            np.ceil(np.array(self.slm.shape) / superpixel_size).astype(int)
        )
        num_superpixels = slm_supershape[0] * slm_supershape[1]

        # Next, we get the size of the window necessary to measure a spot
        # produced by the given superpixel size.
        interference_window = self.wavefront_calibration_superpixel_window(superpixel_size).ravel()
        interference_size = interference_window / self._wavefront_calibration_window_multiplier

        interference_window = (interference_window // 2) * 2 + 1
        interference_size = (interference_size // 2) * 2 + 1

        # Now that we have the supershape, we label each of the pixels with an index.
        # It's sometimes useful to map that index to the xy coordinates of the pixel,
        # hence this function and its inverse.
        def index2coord(index):
            return format_2vectors(
                np.stack((index % slm_supershape[1], index // slm_supershape[1]), axis=0)
            )
        def coord2index(coord):
            coord = np.array(coord)
            return coord[1,:] * slm_supershape[1] + coord[0,:]

        # It's also useful to make an image showing the given indices.
        def index2image(index):
            image = np.zeros(slm_supershape)
            image.ravel()[index] = True
            return image

        # Parse exclude_superpixels
        exclude_superpixels = np.array(exclude_superpixels)

        if exclude_superpixels.shape == slm_supershape:
            exclude_superpixels = exclude_superpixels != 0
        elif exclude_superpixels.size == 2:
            exclude_margin = exclude_superpixels.astype(int)

            # Make the image based on margin values
            exclude_superpixels = np.zeros(slm_supershape)
            exclude_superpixels[:, :exclude_margin[0]] = True
            exclude_superpixels[:, slm_supershape[1]-exclude_margin[0]:] = True
            exclude_superpixels[:exclude_margin[1], :] = True
            exclude_superpixels[slm_supershape[0]-exclude_margin[1]:, :] = True
        else:
            raise ValueError("Did not recognize type for exclude_superpixels")

        num_active_superpixels = int(np.sum(np.logical_not(exclude_superpixels)))

        # Parse calibration_points.
        calibration_points = self._wavefront_calibration_points_parse(
            calibration_points,
            pitch=1.5*np.max(interference_window),
            field_exclusion=np.max(interference_window),
            field_point=field_point,
            field_point_units=field_point_units,
            plot=plot
        )
        num_points = calibration_points.shape[1]

        # Clean the base and field points.
        base_point = np.rint(self.kxyslm_to_ijcam([0, 0])).astype(int)

        if field_point_units != "ij":
            field_blaze = toolbox.convert_vector(
                format_2vectors(field_point),
                from_units=field_point_units,
                to_units="kxy",
                hardware=self.slm
            )

            field_point = self.kxyslm_to_ijcam(field_blaze)
        else:
            field_blaze = toolbox.convert_vector(
                field_point,
                from_units="ij",
                to_units="kxy",
                hardware=self
            )

        field_point = np.rint(format_2vectors(field_point)).astype(int)

        # Use the Fourier calibration to help find points/sizes in the imaging plane.
        if not "fourier" in self.calibrations:
            raise RuntimeError("Fourier calibration must be done before wavefront calibration.")
        calibration_blazes = self.ijcam_to_kxyslm(calibration_points)
        reference_blazes = calibration_blazes.copy()

        # Set the reference superpixels to be centered on the SLM if undefined.
        if reference_superpixels is None:
            all_superpixels = np.arange(num_superpixels)
            all_superpixels_coords = index2coord(all_superpixels)

            distance = np.sum(np.square(all_superpixels_coords - format_2vectors(slm_supershape[::-1])/2), axis=0)
            I = np.argsort(distance)

            reference_superpixels = I[:num_points]
        else:
            reference_superpixels = np.rint(format_2vectors(reference_superpixels)).astype(int)
            reference_superpixels = coord2index(reference_superpixels)

        # Error check the reference superpixels.
        reference_superpixels_coords = index2coord(reference_superpixels)
        reference_superpixels_image = index2image(reference_superpixels)
        if (np.any(np.logical_and(reference_superpixels_image, exclude_superpixels))):
            raise ValueError("reference_superpixels out of range of calibration.")

        # Now we have to solve the challenge of when to measure each target-reference pair.
        num_measurements = num_active_superpixels + ((2*num_points - 2) if phase_steps is not None else 0)

        index_image = np.reshape(np.arange(num_superpixels, dtype=int), slm_supershape)
        active_superpixels = index_image[np.logical_not(exclude_superpixels)].ravel()

        # The base schedule cycles through all indices apart from the base reference index.
        scheduling = np.zeros((num_points, num_measurements), dtype=int)

        scheduling[:, :(num_active_superpixels-1)] = np.mod(
            np.repeat(np.arange(num_active_superpixels-1, dtype=int)[np.newaxis, :] + 1, num_points, axis=0) +
            np.repeat(reference_superpixels[:, np.newaxis], num_active_superpixels-1, axis=1),
            num_active_superpixels
        )

        # Account for some superpixels being excluded.
        scheduling = active_superpixels[scheduling]
        scheduling[:, (num_active_superpixels-1):] = -1

        # Remove conflicts where other calibration pairs are targeting another
        # reference superpixel. Only do this when we are measuring relative phase
        # (if phase_steps is None, then we never write reference superpixels).
        if phase_steps is not None:
            for i in range(num_points):     # Future: Make more efficient.
                # For each calibration point, determine where the reference index is being overwritten.
                reference_index = reference_superpixels[i]

                conflicts = scheduling == reference_index
                conflict_indices = np.array(np.where(conflicts))

                for j in range(int(np.sum(conflicts))):
                    # For each time that the index is overwritten,
                    # reassign the target to empty space.
                    c_index = conflict_indices[:, j]

                    # This is the overwritten target index.
                    displaced_index = scheduling[i, c_index[1]]
                    scheduling[i, c_index[1]] = -1

                    # Find a point in empty space to resettle the index, if it is not already unused.
                    # This algorithm is currently quite slow. Consider speeding?
                    if displaced_index != -1:
                        for k in range(num_active_superpixels-1, num_measurements+1):
                            if k == num_measurements:
                                raise RuntimeError(
                                    "Some unexpected error happened in calibration scheduling."
                                )
                            elif (
                                scheduling[i, k] == -1
                                and not np.any(scheduling[:, k] == reference_index)
                                and not np.any(scheduling[:, k] == displaced_index)
                            ):
                                scheduling[i, k] = displaced_index
                                break

        # Cleanup the scheduling.
        empty_schedules = np.all(scheduling == -1, axis=0)
        scheduling = scheduling[:, np.logical_not(empty_schedules)]
        num_measurements = scheduling.shape[1]

        # Error check whether we expect to be able to see fringes.
        # max_dist_slmpix = np.max(np.hstack((
        #     reference_superpixels - exclude_superpixels,
        #     slm_supershape[::-1, [0]] - exclude_superpixels - reference_superpixels
        # )), axis=1)
        # max_r_slmpix = np.sqrt(np.sum(np.square(max_dist_slmpix)))
        # fringe_period = np.mean(interference_size) / max_r_slmpix / 2

        # if fringe_period < 2:
        #     warnings.warn(
        #         "Non-resolvable interference fringe period "
        #         "for the given SLM calibration extent. "
        #         "Either exclude more of the SLM or magnify the field on the camera."
        #     )

        # Error-check if we're measuring multiple sites at once.
        if num_points > 1:
            calibration_distance = smallest_distance(calibration_points, "euclidean")
            if np.max(interference_window) > calibration_distance:
                message = (
                    "Requested calibration points are too close together. "
                    "The minimum distance {} pix is smaller than the window size {} pix."
                    .format(calibration_distance, interference_window)
                )
                if test_index is None:
                    raise ValueError(message)
                else:
                    self.logger.warning("%s This message will error if running the full calibration.", message)

        # Error check interference point proximity to the 0th order.
        dorder = field_point - base_point
        order_distance = np.inf
        for order in range(-5, 5):
            order_distance_this = smallest_distance(
                np.hstack((
                    calibration_points,     # +1st calibration order
                    base_point + order * dorder,
                )),
                "euclidean"
            )
            if order_distance_this < order_distance:
                order_distance = order_distance_this

        if np.mean(interference_window) > order_distance:
            self.logger.warning(
                "The requested calibration point(s) are close to the expected positions of "
                "the field diffractive orders. Consider moving calibration regions further away."
            )

        # Check proximity to -1th orders.
        calibration_reflections = 2 * base_point - calibration_points
        reflection_distance = smallest_distance(
            np.hstack((
                calibration_points,         # +1st calibration order
                calibration_reflections,    # -1st calibration order
            )),
            "euclidean"
        )

        if np.mean(interference_window)/2 > reflection_distance:
            self.logger.warning(
                "The requested calibration points are close to the expected positions of "
                "the -1th orders of calibration points. Consider shifting the calibration regions "
                "relative to the 0th order. Alternatively, use the avoid_mirrors= parameter "
                "of wavefront_calibration_points"
            )

        # Save the current calibration in case we are just testing (test_index != None)
        amplitude = self.slm._get_source_amplitude()
        phase = self.slm._get_source_phase()
        r2 = self.slm.source.get("r2", None)
        if r2 is not None:
            r2 = np.copy(r2)

        # If we're starting fresh, remove the old calibration such that this does not
        # muddle things. If we're only testing, the stored data above will be reinstated.
        if fresh_calibration:
            self.slm.source.pop("amplitude", "First calibration.")
            self.slm.source.pop("phase", "First calibration.")
            self.slm.source.pop("r2", "First calibration.")

        # Parse phase_steps
        if phase_steps is not None:
            if not np.isclose(phase_steps, int(phase_steps)):
                raise ValueError(f"Expected integer phase_steps. Received {phase_steps}.")
            phase_steps = int(phase_steps)
            if phase_steps <= 0:
                raise ValueError(f"Expected positive phase_steps. Received {phase_steps}.")

        # Interpret the plot command.
        return_movie = plot == 3 and test_index is not None
        if return_movie:
            plot = 1
            if phase_steps is None or phase_steps == 1:
                raise ValueError(
                    "cameraslms.py: Must have phase_steps > 1 to produce a movie."
                )
        verbose = plot >= 0
        plot_fits = plot >= 1
        plot_everything = plot >= 2

        # Build the calibration dict.
        calibration_dict = {
            "calibration_points" : calibration_points,
            "superpixel_size" : superpixel_size,
            "slm_supershape" : slm_supershape,
            "reference_superpixels" : reference_superpixels,
            "phase_steps" : phase_steps,
            "interference_size" : interference_size,
            "interference_window" : interference_window,
            "previous_phase_correction": (
                False if "phase" not in self.slm.source else np.copy(self.slm.source["phase"])
            ),
            "scheduling" : scheduling,
        }

        keys = [
            "power",
            "normalization",
            "background",
            "phase",
            "kx",
            "ky",
            "amp_fit",
            "contrast_fit",
            "r2_fit",
        ]

        for key in keys:
            calibration_dict.update(
                {key: np.full((num_points,) + slm_supershape, np.nan, dtype=np.float32)}
            )

        def superpixels(
            schedule=None,
            reference_phase=None,
            target_phase=None,
            reference_blaze=reference_blazes,
            target_blaze=calibration_blazes,
            phase_baselines=None,
            plot=False
        ):
            """
            Helper function for making superpixel phase masks.

            Parameters
            ----------
            schedule : list of int
                Defines which superpixels to source targets from.
            reference_phase, target_phase : float OR None
                Phase of reference/target superpixel; not rendered if None.
            reference_blaze, target_blaze : (float, float)
                Blaze vector(s) for the given superpixel.
            """
            matrix = blaze(self.slm, field_blaze)

            if reference_phase is not None:
                for i in range(num_points):
                    if schedule is None or schedule[i] != -1:
                        imprint(
                            matrix,
                            np.array([
                                reference_superpixels_coords[0, i], 1,
                                reference_superpixels_coords[1, i], 1
                            ]) * superpixel_size,
                            _blaze_offset,
                            self.slm,
                            # shift=True,
                            vector=reference_blaze[:, [i]],
                            offset=reference_phase  # This is usually zero when not None.
                        )

            if target_phase is not None and schedule is not None:
                target_coords = index2coord(schedule)
                for i in range(num_points):
                    if schedule[i] != -1:
                        phase_baseline = 0 if phase_baselines is None else phase_baselines[i]
                        imprint(
                            matrix,
                            np.array([
                                target_coords[0, i], 1,
                                target_coords[1, i], 1
                            ]) * superpixel_size,
                            _blaze_offset,
                            self.slm,
                            # shift=True,
                            vector=target_blaze[:, [i]],
                            offset=phase_baseline + (target_phase if np.isscalar(target_phase) else target_phase[i])
                        )

            self.slm.set_phase(matrix, settle=True)
            self.cam.flush()
            if plot:
                plt.figure(figsize=(20, 20))
                self.slm.plot()
            return self.cam.get_image()

        def fit_phase(phases, intensities, plot_fits=False):
            """
            Fits a sine function to the intensity vs phase, and extracts best phase and amplitude
            that give constructive interference.
            If fit fails return 0 on all values.

            Parameters
            ----------
            phases : numpy.ndarray
                Phase measurements.
            intensities : numpy.ndarray
                Intensity measurements.
            plot_fits : bool
                Whether to plot fit results.

            Returns
            -------
            best_phase : float
                Phase b [rad] of maximum constructive interference.
            amp : float
                Amplitude a of the interference fringe.
            r2 : float
                R^2 of fit
            contrast : float
                a / (a + c), where c is the background
            """
            guess = [
                phases[np.argmax(intensities)],
                np.max(intensities) - np.min(intensities),
                np.min(intensities),
            ]

            try:
                popt, _ = optimize.curve_fit(cos, phases, intensities, p0=guess)
            except BaseException:
                self.logger.warning("Curve fitting failed; nulling response from this superpixel.")
                return 0, 0, 0, 0

            # Extract phase and amplitude from fit.
            best_phase = popt[0]
            amp = popt[1]
            contrast = popt[1] / (popt[1] + popt[2])

            # Residual and total sum of squares, producing the R^2 metric.
            ss_res = np.sum((intensities - cos(phases, *popt)) ** 2)
            ss_tot = np.sum((intensities - np.mean(intensities)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            if plot_fits:
                plt.scatter(phases / np.pi, intensities, color="k", label="Data")

                phases_fine = np.linspace(0, 2 * np.pi, 100)

                plt.plot(phases_fine / np.pi, cos(phases_fine, *popt), "k-", label="Fit")
                plt.plot(phases_fine / np.pi, cos(phases_fine, *guess), "k--", label="Guess")
                plt.plot(best_phase / np.pi, popt[1] + popt[2], "xr", label="Phase")

                plt.legend(loc="best")
                plt.title("Interference ($R^2$={:.3f})".format(r2))
                plt.grid()
                plt.xlim([0, 2])
                plt.xlabel(r"$\phi$ $[\pi]$")
                plt.ylabel("Signal")

                _slmsuite_plt_show(name="wavefront_calibrate_superpixel_fit_1D")

            return best_phase, amp, r2, contrast

        def fit_phase_image(img, dsuperpixel, plot_fits=True):
            """
            Fits a modulated 2D sinc function to an image, and extracts best phase and
            amplitude that give constructive interference.
            If fit fails return 0 on all values.

            Parameters
            ----------
            img : numpy.ndarray
                2D image centered on the interference point.
            dsuperpixel : ndarray
                Integer distance (dx,dy) between superpixels.

            Returns
            -------
            best_phase : float
                Phase [rad] of maximum constructive interference.
            amp : float
                Amplitude of the interference fringe.
            r2 : float
                R^2 of fit
            contrast : float
                a / (a + c)
            """
            # Future: Cache this outside to avoid repeating memory allocation.
            xy = np.meshgrid(
                *[
                    np.arange(-(img.shape[1 - a] - 1) / 2, +(img.shape[1 - a] - 1) / 2 + 0.5)
                    for a in range(2)
                ]
            )
            xyr = [l.ravel() for l in xy]

            # Process dsuperpixel by rotating it according to the Fourier calibration.
            M = self.fourier_affine.M
            M_norm = M / np.sqrt(np.abs(np.linalg.det(M)))
            dsuperpixel = np.squeeze(np.matmul(M_norm, format_2vectors(dsuperpixel)))

            # Make the guess and bounds.
            d = float(np.amin(img))
            c = 0
            a = float(np.amax(img)) - c
            R = float(np.mean(img.shape)) / 4
            # theta = np.arctan2(M[1, 0],  -M[0, 0])

            guess = [
                R, a, 0, c, d,
                8 * np.pi * dsuperpixel[0] / img.shape[1],
                8 * np.pi * dsuperpixel[1] / img.shape[0]
            ]
            dk = 8 * np.pi * np.max(slm_supershape) / np.min(img.shape)
            lb = [
                .9*R, 0, -4*np.pi, 0, 0,
                guess[5]-dk,
                guess[6]-dk
            ]
            ub = [
                1.1*R, 2*a, 4*np.pi, a, a,
                guess[5]+dk,
                guess[6]+dk
            ]

            # # Restrict sinc2d to be centered (as expected).
            # def sinc2d_local(xy, R, a=1, b=0, c=0, d=0, kx=1, ky=1, theta=0):
            #     # When centered, rotation can be applied to xy, kxy
            #     c = np.cos(theta)
            #     s = np.sin(theta)
            #     rotation = np.array([[c, -s], [s, c]])
            #     kxy = rotation @ np.array([kx, ky])

            #     # If raveled (for optimization)
            #     xy = np.array(xy)
            #     if len(np.array(xy).shape) < 3:
            #         xy_rot = rotation @ xy
            #     # But otherwise not raveled
            #     else:
            #         xy_rot = np.array([rotation @ xy[:, :, i] for i in range(xy.shape[-1])])
            #         xy_rot = np.transpose(xy_rot, (1, 2, 0))

            #     return sinc2d(xy_rot, 0, 0, R, a, b, c, d, kxy[0], kxy[1])

            # Determine the guess phase byt overlapping shifted guesses with the image.
            differences = []
            N = 20
            phases = np.arange(N) * 2 * np.pi / N

            for phase in phases:
                guess[2] = phase
                differences.append(np.sum(np.square(img - sinc2d_centered(xy, *guess))))

            guess[2] = phases[int(np.min(np.argmin(differences)))]

            # Try the fit!
            try:
                popt, _ = optimize.curve_fit(
                    sinc2d_centered,
                    xyr,
                    img.ravel().astype(float),
                    p0=guess,
                    bounds=(lb, ub), #, maxfev=20
                    # method="dogbox",
                    # jac=sinc2d_centered_jacobian
                )
            except BaseException:
                return [np.nan, np.nan, 0, np.nan]

            # Extract phase and amplitude from fit.
            best_phase = popt[2]
            amp = np.abs(popt[1])
            contrast = np.abs(popt[1] / (np.abs(popt[1]) + np.abs(popt[3])))

            # Remove the sinc term when doing the rsquared.
            popt_nomod = np.copy(popt)
            popt_nomod[3] += popt_nomod[1] / 2
            popt_nomod[1] = 0
            img0 = img - sinc2d_centered(xy, *popt_nomod)
            fit0 = sinc2d_centered(xy, *popt) - sinc2d_centered(xy, *popt_nomod)

            # Residual and total sum of squares, producing the R^2 metric.
            ss_res = np.sum((img0 - fit0) ** 2)
            ss_tot = np.sum((img0 - np.mean(img0)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            final = (np.mod(-best_phase, 2*np.pi), amp, r2, contrast)

            # Plot the image, guess, and fit, if desired.
            if plot_fits:
                _, axs = plt.subplots(1, 3, figsize=(20,10))

                axs[0].imshow(img)
                axs[1].imshow(sinc2d_centered(xy, *guess))
                axs[2].imshow(sinc2d_centered(xy, *popt))

                for index, title in enumerate(["Image", "Guess", "Fit"]):
                    axs[index].set_title(title)

                _slmsuite_plt_show(name="wavefront_calibrate_superpixel_fit_2D")

            return final

        def plot_labeled(schedule, img, phase=None, plot=False, title="", plot_zoom=False, focus=None):
            if plot_everything or plot:
                def plot_labeled_rects(ax, points, labels, colors, wh, hh):
                    for point, label, color in zip(points, labels, colors):
                        rect = plt.Rectangle(
                            (float(point[0] - wh/2), float(point[1] - hh/2)),
                            float(wh), float(hh),
                            ec=color, fc="none"
                        )
                        ax.add_patch(rect)
                        ax.annotate(
                            label, (point[0], point[1]),
                            c=color, size="x-small", ha="center", va="center"
                        )

                if return_movie:
                    fig, axs = plt.subplots(1, 3, figsize=(16, 4), facecolor="white")
                else:
                    fig, axs = plt.subplots(1, 3, figsize=(16,4))

                # Plot phase on the first axis.
                if phase is None:
                    phase = self.slm.phase
                axs[0].imshow(
                    np.mod(phase, 2*np.pi),
                    cmap=plt.get_cmap("twilight"),
                    interpolation="none",
                )

                points = []
                labels = []
                colors = []
                center_offset = np.array([superpixel_size/2, superpixel_size/2])

                for i in range(num_points):
                    if schedule is None or schedule[i] != -1:
                        if focus is None:
                            focus = i
                        points.append(reference_superpixels_coords[:, i] * superpixel_size + center_offset)
                        if schedule is not None: points.append((index2coord(schedule[i]) * superpixel_size + center_offset).ravel())
                        if num_points > 1:
                            labels.append("{}".format(i))
                            if schedule is not None: labels.append("{}".format(i))
                        else:
                            labels.append("Reference\nSuperpixel")
                            if schedule is not None: labels.append("Test\nSuperpixel")
                        c1 = (1 if i == focus else .5, .2, 0)
                        colors.append(c1)
                        c2 = (1 if i == focus else .5, 0, .2)
                        if schedule is not None: colors.append(c2)

                plot_labeled_rects(axs[0], points, labels, colors, superpixel_size, superpixel_size)

                # FUTURE: fix for multiple
                # if plot_zoom:
                #     for a in [0, 1]:
                #         ref = reference_superpixels[a] * superpixel_size
                #         test = test_superpixel[a] * superpixel_size

                #         lim = [min(ref, test) - .5 * superpixel_size, max(ref, test) + 1.5 * superpixel_size]

                #         if a:
                #             axs[0].set_ylim([lim[1], lim[0]])
                #         else:
                #             axs[0].set_xlim(lim)

                if img is not None:
                    im = axs[1].imshow(np.log10(img + .1))
                    im.set_clim(0, np.log10(self.cam.bitresolution))

                dpoint = field_point - base_point

                # Assemble points and labels.
                points = [(base_point + N * dpoint).ravel() for N in range(-2, 3)]
                labels = ["-2nd", "-1st", "0th", "1st", "2nd"]
                colors = ["b"] * 5

                focus_point = None

                for i in range(num_points):
                    if schedule is None or schedule[i] != -1:
                        points.append(calibration_points[:, i])
                        if num_points > 1:
                            labels.append("{}".format(i))
                        else:
                            labels.append("Calibration\nPoint")
                        c = (1 if i == focus else .5, 0, 0)
                        colors.append(c)
                        if i == focus:
                            focus_point = calibration_points[:, i]

                # Plot points and labels.
                wh = int(interference_window[0])
                hh = int(interference_window[1])

                plot_labeled_rects(axs[1], points, labels, colors, wh, hh)

                if img is not None:
                    im = axs[2].imshow(np.log10(img + .1))
                    im.set_clim(0, np.log10(self.cam.bitresolution))

                    if self.cam.bitdepth > 10:
                        step = 2
                    else:
                        step = 1

                    bitres_list = np.power(2, np.arange(0, self.cam.bitdepth+1, step), dtype=int)

                    cbar = fig.colorbar(im, ax=axs[2])
                    cbar.ax.set_yticks(np.log10(bitres_list))
                    cbar.ax.set_yticklabels(bitres_list)

                point = focus_point

                axs[2].scatter([point[0]], [point[1]], 5, "r", "*")
                axs[2].set_xlim(point[0] - wh/2, point[0] + wh/2)
                axs[2].set_ylim(point[1] + hh/2, point[1] - hh/2)

                # Axes coloring and colorbar.
                for spine in ["top", "bottom", "right", "left"]:
                    axs[2].spines[spine].set_color("r")
                    axs[2].spines[spine].set_linewidth(1.5)

                axs[0].set_title("SLM Phase")
                axs[1].set_title("Camera Result")
                axs[2].set_title(title)

                if plot_zoom and return_movie:
                    fig.tight_layout()
                    fig.canvas.draw()

                    try:
                        try:
                            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                            image_from_plot = image_from_plot.reshape(
                                fig.canvas.get_width_height()[::-1] + (3,)
                            )
                        except Exception:
                            image_from_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                            image_from_plot = image_from_plot.reshape(
                                fig.canvas.get_width_height()[::-1] + (4,)
                            )[:,:,:3]
                    except Exception:
                        self.logger.warning(
                            "Failed to convert figure to image for wavefront_calibrate movie. "
                            "Returning a blank image instead."
                        )
                        image_from_plot = np.zeros(
                            fig.canvas.get_width_height()[::-1] + (3,),
                            dtype=np.uint8
                        )

                    plt.close()

                    return image_from_plot
                else:
                    title_parsed = title.replace(" ", "_").lower()
                    _slmsuite_plt_show(name=f"wavefront_calibrate_superpixel_{title_parsed}")

        def take_interference_regions(img, integrate=True):
            """Helper function for grabbing the data at the calibration points."""
            return analysis.take(
                img,
                calibration_points,
                interference_window, # / (2 if integrate else 1),
                clip=True,
                integrate=integrate
            )

        def find_centers(img, fit=True):
            """Helper function for finding the center of images around the calibration points."""
            imgs = take_interference_regions(img, integrate=False)  # N x W x H
            centers = analysis.image_positions(imgs)                # 2 x N

            a = np.max(imgs, axis=(1,2))
            R = np.mean(imgs.shape[1:])/4

            guess = np.transpose(
                np.vstack((
                    centers,
                    np.full_like(a, R),
                    a,
                    np.full_like(a, 0),
                ))
            )

            result = analysis.image_fit(imgs, function=_sinc2d_nomod, guess=guess) #, plot=True)

            centers = result[:, 1:3].T

            # if not fit:
            return centers + calibration_points
            # else:
            #     return centers + calibration_points, amps_fit

        def measure(schedule, plot=False):
            # self.cam.flush()

            # Step 0: Measure the background.
            if measure_background:
                back_image = superpixels(schedule, None, None)
                plot_labeled(schedule, back_image, plot=plot, title="Background")
                back = take_interference_regions(back_image)
            else:
                back = [np.nan] * num_points

            # Step 0.5: Measure the power in the reference mode.
            norm_image = superpixels(schedule, 0, None)
            plot_labeled(schedule, norm_image, plot=plot, title="Reference Diffraction")
            norm = take_interference_regions(norm_image)

            # Step 1: Check the target mode, and return if we don't need to correct.
            position_image = superpixels(schedule, None, 0)
            plot_labeled(schedule, position_image, plot=plot, title="Base Target Diffraction")
            if phase_steps is None and not corrected_amplitude:
                pwr = take_interference_regions(position_image)
                return {
                    "power": pwr,
                    "normalization": norm,
                    "background": back,
                    "phase": [np.nan] * num_points,
                    "kx": [np.nan] * num_points,
                    "ky": [np.nan] * num_points,
                    "amp_fit": [np.nan] * num_points,
                    "contrast_fit": [np.nan] * num_points,
                    "r2_fit": [np.nan] * num_points,
                }

            # Step 1.25: Add a blaze to the target mode so that it overlaps with reference mode.
            found_centers = find_centers(position_image)
            blaze_differences = self.ijcam_to_kxyslm(found_centers) - calibration_blazes
            target_blaze_fixed = calibration_blazes - blaze_differences

            # Step 1.5: Measure the power...
            if corrected_amplitude:      # ...in the corrected target mode.
                fixed_image = superpixels(schedule, None, 0, target_blaze=target_blaze_fixed)
                plot_labeled(schedule, fixed_image, plot=plot, title="Corrected Target Diffraction")
                pwr = take_interference_regions(fixed_image)
            else:                       # ...in the uncorrected target mode.
                pwr = take_interference_regions(position_image)

            # Step 1.75: Stop here if we don't need to measure the phase (only save powers).
            if phase_steps is None:
                return {
                    "power": pwr,
                    "normalization": norm,
                    "background": back,
                    "phase": [np.nan] * num_points,
                    "kx": -blaze_differences[0, :],
                    "ky": -blaze_differences[1, :],
                    "amp_fit": [np.nan] * num_points,
                    "contrast_fit": [np.nan] * num_points,
                    "r2_fit": [np.nan] * num_points,
                }

            results = []
            first_index = np.where(schedule != -1)[0][0]

            # target_coords = index2coord(schedule)
            # phase_baselines = np.sum(
            #     2 * np.pi * target_blaze_fixed *
            #     (target_coords - reference_superpixels_coords) *
            #     superpixel_size * self.slm.pitch[:, np.newaxis],
            #     axis=0,
            # )
            phase_baselines = None

            # Step 2: Measure interference and find relative phase. Future: vectorize.
            if phase_steps == 1:
                # Step 2.1: Gather a single image.
                result_img = superpixels(schedule, 0, 0, target_blaze=target_blaze_fixed, phase_baselines=phase_baselines)
                cropped_img = take_interference_regions(result_img, integrate=False)

                # Step 2.2: Fit the data and return.
                coord_difference = index2coord(schedule) - index2coord(reference_superpixels)

                results = [
                    (
                        fit_phase_image(
                            cropped_img[i],
                            coord_difference[:,i],
                            plot_fits=plot and i == first_index
                        )
                        if schedule[i] != -1 else
                        [np.nan] * 4
                    )
                    for i in range(num_points)
                ]
            else:
                # Gather multiple images at different phase offsets.
                phases = np.linspace(0, 2 * np.pi, phase_steps, endpoint=False)
                iresults = []  # list for recording the intensity of the reference point

                # Determine whether to use a progress bar.
                if verbose:
                    description = "phase_measurement"
                    prange = tqdm(phases, position=0, leave=False, desc=description)
                else:
                    prange = phases

                if return_movie:
                    frames = []

                # Step 2.1: Measure phases
                for phase in prange:
                    interference_image = superpixels(schedule, 0, phase, target_blaze=target_blaze_fixed, phase_baselines=phase_baselines)
                    iresults.append(
                        [
                            interference_image[calibration_points[1, i], calibration_points[0, i]]
                            for i in range(num_points)
                        ]
                    )

                    if return_movie:
                        frames.append(
                            plot_labeled(
                                schedule,
                                interference_image,
                                plot=plot,
                                title=r"Phase = ${:1.2f}\pi$".format(phase / np.pi),
                                plot_zoom=True,
                            )
                        )

                iresults = np.array(iresults)

                # Step 2.2: Fit to sine and return.
                for i in range(num_points):
                    results.append(fit_phase(phases, iresults[:, i], plot and i == first_index))

            results = np.array(results)

            phase_fit =     results[:, 0]
            amp_fit =       results[:, 1]
            r2_fit =        results[:, 2]
            contrast_fit =  results[:, 3]

            # Step 2.5: maybe plot a picture of the correct phase.
            if plot:
                interference_image = superpixels(schedule, 0, phase_fit, target_blaze=target_blaze_fixed, phase_baselines=phase_baselines)
                plot_labeled(schedule, interference_image, plot=plot, title="Best Interference")

            # Step 3: Return the result.
            if return_movie:
                return frames

            return {
                "power": pwr,
                "normalization": norm,
                "background": back,
                "phase": phase_fit,
                "kx": -blaze_differences[0, :],
                "ky": -blaze_differences[1, :],
                "amp_fit": amp_fit,
                "contrast_fit": contrast_fit,
                "r2_fit": r2_fit,
            }

        # Correct exposure and position of the reference mode(s).
        # self.cam.flush()
        base_image = superpixels(None, 0, None)
        plot_labeled(None, base_image, plot=plot_everything, title="Base Reference Diffraction")
        found_centers = find_centers(base_image)

        # Correct the original blaze using the measured result.
        reference_blaze_differences = self.ijcam_to_kxyslm(found_centers) - reference_blazes
        np.subtract(reference_blazes, reference_blaze_differences, out=reference_blazes)

        if plot_fits:
            fixed_image = superpixels(None, 0, None)
            plot_labeled(None, fixed_image, plot=plot_everything, title="Corrected Reference Diffraction")

        # If we just want to debug/test one region, then do so.
        if test_index is not None:
            result = measure(scheduling[:, test_index], plot=plot_fits)

            # Reset the phase and amplitude of the SLM to the stored data.
            self.slm.source["amplitude"] = amplitude
            self.slm.source["phase"] = phase
            if r2 is not None:
                self.slm.source["r2"] = r2

            return result

        measurements = range(num_measurements)
        if plot > -1:
            measurements = tqdm(measurements, position=1, leave=True, desc="calibration")

        # Proceed with all of the superpixels.
        for n in measurements:
            schedule = scheduling[:, n]

            # Measure!
            measurement = measure(schedule)

            # Update dictionary.
            coords = index2coord(schedule)
            for i in range(num_points):
                if schedule[i] != -1:
                    for key in measurement.keys():
                        result = measurement[key]
                        if np.size(result) > 1:
                            result = result[i]
                        elif not np.isscalar(result):
                            result = np.squeeze(result)

                        calibration_dict[key][i, coords[1, i], coords[0, i]] = result

        self.calibrations["wavefront_superpixel"] = calibration_dict
        self.calibrations["wavefront_superpixel"].update(self._get_calibration_metadata())

        return calibration_dict

    ### Superpixel Wavefront Calibration Helpers ###

    def wavefront_calibration_superpixel_window(self, superpixel_size):
        """
        Returns the window size for the interference regions.
        This is inversely proportional to the size of the superpixel because the
        superpixel and interference zone are separated by a Fourier transform.
        The computation works by estimating the spot size of the interference beams and
        then enlarging by a stored multiplier
        :attr:`_wavefront_calibration_window_multiplier`
        which defaults to 4.

        Parameters
        ----------
        superpixel_size : int
            The size of the superpixel on the SLM.
        """
        interference_size = np.rint(np.array(
            self.get_farfield_spot_size(
                superpixel_size * self.slm.pitch,
                basis="ij"
            )
        )).astype(int)

        return self._wavefront_calibration_window_multiplier * interference_size

    def wavefront_calibration_superpixel_process(
        self,
        index=0,
        smooth=True,
        r2_threshold=0.9,
        remove_vortices=False,
        remove_blaze=True,
        remove_background=True,
        apply=True,
        plot=False
    ):
        """
        Processes :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations` ``["wavefront"]``
        into the desired phase correction and amplitude measurement. Applies these
        parameters to the respective variables in the SLM if ``apply`` is ``True``.

        Parameters
        ----------
        index : int
            The calibration point index to process, in the case of a multi-point calibration.
            In the future, this should include the option to request an "ij" position,
            then the return will automatically interpolate between the Zernike results
            of the local calibration points.
        smooth : bool OR int
            Whether to blur the correction data to avoid aliasing.
            If ``int``, uses this as the number of smoothing iterations.
            Defaults to ``16`` if ``True``.
        r2_threshold : float
            Threshold for a "good fit". Proxy for whether a datapoint should be used or
            ignored in the final data, depending upon the rsquared value of the fit.
            Should be within ``[0, 1]``.
        remove_vortices : bool
            A wavefront correct should be smooth when using smooth optics (lenses).
            However, incorrect phase wrapping can lead to phase vortices surrounding the
            exceptional points at the ends of improperly chosen branches.
            This is unphysical. If `True`, these exceptional points are eliminated
            halfway through the phase smoothing process. If `smooth=False`, this is
            ignored. The nature of vortex removal might add a global blaze to the
            pattern, so it is recommended to also set ``remove_blaze=True``.
        remove_blaze : bool
            If ``True``, removes the global blaze from the phase correction, as defined
            by the average blaze weighted by the measured power.
        remove_background : bool
            If the experimental background was not measured, this flag estimates the
            interference region's background by looking at the noisefloor of the
            measured power distribution. If the noisefloor is flat enough, the
            power is shifted to have a minimum at zero.
        apply : bool
            Whether to apply the processed calibration to the associated SLM.
            Otherwise, this function only returns and maybe
            plots these results. Defaults to ``True``.
        plot : bool
            Whether to enable debug plots.

        Returns
        -------
        dict
            The updated source dictionary containing the processed source amplitude and phase.
        """
        # Step 0: Initialize helper variables and functions.
        if "wavefront_superpixel" in self.calibrations:
            data = self.calibrations["wavefront_superpixel"]
        elif "wavefront" in self.calibrations:
            data = self.calibrations["wavefront"]
        else:
            raise RuntimeError("Could not find wavefront calibration.")

        if len(data) == 0:
            raise RuntimeError("No raw wavefront data to process. Either load data or calibrate.")

        if not "__version__" in data:
            data["__version__"] = "0.0.1"

        if data["__version__"] == "0.0.1":
            return self._wavefront_calibration_superpixel_process_r001(
                data,
                smooth=smooth,
                r2_threshold=r2_threshold,
                remove_vortices=remove_vortices,
                remove_blaze=remove_blaze,
                remove_background=remove_background,
                apply=apply,
                plot=plot
            )
        else:
            # For now, make a 0.0.1 calibration dict based on a single index.
            slm_supershape = data["slm_supershape"]

            def index2coord(index):
                return format_2vectors(
                    np.stack((index % slm_supershape[1], index // slm_supershape[1]), axis=0)
                )

            reference_superpixel = index2coord(data["reference_superpixels"][index]).ravel()

            correction_dict = {
                "NX": slm_supershape[1],
                "NY": slm_supershape[0],
                "nxref": reference_superpixel[0],
                "nyref": reference_superpixel[1],
                "superpixel_size": data["superpixel_size"],
                "interference_point": data["calibration_points"][:, index],
                "interference_size": data["interference_size"],
            }

            keys = [
                "power",
                "normalization",
                "background",
                "phase",
                "kx",
                "ky",
                "amp_fit",
                "contrast_fit",
                "r2_fit",
            ]

            for key in keys:
                correction_dict.update({key: data[key][index]})

            return self._wavefront_calibration_superpixel_process_r001(
                correction_dict,
                smooth=smooth,
                r2_threshold=r2_threshold,
                remove_vortices=remove_vortices,
                remove_blaze=remove_blaze,
                remove_background=remove_background,
                apply=apply,
                plot=plot,
            )

    def _wavefront_calibration_superpixel_process_r001(
            self,
            data,
            smooth=True,
            r2_threshold=0.9,
            remove_vortices=False,
            remove_blaze=True,
            remove_background=True,
            apply=True,
            plot=False,
        ):
        """
        Wavefront calibration processing, in the release 0.0.1 data format.
        Both branches of the version dispatch route here.
        See docstring for :meth:`wavefront_calibration_superpixel_process`.

        Returns
        -------
        dict
            The updated source dictionary containing the processed source amplitude and phase.
        """
        # Parse smooth.
        if smooth is True:
            smooth = 16
        smooth = int(smooth)
        if smooth < 0:
            raise ValueError("Smoothing iterations must be a non-negative integer.")

        # Parse r2_threshold.
        r2_threshold = float(r2_threshold)

        # Step 0: Initialize helper variables and functions.
        if len(data) == 0:
            raise RuntimeError("No raw wavefront data to process. Either load data or calibrate.")

        NX = data["NX"]
        NY = data["NY"]
        nxref = data["nxref"]
        nyref = data["nyref"]

        def average_neighbors(matrix):
            n = 0
            result = 0
            for xy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                x =  nxref + xy[0]
                y =  nyref + xy[1]

                if x >= 0 and x < NX and y >= 0 and y < NY:
                    result += matrix[y, x]
                    n += 1

            matrix[nyref, nxref] = result / n

        size_blur_k = 1

        # Step 1: Process R^2
        superpixel_size = int(data["superpixel_size"])
        w = superpixel_size * NX
        h = superpixel_size * NY

        r2 = np.copy(data["r2_fit"])
        r2[nyref, nxref] = 1
        r2s = r2

        r2s_large = cv2.resize(r2s, (w, h), interpolation=cv2.INTER_NEAREST)
        r2s_large = r2s_large[: self.slm.shape[0], : self.slm.shape[1]]

        # Step 2: Process the measured amplitude
        # Fix the reference pixel by averaging the 8 surrounding pixels
        pwr = np.copy(data["power"])
        pwr[pwr == np.inf] = np.amax(pwr)
        average_neighbors(pwr)
        if smooth:
            pwr = cv2.GaussianBlur(pwr, (size_blur_k, size_blur_k), 0)

        norm = np.copy(data["normalization"])
        average_neighbors(norm)
        if smooth:
            norm = cv2.GaussianBlur(norm, (size_blur_k, size_blur_k), 0)

        back = np.copy(data["background"])
        back[np.isnan(back)] = 0
        average_neighbors(back)
        if smooth:
            back = cv2.GaussianBlur(back, (size_blur_k, size_blur_k), 0)

        if remove_background:
            is_noise = r2s < r2_threshold
            if np.all(back == 0) and np.sum(is_noise) > 0:
                # Check the area defined as noise.
                pwr_below_r2 = pwr[is_noise]
                pwr_below_r2[np.isnan(pwr_below_r2)] = np.nanmin(pwr_below_r2)
                pwr_below_r2[np.isnan(pwr_below_r2)] = 0

                # If the median is within 0.5 std of the minimum, then we assume the
                # minimum is close to the noise floor.
                pwr_min = np.min(pwr_below_r2)
                norm_ave = np.nanmean(norm)
                norm_min = np.nanmin(norm)
                if (np.median(pwr_below_r2) - pwr_min) / np.nanstd(pwr) < .5 and pwr_min < norm_min:
                    self.logger.warning(
                        "remove_background is enabled and a noise floor was detected; "
                        "removing this background (%.1f%% of the average normalization).",
                        100 * pwr_min/norm_ave
                    )
                    back[:] = pwr_min

        pwr -= back
        norm -= back

        # Normalize and resize
        pwr_norm = np.divide(pwr, norm)

        pwr_norm[np.isnan(pwr_norm)] = 0
        pwr_norm[~np.isfinite(pwr_norm)] = 0
        pwr_norm[pwr_norm < 0] = 0

        pwr_large = cv2.resize(pwr_norm, (w, h), interpolation=cv2.INTER_CUBIC)
        pwr_large = pwr_large[: self.slm.shape[0], : self.slm.shape[1]]

        pwr_large[np.isnan(pwr_large)] = 0
        pwr_large[~np.isfinite(pwr_large)] = 0
        pwr_large[pwr_large < 0] = 0

        if smooth:
            size_blur = 4 * int(superpixel_size) + 1
            pwr_large = cv2.GaussianBlur(pwr_large, (size_blur, size_blur), 0)

        amp_large = np.sqrt(pwr_large)
        amp_large /= np.nanmax(amp_large)

        # Step 3: Process the wavefront
        # Load data.
        kx = np.copy(data["kx"])
        ky = np.copy(data["ky"])
        offset = np.copy(data["phase"])

        # Handle nans.
        kx[np.isnan(kx)] = 0
        ky[np.isnan(ky)] = 0
        offset[np.isnan(offset)] = 0
        r2[np.isnan(r2)] = 0

        # Fix a change in how data is aquired pre-0.3.0.
        # if phase_shift_pre_030:
        #     X = np.arange(float(NX)); X -= np.mean(X)
        #     Y = np.arange(float(NY)); Y -= np.mean(Y)
        #     grid_x, grid_y = np.meshgrid(X, Y)
        #     (dx, dy) = (
        #         2 * np.pi * superpixel_size * self.slm.pitch[0],
        #         2 * np.pi * superpixel_size * self.slm.pitch[1],
        #     )
        #     offset += dx * kx * grid_x + dy * ky * grid_y

        # Fill in the reference pixel with surrounding data.
        real = np.cos(offset)
        imag = np.sin(offset)

        average_neighbors(real)
        average_neighbors(imag)

        average_neighbors(kx)
        average_neighbors(ky)

        offset = np.arctan2(imag, real) + np.pi

        # Apply the R^2 threshold.
        kx[r2s < r2_threshold] = 0
        ky[r2s < r2_threshold] = 0
        offset[r2s < r2_threshold] = 0
        pathing = 0 * r2s - 100

        # Step 3.1: Infer phase for superpixels which do satisfy the R^2 threshold.
        # Go forward and then back along the columns, sweeping each column's rows.
        for nx in list(range(NX)) + list(range(NX - 1, -1, -1)):
            for ny in range(NY):
                if r2s[ny, nx] >= r2_threshold:
                    # Superpixels exceeding the threshold need no correction.
                    pass
                else:
                    # Otherwise, do a majority-vote with adjacent superpixels.
                    kx2 = []
                    ky2 = []
                    offset2 = []
                    source = []

                    (dx0, dy0) = (
                        2 * np.pi * (nx-nxref) * superpixel_size * self.slm.pitch[0],
                        2 * np.pi * (ny-nyref) * superpixel_size * self.slm.pitch[1],
                    )

                    # Loop through the adjacent superpixels (including diagonals).
                    for ax, ay in [
                        (1, 0),
                        (-1, 0),
                        (0, 1),
                        (0, -1),
                        # (1, -1),
                        # (-1, -1),
                        # (1, 1),
                        # (-1, 1),
                    ]:
                        (tx, ty) = (nx + ax, ny + ay)
                        # (dx, dy) = (
                        #     2 * np.pi * ax * superpixel_size * self.slm.pitch[0],
                        #     2 * np.pi * ay * superpixel_size * self.slm.pitch[1],
                        # )

                        # Make sure our adjacent pixel under test is within range and above threshold.
                        if (
                            tx >= 0
                            and tx < NX
                            and ty >= 0
                            and ty < NY
                            and (
                                r2s[ty, tx] >= r2_threshold
                                or pathing[ty, tx] == ny
                                or (abs(pathing[ty, tx] - ny) == 1 and ax != 0)
                            )
                        ):
                            kx3 = kx[ty, tx]
                            ky3 = ky[ty, tx]

                            kx2.append(kx3)
                            ky2.append(ky3)
                            offset2.append(offset[ty, tx] + (dx0 * kx3 + dy0 * ky3))
                            source.append((ax, ay))

                    # Do a majority vote (within std) for the phase.
                    if len(kx2) > 0:
                        kx[ny, nx] = np.mean(kx2)
                        ky[ny, nx] = np.mean(ky2)

                        minstd = np.inf
                        for phi in range(4):
                            shift = phi * np.pi / 2
                            offset3 = np.mod(np.array(offset2) + shift, 2 * np.pi)

                            if minstd > np.std(offset3):
                                minstd = np.std(offset3)
                                offset[ny, nx] = np.mod(np.mean(offset3) - shift, 2 * np.pi)

                        offset[ny, nx] -=  dx0 * kx[ny, nx] + dy0 * ky[ny, nx]
                        pathing[ny, nx] = ny

        # Step 3.2: Make the SLM-sized correction using the compressed data from each superpixel.
        phase = np.zeros(self.slm.shape)
        for nx in range(NX):
            for ny in range(NY):
                imprint(
                    phase,
                    np.array([nx, 1, ny, 1]) * superpixel_size,
                    _blaze_offset,
                    self.slm,
                    # shift=True,
                    vector=(kx[ny, nx], ky[ny, nx]),
                    offset=offset[ny, nx],
                )

        # Step 3.3: Iterative smoothing helps to preserve slopes while avoiding superpixel boundaries.
        # Consider, for instance, a fine blaze which smooths flat.
        if smooth:
            for i in tqdm(range(smooth), desc="smooth"):
                real = np.cos(phase)
                imag = np.sin(phase)

                # Blur the phase to smooth it out
                size_blur = 2 * int(superpixel_size / 4) + 1
                real = cv2.GaussianBlur(real, (size_blur, size_blur), 0)
                imag = cv2.GaussianBlur(imag, (size_blur, size_blur), 0)

                phase = np.arctan2(imag, real) + np.pi

                # If selected, remove vortices halfway through the smoothing.
                if remove_vortices and i == smooth//2:
                    phase = image_remove_vortices(phase)
        else:
            real = np.cos(phase)
            imag = np.sin(phase)
            phase = np.arctan2(imag, real) + np.pi

        # Step 3.4: Pattern cleanup.
        if remove_blaze:
            phase = image_remove_blaze(phase, mask=pwr_large)

        # Shift the final phase to minimize the effect of phase wrapping
        # (only matters when projecting patterns with small dynamic range).
        phase = image_reduce_wraps(phase, mask=pwr_large)

        # Add the old phase correction if it's there.
        if (
            "previous_phase_correction" in data and
            data["previous_phase_correction"] is not None
        ):
            phase += data["previous_phase_correction"]

        # Step 4: Data export.
        # Build the final dict.
        wavefront_calibration = {
            "phase": phase,
            "amplitude": amp_large,
            "r2": r2s_large,
            "r2_threshold": r2_threshold,
        }

        # Step 4.1: Load the correction to the SLM
        if apply:
            self.slm.source.update(wavefront_calibration)

        # Plot the result
        if plot:
            self.slm.plot_source(source=wavefront_calibration)

        return wavefront_calibration

    def _wavefront_calibration_superpixel_plot_raw(self, index=0, r2_threshold=0, phase_detail=True):
        """
        Plots raw data from the superpixel-style wavefront calibration. Specifically,
        plots:

        - The location of the point in the camera plane,
        - The measured source phase at each superpixel,
        - The measured source power at each superpixel,
        - The rsquared of the fit at each superpixel.

        Parameters
        ----------
        index : int OR None:
            For multi-point calibrations, the index of the point to plot data for.
            If ``None``, displays a single plot with the location of all indices.
        r2_threshold : float
            Ignores points with fit quality below this threshold.
        phase_detail : bool
            If ``True``, plots the derivatives of the phase instead of the power and rsquared.
        """
        plt.figure(figsize=(16, 8))

        data = self.calibrations["wavefront_superpixel"]

        if index is None:
            coords = data["calibration_points"]

            plt.subplot(1, 4, 1)
            plt.scatter(coords[0,:], coords[1,:], c="r")
            for i in range(coords.shape[1]):
                plt.annotate(str(i), (coords[0, i], coords[1, i]))
            plt.title("Calibration Points")
            plt.xlabel("Camera $x$ [pix]")
            plt.ylabel("Camera $y$ [pix]")
            plt.xlim([0, self.cam.shape[1]])
            plt.ylim([0, self.cam.shape[0]])
            plt.gca().set_aspect(1)

            return

        # Grab all the data
        coord = data["calibration_points"][:, index].copy()
        phase = data["phase"][index, :, :].copy()
        kx = data["kx"][index, :, :].copy()
        ky = data["ky"][index, :, :].copy()
        power = data["power"][index, :, :] / data["normalization"][index, :, :]
        amp = np.sqrt(power)
        r2 = data["r2_fit"][index, :, :].copy()

        # Threshold the data
        below_thresh = r2 < r2_threshold
        phase[below_thresh] = np.nan
        kx[below_thresh] = np.nan
        ky[below_thresh] = np.nan
        amp[below_thresh] = np.nan

        kscale = np.max([np.nanmax(np.abs(kx)), np.nanmax(np.abs(ky))])

        plt.subplot(1, 4, 1)
        plt.scatter(coord[0], coord[1], c="r")
        plt.annotate(str(index), (coord[0], coord[1]))
        plt.title("Calibration Point {}".format(index))
        plt.xlabel("Camera $x$ [pix]")
        plt.ylabel("Camera $y$ [pix]")
        plt.xlim([0, self.cam.shape[1]])
        plt.ylim([0, self.cam.shape[0]])
        plt.gca().set_aspect(1)

        plt.subplot(1, 4, 2)
        plt.imshow(
            phase,
            clim=(0,2*np.pi),
            cmap=plt.get_cmap("twilight"),
            interpolation="none",
        )
        plt.title(r"Phase Correction $\phi$")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 4, 3)
        if phase_detail:
            plt.imshow(
                kx,
                clim=(-kscale, kscale),
                cmap=plt.get_cmap("twilight"),
                interpolation="none",
            )
            plt.title(r"$k_x \propto \partial\phi/\partial x$")
        else:
            plt.imshow(power)
            plt.title("Measured Beam Power")
        plt.xticks([])
        plt.yticks([])

        plt.subplot(1, 4, 4)
        if phase_detail:
            plt.imshow(
                ky,
                clim=(-kscale, kscale),
                cmap=plt.get_cmap("twilight"),
                interpolation="none",
            )
            plt.title(r"$k_y \propto \partial\phi/\partial y$")
        else:
            plt.imshow(r2, clim=(0,1))
        # plt.contour(r2, [r2_threshold], c="r")
            plt.title("$R^2$")
        plt.xticks([])
        plt.yticks([])

        _slmsuite_plt_show(name="wavefront_calibration_superpixel_plot_raw")

