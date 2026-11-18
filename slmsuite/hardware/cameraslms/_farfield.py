"""
Farfield intensity calibration: diffraction efficiency, zeroth-order scatter, background,
and the fill factor of the SLM's pixels.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from slmsuite._plotting import _slmsuite_plt_show
from slmsuite.holography.toolbox import convert_vector, format_2vectors, phase

# Camera pixels sampled when fitting the pixel's fill factor to the efficiency map.
_FILL_FIT_POINTS = 20000
# Bounds of the fit linear fill, the largest residual (relative to the map's peak) of a fit
# worth applying, and the pattern frequency (in cycles per pixel) the camera must reach.
_FILL_FIT_BOUNDS = (0.1, 1)
_FILL_FIT_RESIDUAL = 0.1
_FILL_FIT_REACH = 0.25


class _FarfieldCalibration:
    """
    Hidden superclass with farfield intensity calibration methods.
    """

    def farfield_calibrate(
        self,
        averaging=10,
        exposures=None,
        plot=0,
    ):
        """
        Calibrates the diffraction efficiency (including aperture cropping),
        the zeroth order scatter, and the camera background in the farfield.

        Parameters
        ----------
        averaging : int
            Number of independent speckle realizations to average over.
        exposures : int OR (int, int) OR None
            High dynamic range (HDR) exposures to use for the farfield image, see
            :meth:`~slmsuite.hardware.cameras.camera.Camera.get_image_hdr()`.
        plot : int OR bool
            If ``>= 1``, plots the processed efficiency map.

        Returns
        -------
        dict
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations` ``["farfield"]``
        """
        # Parse arguments.
        averaging = int(averaging)
        if averaging <= 0:
            raise ValueError("averaging must be positive.")

        # Do the full HDR range by default, using powers of 16.
        if exposures is None:
            exposures = (1, 16)

        # (1) The zeroth order and scatter of a flat pattern, corrected so the spot stays sharp.
        self.slm.set_phase(None, settle=True)

        self.cam.autoexpose()
        exposure_zeroth = self.cam.get_exposure()
        self.cam.flush()

        img_zeroth = self.cam.get_image_hdr(exposures)

        # (2) Fill the farfield with the speckle of random phase, which needs no calibration.
        images = []
        exposure_raw = None

        for i in range(averaging):
            # Average speckle.
            self.slm.set_phase(
                np.random.uniform(0, 2 * np.pi, self.slm.shape),
                settle=True,
                phase_correct=False,
            )

            if i == 0:
                # Fix one exposure for the dimmer speckle, windowed off the zeroth order.
                self.cam.autoexpose(window=img_zeroth < (np.median(img_zeroth) + 1))
                exposure_raw = self.cam.get_exposure()
                self.cam.flush()

            images.append(self.cam.get_image_hdr(exposures))

        # (3) Deflect the power in several directions; each pixel is dark in most of them.
        backgrounds = []
        n_background = 7
        for i in range(n_background):
            vector_base = format_2vectors(
                [np.cos(np.pi * i / n_background), np.sin(np.pi * i / n_background)]
            )

            vector_kxy = convert_vector(
                vector_base / 4,  # Half of the way out in k-space.
                from_units="freq",
                to_units="kxy",
                hardware=self.slm,
            )

            self.slm.set_phase(
                phase.blaze(self.slm, vector_kxy),
                settle=True,
                phase_correct=False,
            )
            backgrounds.append(self.cam.get_image_hdr(exposures))

        # Store the calibration results.
        self.calibrations["farfield"] = {
            "zeroth": img_zeroth,
            "exposure_zeroth": exposure_zeroth,
            "efficiency_raw": np.stack(images),
            "background_raw": np.stack(backgrounds),
            "exposure_raw": exposure_raw,
            # Summed frames saturate here, and the camera may be reconfigured before processing.
            "bitresolution": self.cam.bitresolution,
            "saturation": self.cam.bitresolution
            - (self.cam.averaging if self.cam.averaging is not None else 1),
        }
        self.calibrations["farfield"].update(self._get_calibration_metadata())

        self.farfield_calibration_process()

        if plot >= 1:
            self.get_farfield_efficiency(plot=plot)

        return self.calibrations["farfield"]

    def farfield_calibration_process(self, size_blur=True, background_subtract=True, apply=True):
        r"""
        Processes raw :meth:`farfield_calibrate()` data into a usable efficiency map.
        Averages and blurs the speckle of the raw data, less the background taken as the
        per-pixel median over the deflected frames.

        Random phase scatters evenly over every spatial frequency, so the efficiency map
        is the element pattern of the SLM's pixels. With a Fourier calibration, the linear
        fill of the pixels along each axis is fit to it as
        :math:`\text{sinc}^2(f_x p_x k_x)\,\text{sinc}^2(f_y p_y k_y)` and stored under
        ``"fill"``, along with the rms residual of the fit relative to the map's peak
        under ``"fill_residual"``. A fit which ends on its lower bound, leaves a large
        residual, or comes from a camera that views only the center of the first Nyquist
        zone is stored but not applied.

        Parameters
        ----------
        size_blur : int OR bool
            Amount of blurring to apply to the averaged result, in camera pixels.
            If ``True``, spans the speckle grain given by
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.get_farfield_spot_size()`,
            to a minimum of three pixels.
            Disabled if zero, ``False``, or ``None``.
        background_subtract : bool
            Whether to subtract the background from the averaged speckle.
        apply : bool
            Whether to set the :attr:`~slmsuite.hardware.slms.slm.SLM.pixel` of the SLM
            to the fit fill, keeping the reflectance of any pixel it already has.

        Returns
        -------
        numpy.ndarray
            The processed efficiency map, also stored in
            :attr:`~slmsuite.hardware.cameraslms.FourierSLM.calibrations`
            ``["farfield"]["efficiency"]``.
        """
        if "efficiency_raw" not in self.calibrations.get("farfield", {}):
            raise RuntimeError(
                "Could not find raw farfield data, which is not saved to file. "
                "Run farfield_calibrate() first."
            )

        self._check_calibration_frame("farfield", raise_error=True)

        # The speckle grain is the spot of the whole SLM, not of its illuminated part.
        if isinstance(size_blur, (bool, np.bool_)) and size_blur:
            size_blur = max(
                3,
                (
                    np.max(
                        self.get_farfield_spot_size(
                            slm_size=np.flip(np.squeeze(self.slm.shape))
                            * np.squeeze(self.slm.pitch),
                            basis="ij",
                        )
                    )
                    if "fourier" in self.calibrations
                    else 0
                ),
            )
        if size_blur:
            size_blur = 2 * (int(size_blur) // 2) + 1  # cv2 requires odd kernels.

        def blur(image):
            return cv2.GaussianBlur(image, (size_blur, size_blur), 0) if size_blur else image

        # Each pixel is dark in most deflected patterns; the median is unbiased by noise.
        backgrounds = np.asarray(self.calibrations["farfield"]["background_raw"], dtype=float)
        background = blur(np.median(backgrounds, axis=0))
        self.calibrations["farfield"]["background"] = background

        # Process the raw farfield data, less the floor that is not diffracted light.
        raw = np.asarray(self.calibrations["farfield"]["efficiency_raw"], dtype=float)
        signal = blur(np.mean(raw, axis=0))
        if background_subtract:
            signal = np.maximum(signal - background, 0)

        # Saturated pixels and the zeroth order say nothing about the pattern's brightest spot.
        zeroth = blur(
            np.asarray(self.calibrations["farfield"]["zeroth"], dtype=float)
            * (
                self.calibrations["farfield"]["exposure_raw"]
                / self.calibrations["farfield"]["exposure_zeroth"]
            )
        )
        unusable = np.any(raw >= self.calibrations["farfield"]["saturation"], axis=0) | (
            zeroth > signal
        )

        if size_blur:  # The blur spreads unusable power onto its neighbors.
            unusable = (
                cv2.dilate(unusable.astype(np.uint8), np.ones((size_blur, size_blur), np.uint8)) > 0
            )

        if np.all(unusable):
            self.logger.warning("Farfield calibration has no usable pixels.")
            unusable = np.zeros_like(unusable)

        peak = np.nanmax(signal[~unusable])

        if peak <= 0:
            self.logger.warning("Farfield calibration measured no light.")
            peak = 1

        efficiency = self.calibrations["farfield"]["efficiency"] = signal / peak

        if "fourier" in self.calibrations:
            # The pattern frequency each camera pixel sees, in cycles per SLM pixel.
            step = max(1, int(np.sqrt(efficiency.size / _FILL_FIT_POINTS)))
            (rows, columns) = (np.arange(0, n, step) for n in efficiency.shape)
            usable = ~unusable[np.ix_(rows, columns)].ravel()
            ij = np.array(np.meshgrid(columns, rows)).reshape(2, -1)[:, usable]
            freq = (self.ijcam_to_kxyslm(ij) + self.slm._get_source_blaze()[:, None]) * (
                np.reshape(self.slm.pitch, (2, 1))
            )
            measured = efficiency[np.ix_(rows, columns)].ravel()[usable]

            def residual(parameters):
                (scale, fill_x, fill_y) = parameters
                model = scale * np.sinc(fill_x * freq[0]) ** 2 * np.sinc(fill_y * freq[1]) ** 2
                return model - measured

            (low, high) = _FILL_FIT_BOUNDS
            fit = least_squares(
                residual, (1, 0.9, 0.9), bounds=((0, low, low), (np.inf, high, high))
            )
            fill = fit.x[1:]
            fill_residual = float(np.sqrt(np.mean(fit.fun**2)))
            self.calibrations["farfield"]["fill"] = fill
            self.calibrations["farfield"]["fill_residual"] = fill_residual

            trusted = (
                np.all(fill > low * 1.01)
                and fill_residual < _FILL_FIT_RESIDUAL
                and np.all(np.max(np.abs(freq), axis=1) > _FILL_FIT_REACH)
            )
            if apply and not trusted:
                self.logger.warning(
                    "Fit fill %s (residual %.3g) is not well constrained by this farfield; "
                    "leaving the pixel of the SLM as it is.",
                    fill,
                    fill_residual,
                )

            if apply and trusted:
                pixel = self.slm.pixel
                self.slm.pixel = {
                    "fill": tuple(fill),
                    "reflectance": 1 if pixel is None else pixel.reflectance,
                }

        # The exposure that would perfectly saturate the brightest spot in the farfield.
        self.calibrations["farfield"]["exposure_saturating"] = self.calibrations["farfield"][
            "exposure_raw"
        ] / (peak / self.calibrations["farfield"]["bitresolution"])

        return self.calibrations["farfield"]["efficiency"]

    def get_farfield_efficiency(
        self, fourier_crop=True, efficiency_threshold=None, zeroth_threshold=None, plot=0
    ):
        """
        Returns the **measured** efficiency of the farfield **in the coordinates of the camera**:
        the region which the SLM can illuminate, normalized to one at its brightest.
        Pixels excluded from that normalization, such as the zeroth order, can exceed one.
        If a ``efficiency_threshold`` is given, returns instead a boolean mask where the
        efficiency is above the threshold.

        Parameters
        ----------
        fourier_crop : bool
            If ``True``, crops the efficiency map to the Fourier extent of the farfield region
            (the efficiency is set to zero outside this region).
            If Fourier calibration has not been performed, this is ignored.
        efficiency_threshold : float OR None
            If not ``None``, returns a boolean mask where the efficiency is above this threshold.
        zeroth_threshold : float OR None
            If not ``None``, zeros the returned efficiency data or mask
            where the zeroth order is above this threshold.
        plot : int OR bool
            If ``>= 1``, plots the efficiency map and the thresholded support.

        Returns
        -------
        numpy.ndarray of bool OR float
            Mask of shape :attr:`~slmsuite.hardware.cameras.camera.Camera.shape`.
        """
        if "efficiency" not in self.calibrations.get("farfield", {}):
            raise RuntimeError(
                "No processed farfield calibration. Run farfield_calibrate() and "
                "farfield_calibration_process() first."
            )

        # The stored maps are whole camera frames, meaningful only in their own frame.
        self._check_calibration_frame("farfield", raise_error=True)

        efficiency = np.array(self.calibrations["farfield"]["efficiency"])

        if fourier_crop and "fourier" in self.calibrations:
            efficiency = np.where(
                self.get_farfield_extent(return_mask=True),
                efficiency,
                np.nan,
            )

        if zeroth_threshold is not None:
            zeroth = self.get_farfield_zeroth()
            efficiency = np.where(
                (zeroth < zeroth_threshold),
                efficiency,
                np.nan,
            )

        if efficiency_threshold is not None:
            mask = efficiency > efficiency_threshold
        else:
            mask = efficiency

        if plot >= 1:
            plt.figure()
            plt.imshow(efficiency)

            plt.colorbar(label="Relative Efficiency")

            if efficiency_threshold is not None:
                plt.contour(
                    mask,
                    colors="r",
                    linewidths=0.5,
                )

            plt.title("Farfield Efficiency")
            _slmsuite_plt_show("get_farfield_efficiency")

        return mask

    def get_farfield_zeroth(self):
        r"""The zeroth order image, normalized to the peak of the efficiency
        calibration, which spreads power evenly across the farfield.

        As such, this image will generally have components much larger than one.
        For an ideal SLM mapped to a pixel-matched camera with resolution
        :math:`N \times N`, it would be zero everywhere except for a single
        centered pixel of intensity :math:`N^2`.

        Dividing by the number of spots in a desired pattern gives the strength of
        the zeroth order relative to that pattern.

        See :meth:`farfield_calibrate()`.
        """
        if "exposure_saturating" not in self.calibrations.get("farfield", {}):
            raise RuntimeError("No processed farfield calibration. Run farfield_calibrate() first.")

        # The fraction of full scale that the zeroth order would reach at the exposure
        # which saturates the farfield.
        return (
            self.calibrations["farfield"]["zeroth"]
            * self.calibrations["farfield"]["exposure_saturating"]
            / (
                self.calibrations["farfield"]["exposure_zeroth"]
                * self.calibrations["farfield"]["bitresolution"]
            )
        )

    def get_farfield_background(self):
        """Returns the camera background, in raw counts at the farfield exposure and
        blurred alongside the efficiency, from which it is subtracted before normalizing.

        The deflected orders leave the zeroth order in place, so this image still
        contains whatever the SLM scatters there.

        See :meth:`farfield_calibrate()`.
        """
        if "background" not in self.calibrations.get("farfield", {}):
            raise RuntimeError("No processed farfield calibration. Run farfield_calibrate() first.")

        return self.calibrations["farfield"]["background"]
