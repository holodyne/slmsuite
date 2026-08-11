"""
Farfield intensity calibration: diffraction efficiency, zeroth-order scatter, and background.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from slmsuite.holography.toolbox import convert_vector, format_2vectors, phase
from slmsuite._plotting import _slmsuite_plt_show

class _FarfieldCalibration(object):
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

        # (1) Capture the zeroth order and scatter with a flat phase pattern.
        self.slm.set_phase(None, settle=True, phase_correct=False)

        self.cam.autoexpose(exposure_bounds_s=(0, 1))
        exposure_zeroth = self.cam.get_exposure()
        self.cam.flush()

        img_zeroth = self.cam.get_image_hdr(exposures)

        # (2) Fill the farfield. Random phase scatters power over all of it and
        # needs no calibration; a hologram concentrates it where the camera looks.
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
                self.cam.autoexpose(window=img_zeroth < (np.median(img_zeroth)+1), exposure_bounds_s=(0, 1))
                exposure_raw = self.cam.get_exposure()
                self.cam.flush()

            images.append(self.cam.get_image_hdr(exposures))

        # (3) Deflect the power into diffracted orders in each of several directions; the
        # per-pixel minimum then sees the camera background wherever the orders miss.
        backgrounds = []
        n_background = 7
        for i in range(n_background):
            vector_base = format_2vectors(
                [np.cos(np.pi * i / n_background), np.sin(np.pi * i / n_background)]
            )

            vector_kxy = convert_vector(
                vector_base / 4,    # Half of the way out in k-space.
                from_units="freq",
                to_units="kxy",
                hardware=self.slm
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
            "saturation": self.cam.bitresolution - (
                self.cam.averaging if self.cam.averaging is not None else 1
            ),
        }
        self.calibrations["farfield"].update(self._get_calibration_metadata())

        self.farfield_calibration_process()

        if plot >= 1:
            self.get_farfield_efficiency(plot=plot)

        return self.calibrations["farfield"]

    def farfield_calibration_process(self, size_blur=True):
        """
        Processes raw :meth:`farfield_calibrate()` data into a usable efficiency map.
        Averages and blurs the speckle of the raw data, less the background taken as the
        per-pixel minimum over the deflected frames.

        Parameters
        ----------
        size_blur : int OR bool
            Amount of blurring to apply to the averaged result, in camera pixels.
            If ``True``, spans the speckle grain given by
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.get_farfield_spot_size()`,
            to a minimum of three pixels.
            Disabled if zero, ``False``, or ``None``.

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

        # Blurring away the residual speckle takes a kernel the size of its grain, which
        # is the spot of the whole SLM rather than of its illuminated part.
        if isinstance(size_blur, (bool, np.bool_)) and size_blur:
            size_blur = max(3, (
                np.max(self.get_farfield_spot_size(
                    slm_size=np.flip(np.squeeze(self.slm.shape)) * np.squeeze(self.slm.pitch),
                    basis="ij",
                ))
                if "fourier" in self.calibrations else 0
            ))
        if size_blur:
            size_blur = 2 * (int(size_blur) // 2) + 1   # cv2 requires odd kernels.

        def blur(image):
            return cv2.GaussianBlur(image, (size_blur, size_blur), 0) if size_blur else image

        # The background is the darkest each pixel gets over the deflected patterns.
        backgrounds = np.asarray(self.calibrations["farfield"]["background_raw"], dtype=float)
        background = blur(np.min(backgrounds, axis=0))
        self.calibrations["farfield"]["background"] = background

        # Process the raw farfield data, less the floor that is not diffracted light.
        raw = np.asarray(self.calibrations["farfield"]["efficiency_raw"], dtype=float)
        speckle = blur(np.mean(raw, axis=0))
        signal = np.maximum(speckle - background, 0)

        # Neither pixels the camera railed on nor the zeroth order, which outshines the
        # pattern it is measuring, say anything about the farfield's brightest spot.
        zeroth = np.asarray(self.calibrations["farfield"]["zeroth"], dtype=float) * (
            self.calibrations["farfield"]["exposure_raw"] /
            self.calibrations["farfield"]["exposure_zeroth"]
        )
        unusable = (
            np.any(raw >= self.calibrations["farfield"]["saturation"], axis=0) | (zeroth > speckle)
        )

        if size_blur:   # The blur spreads unusable power onto its neighbors.
            unusable = cv2.dilate(
                unusable.astype(np.uint8), np.ones((size_blur, size_blur), np.uint8)
            ) > 0

        if np.all(unusable):
            self.logger.warning("Farfield calibration has no usable pixels.")
            unusable = np.zeros_like(unusable)

        peak = np.nanmax(signal[~unusable])

        if peak <= 0:
            self.logger.warning("Farfield calibration measured no light.")
            peak = 1

        self.calibrations["farfield"]["efficiency"] = signal / peak

        # The exposure that would perfectly saturate the brightest spot in the farfield.
        self.calibrations["farfield"]["exposure_saturating"] = (
            self.calibrations["farfield"]["exposure_raw"]
            / (peak / self.calibrations["farfield"]["bitresolution"])
        )

        return self.calibrations["farfield"]["efficiency"]

    def get_farfield_efficiency(
        self,
        fourier_crop=True,
        efficiency_threshold=None,
        zeroth_threshold=None,
        plot=0
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

        efficiency = np.array(self.calibrations["farfield"]["efficiency"])

        if fourier_crop and "fourier" in self.calibrations:
            # efficiency *= self.get_farfield_extent(return_mask=True)
            efficiency = np.where(
                self.get_farfield_extent(return_mask=True),
                efficiency,
                np.nan,
            )

        if zeroth_threshold is not None:
            zeroth = self.get_farfield_zeroth()
            # efficiency *= (zeroth < zeroth_threshold)
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

            # if zeroth_threshold is not None:
            #     plt.contour(
            #         zeroth >= zeroth_threshold,
            #         colors="g",
            #         linewidths=0.5,
            #     )

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
            raise RuntimeError(
                "No processed farfield calibration. Run farfield_calibrate() first."
            )

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
            raise RuntimeError(
                "No processed farfield calibration. Run farfield_calibrate() first."
            )

        return self.calibrations["farfield"]["background"]

    def get_farfield_weights(self, ij, weights=None, floor=0.05):
        """
        Relative amplitudes that make spots at camera positions ``ij`` arrive equally
        bright, from the measured farfield efficiency.

        Parameters
        ----------
        ij : array_like
            Camera positions of the spots, of shape ``(2, N)``.
        weights : array_like OR None
            Target amplitudes of the spots, of shape ``(N,)``.
            If ``None``, even power is assumed.
        floor : float
            Clamps the sampled efficiency, capping how far a dim spot is boosted.

        Returns
        -------
        numpy.ndarray
            Amplitudes of shape ``(N,)``, normalized to a maximum of one.
        """
        efficiency = self.get_farfield_efficiency()
        (h, w) = efficiency.shape

        ij = np.rint(format_2vectors(ij)).astype(int)
        sampled = efficiency[
            np.clip(ij[1], 0, h - 1), np.clip(ij[0], 0, w - 1)
        ].astype(float)

        peak = np.nanmax(efficiency)
        if not peak > 0:
            return np.ones(sampled.shape)

        initial_weights = 1 / np.sqrt(np.maximum(sampled, floor * peak))
        if weights is not None:
            initial_weights *= weights

        return initial_weights / np.max(initial_weights)
