"""
Abstract functionality for SLMs.
"""

import os
import time

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = np
import inspect
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from slmsuite._plotting import _slmsuite_plt_show
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

from slmsuite import __version__
from slmsuite.hardware._common import _Common
from slmsuite.holography import analysis, toolbox
from slmsuite.misc import fitfunctions
from slmsuite.misc.files import generate_path, latest_path, load_h5, save_h5


LUT_SIZE = 1 << 16      # Default number of entries in a phase lookup table.


def _xp(array):
    """Return cupy if array is a cupy ndarray, else numpy."""
    if cp is not np and isinstance(array, cp.ndarray):
        return cp
    return np


class SLM(_Common, ABC):
    r"""
    Abstract class for SLMs.

    Attributes
    ----------
    name : str
        Name of the SLM.
    shape : (int, int)
        Stores ``(height, width)`` of the SLM in pixels, the same convention as :attr:`numpy.ndarray.shape`.
    bitdepth : int
        Depth of SLM pixel well in bits. This is useful for converting the floats which
        the user provides to the ``bitdepth``-bit ints that the SLM reads (see the
        private method :meth:`_phase2gray`).
    bitresolution : int
        Stores ``2 ** bitdepth``.
    settle_time_s : float
        Delay in seconds to allow the SLM to settle. This is mostly useful for applications
        requiring high precision. This delay is applied if the user flags ``settle``
        in :meth:`set_phase()`. Defaults to .3 sec for precision.
    pitch_um : (float, float)
        Pixel pitch in microns.
    pitch : (float, float)
        Pixel pitch normalized to wavelengths ``pitch_um / wav_um``. This value is more
        useful than ``pitch_um`` when considering conversions to :math:`k`-space.
    wav_um : float
        Operating wavelength targeted by the SLM in microns. Defaults to 780 nm.
    wav_design_um : float
        Design wavelength for which the maximum settable value corresponds to a
        :math:`2\pi` phase shift.
        Defaults to :attr:`wav_um` if passed ``None``.

        Tip
        ~~~
        :attr:`wav_design_um` is useful for using, for instance, an SLM designed
        at 1064 nm for an application at 780 nm by using only a fraction (780/1064)
        of the full dynamic range. It is especially useful for SLMs which do not have builtin
        capability to change their voltage lookup tables (e.g. Thorlabs).
        Even so, the max lookup wavelength (:attr:`wav_design_um`) could be set larger
        than :attr:`wav_um` should the user want to have a phase range larger than
        :math:`2\pi`, for SLMs with lookup table capability.

    phase_scaling : float
        Wavelength normalized to the phase range of the SLM. See :attr:`wav_design_um`.
        Determined by ``phase_scaling = wav_um / wav_design_um``.
    grid : (numpy.ndarray<float> (height, width), numpy.ndarray<float> (height, width))
        :math:`x` and :math:`y` coordinates of the SLM's pixels in wavelengths
        (see :attr:`wav_um`, :attr:`pitch_um`)
        measured from the center of the :attr:`aperture`.
        Of size :attr:`shape`. A read-only property derived from the
        immutable geometric grid and the :attr:`aperture` center.
    aperture : :class:`~slmsuite.holography.toolbox.Aperture`
        Aperture applied to the SLM's nearfield. Set with :meth:`set_aperture`
        or fitted to a measured amplitude with :meth:`fit_aperture`.
    source : dict
        Stores data describing measured, simulated, or estimated properties of the source,
        such as amplitude and phase.
        Typical keys include:

        ``"amplitude"`` : numpy.ndarray
            Source amplitude (with the dimensions of :attr:`shape`) measured on the SLM via
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate()`.
            Also see :meth:`set_source_analytic()` to set without wavefront calibration.

        ``"phase"`` : numpy.ndarray
            Source phase (with the dimensions of :attr:`shape`) measured on the SLM via
            :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate`.
            Also see :meth:`set_source_analytic()` to set without wavefront calibration.

        For a :class:`.SimulatedSLM`, ``"amplitude_sim"`` and ``"phase_sim"`` keywords
        store the true source properties (defined by the user) used to simulate the SLM's
        far-field.

    xp : module
        The array backend, :mod:`numpy` or :mod:`cupy`, that this SLM stores and processes
        data with. Selected by the ``gpu`` argument.
    phase : numpy.ndarray OR cupy.ndarray
        Last displayed data in units of phase (radians), held on :attr:`xp`.
        If wavefront calibration (`phase_correct=True`) is used, this includes the
        calibration data.
    display : numpy.ndarray OR cupy.ndarray
        Last displayed data in discrete SLM units (integers), held on :attr:`xp`. This is
        the data that is actually displayed by the bit-limited hardware. If wavefront
        calibration (`phase_correct=True`) is used, this includes the calibration data.
    gamma : numpy.ndarray OR cupy.ndarray OR None
        Measured phase response, in units of :math:`2\pi`, resampled onto every one of the
        :attr:`bitresolution` grayscale levels. Set by :meth:`set_gamma`.
    lut : numpy.ndarray OR cupy.ndarray OR None
        Lookup table mapping phase onto the grayscale level which realizes it, inverted
        from :attr:`gamma` by :meth:`set_gamma`. If ``None`` (the default), the private
        method :meth:`_phase2gray` assumes the ideal linear phase response instead.
    settle : bool
        Default behavior for the ``settle`` argument of :meth:`set_phase()`. Defaults to ``False``.
    phase_correct : bool
        Default behavior for the ``phase_correct`` argument of :meth:`set_phase()`. Defaults to ``True``.
    """
    _pickle = [
        "name",
        "shape",
        "bitdepth",
        "bitresolution",
        "pitch_um",
        "pitch",
        "settle_time_s",
        "wav_um",
        "wav_design_um",
        "phase_scaling",
        "aperture",
        "phase_correct",
        "settle",
    ]
    _pickle_data = [
        "source",
        "phase",
        "display",
    ]
    _gamma_sign = -1        # Increasing grayscale decreases phase delay; +1 for the reverse.

    @abstractmethod
    def __init__(
        self,
        resolution,
        bitdepth=8,
        name="",
        wav_um=1,
        wav_design_um=None,
        pitch_um=(8,8),
        settle_time_s=0.3,
        gpu=False,
    ):
        """
        Initialize SLM.

        Parameters
        ----------
        resolution : (int, int)
            The width and height of the SLM in ``(width, height)`` form.

            Important
            ~~~~~~~~~
            This is the opposite of the numpy ``(height, width)``
            convention stored in :attr:`shape`.
        bitdepth : int
            See :attr:`bitdepth`. Defaults to 8.
        name : str
            See :attr:`name`.
        wav_um : float
            See :attr:`wav_um`.
        wav_design_um : float or None
            See :attr:`wav_design_um`.
        pitch_um : float OR (float, float)
            See :attr:`pitch_um`. Defaults to 8 micron square pixels.
        settle_time_s : float
            See :attr:`settle_time_s`.
        gpu : bool or None
            Whether to store and process data with :mod:`cupy` (see :attr:`xp`).
            ``None`` uses :mod:`cupy` if it is installed. Defaults to ``False``.
        """
        # Choose the array backend that this SLM will hold all of its data in.
        if gpu is None:
            self.xp = cp
        elif gpu:
            if cp is np:
                raise ImportError("gpu=True requested, but cupy is not installed.")
            self.xp = cp
        else:
            self.xp = np

        # Empty handles for the phase response and its lookup table (see set_gamma).
        self.gamma = None
        self.lut = None

        # Initialize the common hardware attributes.
        _Common.__init__(
            self,
            resolution=resolution,
            bitdepth=bitdepth,
            name=name,
            pitch_um=pitch_um,
            is_slm=True,
        )

        if self.bitdepth > 12:
            self.logger.warning(
                "Bitdepth %s is greater than 12 and some features "
                "(gamma/LUT, etc) may not be supported.", 
                self.bitdepth
            )

        # Phase and display caches for user reference.
        self.phase = self.xp.zeros(self.shape, dtype=np.float32)
        self.display = self.xp.zeros(self.shape, dtype=self.dtype)

        # By default, target wavelength is the design wavelength
        self.wav_um = float(wav_um)
        if wav_design_um is None:
            self.wav_design_um = float(wav_um)
        else:
            self.wav_design_um = float(wav_design_um)

        if not (.3 < self.wav_um < 2):
            self.logger.warning("SLM operation wavelength of %.2f um is unusual. Was this a typo?", self.wav_um)
        if not (.3 < self.wav_design_um < 2):
            self.logger.warning("SLM design wavelength of %.2f um is unusual. Was this a typo?", self.wav_design_um)

        # Make normalized coordinate grids. ``_grid_base`` is the immutable geometric
        # grid (centered on the SLM); the public ``grid`` property derives the
        # aperture-centered working frame from it (see the ``grid`` property).
        height, width = self.shape
        xpix = (width  - 1) * np.linspace(-0.5, 0.5, width)
        ypix = (height - 1) * np.linspace(-0.5, 0.5, height)
        self._grid_base = [
            g.astype(np.float32)
            for g in np.meshgrid(self.pitch[0] * xpix, self.pitch[1] * ypix)
        ]
        self._grid = None            # cache for the aperture-centered working grid
        self._grid_center = None     # aperture center the cache was built for

        # Aperture defaults to "cropped" (circumscribes the whole grid, so it
        # masks nothing until the user sets a real aperture). See set_aperture().
        self.aperture = toolbox.Aperture(self._grid_base, "cropped")

        # Source profile dictionary
        self.source = {}

        # Now inspect the _set_phase_hw() method to see if it supports the execute and
        # block arguments. We need to do this in init because inspect is expensive.
        self._set_phase_hw_args = inspect.signature(self._set_phase_hw).parameters.keys()
        self._set_phase_hw_block = "block" in self._set_phase_hw_args
        self._set_phase_hw_execute = "execute" in self._set_phase_hw_args

        # Time to delay after writing (allows SLM to stabilize).
        self.settle_time_s = float(settle_time_s)

        # Default settle and phase_correct behavior for set_phase.
        self.phase_correct = True
        self.settle = False

    # Aperture-derived properties
    @property
    def grid(self):
        r"""
        :math:`(x, y)` coordinate meshgrids of the SLM's pixels in normalized units
        (wavelengths), measured from the **aperture center**. This is the working
        coordinate frame that analytic phase functions (lenses, gratings, Zernike, ...)
        are generated in.
        """
        center = (
            None if self.aperture.center is None
            else tuple(float(c) for c in self.aperture.center)
        )
        if self._grid is None or self._grid_center != center:
            if center is None:
                self._grid = self._grid_base
            else:
                self._grid = [
                    self._grid_base[0] - center[0],
                    self._grid_base[1] - center[1],
                ]
            self._grid_center = center
        return self._grid

    @property
    def aperture_mask(self):
        """
        Boolean mask (of :attr:`shape`) of the pixels inside :attr:`aperture`.
        """
        return self.aperture.mask

    @property
    def zernike_scaling(self):
        """
        The ``(x_scale, y_scale)`` lateral scaling mapping :attr:`grid` onto the Zernike
        unit disk, from :attr:`aperture`. Used by
        :meth:`~slmsuite.holography.toolbox.phase.zernike_sum`. This is the SLM-level name
        for the general :attr:`~slmsuite.holography.toolbox.Aperture.scale`.
        """
        return self.aperture.scale

    @property
    def source_radius(self):
        r"""
        The source radius in normalized units, for structured beams such as
        :meth:`~slmsuite.holography.toolbox.phase.laguerre_gaussian`. Derived from the
        :attr:`aperture` scaling as :math:`1 / (2\,s)`, where :math:`s` is the (isotropic)
        lateral scale. Raises :class:`ValueError` for an anisotropic (elliptical) aperture,
        which a single radius cannot describe.
        """
        return float(1.0 / (2.0 * self.aperture._isotropic_scale()))

    def _unpickle(self, data):
        """
        Restores the pickled state which :meth:`__init__` does not take: the display
        defaults, the :attr:`aperture`, the measured phase response, and the displayed
        :attr:`phase`. See :meth:`~slmsuite._pickling._Picklable._unpickle`.

        :attr:`phase_scaling` is not restored, as it is derived from the wavelengths
        which the constructor already fixed. Neither are :attr:`gamma` and :attr:`lut`,
        which are rebuilt from the pixel calibration that measured them; see
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM._pixel_calibration_apply_gamma`.
        """
        super()._unpickle(data)

        if "settle_time_s" in data:
            self.settle_time_s = float(data["settle_time_s"])
        if "phase_correct" in data:
            self.phase_correct = bool(data["phase_correct"])
        if "settle" in data:
            self.settle = bool(data["settle"])

        # The pickled center is already in normalized grid units, so build the Aperture
        # directly rather than through set_aperture(), which reads `center` as pixels.
        aperture = data.get("aperture", None)
        if aperture is not None:
            self.aperture = toolbox.Aperture(
                self._grid_base, aperture["spec"], center=aperture.get("center", None)
            )
            self._grid = None

        # Last, as this renders through everything set above. The stored phase is
        # already corrected, so it must not be corrected a second time.
        phase = data.get("phase", None)
        if phase is not None:
            self.set_phase(phase, phase_correct=False, settle=False)

    @abstractmethod
    def close(self):
        """Abstract method to close the SLM and delete related objects."""
        raise NotImplementedError()

    @staticmethod
    def info(verbose=True):
        """
        Abstract method to load display information. Unsupported by this SLM.

        Parameters
        ----------
        verbose : bool
            Whether or not to print display information.

        Returns
        -------
        list
            An empty list.
        """
        if verbose:
            print(".info() NotImplemented.")
        return []

    def load_vendor_phase_correction(self, file_path):
        """
        Loads vendor-provided phase correction from file,
        setting :attr:`~slmsuite.hardware.slms.slm.SLM.source` ``["phase"]``.
        By default, this is interpreted as an image file and is padded or unpadded to
        the shape of the SLM.
        Subclasses should implement vendor-specific routines for loading and
        interpreting the file (e.g. :class:`Santec` loads a .csv).

        Parameters
        ----------
        file_path : str
            File path for the vendor-provided phase correction.

        Returns
        -------
        numpy.ndarray
            :attr:`~slmsuite.hardware.slms.slm.SLM.source` ``["phase"]``,
            the vendor-provided phase correction.
        """
        # Load an invert the image file (see phase sign convention rules in set_phase).
        phase_correction = self.bitresolution - 1 - np.array(Image.open(file_path), dtype=float)

        if phase_correction.ndim != 2:
            raise ValueError("Expected 2D image; found shape {}.".format(phase_correction.shape))

        phase_correction *= 2 * np.pi / (self.phase_scaling * self.bitresolution)

        # Deal with correction shape
        # (this should be made into a toolbox method to supplement pad, unpad)
        file_shape_error = np.sign(np.array(phase_correction.shape) - np.array(self.shape))

        if np.any(np.abs(np.diff(file_shape_error)) > 1):
            raise ValueError(
                "Note sure how to pad or unpad correction shape {} to SLM shape {}.".format(
                    phase_correction.shape, self.shape
                )
            )

        if np.any(file_shape_error > 0):
            self.source["phase"] = toolbox.unpad(phase_correction, self.shape)
        elif np.any(file_shape_error < 0):
            self.source["phase"] = toolbox.pad(phase_correction, self.shape)
        else:
            self.source["phase"] = phase_correction

        return self.source["phase"]

    def _plot_aperture(self, ax):
        """
        Overlay the outline of the current :attr:`aperture` on a pixel-coordinate axis.
        Drawn only if the aperture actually crops the SLM (the default ``"cropped"``
        aperture, whose mask is all-True, draws nothing).
        """
        if not self.aperture.crops:
            return
        mask = np.asarray(self.aperture_mask)
        if not np.all(mask):
            ax.contour(
                mask.astype(float),
                levels=[0.5],
                colors="r",
                linewidths=1,
                linestyles="--",
            )

    def plot(self, phase=None, limits=None, title="Phase", ax=None, cbar=True, aperture=True):
        """
        Plots the provided phase.

        Parameters
        ----------
        phase : ndarray OR None
            Phase to be plotted. If ``None``, grabs the last written :attr:`phase` from the SLM.
        limits : None OR float OR [[float, float], [float, float]]
            Scales the limits by a given factor or uses the passed limits directly.
        title : str
            Title the axis.
        ax : matplotlib.pyplot.axis OR None
            Axis to plot upon.
        cbar : bool
            Also plot a colorbar. Does not work if ``ax`` is passed.
        aperture : bool
            If ``True`` (default), overlay the outline of the current :attr:`aperture`
            (when it crops the SLM).

        Returns
        -------
        matplotlib.pyplot.axis
            Axis of the plotted phase.
        """
        if phase is None:
            phase = self.phase
        if _xp(phase) is not np:
            phase = phase.get()
        phase = np.array(phase, copy=(False if np.__version__[0] == '1' else None))
        phase = np.mod(phase, 2*np.pi) / np.pi

        (ax, cax, should_show) = self._plot(
            phase, limits, title, ax=ax, cbar=cbar,
            labels=("SLM $n$ [pix]", "SLM $m$ [pix]"),
            clim=[0, 2], cmap="twilight", interpolation="none",
        )

        if cax is not None:
            ticks = [0,1,2]
            cax.set_yticks(ticks)
            cax.set_yticklabels([f"${t}\\pi$" for t in ticks])

        if aperture and phase.shape == self.shape:
            self._plot_aperture(ax)

        if should_show:
            _slmsuite_plt_show(name="slm_plot")

        return ax

    @property
    def pitch(self):
        return self.pitch_um / self.wav_um

    # Phase scaling and LUT methods

    @property
    def phase_scaling(self):
        return self.wav_um / self.wav_design_um

    def interpolate_gamma(self, gamma, levels):
        r"""
        Interpolates a phase response measured at some ``levels`` onto all
        :attr:`bitresolution` grayscale levels, as :meth:`set_gamma` requires. Levels
        outside the sampled range are closed circularly, at the curve's average slope.

        Parameters
        ----------
        gamma : array_like
            Measured phase response, in units of :math:`2\pi`. See :meth:`set_gamma`.
        levels : array_like
            The grayscale levels at which ``gamma`` was sampled, in any order.

        Returns
        -------
        numpy.ndarray
            ``gamma`` sampled at every grayscale level.
        """
        bitresolution = self.bitresolution
        gamma = np.ravel(np.array(gamma, dtype=float))
        levels = np.ravel(np.array(levels, dtype=float))

        if len(levels) != len(gamma):
            raise ValueError(
                f"Expected {len(gamma)} levels to pair with gamma; got {len(levels)}."
            )

        order = np.argsort(levels)
        (levels, gamma) = (levels[order], gamma[order])

        if len(gamma) < 2 or levels[0] == levels[-1]:
            raise ValueError("Expected gamma to sample at least two distinct levels.")
        if levels[-1] - levels[0] >= bitresolution:
            raise ValueError(
                f"Expected levels to span less than the bitresolution {bitresolution}; "
                f"got {levels[0]} to {levels[-1]}."
            )

        span = (gamma[-1] - gamma[0]) * bitresolution / (levels[-1] - levels[0])

        return np.interp(
            np.arange(bitresolution),
            np.concatenate(([levels[-1] - bitresolution], levels, [levels[0] + bitresolution])),
            np.concatenate(([gamma[-1] - span], gamma, [gamma[0] + span])),
        )

    def set_gamma(self, gamma=None, lut_size=LUT_SIZE):
        r"""
        Sets :attr:`lut`, the lookup table mapping a desired phase onto the grayscale
        level which best realizes it, from a measured phase response ``gamma``.
        Measure ``gamma`` with
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.pixel_calibration_process`.
        Its sweep projects grayscale levels directly, so a previously-set :attr:`lut`
        does not disturb the measurement.

        Note
        ~~~~
        The lookup table supersedes :attr:`phase_scaling`, as a measured ``gamma`` already
        contains the phase range which the SLM achieves at :attr:`wav_um`. Phases outside
        this range are mapped to the nearest phase which the SLM can realize.

        Parameters
        ----------
        gamma : array_like OR None
            Magnitude of the phase realized by each of the :attr:`bitresolution` grayscale
            levels, in units of :math:`2\pi`, increasing with level for the usual SLM. Its
            sense is set per class, as most SLMs decrease phase delay with increasing level
            whereas a :class:`~slmsuite.hardware.slms.texasinstruments.PLM` increases it.
            Must be unwrapped, as an SLM with more than :math:`2\pi` of range spans more
            than one unit. Pass a measurement of only some levels through
            :meth:`interpolate_gamma` first. ``None`` clears :attr:`lut`, restoring the
            ideal linear response.
        lut_size : int
            Number of entries in :attr:`lut`. Must be a power of two.

        Returns
        -------
        numpy.ndarray OR cupy.ndarray OR None
            :attr:`lut`.
        """
        if gamma is None:
            self.gamma = self.lut = None
            return self.lut

        if lut_size < 1 or lut_size & (lut_size - 1):
            raise ValueError(f"Expected lut_size {lut_size} to be a positive power of two.")

        if _xp(gamma) is not np:
            gamma = gamma.get()
        gamma = np.ravel(np.array(gamma, dtype=float))

        if len(gamma) != self.bitresolution:
            raise ValueError(
                f"Expected gamma to span all {self.bitresolution} levels; got {len(gamma)}."
            )
        if not np.all(np.isfinite(gamma)):
            raise ValueError("Expected finite gamma; a degenerate fit may produce nan.")

        if self.phase_scaling != 1:
            self.logger.warning(
                "The gamma lookup table supersedes this SLM's phase_scaling of %s.",
                self.phase_scaling,
            )

        # Phase realized by each level, sorted and tiled to bucket phase circularly.
        phase = np.mod(self._gamma_sign * gamma * 2 * np.pi, 2 * np.pi)
        ranking = np.argsort(phase)
        tiled = np.concatenate(
            (phase[ranking] - 2*np.pi, phase[ranking], phase[ranking] + 2*np.pi)
        )

        # Assign each of the lut_size uniformly spaced phases to the nearest level.
        grid = np.arange(lut_size) * (2 * np.pi / lut_size)
        index = np.searchsorted((tiled[:-1] + tiled[1:]) / 2, grid, side="right")

        self.gamma = self.xp.asarray(gamma)
        self.lut = self.xp.asarray(ranking[np.mod(index, self.bitresolution)].astype(self.dtype))

        return self.lut

    @property
    def _phase_to_lut(self):
        """Scale factor from phase in radians onto an index of :attr:`lut`."""
        return np.float64(self.lut.size / (2 * np.pi))

    def _phase2lut(self, phase):
        r"""
        Helper function to index :attr:`lut` with phase in radians, wrapping modulo
        :math:`2\pi`.
        """
        xp = self.xp
        return xp.floor(phase * self._phase_to_lut).astype(xp.int64) & (self.lut.size - 1)

    def _phase2gray(self, phase, out=None):
        r"""
        Helper function to convert an array of phases (units of :math:`2\pi`) to an array of
        :attr:`~slmsuite.hardware.slms.slm.SLM.bitresolution` -scaled and -cropped integers.
        This is used by :meth:`set_phase()`. See special cases described in :meth:`set_phase()`.
        If :attr:`lut` is set, the conversion is a lookup into the measured phase response
        rather than the ideal linear scaling.

        Parameters
        ----------
        phase : numpy.ndarray or cupy.ndarray
            Array of phases in radians.
        out : numpy.ndarray or cupy.ndarray
            Array to store integer values scaled to SLM voltage, i.e. for in-place
            operations.
            If ``None``, an appropriate array will be allocated.

        Returns
        -------
        out
        """
        xp = self.xp

        if out is None:
            out = xp.zeros(self.shape, dtype=self.dtype)

        if self.lut is not None:
            return xp.take(self.lut, self._phase2lut(phase), out=out)

        if self.phase_scaling == 1:
            # Prepare the 2pi -> integer conversion factor and convert.
            factor = self._gamma_sign * (self.bitresolution / 2 / np.pi)
            phase *= factor

            # Cast via signed integers; cupy clamps negative floats cast to unsigned.
            xp.rint(phase, out=phase)
            index = phase.astype(xp.int32)

            # Restore phase (usually self.phase) as these operations are in-place.
            phase *= 1 / factor

            # Shift by one so that phase=0 --> display=max. That way, phase will be more continuous.
            index -= 1

            # This implements modulo much faster than xp.mod().
            if self.bitresolution & (self.bitresolution - 1) == 0:
                active_bits_mask = int(self.bitresolution - 1)
                xp.bitwise_and(index, active_bits_mask, out=index)
            else:
                # Slow backup using xp.mod().
                xp.mod(index, self.bitresolution, out=index)

            # Copy and cast the data to the output (usually self.display)
            xp.copyto(out, index, casting="unsafe")
        else:
            # The bounds and truncation below are written for decreasing phase delay.
            if self._gamma_sign != -1:
                raise NotImplementedError(
                    "An SLM with increasing phase delay must use a gamma lookup table "
                    "(see set_gamma) when phase_scaling is not one."
                )

            # phase_scaling is not included in the scaling.
            factor = -(self.bitresolution * self.phase_scaling / 2 / np.pi)
            phase *= factor

            # Only if necessary, modulo the phase to remain within SLM bounds.
            if xp.amin(phase) <= -self.bitresolution or xp.amax(phase) > 0:
                # Minus 1 is to conform with the in-bound case.
                phase -= 1
                # xp.mod is the slowest step. It could maybe be faster if phase is converted to
                # an integer beforehand, but there is an amount of risk for overflow.
                # For instance, a standard double can represent numbers far larger than
                # even a 64 bit integer. If this optimization is implemented, take care to
                # generate checks for the conversion to long integer / etc before the final
                # conversion to dtype of uint8 or uint16.
                xp.mod(phase, self.bitresolution * self.phase_scaling, out=phase)
                phase += self.bitresolution * (1 - self.phase_scaling)

                # Set values still out of range to the maximum.
                if self.phase_scaling > 1:
                    phase[phase < 0] = self.bitresolution - 1
            else:
                # Go from negative to positive.
                phase += self.bitresolution - 1

            # Copy and cast the data to the output (usually self.display)
            xp.copyto(out, phase, casting="unsafe")

            # Restore phase (though we do not unmodulo)
            phase *= 1 / factor

        return out

    # Writing methods

    @abstractmethod
    def _set_phase_hw(self, display):
        """
        Low-level hardware interface to project integer data onto the SLM.
        When the user calls the :meth:`.SLM.set_phase` method of
        :class:`.SLM`, the ``phase`` argument is error-checked and processed into
        the integer array ``display`` that is passed to :meth:`_set_phase_hw()`.
        When integer data is passed to :meth:`set_phase` instead of floating point, it
        is passed directly to :meth:`_set_phase_hw()` as ``display``.
        We call this parameter ``display`` to distinguish it from the (potentially)
        floating point ``phase`` parameter of :meth:`set_phase`.

        Parameters
        ----------
        display
            Integer data to display on the SLM.
        """
        raise NotImplementedError("SLM subclasses must implement _set_phase_hw().")

    def _format_phase_hw(self, phase):
        """
        Formats the phase data for hardware-specific requirements prior to calling
        :meth:`_set_phase_hw`. By default, performs grayscale conversion via
        :meth:`_phase2gray`. Override in subclasses for custom formatting
        (e.g. converting phase to an electrode bitmap for :class:`.texasinstruments.PLM`).

        Parameters
        ----------
        phase : numpy.ndarray
             See :meth:`set_phase`.

        Returns
        -------
        numpy.ndarray
            Formatted phase data for :meth:`_set_phase_hw`.
        """
        return self._phase2gray(phase, out=self.display)

    def _gray2display(self, gray):
        """
        Helper function to send integer data to a format understood by the SLM.
        For most SLMs, this is a no-op, but for some SLMs (e.g. :class:`.texasinstruments.PLM`),
        this is a more complicated step to convert from grayscale to an electrode bitmap.
        """
        self.xp.copyto(self.display, gray)
        return self.display

    def set_phase(
        self,
        phase,
        phase_correct: bool = None,
        settle: bool = None,
        execute: bool = None,
        block: bool = None,
        **kwargs
    ):
        r"""
        Checks, cleans, and adds to data, then sends the data to the SLM and
        potentially waits for ``settle_time_s`` seconds. This method calls the
        SLM-specific private method :meth:`_format_phase_hw()` (if implemented)
        to format the phase data before calling :meth:`_set_phase_hw()` which
        transfers the data to the SLM.

        Warning
        ~~~~~~~
        Subclasses implementing vendor-specific software *should not* overwrite this
        method. Subclasses *should* overwrite :meth:`_set_phase_hw()` (and
        :meth:`_format_phase_hw()` if required) instead.

        Caution
        ~~~~~~~
        The sign on ``phase`` is flipped before converting to integer data.
        This is to convert between the 'increasing value ==> increasing voltage
        (= decreasing phase delay)' convention in most SLMs and
        :mod:`slmsuite`'s 'increasing value ==> increasing phase delay'
        convention. As a result, zero phase will appear entirely white (255 for
        an 8-bit SLM), and increasing phase will darken the displayed pattern.
        If integer data is passed, this data is displayed directly and the sign
        is *not* flipped.

        Important
        ~~~~~~~~~
        The user does not need to wrap (e.g. :mod:`numpy.mod(data,
        2*numpy.pi)`) the passed phase data, unless they are pre-caching data
        for speed (see below). :meth:`.set_phase()` uses optimized routines to
        wrap the phase (see the private method :meth:`_phase2gray()`). When a
        measured phase response is loaded with :meth:`set_gamma()`, phase is
        instead wrapped and converted by lookup into :attr:`lut`. Otherwise,
        which routine is used depends on :attr:`phase_scaling`:

        -  :attr:`phase_scaling` is one.
            Fast bitwise integer modulo is used. Much faster than the other
            routines which depend on :meth:`numpy.mod()`.

        -  :attr:`phase_scaling` is less than one.
            In this case, the SLM has **more phase tuning range** than
            necessary. If the data is within the SLM range ``[0,
            2*pi/phase_scaling]``, then the data is passed directly. Otherwise,
            the data is wrapped by :math:`2\pi` using the very slow
            :meth:`numpy.mod()`. Try to avoid this in applications where speed
            is important.

        -  :attr:`phase_scaling` is more than one.
            In this case, the SLM has **less phase tuning range** than
            necessary. Processed the same way as the :attr:`phase_scaling` is
            less than one case, with the important exception that phases (after
            wrapping) between ``2*pi/phase_scaling`` and ``2*pi`` are set to
            zero. For instance, a sawtooth blaze would be truncated at the
            tips.

        Caution
        ~~~~~~~
        After scale conversion, data is ``floor()`` ed to integers with
        ``np.copyto``, rather than rounded to the nearest integer
        (``np.rint()`` equivalent). While this is irrelevant for the average
        user, it may be significant in some cases. If this behavior is
        undesired consider either: :meth:`set_phase()` integer data directly or
        modifying the behavior of the private method :meth:`_phase2gray()` in a
        pull request. We have not been able to find an example of ``np.copyto``
        producing undesired behavior, but will change this if such behavior is
        found.

        Parameters
        ----------
        phase : numpy.ndarray OR cupy.ndarray OR
                slmsuite.holography.algorithms.Hologram OR None
            Phase data to display in units of :math:`2\pi`, unless the passed
            data is of integer type and the data is applied directly.

            -  If ``None`` is passed to :meth:`.set_phase()`, data is zeroed.
            -  If a :class:`~slmsuite.holography.algorithms.Hologram` is passed,
               the phase is grabbed from
               :meth:`~slmsuite.holography.algorithms.Hologram.get_phase()`.
            -  If the array has a larger shape than the SLM shape, then the data is
               cropped to size in a centered manner
               (:meth:`~slmsuite.holography.toolbox.unpad`).
            -  If integer data is passed with the same type as :attr:`display`
               (``np.uint8`` for <=8-bit SLMs, ``np.uint16`` otherwise),
               then this data is **directly** passed to the
               SLM, without going through the "phase delay to grayscale" conversion
               defined in the private method :meth:`_phase2gray`. In this situation,
               ``phase_correct`` is **ignored**.
               This is error-checked such that bits with greater significance than the
               bitdepth of the SLM are zero (e.g. the final 6 bits of 16 bit data for a
               10-bit SLM). Integer data with type different from :attr:`display` leads
               to a TypeError.

            Usually, an **exact** stored copy of the data passed by the user
            under ``phase`` is stored in the attribute :attr:`phase`. However,
            in cases where :attr:`phase_scaling` is not one, this copy is modified
            to include how the data was wrapped. If the data was cropped, then
            the cropped data is stored, etc. If integer data was passed, the
            equivalent floating point phase is computed and stored in the
            attribute :attr:`phase`.
        phase_correct : bool OR None
            Whether to add wavefront correction to the pattern. This correction
            is stored in
            :attr:`~slmsuite.hardware.slms.slm.SLM.source` ``["phase"]``. If
            ``None``, defaults to :attr:`phase_correct` (which defaults to
            ``True``).
        settle : bool OR None
            Whether to sleep for
            :attr:`~slmsuite.hardware.slms.slm.SLM.settle_time_s`. If ``None``,
            defaults to :attr:`settle` (which defaults to ``False``).
            If ``block=False``, this parameter is ignored.
        execute : bool OR None
            Whether to actually send the image to the SLM. Most SLMs do not
            support this feature, and will error if ``execute`` is not
            ``None``. Otherwise, ``None`` must default to ``True``. Use case:
            if ``execute=False`` and ``block=True``, only the block is enforced
            and no new data is written.

            Important
            ~~~~~~~~~
            New phase/display data is always
            calculated regardless of the value of ``execute``.
        block : bool OR None
            Some SLM subclasses support non-blocking writes that are triggered
            externally. This parameter will determine whether to block the
            thread until the image is fully written. Most SLMs do not support
            this feature, and will error if ``block`` is not ``None``.
            Otherwise, ``None`` must default to ``True``. Use case: if
            ``execute=True`` and ``block=False``, the write is non-blocking.
        **kwargs
            Passed to the SLM in case the subclass needs to do something
            special. For instance, some SLMs support a ``timeout`` parameter
            that determines how long to wait for the SLM commands to execute
            before raising an error.

        Returns
        -------
        numpy.ndarray
           :attr:`~slmsuite.hardware.slms.slm.SLM.display`, the integer data
           sent to the SLM.

        Raises
        ------
        TypeError
            If integer data is incompatible with the bitdepth or if the passed
            phase is otherwise incompatible (not a 2D array or smaller than the
            SLM shape, etc).
        """
        # Parse execute and block arguments.
        if execute is not None:
            if self._set_phase_hw_execute:
                kwargs["execute"] = bool(execute)
            else:
                raise ValueError(
                    "This SLM does not support the execute argument in set_phase."
                )

        if block is None:
            block = True
        else:
            if self._set_phase_hw_block:
                kwargs["block"] = bool(block)
            else:
                raise ValueError(
                    "This SLM does not support the block argument in set_phase."
                )

        # Start a counter here for the settle time blocking.
        t0 = time.perf_counter()

        # Parse phase.
        if hasattr(phase, "get_phase"):
            # If we passed a hologram, grab the phase from there.
            phase = phase.get_phase()

        xp = self.xp

        if phase is None:
            # Zero the phase pattern.
            self.phase.fill(0)
        else:
            # Move the data onto this SLM's backend; numpy cannot read GPU memory.
            if xp is np and _xp(phase) is not np:
                phase = phase.get()
            phase = xp.asarray(phase)

        # Pass integer data directly to the SLM (no quantize/wrapping).
        if phase is not None and np.issubdtype(phase.dtype, np.integer):
            # First, check the type.
            if phase.dtype != self.dtype:
                raise TypeError(
                    f"Unexpected integer type {phase.dtype}. Expected {self.dtype}."
                )

            # If integer data was passed, check that we are not out of range.
            if xp.any(phase >= self.bitresolution):
                raise TypeError(
                    f"Integer data must be within the bitdepth ({self.bitdepth}-bit) of the SLM."
                )

            # Unpad if necessary.
            if phase.shape != self.shape:
                phase = toolbox.unpad(phase, self.shape)

            # Send the data to self.display.
            self.display = self._gray2display(phase)

            # Update the phase variable with the integer data that we displayed.
            if self.gamma is None:
                realized = phase * (self._gamma_sign * 2 * np.pi
                                    / self.phase_scaling / self.bitresolution)
            else:
                realized = self._gamma_sign * 2 * np.pi * self.gamma[phase]
            xp.copyto(self.phase, xp.mod(realized, 2 * np.pi))
        else:
            # If float data was passed (or the None case).
            # Unpad if necessary.
            if phase is not None:
                if phase.shape != self.shape:
                    phase = toolbox.unpad(phase, self.shape)

                # Copy the data to self.phase.
                xp.copyto(self.phase, phase)

            # Add phase correction if requested.
            if phase_correct is None:
                phase_correct = self.phase_correct
            if phase_correct and ("phase" in self.source):
                self.phase += xp.asarray(self._get_source_phase())

            # Pass the data to self.display.
            # Turn the floats in phase space to integer data for the SLM.
            self.display = self._format_phase_hw(self.phase)

        # Write!
        self._set_phase_hw(self.display, **kwargs)

        # For accurate settle, reset the time to be after the data has actually been sent to the SLM.
        t0 = time.perf_counter()

        # Maybe some of that time will be spent rendering the data in the viewer...
        if self.viewer is not None:
            phase = self.phase if self.xp is np else self.phase.get()
            factor = (self.phase_scaling * self.bitresolution / (2 * np.pi))
            self.viewer.render((phase * factor).astype(self.dtype))

        # Optional delay.
        if settle is None:
            settle = self.settle
        if block and settle:
            time_elapsed = time.perf_counter() - t0
            time_remaining = self.settle_time_s - time_elapsed
            if time_remaining > 0:
                time.sleep(time_remaining)

        return self.display

    def write(
        self,
        phase,
        phase_correct=True,
        settle=False,
        **kwargs,
    ):
        "Backwards-compatibility alias for :meth:`set_phase()`."
        warnings.warn(
            "The backwards-compatible alias SLM.write will be depreciated "
            "in favor of SLM.set_phase in a future release."
        )

        self.set_phase(phase, phase_correct, settle, **kwargs)

    # File saving methods

    def save_phase(self, path=".", name=None):
        """
        Saves :attr:`~slmsuite.hardware.slms.slm.SLM.phase` and
        :attr:`~slmsuite.hardware.slms.slm.SLM.display`
        to a file like ``"path/name_id.h5"``.

        Parameters
        ----------
        path : str
            Path to directory to save in. Default is current directory.
        name : str OR None
            Name of the save file. If ``None``, will use :attr:`name` + ``'-phase'``.

        Returns
        -------
        str
            The file path that the phase was saved to.
        """
        if name is None:
            name = self.name + '_phase'
        file_path = generate_path(path, name, extension="h5")
        save_h5(
            file_path,
            {
                "__version__" : __version__,
                "phase" : self.phase,
                "display" : self.display,
            }
        )

        self.logger.info("Saved phase to '%s'.", file_path)

        return file_path

    def load_phase(self, file_path=None, settle=False):
        """
        Loads :attr:`~slmsuite.hardware.slms.slm.SLM.display`
        from a file and writes to the SLM.

        Parameters
        ----------
        file_path : str OR None
            Full path to the phase file. If ``None``, will
            search the current directory for a file with a name like
            :attr:`name` + ``'-phase'``.
        settle : bool
            Whether to sleep for :attr:`~slmsuite.hardware.slms.slm.SLM.settle_time_s`.

        Returns
        -------
        str
            The file path that the phase was loaded from.

        Raises
        ------
        FileNotFoundError
            If a file is not found.
        Warning
            Warns the user if the stored
            :attr:`~slmsuite.hardware.slms.slm.SLM.phase`
            does not agree with the displayed value.
        """
        if file_path is None:
            path = os.path.abspath(".")
            name = self.name + '_phase'
            file_path = latest_path(path, name, extension="h5")
            if file_path is None:
                raise FileNotFoundError(
                    "Unable to find a phase file like\n{}"
                    "".format(os.path.join(path, name))
                )

        data = load_h5(file_path)

        display = self.xp.asarray(data["display"])
        self.phase = self.xp.asarray(data["phase"], dtype=np.float32)

        self._set_phase_hw(display)

        self.logger.info("Loaded phase from '%s'.", file_path)

        # Verify the file's display against one recomputed from its phase.
        if not self.xp.all(self.xp.isclose(display, self._format_phase_hw(self.phase))):
            self.logger.warning("Integer data in 'display' does not match 'phase' for this SLM.")

        self.display = display

        # Optional delay.
        if settle:
            time.sleep(self.settle_time_s)

        return file_path

    # Triggering

    def set_input_trigger(self, on : bool = False):
        r"""
        **(Not supported by this SLM.)**
        Configures the input trigger of the SLM, where an external electronic signal can
        synchronize the time at which the SLM updates its display.

        Parameters
        ----------
        on : bool
            Subclasses *must* support a boolean configuration argument, but can
            also accept other datatypes or parameters as needed.
        """
        raise NotImplementedError("This SLM does not support input triggering.")

    def set_output_trigger(self, on : bool = False):
        r"""
        **(Not supported by this SLM.)**
        Configures the output trigger of the SLM, where the SLM can send an electronic
        signal upon updating its display.

        Parameters
        ----------
        on : bool
            Subclasses *must* support a boolean configuration argument, but can
            also accept other datatypes or parameters as needed.
        """
        raise NotImplementedError("This SLM does not support output triggering.")

    # Segmentation

    def segment(
        self,
        shape,
    ):
        """
        Splits the area of the SLM into a number of segments.

        Parameters
        ----------
        shape : int OR (int, int)
            Segmentation pattern in ``(rows, columns)``.
            If a single integer is passed, this is assumed to be the number of columns,
            i.e. ``(1, shape)``.
        """
        # Parse shape
        if np.isscalar(shape):
            shape = int(np.rint(shape))
            shape = (1, shape)

        shape = toolbox.format_shape(shape)

        # Get width and height of segments in pixels.
        h, w = segment_shape = [s // p for s, p in zip(self.shape, shape)]

        # Shift the grid so that extra area is on the edges.
        y0, x0 = [((s - p * sp) // 2) for s, p, sp in zip(self.shape, shape, segment_shape)]

        # Import here to avoid circular imports.
        from slmsuite.hardware.slms.segmented import SegmentedSLM

        # Now make all the children and return.
        children = []

        for xi in range(shape[1]):
            for yi in range(shape[0]):
                x = x0 + xi * w
                y = y0 + yi * h

                # The last SLM should handle updates by default.
                child = SegmentedSLM(
                    parent=self,
                    window=(x, w, y, h),
                    name=f"{self.name}_segment_{x}_{y}",
                    refresh=(xi == shape[1] - 1 and yi == shape[0] - 1)
                )

                children.append(child)

        return children

    # Source and calibration methods

    def set_source_analytic(
            self,
            fit_function="gaussian2d",
            units="norm",
            phase_offset=0,
            sim=False,
            **kwargs
        ):
        """
        In the absence of a proper wavefront calibration, sets
        :attr:`~slmsuite.hardware.slms.slm.SLM.source` amplitude and phase using a
        ``fit_function`` from :mod:`~slmsuite.holography.analysis.fitfunctions`.

        Note
        ~~~~
        :class:`~slmsuite.hardware.cameraslms.FourierSLM` includes
        capabilities for wavefront calibration via
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate`.
        This process also measures the amplitude of the source on the SLM
        and stores this in :attr:`source`. :attr:`source` keywords
        are also used for better refinement of holograms during numerical
        optimization. If unable to run
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.wavefront_calibrate`,
        this method allows the user to set an approximation of the complex source.

        Parameters
        ----------
        fit_function : str OR lambda
            Function name from :mod:`~slmsuite.misc.fitfunctions` used to set the
            source profile. The function can also be passed directly.
            Defaults to ``"gaussian2d"``.
        units : str in {"norm", "frac", "nm", "um", "mm", "m"}
            Units for the :math:`(x,y)` grid passed to ``fit_function``. This essentially
            determines the scaling on the normalized grid stored in the SLM which is
            passed to the ``fit_function``.
        sim : bool
            Sets the simulated source distribution if ``True`` or the approximate
            experimental source distribution (in absence of wavefront calibration)
            if ``False``.
        phase_offset : float OR numpy.ndarray
            Additional phase (of shape :attr:`shape`) added to :attr:`source`.
        **kwargs
            Arguments passed to ``fit_function`` in addition to the SLM grid in the
            requested ``units``. If the ``fit_function`` is ``"gaussian2d"`` and no
            keyword arguments have been passed, the radius defaults to 1/2 of the
            smaller of the two SLM dimensions.

        Returns
        --------
        dict
            :attr:`~slmsuite.hardware.slms.slm.SLM.source`.
        """
        # Wavelength normalized
        if units == "norm":
            scaling = (1,1)
        # Fractions of the display
        elif units == "frac":
            scaling = [g.max() - g.min() for g in self.grid]
        # Physical units
        else:
            if units in toolbox.LENGTH_FACTORS.keys():
                factor = toolbox.LENGTH_FACTORS[units]
            else:
                raise RuntimeError("Did not recognize units '{}'".format(units))
            scaling = [factor / self.wav_um, factor / self.wav_um]

        xy = [g / s for g,s in zip(self.grid, scaling)]

        if len(kwargs) == 0 and isinstance(fit_function, str) and fit_function == "gaussian2d":
            w = np.min([np.amax(xy[0]), np.amax(xy[1])]) / 2
            kwargs = {"x0" : 0, "y0" : 0, "a" : 1, "c" : 0, "wx" : w, "wy" : w}

        if isinstance(fit_function, str):
            fit_function = getattr(fitfunctions, fit_function)

        source = fit_function(xy, **kwargs)

        self.source["amplitude_sim" if sim else "amplitude"] = np.abs(source)
        self.source["phase_sim" if sim else "phase"] = np.angle(source) + phase_offset

        return self.source

    def _center_pix_to_norm(self, center_pix):
        """Convert a ``(x, y)`` pixel coordinate to the grid's normalized units."""
        center_pix = np.array(center_pix, dtype=float).ravel()
        return self.pitch * (center_pix - (np.flip(self.shape) - 1) / 2.0)

    def _length_to_norm(self, length, units):
        """Convert a scalar ``length`` in ``units`` to the grid's normalized units."""
        if units == "norm":
            factor = 1.0
        elif units == "frac":
            # Fraction of the half-extent (the smaller of the two half-dimensions).
            factor = float(np.min([
                np.nanmax(self._grid_base[0]), np.nanmax(self._grid_base[1])
            ]))
        elif units in toolbox.LENGTH_FACTORS:
            factor = toolbox.LENGTH_FACTORS[units] / self.wav_um
        else:
            raise RuntimeError("Did not recognize units '{}'".format(units))
        return length * factor

    def set_aperture(self, spec=None, *, radius=None, center=None, units="norm"):
        r"""
        Sets the SLM's :attr:`aperture` which defines the working
        coordinate frame (:attr:`grid` centering), the Zernike lateral scaling
        (:attr:`zernike_scaling`), the in-use mask (:attr:`aperture_mask`), and the
        effective (masked) source amplitude and phase.

        Setting an aperture **always** applies it to the source amplitude and phase: the
        region outside the aperture is masked off everywhere the source is used (e.g.
        holography, :meth:`set_phase`'s wavefront correction). The default aperture
        (``"cropped"``) circumscribes the whole grid and so masks nothing.

        Parameters
        ----------
        spec : :class:`~slmsuite.holography.toolbox.Aperture` OR spec OR None
            The aperture shape/scaling, as accepted by
            :class:`~slmsuite.holography.toolbox.Aperture`
            (``"circular"`` / ``"elliptical"`` / ``"cropped"`` / ``float`` /
            ``(float, float)`` / an :class:`~slmsuite.holography.toolbox.Aperture`).
            Mutually exclusive with ``radius``. If both ``spec`` and ``radius`` are
            ``None``, the current aperture's spec is kept.
        radius : float OR None
            Shorthand for a circular aperture of the given source (:math:`1/e`) radius,
            interpreted in ``units``. The aperture/pupil itself extends to twice this
            radius (the lateral scaling is :math:`1 / (2\,r)`), matching
            :attr:`source_radius`.
        center : (float, float) OR None
            The ``(x, y)`` pixel the aperture is centered on. ``None`` (the default)
            centers the aperture on the geometric center of the SLM.
        units : str
            Units for ``radius``: ``"norm"`` (normalized to wavelengths, the default),
            ``"frac"`` (fraction of the half-extent of the SLM), or a physical length
            (``"um"``, ``"mm"``, ...).

        Returns
        -------
        :class:`~slmsuite.holography.toolbox.Aperture`
            The new :attr:`aperture`.
        """
        if radius is not None:
            if spec is not None:
                raise ValueError("Provide either spec or radius, not both.")
            spec = 1.0 / (2.0 * self._length_to_norm(radius, units))

        # An Aperture may be passed directly; take its spec, and its (already
        # normalized) center unless an explicit pixel ``center`` overrides it.
        spec_center_norm = None
        if isinstance(spec, toolbox.Aperture):
            spec_center_norm = spec.center
            spec = spec.spec

        if spec is None:
            spec = self.aperture.spec

        center_norm = (
            spec_center_norm if center is None else self._center_pix_to_norm(center)
        )

        self.aperture = toolbox.Aperture(self._grid_base, spec, center=center_norm)
        self._grid = None
        return self.aperture

    def fit_aperture(self, method="moments", recenter=True):
        r"""
        Fits the SLM's :attr:`aperture` to the measured source amplitude distribution in
        :attr:`source` ``["amplitude"]`` (analyzed via ``"moments"`` or least-squares
        ``"fit"``). This sets a circular aperture whose source radius is the :math:`1/e`
        field-amplitude radius (:math:`1/e^2` in intensity) of the measured amplitude, and
        (if ``recenter``) whose center matches the amplitude centroid.

        If no source amplitude has been measured, the aperture is set to a circular
        aperture of source radius equal to a quarter of the smallest SLM extent.

        Parameters
        ----------
        method : str {"fit", "moments"}
            Whether to use moment calculations (``"moments"``, faster) or a least-squares
            ``"fit"`` (more accurate) to determine the center and radius.
        recenter : bool
            If ``True``, recenter the aperture on the measured amplitude centroid. If
            ``False``, keep the current aperture center.

        Returns
        -------
        :class:`~slmsuite.holography.toolbox.Aperture`
            The fitted :attr:`aperture`.
        """
        if "amplitude" not in self.source:
            # No measured amplitude: guess a circular aperture from the grid extent.
            radius_norm = .25 * np.min((
                self.shape[1] * self.pitch[0],
                self.shape[0] * self.pitch[1],
            ))
            spec = 1.0 / (2.0 * radius_norm)
            center_norm = self.aperture.center if not recenter else None
            self.aperture = toolbox.Aperture(self._grid_base, spec, center=center_norm)
            self._grid = None
            return self.aperture

        amp = np.abs(self.source["amplitude"])

        if method == "fit":
            result = analysis.image_fit(amp, plot=False)
            radius = np.sqrt(2) * np.array([result[0, 5], result[0, 6]])
            center = np.array([result[0, 1], result[0, 2]])
        elif method == "moments":
            # Do moments in power-space, not amplitude.
            center = analysis.image_positions(np.square(amp))
            radius = np.sqrt(4 * analysis.image_variances(np.square(amp), centers=center)[:2])
            center = np.squeeze(center)
        else:
            raise ValueError(f"method '{method}' not recognized; use 'moments' or 'fit'.")

        # image_positions returns coordinates relative to the image center, which
        # analysis.image_moment defines as (N - 1) / 2 (matching the SLM grid and
        # _center_pix_to_norm). Use the same convention to recover absolute pixels.
        center_pix = np.squeeze(center) + (np.flip(self.shape) - 1) / 2.0

        radius_norm = np.mean(self.pitch * np.squeeze(radius))
        if not np.isfinite(radius_norm) or radius_norm <= 0:
            raise RuntimeError(
                f"fit_aperture found a degenerate source radius ({radius_norm}) with "
                f"method '{method}'; the measured source amplitude carries no usable signal."
            )
        spec = 1.0 / (2.0 * radius_norm)

        center_norm = self.aperture.center
        if recenter:
            center_norm = self._center_pix_to_norm(center_pix)

        self.aperture = toolbox.Aperture(self._grid_base, spec, center=center_norm)
        self._grid = None
        return self.aperture

    def fit_source_amplitude(self, method="moments", extent_threshold=.1, force=True):
        warnings.warn(
            "fit_source_amplitude is deprecated in favor of fit_aperture and "
            "will be removed in a future release."
        )
        self.fit_aperture(method=method, recenter=True)

    def _get_source_amplitude(self):
        """
        The effective source amplitude: the measured amplitude (or unity if unmeasured)
        masked by the :attr:`aperture`.
        """
        if self.source.get("amplitude") is not None:
            amp = self.source["amplitude"]
            if not self.aperture.crops:
                # No cropping: skip the all-True mask multiply. Copy so callers may
                # mutate the result without corrupting self.source["amplitude"].
                return amp.copy()
        else:
            amp = np.ones(self.shape)
            if not self.aperture.crops:
                return amp          # Already a fresh, independent array.
        return amp * _xp(amp).asarray(self.aperture_mask)

    def _get_source_phase(self):
        """
        The effective source phase: the measured phase (or zero if unmeasured) masked by
        the :attr:`aperture`.
        """
        if self.source.get("phase") is not None:
            phase = self.source["phase"]
            if not self.aperture.crops:
                # No cropping: skip the all-True mask multiply. Copy so callers may
                # mutate the result without corrupting self.source["phase"].
                return phase.copy()
        else:
            phase = np.zeros(self.shape)
            if not self.aperture.crops:
                return phase        # Already a fresh, independent array.
        return phase * _xp(phase).asarray(self.aperture_mask)

    def plot_source(self, source=None, sim=False, power=False, aperture=True):
        """
        Plots measured or simulated amplitude and phase distribution
        of the SLM illumination. Also plots the rsquared goodness of fit value if available.

        Parameters
        ----------
        source : dict OR None
            The data to plot. If ``None``, uses :attr:`source`.
        sim : bool
            Plots the simulated source distribution if ``True`` or the measured
            source distribution if ``False``.
        power : bool
            If ``True``, plot the power (amplitude squared) instead of the amplitude.
        aperture : bool
            If ``True`` (default), overlay the outline of the current :attr:`aperture`
            (when it crops the SLM) on the phase and amplitude panels.

        Returns
        --------
        matplotlib.axes.Axes
            Axis handles for the generated plot.
        """
        if source is None:
            source = self.source

        # Check if proper source keywords are present.
        if sim and not np.all([k in source for k in ("amplitude_sim", "phase_sim")]):
            raise RuntimeError("Simulated amplitude and/or phase keywords missing from slm.source!")
        elif not sim and not np.all([k in source for k in ("amplitude", "phase")]):
            raise RuntimeError(
                "'amplitude' or 'phase' keywords missing from slm.source! Run "
                ".wavefront_calibrate() or .set_source_analytic() to set a source profile."
            )

        # Handle whether we're going to plot the R^2.
        plot_r2 = not sim and "r2" in source
        r2_full_shape = plot_r2 and source["r2"].shape == self.shape
        plot_r2_contour = plot_r2 and r2_full_shape and "r2_threshold" in source

        def r2_contour(ax):
            if plot_r2_contour:
                ax.contour(
                    source["r2"],
                    levels=[source["r2_threshold"]],
                    colors="red",
                    linewidths=1,
                )

        # Make the subplots.
        _, axs = plt.subplots(1, 3 if plot_r2 else 2, figsize=(10, 6))

        # Panel 1: Phase
        im = axs[0].imshow(
            np.mod(source["phase_sim" if sim else "phase"], 2*np.pi),
            cmap=plt.get_cmap("twilight"),
            interpolation="none",
        )
        r2_contour(axs[0])
        if aperture:
            self._plot_aperture(axs[0])
        axs[0].set_title("Simulated Source Phase" if sim else "Source Phase")
        axs[0].set_xlabel("SLM $x$ [pix]")
        axs[0].set_ylabel("SLM $y$ [pix]")
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im.set_clim([0, 2*np.pi])
        plt.colorbar(im, cax=cax)

        # Panel 2: Amplitude or Power
        if power:
            im = axs[1].imshow(
                np.square(source["amplitude_sim" if sim else "amplitude"]),
                clim=(0, 1)
            )
            axs[1].set_title("Simulated Source Power" if sim else "Source Power")
        else:
            im = axs[1].imshow(source["amplitude_sim" if sim else "amplitude"], clim=(0, 1))
            axs[1].set_title("Simulated Source Amplitude" if sim else "Source Amplitude")
        r2_contour(axs[1])
        if aperture:
            self._plot_aperture(axs[1])
        axs[1].set_xlabel("SLM $x$ [pix]")
        axs[1].set_ylabel("SLM $y$ [pix]")
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        # Panel 3: R^2
        if plot_r2:
            im = axs[2].imshow(source["r2"], clim=(0, 1))
            r2_contour(axs[2])
            axs[2].set_title("Cal Fitting $R^2$")
            if r2_full_shape:
                axs[2].set_xlabel("SLM $x$ [pix]")
                axs[2].set_ylabel("SLM $y$ [pix]")
            else:
                axs[2].set_xlabel("SLM $x$ [superpix]")
                axs[2].set_ylabel("SLM $y$ [superpix]")
            divider = make_axes_locatable(axs[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

        # Finalize the plot and return the axes.
        plt.tight_layout()
        _slmsuite_plt_show(name="plot_source")

        return axs

    def get_point_spread_function_knm(self, padded_shape=None):
        """
        Fourier transforms the wavefront calibration's measured amplitude to directly compute
        the expected diffraction-limited performance of the system in ``"knm"`` space.

        Parameters
        ----------
        padded_shape : (int, int) OR None
            The point spread function changes in resolution depending on the padding.
            Use this variable to provide this padding.
            If ``None``, do not pad.

        Returns
        -------
        numpy.ndarray
            The point spread function of shape ``padded_shape``.
        """
        nearfield = toolbox.pad(self._get_source_amplitude(), padded_shape)
        farfield = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(nearfield), norm="ortho")))

        return farfield

    def get_spot_radius_kxy(self):
        """
        Approximates the expected standard deviation radius of farfield spots in the
        ``"kxy"`` basis based on the near-field amplitude distribution
        stored in :attr:`source`.
        For a Gaussian source, this is the :math:`1/e` amplitude radius
        (:math:`1/e^2` power radius).

        Returns
        -------
        float
            Radius of the farfield spot.
        """
        rad_norm = self.source_radius
        rad_pix = rad_norm / np.mean(self.pitch)
        rad_freq = np.reciprocal(rad_pix)

        psf_kxy = toolbox.convert_vector(
            [rad_freq, rad_freq],
            from_units="freq",
            to_units="kxy",
            hardware=self,
            shape=self.shape,
        )

        return np.mean(psf_kxy)

    # Self-test method to test everything above.

    def test(self):
        """
        Tests the core hardware methods of :class:`SLM`.
        Validates that the SLM is connected correctly and all hardware
        features are supported.
        """
        print(f"Testing SLM: {self.name}")

        print("  Testing set_phase...")



        # Benchmark set_phase.
        n_iter = 20
        phase = np.random.rand(n_iter, *self.shape) * 2 * np.pi
        t0 = time.time()
        for i in range(n_iter):
            self.set_phase(phase[i,:,:], phase_correct=False)
        elapsed = time.time() - t0
        fps = n_iter / elapsed
        print(f"    set_phase benchmark: {fps:.1f} Hz ({elapsed/n_iter*1e3:.2f} ms/frame)")

        print("  Testing set_input_trigger...")
        for val in [True, False]:
            try:
                self.set_input_trigger(val)
                print(f"    set_input_trigger({val}): OK")
            except NotImplementedError:
                print(f"    set_input_trigger({val}): NotImplementedError (expected for base SLM)")

        print("  Testing set_output_trigger...")
        for val in [True, False]:
            try:
                self.set_output_trigger(val)
                print(f"    set_output_trigger({val}): OK")
            except NotImplementedError:
                print(f"    set_output_trigger({val}): NotImplementedError (expected for base SLM)")

        return True
