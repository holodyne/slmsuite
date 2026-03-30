"""
Abstract functionality for SLMs.
"""

import time
import os
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from PIL import Image
import inspect
from abc import ABC, abstractmethod

from slmsuite import __version__
from slmsuite.hardware import _Picklable
from slmsuite.holography import toolbox
from slmsuite.misc import fitfunctions
from slmsuite.misc.math import REAL_TYPES
from slmsuite.holography import analysis
from slmsuite.misc.files import generate_path, latest_path, save_h5, load_h5


def _xp(array):
    """Return cupy if array is a cupy ndarray, else numpy."""
    if cp is not np and isinstance(array, cp.ndarray):
        return cp
    return np


class SLM(_Picklable, ABC):
    r"""
    Abstract class for SLMs.

    Attributes
    ------
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
    pitch : float
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
        measured from the center of the SLM.
        Of size :attr:`shape`. Produced by :meth:`numpy.meshgrid`.
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

        When :meth:`.fit_source_amplitude()` is called,
    phase : numpy.ndarray
        Last displayed data in units of phase (radians). If wavefront calibration
        (`phase_correct=True`) is used, this includes the calibration data.
    display : numpy.ndarray
        Last displayed data in discrete SLM units (integers). This is the data
        that is actually displayed by the bit-limited hardware. If wavefront calibration
        (`phase_correct=True`) is used, this includes the calibration data.
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
    ]
    _pickle_data = [
        "source",
        "phase",
        "display",
    ]

    @abstractmethod
    def __init__(
        self,
        resolution,
        bitdepth=8,
        name="SLM",
        wav_um=1,
        wav_design_um=None,
        pitch_um=(8,8),
        settle_time_s=0.3,
    ):
        """
        Initialize SLM.

        Parameters
        ----------
        resolution
            The width and height of the camera in ``(width, height)`` form.

            Important
            ~~~~~~~~~
            This is the opposite of the numpy ``(height, width)``
            convention stored in :attr:`shape`.
        bitdepth
            See :attr:`bitdepth`. Defaults to 8.
        name
            See :attr:`name`.
        wav_um
            See :attr:`wav_um`.
        wav_design_um
            See :attr:`wav_design_um`.
        pitch_um
            See :attr:`pitch_um`. Defaults to 8 micron square pixels.
        settle_time_s
            See :attr:`settle_time_s`.
        """
        self.name = str(name)
        width, height = resolution
        self.shape = (int(height), int(width))

        # By default, target wavelength is the design wavelength
        self.wav_um = float(wav_um)
        if wav_design_um is None:
            self.wav_design_um = float(wav_um)
        else:
            self.wav_design_um = float(wav_design_um)

        # Multiplier for when the target wavelengths differ from the design wavelength.
        self.phase_scaling = self.wav_um / self.wav_design_um

        # Bit depth of SLM pixels.
        self.bitdepth = int(bitdepth)

        # time to delay after writing (allows SLM to stabilize).
        self.settle_time_s = float(settle_time_s)

        # Spatial dimensions
        if isinstance(pitch_um, REAL_TYPES):
            pitch_um = [pitch_um, pitch_um]
        self.pitch_um = np.squeeze(pitch_um)
        if len(self.pitch_um) != 2 or np.any(self.pitch_um <= 0):
            raise ValueError("Expected positive (float, float) for pitch_um")
        self.pitch_um = np.array([float(self.pitch_um[0]), float(self.pitch_um[1])])

        self.pitch = self.pitch_um / self.wav_um

        # Make normalized coordinate grids.
        xpix = (width  - 1) * np.linspace(-0.5, 0.5, width)
        ypix = (height - 1) * np.linspace(-0.5, 0.5, height)
        self.grid = list(np.meshgrid(self.pitch[0] * xpix, self.pitch[1] * ypix))

        # Source profile dictionary
        self.source = {}

        # Decide dtype
        if self.bitdepth <= 8:
            self.dtype = np.dtype(np.uint8)
        else:
            self.dtype = np.dtype(np.uint16)

        # Phase and display caches for user reference.
        self.phase = np.zeros(self.shape)
        self.display = np.zeros(self.shape, dtype=self.dtype)

        # Now inspect the _set_phase_hw() method to see if it supports the execute and
        # block arguments. We need to do this in init because inspect is expensive.
        self._set_phase_hw_args = inspect.signature(self._set_phase_hw).parameters.keys()
        self._set_phase_hw_block = "block" in self._set_phase_hw_args
        self._set_phase_hw_execute = "execute" in self._set_phase_hw_args

        # Default settle and phase_correct behavior for set_phase.
        self.phase_correct = True
        self.settle = False

    @property
    def bitresolution(self):
        return 2**self.bitdepth

    @abstractmethod
    def close(self):
        """Abstract method to close the SLM and delete related objects."""
        raise NotImplementedError()

    def __del__(self):
        try:
            self.close()
        except:
            pass

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
        setting :attr:`~slmsuite.hardware.slms.slm.SLM.source["phase"]`.
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
            :attr:`~slmsuite.hardware.slms.slm.SLM.source["phase"]`,
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

        if np.any(file_shape_error > 1):
            self.source["phase"] = toolbox.unpad(phase_correction, self.shape)
        elif np.any(file_shape_error < 1):
            self.source["phase"] = toolbox.pad(phase_correction, self.shape)
        else:
            self.source["phase"] = phase_correction

        return self.source["phase"]

    def plot(self, phase=None, limits=None, title="Phase", ax=None, cbar=True):
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
            Also plot a colorbar.

        Returns
        -------
        matplotlib.pyplot.axis
            Axis of the plotted phase.
        """
        if phase is None:
            phase = self.phase
        phase = np.array(phase, copy=(False if np.__version__[0] == '1' else None))
        phase = np.mod(phase, 2*np.pi) / np.pi

        if len(plt.get_fignums()) > 0:
            fig = plt.gcf()
        else:
            fig = plt.figure(figsize=(20,8))

        if ax is not None:
            plt.sca(ax)

        im = plt.imshow(phase, clim=[0, 2], cmap="twilight", interpolation="none")
        ax = plt.gca()

        if cbar:
            cax = make_axes_locatable(ax).append_axes("right", size="2%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")
            ticks = [0,1,2]
            cax.set_yticks([0,1,2])
            cax.set_yticklabels([f"${t}\\pi$" for t in ticks])

        # ax.invert_yaxis()
        ax.set_title(title)

        if limits is not None and limits != 1:
            if np.isscalar(limits):
                axlim = [ax.get_xlim(), ax.get_ylim()]

                centers = np.mean(axlim, axis=1)
                deltas = np.squeeze(np.diff(axlim, axis=1)) * limits / 2

                limits = np.vstack((centers - deltas, centers + deltas)).T
            elif np.shape(limits) == (2,2):
                pass
            else:
                raise ValueError(f"limits format {limits} not recognized; provide a scalar or limits.")

            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])

        if phase.shape == self.shape:
            ax.set_xlabel("SLM $n$ [pix]")
            ax.set_ylabel("SLM $m$ [pix]")

        plt.sca(ax)

        return ax

    # Writing methods

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
        wrap the phase (see the private method :meth:`_phase2gray()`). Which
        routine is used depends on :attr:`phase_scaling`:

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
        phase : numpy.ndarray OR cupy.cparray OR
                slmsuite.holography.algorithms.Hologram OR None
        Phase data to display in units of :math:`2\pi`, unless the passed
        data is of integer type and the data is applied directly.

            -  If ``None`` is passed to :meth:`.set_phase()`, data is zeroed.
            -  If a :class:`~slmsuite.holography.algorithms.Hologram` is passed,
               the phase is grabbed from
               :meth:`~slmsuite.holography.algorithms.Hologram.get_phase()`.
            -  If the array has a larger shape than the SLM shape, then the data is
               cropped to size in a centered manner
               (:attr:`~slmsuite.holography.toolbox.unpad`).
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
            in cases where :attr:`phase_scaling` not one, this copy is modified
            to include how the data was wrapped. If the data was cropped, then
            the cropped data is stored, etc. If integer data was passed, the
            equivalent floating point phase is computed and stored in the
            attribute :attr:`phase`.
        phase_correct : bool OR None
            Whether to add wavefront correction to the pattern. This correction
            is stored in
            :attr:`~slmsuite.hardware.slms.slm.SLM.source```["phase"]``. If
            ``None``, defaults to :attr:`phase_correct` (which defaults to
            ``True``).
        settle : bool OR None
            Whether to sleep for
            :attr:`~slmsuite.hardware.slms.slm.SLM.settle_time_s`. If ``None``,
            defaults to :attr:`settle` (which defaults to ``False``).
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
        if execute is None:
            execute = True
        else:
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

        # Parse phase.
        if hasattr(phase, "get_phase"):
            # If we passed a hologram, grab the phase from there.
            phase = phase.get_phase()

        if phase is None:
            # Zero the phase pattern.
            self.phase.fill(0)
            xp = _xp(self.phase)
        else:
            # Make sure the array is an ndarray (cupy or numpy).
            xp = _xp(phase)
            phase = xp.asarray(phase)

            # If internal structures are already on GPU but input is numpy, upgrade the input.
            if cp is not np and xp is np and isinstance(self.phase, cp.ndarray):
                xp = cp
                phase = cp.asarray(phase)

            # Promote self.phase and self.display to GPU if input is cupy.
            if xp is cp:
                if not isinstance(self.phase, cp.ndarray):
                    self.phase = cp.zeros(self.shape)
                if not isinstance(self.display, cp.ndarray):
                    self.display = cp.zeros(self.shape, dtype=self.dtype)

        if phase is not None and np.issubdtype(phase.dtype, np.integer):
            # Check the type.
            if phase.dtype != self.display.dtype:
                raise TypeError(
                    "Unexpected integer type {}. Expected {}.".format(
                        phase.dtype, self.display.dtype
                    )
                )

            # If integer data was passed, check that we are not out of range.
            if xp.any(phase >= self.bitresolution):
                raise TypeError(
                    "Integer data must be within the bitdepth ({}-bit) of the SLM.".format(
                        self.bitdepth
                    )
                )

            # Copy the pattern and unpad if necessary.
            if phase.shape != self.shape:
                xp.copyto(self.display, toolbox.unpad(phase, self.shape))
            else:
                xp.copyto(self.display, phase)

            # Update the phase variable with the integer data that we displayed.
            self.phase = 2 * np.pi - self.display * (
                2 * np.pi / self.phase_scaling / self.bitresolution
            )
        else:
            # If float data was passed (or the None case).
            # Copy the pattern and unpad if necessary.
            if phase is not None:
                if phase.shape != self.shape:
                    xp.copyto(self.phase, toolbox.unpad(phase, self.shape))
                else:
                    xp.copyto(self.phase, phase)

            # Add phase correction if requested.
            if phase_correct is None:
                phase_correct = self.phase_correct
            if phase_correct and ("phase" in self.source):
                self.phase += xp.asarray(self.source["phase"])

            # Pass the data to self.display.
            # Turn the floats in phase space to integer data for the SLM.
            self.display = self._format_phase_hw(self.phase)

        # Write!
        if execute:
            self._set_phase_hw(self.display, **kwargs)

        # Optional delay.
        if settle is None:
            settle = self.settle
        if execute and settle:
            time.sleep(self.settle_time_s)

        return self.display

    def _phase2gray(self, phase, out=None):
        r"""
        Helper function to convert an array of phases (units of :math:`2\pi`) to an array of
        :attr:`~slmsuite.hardware.slms.slm.SLM.bitresolution` -scaled and -cropped integers.
        This is used by :meth:`set_phase()`. See special cases described in :meth:`set_phase()`.
        Supports both :mod:`numpy` and :mod:`cupy` arrays for GPU acceleration.

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
        xp = _xp(phase)

        if out is None:
            out = xp.zeros(self.shape, dtype=self.dtype)
        elif xp is cp and not isinstance(out, cp.ndarray):
            out = cp.zeros(self.shape, dtype=self.dtype)
            self.display = out

        if self.phase_scaling == 1:
            # Prepare the 2pi -> integer conversion factor and convert.
            factor = -(self.bitresolution / 2 / np.pi)
            phase *= factor

            # There is some randomness involved in casting positive floats to integers.
            # Avoid this by going all negative.
            maximum = xp.amax(phase)
            if maximum >= 0:
                toshift = self.bitresolution * 2 * float(xp.ceil(maximum / self.bitresolution))
                phase -= toshift

            # Copy and cast the data to the output (usually self.display)
            xp.rint(phase, out=phase)
            xp.copyto(out, phase, casting="unsafe")

            # Restore phase (usually self.phase) as these operations are in-place.
            phase *= 1 / factor

            # Shift by one so that phase=0 --> display=max. That way, phase will be more continuous.
            out -= 1

            # This part (along with the choice of type), implements modulo much faster than xp.mod().
            if self.bitresolution & (self.bitresolution - 1) == 0:
                active_bits_mask = int(self.bitresolution - 1)
                xp.bitwise_and(out, active_bits_mask, out=out)
            else:
                # Slow backup using xp.mod().
                xp.mod(out, self.bitresolution, out=out)
        else:
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

                # Set values still out of range to zero.
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

        self._set_phase_hw(data["display"])
        self.display = data["display"]
        self.phase = data["phase"]

        if not np.all(np.isclose(data["display"], self._format_phase_hw(data["phase"]))):
            warnings.warn("Integer data in 'display' does not match 'phase' for this SLM.")

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

    def fit_source_amplitude(self, method="moments", extent_threshold=.1, force=True):
        """
        Extracts various :attr:`source` parameters from the source for use in
        analytic functions. This is done by analyzing the :attr:`source` ``["amplitude"]``
        distribution with ``"moments"`` or least squares ``"fit"``.
        These parameters include the following keys:

        -   ``"amplitude_center_pix"`` : (float, float)

            Pixel corresponding to the center of the source.
            The grid is also changed to be centered on this pixel.

        -   ``"amplitude_radius"`` : float

            The radial standard deviation of the amplitude distribution in normalized units.
            For a Gaussian source, this is the :math:`1/e` amplitude radius
            (:math:`1/e^2` power radius).
            This is scalar and averages the :math:`x` and :math:`y` distributions.
            This is used to set the source radius for
            :meth:`~slmsuite.holography.toolbox.phase.laguerre_gaussian()`
            and similar.

        -   ``"amplitude_extent"`` : (float, float)

            The box radii of the smallest rectangle which covers all amplitude
            larger than ``extent_threshold``, where the maximum of the distribution is
            normalized to one.

        -   ``"amplitude_extent_radius"`` : float

            Smallest scalar radius about the center of the that covers all amplitude
            larger than ``extent_threshold``, where the maximum of the distribution is
            normalized to one.
            This is used to determine the scaling for
            :meth:`~slmsuite.holography.toolbox.phase.zernike_aperture()`:
            Too small of a scaling is not good because amplitude would
            overlap outside where Zernike is defined, with divergent phase for higher
            order Zernike polynomials.
            Too large of a scaling is not good because one needs to use high order
            Zernike to attain sufficient spatial resolution at the center of the distribution.

        Important
        ~~~~~~~~~
        If :attr:`source` ``["amplitude"]`` is not set, then the parameters are guessed
        as fractions of the grid:

        -   ``"amplitude_center_pix"``
            Unchanged from current center.

        -   ``"amplitude_radius"``
            Guessed as 1/4 of the smallest extent.

        -   ``"amplitude_extent"``
            Guessed as the the rectangle that circumscribes the SLM field.

        -   ``"amplitude_extent_radius"``
            Guessed as the the radius that circumscribes the SLM field.

        Important
        ~~~~~~~~~
        The ``grid`` is recentered upon the detected center of the source.
        This ``grid`` is used to generated phase functions like
        :meth:`~slmsuite.holography.toolbox.phase.lens()` or
        :meth:`~slmsuite.holography.toolbox.phase.laguerre_gaussian()`.
        Such generation works best when centered upon the source; a
        :meth:`~slmsuite.holography.toolbox.phase.lens()` focuses coaxially and a
        :meth:`~slmsuite.holography.toolbox.phase.laguerre_gaussian()` appears symmetric.

        Parameters
        ----------
        method : str {"fit", "moments"}
            Whether to use moment calculations ``"moments"``
            or a least squares ``"fit"`` to determine
            ``"amplitude_center_pix"`` and ``"amplitude_radius"``.
            ``"moments"`` is faster but ``"fit"`` is more accurate.
        extent_threshold : float
            Fraction of the maximal amplitude to use as
            the full extent of the amplitude distribution.
        force : bool
            If ``False``, does not calculate if these quantities already exist.
            ``True`` forces recomputation.
        """
        # If we have already done a fit, and we don't want to force a new one, then return.
        if "amplitude_center_pix" in self.source and not force:
            return

        center_grid = np.array(
            [np.argmin(np.abs(self.grid[0][0,:])), np.argmin(np.abs(self.grid[1][:,0]))]
        )

        if not "amplitude" in self.source:
            # If there is no measured source amplitude, then make guesses based off of the grid.
            self.source["amplitude_center_pix"] = center_grid
            self.source["amplitude_radius"] = .25 * np.min((
                self.shape[1] * self.pitch[0],
                self.shape[0] * self.pitch[1]
            ))
            self.source["amplitude_extent"] = np.array(
                [np.max(np.abs(self.grid[0])), np.max(np.abs(self.grid[1]))]
            )
            self.source["amplitude_extent_radius"] = np.sqrt(np.amax(
                np.square(self.grid[0]) + np.square(self.grid[1])
            ))
        else:
            # Otherwise, use the measured amplitude distribution.
            amp = np.abs(self.source["amplitude"])

            # Parse extent_threshold
            if extent_threshold > 1:
                raise RuntimeError("extent_threshold cannot exceed 1 (100%). Use a small value.")

            if method == "fit":
                result = analysis.image_fit(amp, plot=False)
                std = np.array([result[0,5], result[0,6]])

                center = np.array([result[0,1], result[0,2]])
            elif method == "moments":
                # Do moments in power-space, not amplitude.
                center = analysis.image_positions(np.square(amp))
                std = np.sqrt(2 * analysis.image_variances(np.square(amp), centers=center)[:2])

                center = np.squeeze(center)

            center += np.flip(self.shape)/2

            self.source["amplitude_center_pix"] = center
            self.source["amplitude_radius"] = np.mean(self.pitch * np.squeeze(std))

            # Handle centering.
            dcenter = center_grid - center

            self.grid[0] += dcenter[0] * self.pitch[0]
            self.grid[1] += dcenter[1] * self.pitch[1]

            center_grid = np.array(
                [np.argmin(self.grid[0][0,:]), np.argmin(self.grid[1][:,0])]
            )

            extent_mask = amp > (extent_threshold * np.amax(amp))

            self.source["amplitude_extent"] = np.array([
                np.max(np.abs(self.grid[0][extent_mask])),
                np.max(np.abs(self.grid[1][extent_mask]))
            ])
            self.source["amplitude_extent_radius"] = np.sqrt(np.amax(
                np.square(self.grid[0][extent_mask]) + np.square(self.grid[1][extent_mask])
            ))

    def set_source_aperture(
        self,
        amplitude_center_pix=None,
        amplitude_radius=None,
        amplitude_extent=None,
        amplitude_extent_radius=None,
    ):
        """
        Sets source aperture parameters measured by :meth:`fit_source_amplitude()` and
        takes appropriate follow-on actions like shifting the grid.

        Parameters
        ----------
        amplitude_center_pix : (float, float) OR None
            Pixel corresponding to the center of the source.
            If provided, the grid is recentered on this pixel.
        amplitude_radius : float OR None
            The radial standard deviation of the amplitude distribution in normalized units.
            For a Gaussian source, this is the :math:`1/e` amplitude radius
            (:math:`1/e^2` power radius).
        amplitude_extent : (float, float) OR None
            The box radii of the smallest rectangle which covers the desired amplitude extent
            in normalized units.
        amplitude_extent_radius : float OR None
            Scalar radius about the center that covers the desired amplitude extent
            in normalized units.

        Returns
        -------
        dict
            :attr:`~slmsuite.hardware.slms.slm.SLM.source`.
        """
        # Handle center repositioning with grid shift
        if amplitude_center_pix is not None:
            amplitude_center_pix = np.array(amplitude_center_pix)

            # Get current center
            current_center = np.array(
                [np.argmin(np.abs(self.grid[0][0,:])), np.argmin(np.abs(self.grid[1][:,0]))]
            )

            # Calculate shift and update grid
            dcenter = current_center - amplitude_center_pix
            self.grid[0] += dcenter[0] * self.pitch[0]
            self.grid[1] += dcenter[1] * self.pitch[1]

            # Store the new center
            self.source["amplitude_center_pix"] = amplitude_center_pix

        # Set the other parameters directly
        if amplitude_radius is not None:
            self.source["amplitude_radius"] = float(amplitude_radius)

        if amplitude_extent is not None:
            self.source["amplitude_extent"] = np.array(amplitude_extent)

        if amplitude_extent_radius is not None:
            self.source["amplitude_extent_radius"] = float(amplitude_extent_radius)

        return self.source

    def get_source_radius(self):
        """
        Extracts the source radius in normalized units for functions like
        :meth:`~slmsuite.holography.toolbox.phase.laguerre_gaussian()`
        from the scalars computed in
        :meth:`~slmsuite.hardware.slms.slm.SLM.fit_source_amplitude()`.
        """
        self.fit_source_amplitude(force=False)
        return self.source["amplitude_radius"]

    def get_source_zernike_scaling(self):
        """
        Extracts the scaling for
        :meth:`~slmsuite.holography.toolbox.phase.zernike_aperture()`
        from the scalars computed in
        :meth:`~slmsuite.hardware.slms.slm.SLM.fit_source_amplitude()`.
        """
        self.fit_source_amplitude(force=False)
        return np.reciprocal(2 * self.source["amplitude_radius"])

    def get_source_center(self):
        """
        Extracts the scaling for
        :meth:`~slmsuite.holography.toolbox.phase.zernike_aperture()`
        from the scalars computed in
        :meth:`~slmsuite.hardware.slms.slm.SLM.fit_source_amplitude()`.
        """
        self.fit_source_amplitude(force=False)
        return self.source["amplitude_center_pix"]

    def _get_source_amplitude(self):
        """Deals with the case of an unmeasured source amplitude."""
        if "amplitude" in self.source:
            return self.source["amplitude"]
        else:
            return np.ones(self.shape)

    def _get_source_phase(self):
        """Deals with the case of an unmeasured source phase."""
        if "phase" in self.source:
            return self.source["phase"]
        else:
            return np.zeros(self.shape)

    def plot_source(self, source=None, sim=False, power=False):
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

        Returns
        --------
        matplotlib.pyplot.axis
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
        plt.show()

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
        self.fit_source_amplitude(force=False)

        rad_norm = self.source["amplitude_radius"]
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
