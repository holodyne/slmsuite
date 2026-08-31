"""
A segment of a larger SLM.
This class allows the user to work with a specific region
of a parent SLM as if it were a separate SLM.
"""
import numpy as np

from slmsuite.hardware.slms.slm import SLM
from slmsuite.holography.toolbox import window_extent, window_slice

class SegmentedSLM(SLM):
    """
    A segment of a larger SLM.

    This class allows the user to work with a specific region of a parent SLM
    as if it were a separate SLM.

    Attributes
    ----------
    parent : SLM
        The parent SLM of which this is a segment.
    refresh : bool
        If ``True``, this segment will by default project the entire parent SLM's
        display after being updated.
    extent_slice : tuple of slice
        The rectangular extent of this segment on the parent SLM in slice format.
    subwindow : None OR tuple of arrays of indices
        If the window of this segment is non-rectangular, this stores the indices of the
        pixels in the segment's window within the rectangular extent. If the window is
        rectangular, this is ``None``.
    """

    def __init__(
        self,
        parent,
        window,
        name,
        refresh=False,
    ):
        r"""
        Initialize SLM and attributes.

        Parameters
        ----------
        parent : SLM
            The SLM to be segmented.
        window : (int, int, int, int) OR (array_like, array_like) OR array_like
            Format used by :func:`~slmsuite.holography.toolbox.window_slice`
            to define the window of interest on the parent SLM.
        name : str
            Name of this segment of the SLM.
        refresh : bool, optional
            If ``True``, this segment will by default project the entire parent SLM's
            display after being updated.
        """
        # Parse parent.
        if not isinstance(parent, SLM):
            raise ValueError("Parent must be an instance of SLM.")
        self.parent = parent
        self.refresh = bool(refresh)

        # Parse window — preserve original for unclipped bounds checking.
        window_raw = window
        window = window_slice(window, shape=parent.shape)  # 2 slice, 2 indices, or boolean array format

        # Get the rectangular extent of the window.
        self.subwindow = None
        if isinstance(window[0], slice):
            # Rectangular window: build (x, w, y, h) extent from the ORIGINAL (unclipped)
            # coordinates so the bounds check below can detect out-of-bounds windows.
            xi = int(window_raw[0])
            xf = xi + int(window_raw[1])
            yi = int(window_raw[2])
            yf = yi + int(window_raw[3])
            extent = (xi, xf - xi, yi, yf - yi)
            self.extent_slice = window               # clipped slices for actual indexing
        else:
            extent = window_extent(window)           # (x, w, y, h) format
            self.extent_slice = window_slice(extent) # 2 slice format

            # Handle the case where the window is not rectangular.
            if isinstance(window, np.ndarray):    # Boolean array
                self.subwindow = window[tuple(self.extent_slice)]
            else:                                 # Lists of indices (y_ind, x_ind)
                self.subwindow = (
                    window[0] - extent[2],        # y_ind - y_start
                    window[1] - extent[0],        # x_ind - x_start
                )

        # Error check the window against the parent SLM's shape.
        if (
            extent[0] < 0 or extent[0] + extent[1] > parent.shape[1] or
            extent[2] < 0 or extent[2] + extent[3] > parent.shape[0]
        ):
            raise ValueError("Window is out of bounds of the parent SLM.")

        # Instantiate the superclass, sharing the parent's backend and phase conventions.
        self._gamma_sign = parent._gamma_sign
        super().__init__(
            (extent[1], extent[3]),
            bitdepth=parent.bitdepth,
            name=name,
            wav_um=parent.wav_um,
            wav_design_um=parent.wav_design_um,
            pitch_um=parent.pitch_um,
            settle_time_s=parent.settle_time_s,
            gpu=(parent.xp is not np),
        )

        # Load source data from the parent SLM when available.
        for key in ("amplitude", "phase"):
            if key in self.parent.source:
                self.source[key] = self.parent.source[key][tuple(self.extent_slice)]

    @property
    def gamma(self):
        """This segment's own phase response, falling back to the parent's when unset."""
        return self.parent.gamma if self._gamma is None else self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @property
    def lut(self):
        """This segment's own lookup table, falling back to the parent's when unset."""
        return self.parent.lut if self._lut is None else self._lut

    @lut.setter
    def lut(self, lut):
        self._lut = lut

    def close(self):
        """Raise an error when attempting to close a segmented SLM."""
        raise RuntimeError("Close the parent SLM instead of the segmented SLM.")

    @staticmethod
    def info(verbose=True):
        """
        Prints instructions on how to use segmented SLMs.
        """
        if verbose:
            print("Call slm.segment() to produce child SegmentedSLMs.")
        return []

    def _set_phase_hw(
        self,
        display,
        refresh=None,
    ):
        """
        Overwrites the phase data in the parent SLM's display
        and writes the full parent display to hardware if desired.

        Parameters
        ----------
        display
            Integer data to display on the SLM. See :meth:`.SLM._set_phase_hw`.
        refresh : bool, optional
            Whether to update the full parent SLM.
            If ``None``, uses the value of ``self.refresh``, which is ``True``
            for the final segment of a segmented SLM by default.
        """
        # Update the parent SLM's display and phase data.
        if self.subwindow is None:                  # Rectangular window case
            self.parent.display[tuple(self.extent_slice)] = display
            if self.phase is not None:
                self.parent.phase[tuple(self.extent_slice)] = self.phase
        else:                                       # Non-rectangular window case
            self.parent.display[tuple(self.extent_slice)][self.subwindow] = display[self.subwindow]
            if self.phase is not None:
                self.parent.phase[tuple(self.extent_slice)][self.subwindow] = (
                    self.phase[self.subwindow]
                )

        # Update the parent SLM's hardware if desired.
        if refresh is None:
            refresh = self.refresh
        if refresh:
            self.parent._set_phase_hw(self.parent.display)

    def set_input_trigger(self, on : bool = False):
        r"""
        Program the input trigger on the parent SLM.

        Raises
        ------
        RuntimeError
        """
        raise RuntimeError("Program the input trigger on the parent SLM.")

    def set_output_trigger(self, on : bool = False):
        r"""
        Program the output trigger on the parent SLM.

        Raises
        ------
        RuntimeError
        """
        raise RuntimeError("Program the output trigger on the parent SLM.")
