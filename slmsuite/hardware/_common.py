from abc import ABC, abstractmethod

import numpy as np
from yaml import warnings

from slmsuite.hardware._viewer import _Viewable
from slmsuite._logging import _Loggable
from slmsuite.holography.toolbox import format_shape
from slmsuite.misc.math import REAL_TYPES

class _Common(_Viewable, _Loggable, ABC):
    """
    Handles common properties and methods for both cameras and SLMs.
    """
    def __init__(
        self,
        resolution,
        bitdepth,
        name,
        pitch_um,
        is_slm,
    ):
        # Parse shape.
        width, height = format_shape(resolution)
        self.shape = (height, width)

        # Parse datatype variables.
        self.bitdepth = int(bitdepth)
        self.dtype = self._get_dtype()

        # Remember the name.
        self.name = str(name)
        if len(self.name) == 0:
            self.name = str(self.__class__.__name__)

        # Parse spatial dimensions.
        if pitch_um is None:
            if is_slm:
                raise ValueError("SLMs must have a pitch_um specified.")
            self.pitch_um = None
        else:
            if isinstance(pitch_um, REAL_TYPES):
                pitch_um = [pitch_um, pitch_um]
            self.pitch_um = np.squeeze(pitch_um)
            if len(self.pitch_um) != 2 or np.any(self.pitch_um <= 0):
                raise ValueError("Expected positive (float, float) for pitch_um")
            self.pitch_um = np.array([float(self.pitch_um[0]), float(self.pitch_um[1])])

        # Empty handle for the live viewer.
        self.viewer = None

        # Whether this is an SLM or not, used for some viewer settings.
        self.is_slm = bool(is_slm)

        # Initialize viewer.
        _Viewable.__init__(self)

        # Initialize logger.
        _Loggable.__init__(self)
        self.log_state()

    @abstractmethod
    def close(self):
        """Abstract method to close the hardware."""
        raise NotImplementedError()

    def __del__(self):
        try:
            self.close()
        except:
            pass

    @property
    def bitresolution(self):
        return 2**self.bitdepth     # Overwritten in Camera to account for averaging.

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    @property
    def resolution(self):
        return (self.shape[1], self.shape[0])

    def _get_dtype(self, test_data=None):
        """
        Determines and sets the dtype appropriate for the hardware's bitdepth.
        If ``test_data`` is not provided, attempts to get a sample image from the
        hardware; if that also fails, infers the dtype from bitdepth alone.
        """
        if test_data is None and hasattr(self, "_get_image_hw_tolerant"):
            test_data = self._get_image_hw_tolerant

        try:
            if test_data is None:
                raise ValueError("No test data provided for dtype inference.")

            if callable(test_data):
                test_data = test_data()

            self.dtype = np.dtype(
                np.array(
                    test_data
                ).dtype
            )   # Future: check if cameras change dtype after init.
        except:
            if self.bitdepth <= 0:
                raise ValueError("Non-positive bitdepth does not make sense.")
            elif self.bitdepth <= 8:
                self.dtype = np.dtype(np.uint8)
            elif self.bitdepth <= 16:
                self.dtype = np.dtype(np.uint16)
            elif self.bitdepth <= 32:
                self.dtype = np.dtype(np.uint32)
            elif self.bitdepth <= 64:
                self.dtype = np.dtype(np.uint64)
            else:
                self.dtype = np.dtype(float)

        try:
            # Determine the bitdepth of the datatype.
            if self.dtype.kind == "i" or self.dtype.kind == "u":
                dtype_bitdepth = self.dtype(0).nbytes * 8
                if self.dtype.kind == "i":
                    dtype_bitdepth -= 1
            elif self.dtype.kind == "f":
                dtype_bitdepth = np.inf

            # Warn the user if something is wrong.
            if dtype_bitdepth < self.bitdepth:
                raise warnings.warn(
                    f"Hardware '{self.name}' bitdepth of {self.bitdepth} does not conform "
                    f"with the image type {self.dtype} with {self.dtype.itemsize} bytes."
                )
        except:     # The above sometimes fails for non-numpy datatypes.
            pass

        return self.dtype
