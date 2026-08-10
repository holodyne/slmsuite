from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        # Remember the name.
        self.name = str(name)
        if len(self.name) == 0:
            self.name = str(self.__class__.__name__)

        # Initialize logger.
        _Loggable.__init__(self)

        # Parse shape.
        width, height = format_shape(resolution)
        self.shape = (height, width)
        
        if not np.all((np.array(self.shape) > 100) & (np.array(self.shape) < 1e4)):
            self.logger.warning("Resolution of %s is unusual. Was this a typo?", resolution)


        # Parse datatype variables.
        self.bitdepth = int(bitdepth)
        self.dtype = self._get_dtype()  # bitdepth is error-checked here.

        # Parse spatial dimensions.
        if pitch_um is None:
            if is_slm:
                raise ValueError("SLMs must have a pitch_um specified.")
            self.pitch_um = None
        else:
            if isinstance(pitch_um, REAL_TYPES):
                pitch_um = [pitch_um, pitch_um]
            pitch_um = np.squeeze(pitch_um)
            if len(pitch_um) != 2 or np.any(pitch_um <= 0):
                raise ValueError("Expected positive (float, float) for pitch_um")
            self.pitch_um = np.array([float(pitch_um[0]), float(pitch_um[1])])

        if self.pitch_um is not None and not np.all(
            (self.pitch_um > 1) & (self.pitch_um < 50)
        ):
            self.logger.warning(
                "Pixel pitch of %.2f x %.2f um is unusual. Was this a typo?",
                self.pitch_um[0], self.pitch_um[1],
            )

        # Whether this is an SLM or not, used for some viewer settings.
        self.is_slm = bool(is_slm)

        # Initialize viewer.
        self.viewer = None

        # With all the init variables filled in, log the state.
        self.log_state()

    @abstractmethod
    def close(self):
        """Abstract method to close the hardware."""
        raise NotImplementedError()

    def __del__(self):
        try:
            self.close()
        except Exception:
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
        # One quiet attempt: hardware that cannot yet capture falls back on bitdepth below.
        if test_data is None and hasattr(self, "_get_image_hw"):
            test_data = lambda: self._get_image_hw(timeout_s=1)

        dtype = None

        if test_data is not None:
            try:
                if callable(test_data):
                    test_data = test_data()

                probed = np.dtype(np.array(test_data).dtype)
                if probed.kind in "iuf":   # else a non-numeric probe would mistype the hardware.
                    dtype = probed
            except Exception as error:
                self.logger.debug("Could not probe '%s' for a dtype: %s", self.name, error)

        if dtype is None:
            if self.bitdepth <= 0:
                raise ValueError("Non-positive bitdepth does not make sense.")
            elif self.bitdepth <= 8:
                dtype = np.dtype(np.uint8)
            elif self.bitdepth <= 16:
                dtype = np.dtype(np.uint16)
            elif self.bitdepth <= 32:
                dtype = np.dtype(np.uint32)
            elif self.bitdepth <= 64:
                dtype = np.dtype(np.uint64)
            else:
                dtype = np.dtype(float)

            if self.bitdepth > 16:
                self.logger.warning("Bitdepth %s is unusually high.", self.bitdepth)

        self.dtype = dtype

        # Warn the user if the image type cannot represent the full bitdepth.
        if dtype.kind == "i" or dtype.kind == "u":
            dtype_bitdepth = dtype.itemsize * 8
            if dtype.kind == "i":
                dtype_bitdepth -= 1

            if dtype_bitdepth < self.bitdepth:
                warnings.warn(
                    f"Hardware '{self.name}' bitdepth of {self.bitdepth} does not conform "
                    f"with the image type {dtype} with {dtype.itemsize} bytes."
                )

        return self.dtype

    def _plot(self, data, limits, title, *, ax, cbar, labels, **kwargs):
        """
        Plots ``data`` on a pixel axis, passing ``kwargs`` to ``imshow`` and applying
        ``labels`` only when ``data`` fills the hardware.

        Returns
        -------
        (matplotlib.pyplot.axis, matplotlib.pyplot.axis OR None, bool)
            Data axis, colorbar axis, and whether a new figure awaits display.
        """
        should_show = False
        if ax is None:
            if len(plt.get_fignums()) > 0:
                fig = plt.gcf()
            else:
                fig = plt.figure(figsize=(20,8))
                should_show = True
        else:
            fig = None
            plt.sca(ax)

        im = plt.imshow(data, **kwargs)
        ax = plt.gca()

        cax = None
        if cbar and fig is not None:
            cax = make_axes_locatable(ax).append_axes("right", size="2%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")
            plt.sca(ax)

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

        if data.shape == self.shape:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])

        return (ax, cax, should_show)
