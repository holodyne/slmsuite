"""
A simulated SLM.
"""

import numpy as np
from slmsuite.hardware.slms.slm import SLM

class SimulatedSLM(SLM):
    """
    A simulated SLM to emulate physical artifacts of actual SLMs.

    Attributes
    ----------
    source : dict
        For a :class:`SimulatedSLM()`, :attr:`source` stores ``"amplitude_sim"`` and ``"phase_sim"``,
        which are used to compute the SLM's simulated far-field.

        ``"amplitude_sim"`` : numpy.ndarray
            User-defined source amplitude (with the dimensions of :attr:`shape`) on the SLM.
        ``"phase_sim"`` : numpy.ndarray
            User-defined source phase (with the dimensions of :attr:`shape`) on the SLM.
    """

    def __init__(self, resolution, pitch_um=(8,8), source=None, **kwargs):
        r"""
        Initialize simulated slm.

        Parameters
        ----------
        resolution
            The width and height of the SLM in ``(width, height)`` form.

            Important
            ~~~~~~~~~
            This is the opposite of the numpy ``(height, width)``
            convention stored in :attr:`shape`.
        pitch_um : (float, float)
            Pixel pitch in microns. Defaults to 8 micron square pixels.
        source : dict
            See :attr:`source`. Defaults to uniform illumination with a flat phase if ``None``.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """
        super().__init__(resolution, pitch_um=pitch_um, settle_time_s=0, **kwargs)

        if source is None:
            self.source["amplitude_sim"] = np.ones_like(self.grid[0])
            self.source["phase_sim"] = np.zeros_like(self.grid[0])
        else:
            # assert np.all([source[kw].shape == self.shape for kw in source.keys()]
            # ), "The shape of the provided phase profile must match the SLM resolution!"
            self.source.update(source)

            # Handle case where `source` only has real values from experiment
            if "amplitude_sim" not in source.keys():
                self.source["amplitude_sim"] = self.source["amplitude"]
                self.source["phase_sim"] = -self.source["phase"]

        self.set_phase(None)

    def close(self):
        pass

    def _set_phase_hw(self, display):
        """Updates SLM.display to implement various physical artifacts of SLMs."""

        # FUTURE: apply physical effects directly to SLM.display

        return
