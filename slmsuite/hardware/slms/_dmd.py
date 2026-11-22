"""
Texas Instruments DMD chips, shared by the drivers of the controllers that address them.

:class:`_DMDModel` reads a chip's shape, pitch and mirror tilt from
``texasinstruments.yaml`` and models its two mirror states as the element patterns of a
:class:`~slmsuite.holography.toolbox.Pixel`.
"""

import numpy as np

from slmsuite.hardware.slms._plm import _DLPModel
from slmsuite.hardware.slms.slm import SLM
from slmsuite.misc.xp import as_numpy


class _DMDModel(_DLPModel):
    r"""
    Mixin for :class:`~slmsuite.hardware.slms.slm.SLM` drivers of Texas Instruments DMD
    chips: the models of ``texasinstruments.yaml`` with a mirror tilt.

    A DMD is a one-bit SLM whose levels differ in amplitude, not in phase: a phase is
    displayed as the binary amplitude hologram which turns a mirror on where the cosine
    of its phase is positive, and an integer mask is displayed as it is. The
    :attr:`~slmsuite.hardware.slms.slm.SLM.pixel` is the tilted on-state mirror; the
    light of the off state is not modeled. Assigning a dict to
    :attr:`~slmsuite.hardware.slms.slm.SLM.pixel` adds to the pixel of the model, and
    ``None`` restores it.
    """

    _model_key = "tilt"
    gamma_amplitude = (0, 1)

    def _get_model_pixel(self, pixel=None):
        """
        The pixel of this model: the ``pixel`` keys of :attr:`model_config` with the
        tilt of the on state.

        Parameters
        ----------
        pixel : Pixel OR dict OR None
            A dict adds to or replaces keywords, such as ``incidence`` and ``order``.
            A :class:`~slmsuite.holography.toolbox.Pixel` is returned as it is.

        Returns
        -------
        Pixel OR dict
        """
        if pixel is not None and not isinstance(pixel, dict):
            return pixel

        direction = np.array(self.model_config["tilt_direction"], dtype=float)
        on = np.radians(self.model_config["tilt"]) * direction / np.hypot(*direction)

        return {**self.model_config.get("pixel", {}), "tilt": on, **(pixel or {})}

    @property
    def pixel(self):
        return self._pixel

    @pixel.setter
    def pixel(self, pixel):
        SLM.pixel.fset(self, self._get_model_pixel(pixel))

    @property
    def _viewer_frame(self):
        """The mirror states on display."""
        return as_numpy(self.xp.cos(self._phase) > 0).astype(self.dtype)

    def set_gamma(self, gamma=None, levels=None, lut_size=None):
        """Raises :class:`NotImplementedError`: mirror states have no phase response."""
        raise NotImplementedError("A DMD has no phase response to set.")
