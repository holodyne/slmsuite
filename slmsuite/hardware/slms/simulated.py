"""
A simulated SLM.
"""

import numpy as np
from slmsuite.hardware.slms.slm import SLM
from slmsuite.misc.xp import as_backend, get_array_module

class SimulatedSLM(SLM):
    r"""
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
    gamma_sim : numpy.ndarray OR None
        User-defined phase response actually realized by each grayscale level, in units of
        :math:`2\pi`. ``None`` simulates the ideal linear response. This is the truth which
        :meth:`~slmsuite.hardware.cameraslms.FourierSLM.pixel_calibrate` measures, as
        opposed to the :attr:`~slmsuite.hardware.slms.slm.SLM.gamma` it recovers.
    """
    _pickle_data = SLM._pickle_data + ["gamma_sim"]

    def __init__(self, resolution, pitch_um=(8,8), source=None, gamma_sim=None, **kwargs):
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
            See :attr:`source`. Defaults to uniform illumination with a flat phase if
            ``None``. A measured source (``"amplitude"``/``"phase"``, as a wavefront
            calibration produces) is accepted in place of ``"amplitude_sim"`` and
            ``"phase_sim"``: the measured phase is a *correction*, so the aberration
            simulated is its negative, and an unmeasured half defaults to ideal.
        gamma_sim : array_like OR None
            See :attr:`gamma_sim`. Must span every one of the ``bitresolution`` levels;
            interpolate a sparse measurement with
            :meth:`~slmsuite.hardware.slms.slm.SLM.interpolate_gamma` first.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """
        kwargs.setdefault("settle_time_s", 0)
        super().__init__(resolution, pitch_um=pitch_um, **kwargs)

        self.gamma_sim = gamma_sim

        if not source:
            self.source["amplitude_sim"] = self.xp.ones_like(self.grid[0])
            self.source["phase_sim"] = self.xp.zeros_like(self.grid[0])
        else:
            # assert np.all([source[kw].shape == self.shape for kw in source.keys()]
            # ), "The shape of the provided phase profile must match the SLM resolution!"
            self.source.update(source)

            # Handle case where `source` only has real values from experiment. A
            # measured phase is the *correction* for an aberration, so the aberration
            # this SLM simulates is its negative. Whichever half was never measured
            # defaults to ideal illumination.
            # source is a _Source, so anything stored above already landed on self.xp.
            if "amplitude_sim" not in self.source:
                amplitude = self.source.get("amplitude", None)
                phase = self.source.get("phase", None)

                self.source["amplitude_sim"] = (
                    self.xp.ones_like(self.grid[0]) if amplitude is None else amplitude
                )
                self.source["phase_sim"] = (
                    self.xp.zeros_like(self.grid[0]) if phase is None else -phase
                )

        self.set_phase(None)

    @property
    def gamma_sim(self):
        """The phase response that this SLM simulates, or ``None`` for the ideal one."""
        return self._gamma_sim

    @gamma_sim.setter
    def gamma_sim(self, gamma_sim):
        if gamma_sim is None:
            self._gamma_sim = None
            return

        gamma_sim = np.ravel(np.array(gamma_sim, dtype=float))
        if len(gamma_sim) != self.bitresolution:
            raise ValueError(
                f"Expected gamma_sim to span all {self.bitresolution} levels; "
                f"got {len(gamma_sim)}."
            )
        if not np.all(np.isfinite(gamma_sim)):
            raise ValueError("Expected finite gamma_sim.")

        self._gamma_sim = gamma_sim

    def _unpickle(self, data):
        """
        Restores :attr:`gamma_sim` alongside the base SLM state. Set before ``super()``,
        which re-displays the pickled phase through this simulated response.
        """
        self.gamma_sim = data.get("gamma_sim", None)
        super()._unpickle(data)

    def _display2phase(self, display, dtype=np.float32):
        """
        Converts integer display array to floating-point phase realizing the simulated
        phase response (gamma_sim) or ideal linear response.
        """
        xp = get_array_module(display)
        if self.gamma_sim is None:
            return (
                display.astype(dtype)
                * (self._gamma_sign * 2 * np.pi / self.phase_scaling / self.bitresolution)
            )

        gamma_sim = as_backend(self.gamma_sim, xp).astype(dtype)
        return gamma_sim[display] * (self._gamma_sign * 2 * np.pi)

    def close(self):
        pass

    def _set_phase_hw(self, display):
        """Updates SLM.display to implement various physical artifacts of SLMs."""

        # FUTURE: apply physical effects directly to SLM.display

        return
