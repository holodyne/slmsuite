"""
Connects to an SLM on a remote :class:`~slmsuite.hardware.remote.Server`.
"""

from typing import ClassVar

from slmsuite.hardware.remote import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_TIMEOUT, _Client
from slmsuite.hardware.slms.slm import SLM


class RemoteSLM(_Client, SLM):
    """
    Connects to an SLM on a remote :class:`~slmsuite.hardware.remote.Server`.
    """

    _pickle: ClassVar[list[str]] = [*SLM._pickle, "server_attributes", "host", "port", "timeout", "latency_s"]

    def __init__(
        self,
        name: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
        wav_um: float | None = None,
        settle_time_s: float | None = None,
    ):
        r"""
        Connects to an SLM on a remote :class:`~slmsuite.hardware.remote.Server`.

        This client only (1) reads the SLM's attributes on initialization and (2) forwards
        :meth:`._set_phase_hw` commands. Class attributes are not concurrent (not kept up-to-date).
        Any vendor-specific functionality beyond :meth:`._set_phase_hw` must be handled on
        the server, as a security precaution.

        :param name:
            Name of the SLM on the server to connect to.
        :param host:
            Hostname or IP address of the server. Defaults to ``"localhost"``.
        :param port:
            Port number of the server. Defaults to ``5025`` (commonly used for instrument control).
        :param timeout:
            Timeout in seconds for the connection. Defaults to ``1.0``.
        :param wav_um:
            Wavelength of operation in microns. Defaults to whatever is set on the server.
        :param settle_time_s:
            Settle time in seconds. Defaults to whatever is set on the server.
        """
        # Connect to the server.
        _Client.__init__(self, name, "slm", host, port, timeout)

        # Parse information about the SLM from the server.
        pickled = self.server_attributes["__meta__"]

        # Instantiate the superclass using this information.
        SLM.__init__(
            self,
            resolution=(pickled["shape"][1], pickled["shape"][0]),
            bitdepth=pickled["bitdepth"],
            name=self.name,
            wav_um=pickled["wav_um"] if wav_um is None else wav_um,
            wav_design_um=pickled["wav_design_um"],
            pitch_um=pickled["pitch_um"],
            settle_time_s=pickled["settle_time_s"] if settle_time_s is None else settle_time_s,
        )

    def close(self):
        pass

    def _set_phase_hw(self, display, **kwargs):
        """
        Hardware-specific implementation for remote SLM connection.

        See :meth:`SLM._set_phase_hw` for the base class documentation.

        Parameters
        ----------
        display
            Integer data to display on the SLM. See :meth:`.SLM._set_phase_hw`.
        """
        self._com(command="_set_phase_hw", kwargs=dict(display=display, **kwargs))
