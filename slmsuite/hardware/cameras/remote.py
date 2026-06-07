"""
Connects to a camera on a remote :class:`~slmsuite.hardware.remote.Server`.
"""
import warnings, time
from slmsuite.hardware.cameras.camera import Camera
from slmsuite.hardware.remote import _Client, DEFAULT_HOST, DEFAULT_PORT, DEFAULT_TIMEOUT

class RemoteCamera(_Client, Camera):
    """
    Connects to a camera on a remote :class:`~slmsuite.hardware.remote.Server`.
    """

    _pickle = Camera._pickle + [
        "server_attributes",
        "host",
        "port",
        "timeout",
        "latency_s",
    ]

    def __init__(
        self,
        name: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
        **kwargs
    ):
        r"""
        Connects to a camera on a remote :class:`~slmsuite.hardware.remote.Server`.

        This client only (1) reads the camera's attributes on initialization and (2) forwards
        :meth:`._get_image_hw`, :meth:`._get_images_hw`,
        :meth:`._set_exposure_hw`, :meth:`._get_exposure_hw`, and :meth:`.flush` commands.
        Class attributes are not concurrent (not kept up-to-date).
        Any vendor-specific functionality beyond those allowed commands must be handled on
        the server, as a security precaution.

        :param name:
            Name of the camera on the server to connect to.
        :param host:
            Hostname or IP address of the server. Defaults to ``"localhost"``.
        :param port:
            Port number of the server. Defaults to ``5025`` (commonly used for instrument control).
        :param timeout:
            Timeout in seconds for the connection. Defaults to ``1.0``.
        :param \*\*kwargs:
            See :meth:`.Camera.__init__` for permissible options, except for
            ``resolution``, ``bitdepth``, and ``pitch_um`` which are set by the server.
        """
        # Connect to the server.
        _Client.__init__(self, name, "camera", host, port, timeout)

        # Parse information about the SLM from the server.
        pickled = self.server_attributes["__meta__"]

        # Instantiate the superclass using this information.
        Camera.__init__(
            self,
            resolution=(pickled["shape"][1], pickled["shape"][0]),
            bitdepth=pickled["bitdepth"],
            pitch_um=pickled["pitch_um"],
            name=self.name,
            **kwargs
        )

    def close(self):
        pass

    ### Property Configuration ###

    def flush(self, timeout_s=1):
        """See :meth:`.Camera.flush`."""
        return self._com(
            command="flush",
            kwargs=dict(timeout_s=timeout_s)
        )

    def _get_exposure_hw(self):
        """See :meth:`.Camera._get_exposure_hw`."""
        return self._com(
            command="_get_exposure_hw",
        )

    def _set_exposure_hw(self, exposure_s):
        """See :meth:`.Camera._set_exposure_hw`."""
        return self._com(
            command="_set_exposure_hw",
            kwargs=dict(exposure_s=exposure_s)
        )

    def _set_woi_hw(self, woi):
        """See :meth:`.Camera._set_woi_hw`."""
        return self._com(command="_set_woi_hw", kwargs=dict(woi=woi))

    def _get_woi_hw(self):
        """See :meth:`.Camera._get_woi_hw`."""
        return self._com(command="_get_woi_hw")

    def _set_binning_hw(self, binning):
        """See :meth:`.Camera._set_binning_hw`."""
        return self._com(command="_set_binning_hw", kwargs=dict(binning=binning))

    def _get_binning_hw(self):
        """See :meth:`.Camera._get_binning_hw`."""
        return self._com(command="_get_binning_hw")

    def _get_image_hw(self, timeout_s):
        """See :meth:`.Camera._get_image_hw`."""
        t = time.perf_counter()
        img = self._com(
            command="_get_image_hw",
            kwargs=dict(timeout_s=timeout_s)
        )
        print(time.perf_counter()-t)
        return img

    def _get_images_hw(self, image_count, timeout_s, out=None):
        """See :meth:`.Camera._get_images_hw`."""
        if out is not None:
            warnings.warn("Remote camera does not support in-place operations.")

        return self._com(
            command="_get_images_hw",
            kwargs=dict(image_count=image_count, timeout_s=timeout_s)
        )