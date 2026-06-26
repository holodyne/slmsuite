"""
Deprecated. Use :class:`.Santec` with ``backend="usb"`` instead::

    from slmsuite.hardware.slms.santec import Santec
    slm = Santec("AB000001", backend="usb")
"""
import warnings as _warnings
from .santec import Santec as _Santec


def SantecUSB(*args, **kwargs):
    """
    Deprecated alias. Use :class:`.Santec` with ``backend="usb"`` instead.

    Args:
        *args: Forwarded to :class:`.Santec`.
        **kwargs: Forwarded to :class:`.Santec`.

    Returns:
        Santec: Instance opened with the USB backend.
    """
    _warnings.warn(
        "SantecUSB is deprecated; use Santec(backend='usb') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _Santec(*args, backend="usb", **kwargs)


__all__ = ["SantecUSB"]
