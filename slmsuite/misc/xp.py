"""
Helpers for bridging the two array backends that :mod:`slmsuite` runs on:
:mod:`numpy` on the host and :mod:`cupy` on the GPU.

Note
~~~~
This is a placeholder for future fully-featured backend handling in slmsuite.
"""

import numpy as np

try:
    import cupy as cp                                   # type: ignore
except ImportError:
    cp = np


def is_gpu_array(array):
    """
    Whether ``array`` is backed by CUDA device memory.
    """
    return cp is not np and (
        isinstance(array, cp.ndarray) or hasattr(array, "__cuda_array_interface__")
    )


def get_array_module(array):
    """
    The array module backing ``array``: :mod:`cupy` if it lives on the device,
    :mod:`numpy` otherwise.
    """
    return cp if is_gpu_array(array) else np


def as_numpy(array):
    """
    ``array`` as a host :class:`numpy.ndarray`, copying off the device if needed.
    ``None`` passes through.
    """
    if array is None:
        return None
    if is_gpu_array(array):
        return array.get()
    return np.asarray(array)


def as_backend(array, xp):
    """
    ``array`` on the ``xp`` backend (:mod:`numpy` or :mod:`cupy`), leaving it alone if
    it is already there. ``None`` passes through.
    """
    if array is None:
        return None
    if xp is np:
        return as_numpy(array)
    return xp.asarray(array)
