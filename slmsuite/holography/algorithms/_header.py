import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
from tqdm.auto import tqdm
import warnings
import pprint
from collections import OrderedDict

# Import numpy and scipy dependencies.
import numpy as np
import scipy.fft as spfft
from scipy.ndimage import gaussian_filter1d as sp_gaussian_filter1d
from scipy.ndimage import affine_transform as sp_affine_transform
from scipy.ndimage import gaussian_filter as sp_gaussian_filter

# Try to import cupy, but revert to base numpy/scipy upon ImportError.
try:
    import cupy as cp                                                           # type: ignore
    import cupyx.scipy.fft as cpfft                                             # type: ignore
    from cupyx import zeros_pinned as cp_zeros_pinned                           # type: ignore
    from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d   # type: ignore
    from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter       # type: ignore
    from cupyx.scipy.ndimage import affine_transform as cp_affine_transform     # type: ignore
except ImportError:
    cp = np
    cpfft = spfft
    cp_zeros_pinned = np.zeros
    cp_gaussian_filter1d = sp_gaussian_filter1d
    cp_gaussian_filter = sp_gaussian_filter
    cp_affine_transform = sp_affine_transform
    warnings.warn(
        "cupy is not installed; using numpy. Install cupy for faster GPU-based holography."
    )

# Warm up cupy's cuBLAS handle before PyTorch is imported, else CUBLAS_STATUS_INVALID_VALUE.
if cp is not np:
    try:
        _w = cp.zeros((2, 2), dtype=cp.float32)
        _w @ _w
        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass

try:
    import torch
except Exception:
    torch = None

# Import helper functions
from slmsuite.holography import analysis, toolbox
from slmsuite.holography.toolbox import phase as tphase
from slmsuite.holography.toolbox.phase import CUDA_KERNELS, _zernike_populate_basis_map, zernike_sum, _load_cuda
from slmsuite.misc.math import REAL_TYPES
from slmsuite.misc.files import save_h5, load_h5

# List of algorithms and default parameters.
# See algorithm documentation for parameter definitions.
# Tip: In general, decreasing the feedback exponent (from 1) improves
#      stability at the cost of slower convergence. The default (0.8)
#      is an empirically derived value for a reasonable tradeoff.
ALGORITHM_DEFAULTS = {
    "GS": {"feedback": "computational"},  # No feedback for bare GS, but initializes var.
    "WGS-Leonardo": {"feedback": "computational", "feedback_exponent": 0.8},
    "WGS-Kim": {
        "feedback": "computational",
        "fix_phase_efficiency": None,
        "fix_phase_iteration": 10,
        "feedback_exponent": 0.8,
    },
    "WGS-Nogrette": {"feedback": "computational", "feedback_factor": 0.1},
    "WGS-Wu": {"feedback": "computational", "feedback_exponent": .5},
    "WGS-tanh": {"feedback": "computational", "feedback_factor": .2, "feedback_exponent": .5},
    "CG" : {
        "feedback": "computational",
        "optimizer": "Adam",
        "optimizer_kwargs": {"lr": .1},
        "loss": None
    }
}

# List of feedback options. See the documentation for the feedback keyword in optimize().
FEEDBACK_OPTIONS = [
    "computational",
    "computational_spot",
    "experimental",
    "experimental_spot",
    "external_spot",
]

# List of statistics groups. See the documentation for the stat_groups keyword in optimize().
STAT_GROUP_OPTIONS = FEEDBACK_OPTIONS + ["experimental_ij", "experimental_knm"]


class LRUCache:
    """
    Small least-recently-used cache backing the transparent caches that memoize expensive
    per-geometry setup, such as the ``"ij"`` -> ``"knm"`` transformation in
    :meth:`~slmsuite.holography.algorithms.FeedbackHologram.ijcam_to_knmslm`.

    Entries hold device arrays, so the capacity is the point: it bounds how much GPU
    memory the cache can pin for the life of the process. :meth:`clear` releases all of
    it. Size ``maxsize`` to how large an entry is, not just to the hit rate.

    Parameters
    ----------
    maxsize : int
        Number of entries to retain. The least recently used are evicted beyond this.
    """

    def __init__(self, maxsize):
        self.maxsize = int(maxsize)
        self._data = OrderedDict()

    def get(self, key):
        """The entry for ``key``, marked most-recently-used, or ``None`` on a miss."""
        value = self._data.get(key)
        if value is not None:
            self._data.move_to_end(key)
        return value

    def put(self, key, value):
        """Store ``value`` under ``key``, evict past capacity, and return ``value``."""
        self._data[key] = value
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)
        return value

    def clear(self):
        """Drop every entry, releasing whatever memory they hold."""
        self._data.clear()

    def __len__(self):
        return len(self._data)
