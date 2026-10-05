import pprint
import warnings

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import numpy and scipy dependencies.
import numpy as np
import scipy.fft as spfft
from scipy.ndimage import (
    affine_transform as sp_affine_transform,
    gaussian_filter as sp_gaussian_filter,
    gaussian_filter1d as sp_gaussian_filter1d,
)

from slmsuite import tqdm

# Shared host/device helpers, re-exported to every algorithms module via `import *`.
from slmsuite.misc.xp import as_backend, as_numpy, get_array_module, is_gpu_array

# Try to import cupy, but revert to base numpy/scipy upon ImportError.
try:
    import cupy as cp  # type: ignore[import-not-found]
    from cupyx import zeros_pinned as cp_zeros_pinned  # type: ignore[import-not-found]
    import cupyx.scipy.fft as cpfft  # type: ignore[import-not-found]
    from cupyx.scipy.ndimage import (
        affine_transform as cp_affine_transform,  # type: ignore[import-not-found]
        gaussian_filter as cp_gaussian_filter,  # type: ignore[import-not-found]
        gaussian_filter1d as cp_gaussian_filter1d,  # type: ignore[import-not-found]
    )
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
from slmsuite.holography.toolbox.phase import (
    CUDA_KERNELS,
    _load_cuda,
    _zernike_populate_basis_map,
    zernike_sum,
)
from slmsuite.misc.files import load_h5, save_h5
from slmsuite.misc.math import REAL_TYPES

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
    "WGS-Wu": {"feedback": "computational", "feedback_exponent": 0.5},
    "WGS-tanh": {"feedback": "computational", "feedback_factor": 0.2, "feedback_exponent": 0.5},
    "CG": {
        "feedback": "computational",
        "optimizer": "Adam",
        "optimizer_kwargs": {"lr": 0.1},
        "loss": None,
    },
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
STAT_GROUP_OPTIONS = [*FEEDBACK_OPTIONS, "experimental_ij", "experimental_knm"]
