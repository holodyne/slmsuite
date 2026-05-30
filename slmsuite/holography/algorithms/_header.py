import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
from tqdm.auto import tqdm
import warnings
import pprint

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

# Import the array-backend abstraction (numpy / cupy / torch dispatch) BEFORE ``import torch``.
# backend warms up cupy's cuBLAS before it imports torch, working around a Windows cuBLAS
# load-order conflict (see slmsuite.misc.backend). Importing it first guarantees the warmup
# runs before torch's cuBLAS is loaded into the process.
from slmsuite.misc import backend
from slmsuite.misc.backend import get_module, is_torch, is_cupy, is_autograd, to_numpy, resolve_backend

try:
    import torch
except:
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
# Caution: The order of these algorithms is used in other parts of the code
#          such as ALGORITHM_INDEX to numerically encode feedback methods.
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
ALGORITHM_INDEX = {key : i for i, key in enumerate(ALGORITHM_DEFAULTS.keys())}

# List of feedback options. See the documentation for the feedback keyword in optimize().
FEEDBACK_OPTIONS = [
    "computational",
    "computational_spot",
    "experimental",
    "experimental_spot",
    "external_spot",
]