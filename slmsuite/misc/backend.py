"""
Array-backend abstraction for :mod:`slmsuite`.

:mod:`slmsuite` is *backend-polymorphic*: the same public methods (``set_phase``,
``get_image``, ``optimize``, ...) operate on whichever array library backs the data
that was passed in. 

The guiding principle is **dispatch on the array, not on the method name**: callers
resolve the active namespace with :func:`get_module` (analogous to
:func:`cupy.get_array_module`, extended to torch) and branch only where an operation is
genuinely backend-specific (in-place ``out=`` buffers, host transfer, integer casting).

This module is intentionally dependency-light (only :mod:`numpy`, with optional
:mod:`cupy`/:mod:`torch`) and lives in :mod:`slmsuite.misc` so that both
:mod:`slmsuite.holography.toolbox` and :mod:`slmsuite.holography.algorithms` can import
it without a circular dependency.
"""

import operator as _operator

import numpy as np

# Optional cupy. 
try:
    import cupy as cp
except ImportError:
    cp = np

def _warmup_cupy_cublas_before_torch():
    """
    Establish cupy's cuBLAS handle *before* :mod:`torch` is imported.

    On Windows, when cupy's bundled cuBLAS is loaded into the process and torch's cuBLAS is
    then loaded afterwards, cupy's ``gemmEx`` can succeed exactly once and then fail every
    subsequent call with ``CUBLAS_STATUS_INVALID_VALUE`` (the two cuBLAS DLLs conflict). Issuing
    one tiny cupy matmul here -- after cupy is imported but before ``import torch`` below --
    forces cupy to bind its own cuBLAS handle first, which then survives torch's later load.
    This is a no-op when cupy/GPU is unavailable, and any failure is swallowed (the warmup is a
    best-effort mitigation, never a hard dependency).
    """
    if cp is np:
        return
    try:
        _w = cp.zeros((2, 2), dtype=cp.float32)
        _w @ _w
        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


_warmup_cupy_cublas_before_torch()

# Optional torch. 
try:
    import torch
    import torch.fft
except ImportError:
    torch = None


#: Public API. Several names intentionally shadow Python builtins / numpy functions
#: (``abs``, ``copy``, ``power``, ``where``, ``clip``, ``real``, ``imag``, ``conj``, ``angle``)
#: so callers can write ``backend.<name>`` as a drop-in for the numpy equivalent. ``cp`` and
#: ``torch`` are re-exports so dependents can resolve the active modules / sentinels.
__all__ = [
    "cp", "torch",
    # Dispatch primitives.
    "get_module", "is_torch", "is_cupy", "is_autograd", "is_complex", "to_numpy", "to_backend",
    "resolve_backend", "zeros", "asarray",
    # Elementwise ops.
    "add", "subtract", "multiply", "divide", "power", "reciprocal", "exp", "tanh",
    "abs", "conj", "angle", "real", "imag", "sinc", "arctan2", "clip", "mod",
    "nan_to_num", "nanmean", "nansum", "isclose", "where", "where_replace", "norm",
    # Array construction / manipulation.
    "copy", "ones_like", "zeros_like", "vstack", "scatter_update", "pad", "embed", "unpad_indices",
    # FFT family.
    "fft2", "ifft2", "fftshift", "ifftshift",
]

# ===================
# Dispatch primitives
# ===================

def get_module(array):
    """
    Return the array namespace that should be used to operate on ``array``.

    This is the torch-aware generalization of :func:`cupy.get_array_module`.

    Parameters
    ----------
    array : numpy.ndarray OR cupy.ndarray OR torch.Tensor OR scalar
        The array whose backend should be resolved. Scalars and unrecognized
        types fall back to :mod:`numpy`.

    Returns
    -------
    module
        :mod:`torch`, :mod:`cupy`, or :mod:`numpy`.
    """
    if torch is not None and isinstance(array, torch.Tensor):
        return torch
    if cp is not np and isinstance(array, cp.ndarray):
        return cp
    return np


def is_torch(array):
    """Return ``True`` if ``array`` is a :class:`torch.Tensor`."""
    return torch is not None and isinstance(array, torch.Tensor)


def is_cupy(array):
    """Return ``True`` if ``array`` is a :class:`cupy.ndarray`."""
    return cp is not np and isinstance(array, cp.ndarray)


def is_autograd(array):
    """
    Return ``True`` if ``array`` participates in an autograd graph.

    Used to decide whether in-place/``out=`` optimizations are safe (they are not, for
    a tensor that requires gradients).
    """
    return is_torch(array) and (array.requires_grad or array.grad_fn is not None)


def is_complex(array):
    """Backend-aware check for complex type."""
    if is_torch(array):
        return torch.is_complex(array)
    xp = get_module(array)
    return xp.iscomplexobj(array)


def to_numpy(array):
    """
    Move ``array`` to host as a :mod:`numpy` array, for any supported backend.

    Generalizes the scattered ``if cp != np: array = array.get()`` idiom and additionally
    detaches torch tensors from the autograd graph.

    Parameters
    ----------
    array : numpy.ndarray OR cupy.ndarray OR torch.Tensor

    Returns
    -------
    numpy.ndarray
    """
    if is_torch(array):
        return array.detach().cpu().numpy()
    if is_cupy(array):
        return array.get()
    return np.asarray(array)


def to_backend(data, like_array):
    """
    Converts data to match the backend, device, and dtype of the reference like_array.
    """
    if data is None:
        return None
    
    # Fast path: If data is already a standard Python scalar or a numpy scalar
    # (which we convert to a Python scalar via .item()), return it directly.
    # This avoids expensive 0D array/tensor creation and device transfers.
    if isinstance(data, (int, float, complex)):
        return data
    if isinstance(data, (np.number, np.bool_)):
        return data.item()
    
    xp = get_module(like_array)
    if xp is torch:
        if isinstance(data, torch.Tensor):
            # Fast path: already on the target device/dtype -> return as-is. ``.to()`` is not free
            # and the binary op factories call this on both operands every iteration of a GS/WGS
            # loop, where one operand is invariably already the reference tensor.
            if data.device == like_array.device and data.dtype == like_array.dtype:
                return data
            return data.to(device=like_array.device, dtype=like_array.dtype)
        t = _get_torch_tensor_from_cupy(data)
        return t.to(device=like_array.device, dtype=like_array.dtype)
    else:
        if is_torch(data):
            data = to_numpy(data)
        if xp is cp:
            return cp.asarray(data, dtype=like_array.dtype)
        else:
            return np.asarray(data, dtype=like_array.dtype)


def resolve_backend(name=None, device=None):
    """
    Resolve a backend selector (and optional device) to ``(module, device)``.

    Parameters
    ----------
    name : str OR None
        ``None`` / ``"auto"`` selects the default (``cupy`` if installed, else ``numpy``).
        ``"numpy"``/``"np"`` forces numpy; ``"cupy"``/``"cp"``/``"gpu"`` forces cupy;
        ``"torch"``/``"pytorch"`` selects torch.
    device : str OR torch.device OR None
        Only meaningful for torch. Defaults to ``"cuda"`` if available, else ``"cpu"``.

    Returns
    -------
    (module, device)
        ``module`` is :mod:`numpy`, :mod:`cupy`, or :mod:`torch`. ``device`` is a
        :class:`torch.device` for the torch backend, else ``None``.
    """
    n = name.lower() if isinstance(name, str) else name
    if n is None or n == "auto":
        return cp, None
    if n in ("numpy", "np"):
        return np, None
    if n in ("cupy", "cp", "gpu"):
        if cp is np:
            raise ValueError("backend='cupy' requested but cupy is not installed.")
        return cp, None
    if n in ("torch", "pytorch"):
        if torch is None:
            raise ValueError("backend='torch' requested but torch is not installed.")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch, torch.device(device)
    raise ValueError(f"Unrecognized backend {name!r}; use 'numpy', 'cupy', or 'torch'.")


def zeros(shape, xp, dtype, device=None):
    """Backend-aware zero array of ``dtype`` (a numpy dtype) in namespace ``xp``."""
    if xp is torch:
        return torch.zeros(tuple(shape), dtype=_torch_dtype(dtype), device=device)
    return xp.zeros(shape, dtype=dtype)


def asarray(data, xp, dtype, device=None):
    """
    Create/convert ``data`` as an array in namespace ``xp`` with numpy ``dtype`` (on ``device``
    for torch). Bridges numpy<->cupy<->torch via :func:`to_numpy`/dlpack as needed.
    """
    if xp is torch:
        td = _torch_dtype(dtype)
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=td)
        return torch.as_tensor(to_numpy(data), dtype=td, device=device)
    if xp is cp:
        if is_torch(data):
            data = to_numpy(data)
        return cp.asarray(data, dtype=dtype)
    return np.asarray(to_numpy(data), dtype=dtype)


# Internal dispatch helpers

def _out_for(out, xp):
    """
    Return ``out`` only if it is a preallocated buffer matching the active numpy/cupy namespace
    ``xp``; otherwise ``None``.

    This lets call sites pass ``out=self.buffer`` *unconditionally*: the torch branch of each op
    never reaches here (it computes functionally), and a stale buffer left over from a different
    backend -- e.g. a torch tensor still in ``self.phase`` during a later numpy pass -- is safely
    ignored (a fresh array is allocated) instead of raising. It centralizes the autograd/backend
    ``out=`` safety that previously lived as ``out=None if is_autograd(...) else buf`` ternaries at
    every call site.
    """
    if out is None:
        return None
    return out if get_module(out) is xp else None


def _get_torch_tensor_from_cupy(array):
    if torch is None:
        raise RuntimeError("Cannot get torch tensor without torch. Something is wrong.")

    if array is None:
        return None
    
    if cp is np or not isinstance(array, cp.ndarray):
        return torch.from_numpy(np.asarray(array))
    else:
        if not array.flags.c_contiguous:
            array = cp.ascontiguousarray(array)
        # Use standard modern from_dlpack directly to avoid deprecation warnings
        return torch.from_dlpack(array)


def _torch_dtype(dtype):
    """Map a numpy real/complex dtype to the matching :class:`torch.dtype`."""
    return {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.complex64): torch.complex64,
        np.dtype(np.complex128): torch.complex128,
    }[np.dtype(dtype)]


# --- Elementwise op factories -------------------------------------------------------------
#
# Most elementwise wrappers follow one of two patterns that differ only by name:
#   * numpy/cupy: dispatch to ``xp.<name>(..., out=out)``, preserving the in-place fast path.
#   * torch:      drop ``out=`` (in-place writes sever the autograd graph) and compute
#                 functionally, coercing operands to a common device/dtype first.

def _coerce_pair(a, b):
    """Coerce ``(a, b)`` onto the torch tensor's device/dtype when either operand is torch."""
    ref = a if is_torch(a) else b
    return to_backend(a, ref), to_backend(b, ref)


def _make_binary(np_name, torch_op):
    """Build a backend-aware binary op (see module note above)."""
    def op(a, b, out=None):
        if is_torch(a) or is_torch(b):
            a, b = _coerce_pair(a, b)
            return torch_op(a, b)
        xp = get_module(a)
        return getattr(xp, np_name)(a, b, out=_out_for(out, xp))
    op.__name__ = op.__qualname__ = np_name
    op.__doc__ = (
        f"Backend-aware {np_name}. ``out=`` honored on numpy/cupy (in-place); "
        "dropped on torch (autograd-safe, functional)."
    )
    return op


def _make_unary(np_name, torch_name, has_out=True):
    """Build a backend-aware unary op. ``has_out=False`` for non-ufunc numpy fns (angle, sinc)."""
    if has_out:
        def op(x, out=None):
            if is_torch(x):
                return getattr(torch, torch_name)(x)
            xp = get_module(x)
            return getattr(xp, np_name)(x, out=_out_for(out, xp))
    else:
        def op(x):
            if is_torch(x):
                return getattr(torch, torch_name)(x)
            return getattr(get_module(x), np_name)(x)
    op.__name__ = op.__qualname__ = np_name
    op.__doc__ = (
        f"Backend-aware {np_name}."
        + (" ``out=`` honored on numpy/cupy; dropped on torch." if has_out else "")
    )
    return op


# Binary ops. ``operator.*`` maps cleanly onto torch's overloaded dunder methods
# (e.g. ``operator.pow(a, b)`` -> ``a ** b`` -> ``torch.pow``), so no torch reference is
# needed at bind time and ``power`` correctly resolves to ``torch.pow`` (not the nonexistent
# ``torch.power``).
add = _make_binary("add", _operator.add)
subtract = _make_binary("subtract", _operator.sub)
multiply = _make_binary("multiply", _operator.mul)
divide = _make_binary("divide", _operator.truediv)
power = _make_binary("power", _operator.pow)

# Unary ops.
reciprocal = _make_unary("reciprocal", "reciprocal")
exp = _make_unary("exp", "exp")
tanh = _make_unary("tanh", "tanh")
abs = _make_unary("abs", "abs")
conj = _make_unary("conj", "conj")
angle = _make_unary("angle", "angle", has_out=False)   # numpy.angle is not a ufunc (no out=)
sinc = _make_unary("sinc", "sinc", has_out=False)      # numpy.sinc is not a ufunc (no out=)


# --- Hand-coded Elementwise Math ---

def arctan2(y, x, out=None):
    """Backend-aware arctan2 function."""
    if is_torch(y) or is_torch(x):
        ref = y if is_torch(y) else x
        return torch.arctan2(to_backend(y, ref), to_backend(x, ref))
    xp = get_module(y)
    return xp.arctan2(y, x, out=_out_for(out, xp))


def clip(array, a_min, a_max, out=None):
    """Backend-aware clip. ``out=`` honored on numpy/cupy; dropped on torch (autograd-unsafe)."""
    if is_torch(array):
        return torch.clamp(array, min=a_min, max=a_max)
    xp = get_module(array)
    return xp.clip(array, a_min, a_max, out=_out_for(out, xp))


def mod(x, y, out=None):
    """Backend-aware modulo. ``out=`` honored on numpy/cupy; dropped on torch (autograd-unsafe)."""
    if is_torch(x):
        return torch.remainder(x, y)
    xp = get_module(x)
    return xp.mod(x, y, out=_out_for(out, xp))


# --- NaN-Safe Operations & Reductions ---

def nan_to_num(array, copy=True, nan=0.0, posinf=None, neginf=None, out=None):
    """Backend-aware nan_to_num function."""
    if is_torch(array):
        if is_autograd(array) or copy:
            return torch.nan_to_num(array, nan=nan, posinf=posinf, neginf=neginf)
        else:
            if out is None:
                out = array
            return torch.nan_to_num(array, nan=nan, posinf=posinf, neginf=neginf, out=out)
    xp = get_module(array)
    try:
        return xp.nan_to_num(array, copy=copy, nan=nan, posinf=posinf, neginf=neginf)
    except TypeError:
        return xp.nan_to_num(array, copy=copy, nan=nan)


def nanmean(array, axis=None, keepdims=False):
    """Backend-aware nanmean function."""
    if is_torch(array):
        if axis is None:
            return torch.nanmean(array)
        else:
            return torch.nanmean(array, dim=axis, keepdim=keepdims)
    xp = get_module(array)
    return xp.nanmean(array, axis=axis, keepdims=keepdims)


def nansum(array, axis=None, keepdims=False):
    """Backend-aware nansum function."""
    if is_torch(array):
        if axis is None:
            return torch.nansum(array)
        else:
            return torch.nansum(array, dim=axis, keepdim=keepdims)
    xp = get_module(array)
    return xp.nansum(array, axis=axis, keepdims=keepdims)


# --- Comparisons and Logical Operations ---

def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Backend-aware isclose function."""
    xp = get_module(a)
    if xp is torch:
        return torch.isclose(a, to_backend(b, a), rtol=rtol, atol=atol, equal_nan=equal_nan)
    return xp.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def where(condition, x, y):
    """Backend-aware where function."""
    xp = get_module(x)
    if xp is torch:
        return torch.where(condition, x, to_backend(y, x))
    return xp.where(condition, x, y)


def where_replace(array, condition, value):
    """
    Unified conditional replacement.
    NumPy/CuPy: modifies array in-place where condition is True, returns it.
    PyTorch: uses out-of-place torch.where (to preserve autograd), returns it.
    """
    if is_torch(array):
        val_tensor = torch.as_tensor(value, dtype=array.dtype, device=array.device)
        return torch.where(condition, val_tensor, array)
    xp = get_module(array)
    array[condition] = value
    return array


# --- Vector / Matrix Norm ---

def norm(matrix):
    """
    Computes the root of the sum of squares of the given matrix.
    Generalized backend-aware version of Hologram._norm.
    """
    xp = get_module(matrix)
    if is_complex(matrix):
        return xp.sqrt(nansum(xp.square(xp.abs(matrix))))
    else:
        return xp.sqrt(nansum(xp.square(matrix)))


# ===================================
# Array construction and manipulation
# ===================================

def copy(array, dtype=None):
    """
    Backend-aware copy/clone of an array.
    Preserves autograd graphs for PyTorch, and copies for NumPy/CuPy.
    """
    if is_torch(array):
        res = array.clone()
        if dtype is not None:
            res = res.to(dtype=dtype)
        return res
    xp = get_module(array)
    return xp.array(array, copy=True, dtype=dtype)


def real(array):
    """Real part of a (possibly complex) array. ``.real`` is uniform across numpy/cupy/torch."""
    return array.real


def imag(array):
    """Imaginary part of a (possibly complex) array. ``.imag`` is uniform across backends."""
    return array.imag


def ones_like(array):
    """Array of ones with same shape/backend/device/dtype."""
    if is_torch(array):
        return torch.ones_like(array)
    xp = get_module(array)
    return xp.ones_like(array)


def zeros_like(array):
    """Array of zeros with same shape/backend/device/dtype."""
    if is_torch(array):
        return torch.zeros_like(array)
    xp = get_module(array)
    return xp.zeros_like(array)


def vstack(tensors):
    """Backend-aware vstack function."""
    xp = get_module(tensors[0])
    if xp is torch:
        return torch.vstack(tensors)
    return xp.vstack(tensors)


def scatter_update(array, indices, values):
    """
    Updates array[indices] = values.
    NumPy/CuPy: In-place assignment.
    PyTorch: Out-of-place clone and index assignment (to preserve autograd).
    """
    if is_torch(array):
        new_array = array.clone()
        new_array[indices] = values
        return new_array
    else:
        array[indices] = values
        return array


def pad(matrix, pad_width):
    """
    Backend-aware centered zero-pad, matching :func:`numpy.pad` semantics.

    Parameters
    ----------
    matrix : numpy.ndarray OR cupy.ndarray OR torch.Tensor
        2D array to pad.
    pad_width : ((int, int), (int, int))
        ``((before_0, after_0), (before_1, after_1))`` as accepted by :func:`numpy.pad`.

    Returns
    -------
    Padded array, of the same backend as ``matrix``.
    """
    if is_torch(matrix):
        # torch.nn.functional.pad takes the last axis first: (left, right, top, bottom).
        (b0, a0), (b1, a1) = pad_width
        return torch.nn.functional.pad(matrix, (b1, a1, b0, a0), mode="constant", value=0)
    xp = get_module(matrix)
    return xp.pad(matrix, pad_width, mode="constant", constant_values=0)


def unpad_indices(outer_shape, inner_shape):
    """
    Helper to compute centered slicing indices to extract or insert
    an inner_shape array inside an outer_shape array.
    Matches standard centered unpadding.
    """
    d0 = (inner_shape[0] - outer_shape[0]) / 2.0
    d1 = (inner_shape[1] - outer_shape[1]) / 2.0

    pad_b = int(np.floor(-d0))
    pad_t = int(outer_shape[0] - np.ceil(-d0))
    pad_l = int(np.floor(-d1))
    pad_r = int(outer_shape[1] - np.ceil(-d1))

    return (pad_b, pad_t, pad_l, pad_r)


def embed(src, dst_shape, active_shape, out=None):
    """
    Centered pad-and-embed src (of active_shape) into dst_shape.
    For NumPy/CuPy, if `out` is provided, fills in-place.
    For PyTorch, operates functionally to preserve autograd gradients.
    """
    if is_torch(src):
        d0 = dst_shape[0] - active_shape[0]
        d1 = dst_shape[1] - active_shape[1]
        pad_b = d0 // 2
        pad_t = d0 - pad_b
        pad_l = d1 // 2
        pad_r = d1 - pad_l
        # Torch pad expects (left, right, top, bottom)
        return torch.nn.functional.pad(src, (pad_l, pad_r, pad_b, pad_t), mode="constant", value=0)

    xp = get_module(src)
    if out is None or out.shape != dst_shape or is_torch(out):
        out = xp.zeros(dst_shape, dtype=src.dtype)
    else:
        out.fill(0)

    i0, i1, i2, i3 = unpad_indices(dst_shape, active_shape)
    out[i0:i1, i2:i3] = src
    return out


# =============
# FFT Interface
# =============

def fft2(x, norm="ortho"):
    """2D Fast Fourier Transform."""
    if is_torch(x):
        return torch.fft.fft2(x, norm=norm)
    xp = get_module(x)
    return xp.fft.fft2(x, norm=norm)


def ifft2(x, norm="ortho"):
    """Inverse 2D Fast Fourier Transform."""
    if is_torch(x):
        return torch.fft.ifft2(x, norm=norm)
    xp = get_module(x)
    return xp.fft.ifft2(x, norm=norm)


def fftshift(x, axes=None):
    """Shift the zero-frequency component to the center of the spectrum."""
    if is_torch(x):
        return torch.fft.fftshift(x, dim=axes)
    xp = get_module(x)
    return xp.fft.fftshift(x, axes=axes)


def ifftshift(x, axes=None):
    """Inverse shift the zero-frequency component."""
    if is_torch(x):
        return torch.fft.ifftshift(x, dim=axes)
    xp = get_module(x)
    return xp.fft.ifftshift(x, axes=axes)
