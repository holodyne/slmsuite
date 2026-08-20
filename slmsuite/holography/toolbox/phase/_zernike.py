"""
Zernike polynomials and related functions.
"""
import os
import threading
import warnings
import weakref
from collections import OrderedDict
from functools import cached_property

import numpy as np

try:
    import cupy as cp  # type: ignore
except ImportError:
    cp = np
from math import comb, factorial

import matplotlib.pyplot as plt
from scipy import special

from slmsuite.holography.toolbox import Aperture, _process_grid
from slmsuite.misc.xp import get_array_module
from slmsuite._plotting import _slmsuite_plt_show
from slmsuite._logging import make_logger

logger = make_logger(__name__)

# Load CUDA code. This is used for cupy.RawKernels in this file and elsewhere.

def _load_cuda():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda.cu"), 'r') as file:
        CUDA_KERNELS = file.read()

    return CUDA_KERNELS

try:
    CUDA_KERNELS = _load_cuda()
except Exception:
    warnings.warn("Unable to load toolbox/cuda.cu; cannot use custom GPU kernels.")
    CUDA_KERNELS = None

# Zernike.

ZERNIKE_INDEXING_DIMENSION = {"ansi" : 1, "noll" : 1, "fringe" : 1, "wyant" : 1, "radial" : 2}
ZERNIKE_INDEXING = ZERNIKE_INDEXING_DIMENSION.keys()
ZERNIKE_INDEX_UNDEFINED = np.iinfo(int).min
ZERNIKE_NAMES = [
    # Oth order
    "Piston",

    # 1st order
    "Vertical tilt",
    "Horizontal tilt",

    # 2nd order
    "Oblique astigmatism",
    "Defocus",
    "Vertical astigmatism",

    # 3rd order
    "Vertical trefoil",
    "Vertical coma",
    "Horizontal coma",
    "Oblique trefoil",

    # 4th order
    "Oblique quadrafoil",
    "Oblique secondary astigmatism",
    "Spherical aberration",
    "Vertical secondary astigmatism",
    "Vertical quadrafoil",

    # 5th order
    "Vertical pentafoil",
    "Vertical secondary trefoil",
    "Vertical secondary coma",
    "Horizontal secondary coma",
    "Oblique secondary trefoil",
    "Oblique pentafoil",

    # 6th order
    "Oblique hexafoil",
    "Oblique secondary quadrafoil",
    "Oblique tertiary astigmatism",
    "Secondary spherical aberration",
    "Vertical tertiary astigmatism",
    "Vertical secondary quadrafoil",
    "Vertical hexafoil",
]



def zernike_order_number(radial_order):
    """
    Get the number of Zernike polynomials under (inclusive) a given radial order.

    For radial order :math:`n`, this evaluates to :math:`(n+1)(n+2)/2`.

    Parameters
    ----------
    radial_order : int
        Maximum radial order to include.

    Returns
    -------
    max_index : int
        Maximum Zernike index for this radial order (ANSI 0-based)
    """
    return (radial_order + 1) * (radial_order + 2) // 2


def zernike_convert_index(indices, from_index="ansi", to_index="ansi"):
    """
    Helper function for converting between Zernike indexing conventions.

    Currently supported conventions:

    -  ``"radial"``
        The standard 2-dimensional :math:`n,l` indexing for
        `Zernike polynomials <https://en.wikipedia.org/wiki/Zernike_polynomials>`_,
        where :math:`n` is the radial index
        and :math:`l` is the azimuthal index.
        Denoted :math:`Z_n^l`.

    -  ``"ansi"``
        1-dimensional (0-indexed) `ANSI indices
        <https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices>`_.
        **This is the default** :mod:`slmsuite` **index.**
        Denoted :math:`Z_i`.

    -  ``"noll"``
        1-dimensional (1-indexed) `Noll indices
        <https://en.wikipedia.org/wiki/Zernike_polynomials#Noll's_sequential_indices>`_.

    -  ``"fringe"``
        1-dimensional (1-indexed) `Fringe indices
        <https://en.wikipedia.org/wiki/Zernike_polynomials#Fringe/University_of_Arizona_indices>`_.
        Defined only for the canonical 37-term set, where the last term
        :math:`Z_{37} = Z_{12}^0` breaks the pattern of the first 36.

    -  ``"wyant"``
        1-dimensional (0-indexed) `Wyant indices
        <https://en.wikipedia.org/wiki/Zernike_polynomials#Wyant_indices>`_.
        Equivalent to ``"fringe"``, except starting with zero instead of one.

    Parameters
    ----------
    indices : array_like
        List of indices of shape ``(N, D)`` where ``D`` is the dimension of the indexing
        (1, apart from ``"radial"`` indexing which has a dimension of 2).
    from_index, to_index : str
        Zernike index convention. Must be supported.

    Returns
    -------
    indices_converted : numpy.ndarray
        List of indices of shape ``(N, D)`` where ``D`` is the dimension of the indexing
        (1, apart from ``"radial"`` indexing which has a dimension of 2).
        A polynomial outside the 37-term Fringe/Wyant set has no index in these conventions
        and is reported as :data:`ZERNIKE_INDEX_UNDEFINED`, which every further conversion
        propagates unchanged.

    Raises
    ------
    ValueError
        If an invalid index number or index type is given,
        or an invalid indices shape is given.
    """
    # Parse arguments.
    if from_index not in ZERNIKE_INDEXING:
        raise ValueError(
            f"From index '{from_index}' not recognized as a valid unit. "
            f"Options: {ZERNIKE_INDEXING}."
        )
    if to_index not in ZERNIKE_INDEXING:
        raise ValueError(
            f"To index '{to_index}' not recognized as a valid unit. "
            f"Options: {ZERNIKE_INDEXING}."
        )

    dimension = ZERNIKE_INDEXING_DIMENSION[from_index]

    indices = np.array(indices, dtype=int, copy=(False if np.__version__[0] == '1' else None))
    if indices.size == dimension:
        indices = indices.reshape((1, dimension))
    if dimension > 1 and indices.shape[1] != dimension:
        raise ValueError(f"Expected dimension (N, {dimension}); found {indices.shape}")

    if from_index == to_index:
        return indices

    # Undefined indices are carried through rather than converted.
    undefined = indices == ZERNIKE_INDEX_UNDEFINED
    if np.any(undefined):
        result = zernike_convert_index(np.where(undefined, 0, indices), from_index, to_index)
        result[np.any(np.reshape(undefined, (-1, dimension)), axis=1), ...] = ZERNIKE_INDEX_UNDEFINED
        return result

    # Convert all cases to radial indices n, l.
    if from_index == "radial":
        n = indices[:,0]
        l = indices[:,1]
    elif from_index == "noll" or from_index == "fringe" or from_index == "wyant":
        ansi = _zernike_index_inverse(indices, from_index)
        unmapped = ansi == ZERNIKE_INDEX_UNDEFINED
        result = zernike_convert_index(np.where(unmapped, 0, ansi), "ansi", to_index)
        result[unmapped, ...] = ZERNIKE_INDEX_UNDEFINED
        return result
    elif from_index == "ansi":
        n = np.floor(.5 * np.sqrt(8*indices + 1) - .5).astype(int)
        l = 2*indices - n*(n+2)

    # Error check n,l
    if np.any((n + l) % 2):
        raise ValueError(f"Invalid Zernike index n,l. n+l must be even. n={n}, l={l}.")
    if np.any(np.abs(l) > n):
        raise ValueError(f"Invalid Zernike index n,l. |l| cannot be larger than n. n={n}, l={l}.")
    if np.any(n < 0):
        raise ValueError(f"Invalid Zernike index n,l. n must be non-negative. n={n}, l={l}.")

    # Convert to the desired indices.
    if to_index == "radial":
        result = np.vstack((n, l)).T
    elif to_index == "noll":
        result = (n * (n + 1)) // 2 + np.abs(l)
        result += np.logical_and(l >= 0, np.mod(n, 4) > 1)
        result += np.logical_and(l <= 0, np.mod(n, 4) <= 1)
    elif to_index == "wyant" or to_index == "fringe":
        is_wyant = int(to_index == "wyant")
        index = np.square(1 + (n + np.abs(l)) / 2).astype(int) - 2 * np.abs(l) + (l < 0)
        piston37 = np.logical_and(n == 12, l == 0)
        # Terms beyond the 37-term set have no equivalent.
        result = np.where(
            piston37,
            37 - is_wyant,
            np.where(index >= 37, ZERNIKE_INDEX_UNDEFINED, index - is_wyant),
        )
    elif to_index == "ansi":
        result = (n * (n + 2) + l) // 2

    return result


# Lazy inverses of the 1D conventions. {convention : (built_to, {index : ansi})}
_zernike_inverse_cache = {}


def _zernike_index_inverse(indices, from_index):
    """
    ANSI indices for a 1-dimensional Zernike convention, inverted by table lookup
    over the range where the forward map is defined.
    Unmapped indices return :data:`ZERNIKE_INDEX_UNDEFINED`.
    """
    # The Fringe/Wyant piston term Z_37 requires ANSI indices through radial order 12.
    D = max(int(np.max(indices, initial=0)) + 1, zernike_order_number(12))
    (built_to, table) = _zernike_inverse_cache.get(from_index, (0, {}))

    if built_to < D:
        ansi = np.arange(D)
        forward = np.ravel(zernike_convert_index(ansi, "ansi", from_index))
        table = {int(f): int(a) for (a, f) in zip(ansi, forward) if f >= 0}
        _zernike_inverse_cache[from_index] = (D, table)

    return np.array(
        [table.get(int(i), ZERNIKE_INDEX_UNDEFINED) for i in np.ravel(indices)], dtype=int
    )


def zernike(grid, index, weight=1, **kwargs):
    r"""
    Returns a single real
    `Zernike polynomial <https://en.wikipedia.org/wiki/Zernike_polynomials>`_
    as a subset of :meth:`.zernike_sum()`.
    These polynomials are commonly used as an orthonormal basis for optical aberration
    and are used in a number of places inside :mod:`slmsuite` for aberration
    compensation.

    Under the hood, this calls :meth:`.zernike_sum()` with a single term. See
    :meth:`.zernike_sum()` for more information about normalization and scaling.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    index : int
        ANSI Zernike index defining the polynomial.
    weight : float
        Amplitude of the polynomial.
    **kwargs
        Passed to :meth:`.zernike_sum()`.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    return zernike_sum(grid, (int(index),), (float(weight),), **kwargs)


def zernike_get_string(index, derivative=(0,0)):
    r"""
    Returns a :math:`\LaTeX` string corresponding to the cartesian representation of the
    Zernike polynomial of the given index. The monomials are sorted in reverse Cantor order.

    Parameters
    ----------
    index : int
        ANSI Zernike index.
    derivative : (int, int)
        If non-negative, returns the Zernike derivative of the given order. For instance,
        ``(1, 0)`` corresponds to the first derivative in the :math:`x` direction.
    """
    cxy, cw = _zernike_get_cantor([index], [[1]], derivative)
    result = ""

    # Sum the monomial terms together.
    for i, w in zip(reversed(range(len(cw))), reversed(cw[:, 0])):
        result += "{0:+}".format(int(w))

        for j, n in enumerate(["x", "y"]):
            if cxy[i, j] >= 1:
                result += n

                if cxy[i, j] > 1:
                    result += f"^{cxy[i, j]}"

    if len(result) == 0:
        result = "0"

    return result.strip("+")    # Remove potential leading +


def _zernike_get_cantor(indices, weights, derivative=(0,0)):
    indices = np.array(indices)
    weights = np.array(weights)

    # Separate the negative indices (special cases) before processing.
    negative_mask = indices < 0
    positive_mask = indices >= 0

    negative_indices = indices[negative_mask]
    indices = indices[positive_mask]

    negative_weights = weights[negative_mask, :]
    weights = weights[positive_mask, :]

    # Grab the zernike-cantor transformation from the cache.
    _zernike_build_indices(indices)
    zernike_cantor = _zernike_cache_vectorized[indices, :]   # (D, M)
    M = zernike_cantor.shape[1]
    cantor_indices = np.arange(M)

    # Remove vectors with all zeros.
    nonzero = np.any(zernike_cantor, axis=0)    # Which D are nonzero for given m in M
    cantor_indices = cantor_indices[nonzero]    # M -> M'
    zernike_cantor = zernike_cantor[:, nonzero] # (D, M')

    cantor_pairing = _inverse_cantor_pairing(cantor_indices)    # (M', 2)

    # Differentiate the terms if needed.
    if np.any(derivative):

        for j in [0, 1]:
            if derivative[j] > 0:
                power = cantor_pairing[:, j].astype(int)    # (M',) per-monomial x/y power

                # Apply the power rule.
                if derivative[j] == 1:
                    zernike_cantor = zernike_cantor * power[np.newaxis, :]
                elif derivative[j] > 1:
                    keep = power >= derivative[j]
                    factor = np.zeros_like(power)
                    factor[keep] = (
                        special.factorial(power[keep]) / special.factorial(power[keep] - derivative[j])
                    ).astype(int)
                    zernike_cantor = zernike_cantor * factor[np.newaxis, :]

                # Reduce the power of the term.
                cantor_pairing[:, j] -= derivative[j]
                cantor_pairing[cantor_pairing[:, j] < 0, j] = 0

        # Remove terms with all zeros
        nonzero = np.any(zernike_cantor, axis=0)        # Which D are nonzero for given m in M'
        cantor_pairing = cantor_pairing[nonzero, :]     # M' -> M''
        zernike_cantor = zernike_cantor[:, nonzero]     # (D, M'')

    # Reshape the weights into this new basis.
    cantor_weights = np.matmul(zernike_cantor.T, weights)  # (M' or M'', D) x (D, N) = (M' or M'', N)

    # Add in the negative indices.
    (M, N) = cantor_weights.shape
    MM = M + np.sum(negative_mask)

    final_pairing = np.zeros((MM, 2), dtype=int)
    final_pairing[:M, :] = cantor_pairing
    final_pairing[M:, 0] = negative_indices

    final_weights = np.zeros((MM, N))
    final_weights[:M, :] = cantor_weights
    final_weights[M:, :] = negative_weights

    return final_pairing, final_weights


def _zernike_indices_parse(indices=None, D=None, smaller_okay=False):
    """
    Parse Zernike indices applied to data expecting size D.
    """
    # Deal with the scalar case: a request for DD indices.
    if np.isscalar(indices):
        DD = int(indices)
        if D is None:
            if not smaller_okay:
                D = DD
        elif not ((smaller_okay and D <= DD) or D == DD):
            raise ValueError(f"Expected data (dimension {D}) to have common size with indices (requested {DD}).")

        D = DD

        # Fill in indices based on D now.
        indices = None

    # If None, assume list based on D.
    if indices is None:
        if D is None:
            raise ValueError("Either dimension or indices must be defined.")
        elif D == 2:
            indices = np.array([2,1])
        elif D == 3:
            indices = np.array([2,1,4])
        elif D == 4:
            indices = np.array([2,1,4,3])
        else:
            indices = np.hstack((np.array([2,1,4,3]), np.arange(5, D+1)))

    # Final checks.
    indices = np.ravel(indices)
    if indices.ndim == 0:
        indices = np.array([indices])
    if D is not None and not ((smaller_okay and D <= len(indices)) or D == len(indices)):
        raise ValueError(f"Expected data (dimension {D}) to have common size with indices (length {len(indices)}).")

    return indices


class ZernikeBasis:
    r"""
    A precomputed, reusable basis of Zernike polynomial images.

    Building Zernike images is the expensive part of :meth:`zernike_sum` and
    :meth:`~slmsuite.holography.analysis.image_zernike_fit`: the polynomial
    coefficients must be gathered and evaluated across the grid. When the same
    grid and basis are used many times -- as in iterative wavefront retrieval --
    this work should be done **once**. A :class:`ZernikeBasis` does exactly that,
    holding the flattened basis images (and, lazily, their Gram matrix) on the
    array module (:mod:`numpy` or :mod:`cupy`) of the grid it was built from.

    The resulting object is a pure cache: it carries no fitting or synthesis
    logic. Pass it to :meth:`zernike_sum` to synthesize a weighted sum, or to
    :meth:`~slmsuite.holography.analysis.image_zernike_fit` to fit coefficients.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized coordinates, in the same form accepted by
        :meth:`zernike_sum`. If the grid arrays are :mod:`cupy` arrays, the basis
        is built and stored on the GPU.
    indices : array_like of int OR None
        ANSI indices of the Zernike polynomials in the basis, of shape ``(D,)``.
        Parsed with :meth:`_zernike_indices_parse`.
    aperture : :class:`~slmsuite.holography.toolbox.Aperture` OR spec OR None
        The aperture defining the lateral scaling of the Zernike polynomials. Resolved
        with :meth:`~slmsuite.holography.toolbox.Aperture.resolve`.
    use_mask : bool
        Whether to zero the region outside the standard Zernike pupil. Defaults
        to ``True`` so that overlap integrals reduce to plain matrix products.

    Attributes
    ----------
    indices : numpy.ndarray
        ANSI indices of the basis modes, of shape ``(D,)``.
    aperture : object
        The ``aperture`` argument, retained for reference.
    grid_shape : tuple of int
        The ``(h, w)`` shape of the grid.
    basis : numpy.ndarray OR cupy.ndarray
        The basis images, of shape ``(D, h, w)``.
    basis_flat : numpy.ndarray OR cupy.ndarray
        The basis images flattened to ``(D, h*w)``.
    mask : numpy.ndarray OR cupy.ndarray
        Boolean ``(h, w)`` mask of the standard Zernike pupil.
    """

    def __init__(self, grid, indices, aperture=None, use_mask=True):
        indices = np.ravel(_zernike_indices_parse(indices))
        (x_grid, _) = _process_grid(grid)

        # Resolve the aperture object
        aperture = Aperture.resolve(grid, aperture)

        # One single-mode Zernike image per basis index, stacked as (D, h, w).
        # Build directly (not via zernike_sum) so basis construction does not
        # re-enter the cache and recurse.
        basis = _zernike_sum_direct(
            grid,
            indices[np.newaxis, :],
            np.eye(len(indices)),
            aperture,
            bool(use_mask),
            (0, 0),
            None,
        )
        self._set_core(indices, aperture, tuple(x_grid.shape), basis, aperture.mask)

    def _set_core(self, indices, aperture, grid_shape, basis, mask):
        """Assign the core (eagerly-held) attributes; lazy quantities recompute on access."""
        self.indices = np.atleast_1d(indices)
        # Restore the stack axis if a single mode was selected (e.g. basis[i]).
        self.basis = basis if basis.ndim == 3 else basis[np.newaxis, :, :]
        self.basis_flat = self.basis.reshape(len(self.indices), -1)
        self.aperture = aperture
        self.grid_shape = grid_shape
        self.mask = mask
        self._xp = get_array_module(self.basis)

    @cached_property
    def gram(self):
        """Gram matrix ``basis_flat @ basis_flat.T``, shape ``(D, D)``."""
        return self.basis_flat @ self.basis_flat.T

    @cached_property
    def norm(self):
        """Per-mode self-overlap ``<Z_i, Z_i>``, shape ``(D,)``."""
        return self._xp.einsum("dp,dp->d", self.basis_flat, self.basis_flat)

    @cached_property
    def gram_inv(self):
        """Inverse of the Gram matrix."""
        return self._xp.linalg.inv(self.gram)

    @cached_property
    def grad_mask(self):
        """
        Boolean ``(h, w)`` mask of pupil pixels whose four nearest neighbours are
        also inside the pupil. This is the pupil :attr:`mask` eroded by one pixel:
        the boundary ring is dropped because a central-difference gradient there
        would mix in-pupil values with the zeroed exterior.
        """
        m = self.mask.astype(bool)
        e = self._xp.zeros_like(m)
        # Interior pixel kept iff it and all four neighbours are in-pupil.
        e[1:-1, 1:-1] = (
            m[1:-1, 1:-1]
            & m[2:, 1:-1] & m[:-2, 1:-1]
            & m[1:-1, 2:] & m[1:-1, :-2]
        )
        return e

    @cached_property
    def grad_basis_flat(self):
        """
        Gradient basis, of shape ``(D, 2P)``, where ``P`` is the number of
        :attr:`grad_mask` pixels. Each row is the raw two-pixel central-difference
        gradient of a basis image -- the same stencil applied to the data in
        :meth:`~slmsuite.holography.analysis.image_zernike_fit`'s gradient mode --
        with the ``x`` and ``y`` components concatenated.
        """
        m = self.grad_mask
        # Raw central differences; grad_mask only selects interior pixels,
        # so m[:, 1:-1] and m[1:-1, :] enumerate the same P pixels in order.
        bx = self.basis[..., :, 2:] - self.basis[..., :, :-2]   # (D, h, w-2)
        by = self.basis[..., 2:, :] - self.basis[..., :-2, :]   # (D, h-2, w)
        gx = bx[:, m[:, 1:-1]]                                  # (D, P)
        gy = by[:, m[1:-1, :]]                                  # (D, P)
        return self._xp.concatenate([gx, gy], axis=1)

    @cached_property
    def grad_gram(self):
        """Gradient Gram matrix ``grad_basis_flat @ grad_basis_flat.T``, shape ``(D, D)``."""
        return self.grad_basis_flat @ self.grad_basis_flat.T

    @cached_property
    def grad_norm(self):
        """Per-mode gradient self-overlap, shape ``(D,)``."""
        return self._xp.einsum("dp,dp->d", self.grad_basis_flat, self.grad_basis_flat)

    @cached_property
    def grad_gram_inv(self):
        """Inverse of the gradient Gram matrix."""
        if np.any(np.atleast_1d(self.indices) == 0):
            raise ValueError(
                "The gradient Gram matrix is singular because the basis contains the "
                "piston term (Zernike ANSI index 0), whose gradient is identically zero. "
                "Omit piston from the basis."
            )
        return self._xp.linalg.inv(self.grad_gram)

    @cached_property
    def grad_idx_x(self):
        """Flat integer indices for the eroded pupil dx mask."""
        return self._xp.where(self.grad_mask[:, 1:-1].ravel())[0]

    @cached_property
    def grad_idx_y(self):
        """Flat integer indices for the eroded pupil dy mask."""
        return self._xp.where(self.grad_mask[1:-1, :].ravel())[0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        """Make the basis sliceable."""
        sub = object.__new__(ZernikeBasis)
        # Fresh __dict__: the cached_property lazy quantities recompute on demand.
        sub._set_core(
            self.indices[key], self.aperture, self.grid_shape, self.basis[key], self.mask
        )
        return sub


def _zernike_sum_from_basis(basis, weights, out=None):
    """Synthesize a weighted sum of polynomials from a precomputed :class:`ZernikeBasis`."""
    xp = basis._xp
    weights = xp.asarray(weights)
    D = len(basis)

    if weights.ndim == 0:
        weights = weights.reshape(1)
    if weights.ndim == 1:
        if len(weights) != D:
            raise ValueError("Expected weights to have a common dimension with the basis.")
        weights = weights.reshape(D, 1)
    elif weights.ndim == 2:
        if weights.shape[0] != D:
            raise ValueError("Expected weights to have a common dimension with the basis.")
    else:
        raise ValueError("Expected weights to be 1D or 2D.")

    N = weights.shape[1]

    # (N, D) @ (D, h*w) -> (N, h*w).
    result = weights.T @ basis.basis_flat

    if out is not None:
        if not xp.issubdtype(out.dtype, xp.inexact):
            raise ValueError(f"out must be a floating-point or complex buffer, got {out.dtype}.")
        # Normalize a possibly-flattened out buffer to (N, h, w).
        out = out.reshape((N,) + basis.grid_shape)
        out[...] = result.reshape((N,) + basis.grid_shape)
        return out.reshape(basis.grid_shape) if N == 1 else out

    if N == 1:
        return result.reshape(basis.grid_shape)
    return result.reshape((N,) + basis.grid_shape)


# Transparent ZernikeBasis cache, keyed on grid + indices + aperture.
_ZERNIKE_BASIS_CACHE = OrderedDict()     # key -> ZernikeBasis
_ZERNIKE_BASIS_CACHE_MAX = 32            # LRU cap
_ZERNIKE_BASIS_CACHE_LOCK = threading.Lock()


def clear_zernike_basis_cache():
    """Empty the transparent :class:`ZernikeBasis` cache used by :meth:`zernike_sum`
    and :meth:`~slmsuite.holography.analysis.image_zernike_fit`. Useful after a large
    one-off synthesis, or to free GPU memory held by cached bases."""
    _ZERNIKE_BASIS_CACHE.clear()


def _aperture_key(aperture):
    """Hashable key for an :class:`~slmsuite.holography.toolbox.Aperture`."""
    spec = aperture.spec
    if isinstance(spec, np.ndarray):
        spec = ("arr", tuple(spec.ravel().tolist()))
    elif isinstance(spec, (list, tuple)):
        spec = ("seq", tuple(spec))
    center = aperture.center
    center = None if center is None else tuple(np.ravel(center).tolist())
    return (spec, center)


def _zernike_get_basis(grid, indices, aperture=None, use_mask=True):
    """
    Return a (cached) :class:`ZernikeBasis` for ``grid``, ``indices``, ``aperture``,
    and ``use_mask``, building and caching it on a miss. The cache is keyed on the
    identity of the grid array (plus its shape and both grids' corner values), so
    distinct grids -- e.g. two SLMs -- map to separate entries and coexist. Access is
    serialized by a lock, and entries are evicted least-recently-used once the cache
    exceeds ``_ZERNIKE_BASIS_CACHE_MAX``.
    """
    (x_grid, y_grid) = _process_grid(grid)
    indices = np.ravel(_zernike_indices_parse(indices))
    aperture = Aperture.resolve(grid, aperture)

    key = (
        id(x_grid),
        tuple(x_grid.shape),
        (
            x_grid[0, 0].item(), x_grid[-1, -1].item(),
            y_grid[0, 0].item(), y_grid[-1, -1].item(),
        ),
        tuple(int(i) for i in indices),
        _aperture_key(aperture),
        bool(use_mask),
    )

    with _ZERNIKE_BASIS_CACHE_LOCK:
        basis = _ZERNIKE_BASIS_CACHE.get(key)
        if basis is not None:
            _ZERNIKE_BASIS_CACHE.move_to_end(key)
            return basis

        basis = ZernikeBasis(grid, indices, aperture=aperture, use_mask=use_mask)
        _ZERNIKE_BASIS_CACHE[key] = basis

        # Drop the entry when the grid array dies; guards against id() reuse after GC.
        # The weakref must be kept alive for its callback to fire, so stash it on the
        # cached basis -- its lifetime is then exactly that of the cache entry.
        try:
            basis._grid_ref = weakref.ref(
                x_grid, lambda _ref, k=key: _ZERNIKE_BASIS_CACHE.pop(k, None)
            )
        except TypeError:
            pass    # Array type does not support weakref; rely on the LRU cap below.

        # Trim oldest entries beyond the cap.
        while len(_ZERNIKE_BASIS_CACHE) > _ZERNIKE_BASIS_CACHE_MAX:
            _ZERNIKE_BASIS_CACHE.popitem(last=False)

        return basis


def _zernike_parse_weights_indices(indices, weights):
    """
    Parse ``weights`` to shape ``(D, N)`` and resolve ``indices`` to a concrete
    ``(D,)`` array, the common form consumed by both the cached and direct paths.
    """
    weights = np.squeeze(weights)
    if weights.ndim <= 1:
        if weights.ndim == 0:
            weights = np.array([weights])

        if indices is None:
            D = None
        else:
            indices = np.squeeze(indices)
            if indices.ndim == 0:
                indices = np.array([indices])

            D = len(indices)

        if D is None or len(weights) == D:
            weights = np.reshape(weights, (-1, 1))
        else:
            raise ValueError("Expected weights to have a common dimension with indices.")
    elif weights.ndim == 2:
        pass
    else:
        raise ValueError("Expected weights to be 1D or 2D.")

    (D, N) = weights.shape
    indices = _zernike_indices_parse(indices, D)
    return indices, weights, D, N


def _zernike_sum_direct(grid, indices, weights, aperture, use_mask, derivative, out):
    """
    Direct (uncached) Zernike summation: gather the combined monomial (cantor)
    coefficients and evaluate them on the grid in one pass. This is the path used
    when the cache cannot represent the request (``np.nan`` masking, a derivative)
    and the path used to build basis images themselves (so basis construction does
    not re-enter -- and recurse through -- the cache).
    """
    (x_grid, y_grid) = _process_grid(grid)
    aperture = Aperture.resolve(grid, aperture)
    (x_scale, y_scale) = aperture.scale
    xp = get_array_module(x_grid)

    indices, weights, _, N = _zernike_parse_weights_indices(indices, weights)

    # Parse out.
    out = _parse_out(x_grid, out, stack=N)

    # At the end, we're going to set the values outside the aperture to zero.
    # Make a mask for this if it's necessary.
    if use_mask is False:
        mask = None
    else:
        mask = aperture.mask
        mask_value = 0
        if np.isnan(use_mask):
            use_mask = True
            mask_value = np.nan
        use_mask = use_mask and bool(xp.any(mask == 0))

    # Make the new grids.
    if use_mask:
        x_grid_scaled = x_grid[mask] * x_scale
        y_grid_scaled = y_grid[mask] * y_scale
    else:
        # Special case to avoid copying grids in the case of no scaling.
        if x_scale == 1:    x_grid_scaled = x_grid
        else:               x_grid_scaled = x_grid * x_scale
        if y_scale == 1:    y_grid_scaled = y_grid
        else:               y_grid_scaled = y_grid * y_scale

    # Gather the Zernike information.
    cantor_terms, cantor_weights = _zernike_get_cantor(indices, weights, derivative)

    # The masked case only computes on a fraction of the full space.
    if use_mask:
        out.fill(mask_value)
        out[:, mask] = polynomial(
            grid=(x_grid_scaled, y_grid_scaled),
            weights=cantor_weights,
            terms=cantor_terms,
            out=out[:, mask]
        )
    else:
        out = polynomial(
            grid=(x_grid_scaled, y_grid_scaled),
            weights=cantor_weights,
            terms=cantor_terms,
            out=out
        )

    if N == 1:
        return out.reshape(x_grid.shape)
    else:
        return out


def zernike_sum(grid, indices, weights, aperture=None, use_mask=True, derivative=(0,0), out=None):
    r"""
    Returns a summation of
    `Zernike polynomials <https://en.wikipedia.org/wiki/Zernike_polynomials>`_
    in a computationally-efficient manner.
    These polynomials are commonly used as an orthonormal basis for optical aberration
    and are used in a number of places inside :mod:`slmsuite` for aberration compensation.
    This function returns a sum of polynomials:

    .. math:: \phi(\vec{x}) = \sum_k w_k Z_{J_k}(\vec{x}).

    where :math:`J_k` are the 1-dimensional `ANSI
    <https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices>`_
    ``indices`` of the polynomials and
    :math:`w_k` are the floating point ``weights`` of each polynomial.

    Important
    ~~~~~~~~~
    These polynomials :math:`Z_j` are normalized within the edge of a standard
    Zernike pupil to a peak-to-valley amplitude of 2, corresponding to :math:`\pm 1`,
    with two families of exception: the piston, which is identically 1, and the
    rotationally symmetric terms of order :math:`n` divisible by four
    (:math:`Z_{12}`, :math:`Z_{40}`, ...), which reach :math:`+1` but bottom out above
    :math:`-1` (:math:`-1/2` for :math:`Z_{12}`).
    When ``use_mask=False``, the polynomial is not cropped outside the standard Zernike pupil.
    This should be used carefully, as polynomials outside the unit circle quickly explode with
    :math:`r^O` for terms of order :math:`O`.

    Tip
    ~~~
    See the below example to generate
    :math:`Z_1 - Z_2 + Z_4 = Z_1^{-1} - Z_1^1 + Z_2^0`,
    where the standard radial Zernike indexing :math:`Z_n^l`
    is instead represented as :math:`Z_j` by the 1-dimensional `ANSI
    <https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices>`_.
    index :math:`j`.

    .. highlight:: python
    .. code-block:: python

        zernike_sum_phase = toolbox.phase.zernike_sum(
            grid=slm,
            indices=(1,  2,  4),    # Define Z_1, Z_2, Z_4
            weights=(1, -1,  1),    # Request Z_1 - Z_2 + Z_4
            aperture="circular"
        )

    To improve performance, especially for higher order polynomials,
    we store a cache of Zernike coefficients to avoid regeneration.

    Tip
    ~~~
    slmsuite uses `ANSI
    <https://en.wikipedia.org/wiki/Zernike_polynomials#OSA/ANSI_standard_indices>`_
    by default, but the user can convert between other common indexing conventions with
    :meth:`~slmsuite.holography.toolbox.phase.zernike_convert_index()`

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM` OR :class:`ZernikeBasis`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.

        A precomputed :class:`ZernikeBasis` may also be passed. In this case the
        sum is evaluated as a single matrix product against the cached basis
        images, and ``indices``, ``aperture``, ``use_mask``, and ``derivative``
        are ignored (they are fixed when the basis is built).
    indices : array_like of int OR None
        Which Zernike polynomials to sum, defined by ANSI indices. Of shape ``(D,)``.

        Use :meth:`~slmsuite.holography.toolbox.phase.zernike_convert_index()`
        to convert to ANSI from various other common indexing conventions.

        If ``None`` is passed, the assumed Zernike basis depends on the
        dimensionality of the provided spots:

        -   If ``D == 2``, then the basis is assumed to be ``[2,1]``
            corresponding to the :math:`x = Z_2 = Z_1^1`
            and :math:`y = Z_1 = Z_1^{-1}` tilt terms.

        -   If ``D == 3``, then the basis is assumed to be ``[2,1,4]``
            corresponding to the previous, with the addition of the
            :math:`Z_4 = Z_2^0` focus term.

        -   If ``D > 3``, then the basis is assumed to be ``[2,1,4,3,5,6...,D]``.
            The piston term (Zernike index 0) is ignored as this constant phase is
            not relevant.

    weights : array_like of float
        The weight for each given index. Of shape ``(D,)``.
        If a stack of Zernike sums is desired, then use shape ``(D, N)``.
    aperture : :class:`~slmsuite.holography.toolbox.Aperture` OR spec OR None
        The aperture defining how the Zernike polynomials are laterally scaled and
        cropped. Pass a first-class :class:`~slmsuite.holography.toolbox.Aperture`
        (e.g. ``Aperture("circular")``), or a legacy shorthand spec
        (``"circular"`` / ``"elliptical"`` / ``"cropped"`` / ``float`` /
        ``(float, float)``) which is wrapped by
        :meth:`~slmsuite.holography.toolbox.Aperture.resolve`. If ``None`` and ``grid``
        is an :class:`~slmsuite.hardware.slms.slm.SLM`, the SLM's
        :attr:`~slmsuite.hardware.slms.slm.SLM.aperture` is used.

        Important
        ~~~~~~~~~
        Read the documentation and tips in
        :class:`~slmsuite.holography.toolbox.Aperture`
        to avoid subtle issues with lateral scaling.

    use_mask : bool OR np.nan
        If ``True``, sets the area where standard Zernike polynomials are undefined to zero.
        If ``False``, the polynomial is not cropped. This should be used carefully, as
        polynomials outside the unit circle quickly explode with
        :math:`r^O` for terms of order :math:`O`.
        If ``np.nan``, the clipped area is set to ``np.nan`` instead of zero;
        this is used for plotting transparency in this undefined region.
    derivative : (int, int)
        If non-negative, returns the Zernike derivative of the given order. For instance,
        ``(1, 0)`` corresponds to the first derivative in the :math:`x` direction.
        This is fast and accurate because the derivative is computed via power rule before
        generating Zernike images.
    out : array_like OR None
        Memory to be used for the phase output. Allocated separately if ``None``.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    if len(derivative) != 2:
        raise ValueError("Expected derivative to be a (int, int)")

    # Fast path: synthesize directly from a precomputed ZernikeBasis. The basis
    # images are already evaluated, so the sum is a single matrix product. The
    # indices, aperture, and masking are baked into the basis and thus ignored.
    if isinstance(grid, ZernikeBasis):
        if any(derivative):
            raise ValueError(
                "derivative is not supported when grid is a ZernikeBasis; "
                "build the basis from a coordinate grid instead."
            )
        return _zernike_sum_from_basis(grid, weights, out)

    # Transparent fast path: reuse (or build and cache) a ZernikeBasis and evaluate
    # the sum as a single matrix product. Only valid when a flat masked/unmasked
    # basis can represent the request; np.nan masking and derivatives take the
    # direct path instead.
    if not any(derivative) and (use_mask is True or use_mask is False):
        indices, weights, _, _ = _zernike_parse_weights_indices(indices, weights)
        basis = _zernike_get_basis(grid, indices, aperture, use_mask)
        return _zernike_sum_from_basis(basis, weights, out)

    return _zernike_sum_direct(grid, indices, weights, aperture, use_mask, derivative, out)


def zernike_pyramid_plot(
        grid,
        order,
        scale=1,
        titles=None,
        cmap="twilight_shifted",
        noborder=False,
        **kwargs
    ):
    r"""
    Plots :meth:`.zernike()` on a pyramid of subplots corresponding to the radial and
    azimuthal order. The user can resize the figure with ``plt.figure()`` beforehand.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    order : int
        Maximum radial order to plot.
    scale : float
        Scales the subplots to ``[-scale, scale]``.
    titles : list of str OR None
        Which titles to plot. Defaults to all of the options:

        -   ``"ansi"`` the ANSI singleton index,
        -   ``"radial"`` the radial index pair,
        -   ``"latex"`` the Cartesian representation of the polynomial,
        -   ``"name"`` the name of the aberration produced by the polynomial.
    cmap : str
        Colormap to use in plotting.
    noborder : bool
        If ``True``, does not plot the axis border and removes color from clipped areas.
    **kwargs
        Passed to :meth:`.zernike()`.
    """
    if titles is None: titles = ["ansi", "radial", "latex", "name"]

    order = int(order + 1)
    indices_ansi = np.arange((order * (order + 1)) // 2)
    indices_radial = zernike_convert_index(indices_ansi, from_index="ansi", to_index="radial")
    derivative = kwargs["derivative"] if "derivative" in kwargs else (0,0)

    # Get the pitch of the subplots for later.
    a1 = plt.subplot(order, order, 1)
    a2 = plt.subplot(order, order, 2)

    pitch = a2.get_position().xmin - a1.get_position().xmin

    a1.remove()
    a2.remove()

    # Grab all the phases as a stack.
    grid_ = _process_grid(grid)
    phases = np.zeros((len(indices_ansi), *grid_[0].shape))

    if noborder:
        if "use_mask" in kwargs and kwargs["use_mask"] is False:
            pass
        else:
            kwargs["use_mask"] = np.nan

    phases = zernike_sum(
        grid,
        indices_ansi[np.newaxis, :],
        np.diag(np.ones_like(indices_ansi)),
        out=phases,
        **kwargs
    )

    axes = []

    for i in indices_ansi:
        n, l = indices_radial[i, :]
        m = (n + l) // 2

        a = plt.subplot(order, order, 1 + m + n*order)
        axes.append(a)

        # Plot the phase.
        plt.imshow(phases[i], cmap=cmap)

        # Construct the title.
        title = ""

        if "ansi" in titles:
            title += f"{i}\n"
        if "radial" in titles:
            title += f"({n}, {l})\n"
        if "latex" in titles:
            latex = zernike_get_string(i, derivative)
            title += "$" + latex + "$\n"
        if derivative == (0,0) and "name" in titles and i < len(ZERNIKE_NAMES):
            title += ZERNIKE_NAMES[i]

        plt.title(title.strip("\n"))

        # Set scales.
        plt.clim([-scale, scale])
        plt.xticks([])
        plt.yticks([])

        if noborder:
            a.axis("off")

    # Center the axes.
    for i, a in enumerate(axes):
        n, l = indices_radial[i, :]
        m = (n + l) // 2

        dx = .5 * (order - 1 - n)
        box = a.get_position()
        box = box.translated(dx * pitch, 0)
        a.set_position(box)

    _slmsuite_plt_show(name="zernike_pyramid_plot")


def _zernike_cache_plot():
    plt.imshow(np.log2(_zernike_cache_vectorized))
    plt.ylabel("Zernike Index (ANSI)")
    plt.xlabel("Monomial Index (Cantor)")
    _slmsuite_plt_show(name="zernike_cache_plot")


# Old style dictionary.     {(n,m) : {(nx, ny) : w, ... }, ... }
_zernike_cache = {}

# New style matrix.         N x M, N: ANSI Zernike, M: cantor polynomial.
_zernike_cache_vectorized = np.array([[]], dtype=float)

# Radial order above which float64 monomial evaluation is untrustworthy.
_ZERNIKE_ORDER_PRECISION = 38
_zernike_precision_warned = False


def _zernike_build_order(n):
    """Pre-caches Zernike polynomial coefficients up to order :math:`n`."""
    N = (n+1) * (n+2) // 2
    for i in range(N):
        _zernike_coefficients(i)


def _zernike_build_indices(indices):
    """Pre-caches Zernike polynomial coefficients for the given ANSI ``indices``."""
    for i in indices:
        _zernike_coefficients(i)


def _zernike_coefficients(index):
    r"""
    Returns the coefficients for the :math:`x^ay^b` terms of the real Zernike polynomial
    of ANSI index ``i``. This is returned as a dictionary of form ``{(a,b) : coefficient}``.
    Uses `this algorithm <https://doi.org/10.1117/12.294412>`_.
    """
    index = int(index)

    # Generate coefficients only if we have not already generated.
    if index not in _zernike_cache:
        zernike_this = {}

        (n, l) = zernike_convert_index(index, to_index="radial")[0]
        l = -l

        global _zernike_precision_warned
        if n >= _ZERNIKE_ORDER_PRECISION and not _zernike_precision_warned:
            _zernike_precision_warned = True
            warnings.warn(
                f"Zernike radial order {n} exceeds the float64 precision of the monomial "
                "representation; the evaluated polynomial will be inaccurate."
            )

        # Define helper variables.
        if l % 2:   # If odd
            q = int((abs(l) - 1) / 2)
        else:
            if l > 0:
                q = int(abs(l)/2 - 1)
            else:
                q = int(abs(l)/2)

        if l <= 0:
            p = 0
        else:
            p = 1

        l = abs(l)
        m = int((n-l)/2)

        # Finding the coefficients is a summed combinatorial search.
        # This is why we cache: so we don't have to do this many times,
        # especially for higher order polynomials and the corresponding cubic scaling.
        for i in range(q+1):
            for j in range(m+1):
                for k in range(m-j+1):
                    factor = -1 if (i + j) % 2 else 1
                    factor *= comb(l, 2 * i + p)
                    factor *= comb(m - j, k)
                    factor *= (
                        factorial(n - j)
                        // (factorial(j) * factorial(m - j) * factorial(n - m - j))
                    )

                    power_key = (int(n - 2*(i + j + k) - p), int(2 * (i + k) + p))

                    # Add this coefficient to the element in the dictionary
                    # corresponding to the right power.
                    if power_key in zernike_this:
                        zernike_this[power_key] += factor
                    else:
                        zernike_this[power_key] = factor

        # Remove all factors that have cancelled out (== 0).
        coefficients = {
            power_key: factor
            for power_key, factor in zernike_this.items()
            if factor != 0
        }

        # If we need to, enlarge the vector cache.
        N = (n+1) * (n+2) // 2      # The Zernike order determines the size of the cache.
        global _zernike_cache_vectorized

        if _zernike_cache_vectorized.shape[1] < N:
            _zernike_cache_vectorized = np.pad(
                _zernike_cache_vectorized,
                (
                    (0, N - _zernike_cache_vectorized.shape[0]),
                    (0, N - _zernike_cache_vectorized.shape[1])
                ),
                constant_values=0
            )

        # Update the vectorized dict, then publish to the dictionary cache.
        for power_key, factor in coefficients.items():
            cantor_index = _cantor_pairing(power_key)
            _zernike_cache_vectorized[index, cantor_index] = factor

        _zernike_cache[index] = coefficients

    return _zernike_cache[index]


def _zernike_populate_basis_map(indices):
    """
    This generates helper maps ``c_md``, ``i_md``, ``pxy_m`` for use in GPU kernels
    (see ``populate_basis`` in cuda.cu).
    """
    indices = np.squeeze(indices)
    D = len(indices)

    # Omit negative indices (special cases)
    zernike_indices = indices[indices >= 0]
    other_indices = indices[indices < 0]

    # Make sure all coefficients are generated.
    for i in zernike_indices:
        _zernike_coefficients(i)

    # Determine the cantor indices.
    nonzero_cantor_indices = np.any(_zernike_cache_vectorized[zernike_indices, :], axis=0)
    cantor_indices = np.arange(len(nonzero_cantor_indices), dtype=int)[nonzero_cantor_indices]

    M = len(cantor_indices)

    pxy_m = _inverse_cantor_pairing(cantor_indices).astype(np.int32)

    # Find an optimal sort pattern for constructing the polynomials.
    # msort = _term_pathing(pxy_m)
    msort = np.arange(M)
    pxy_m = pxy_m[msort, :]

    # Reinsert the other cases.
    if len(other_indices) > 0:
        pxy_m = np.pad(pxy_m, ((0, len(other_indices)), (0,0)))
        pxy_m[len(zernike_indices):, 0] = other_indices     # Other indices go into nx.
        raise NotImplementedError(
            "Special (negative) Zernike indices are not supported in the GPU "
            "CompressedSpotHologram basis map. Use only non-negative Zernike indices "
            "(the CPU zernike_sum / polynomial path does support special indices)."
        )

    # Populate the results.
    c_md = _zernike_cache_vectorized[zernike_indices, :][:, cantor_indices[msort]].T.astype(np.float32)
    i_md = np.full((M, D), -1, dtype=np.int32)

    darange = np.arange(len(zernike_indices))

    for m in msort:
        nonzero = darange[c_md[m, :] != 0]
        i_md[m, :len(nonzero)] = nonzero

    return c_md, i_md, pxy_m.T


def _zernike_test(grid, indices):
    _zernike_test_kernel = cp.RawKernel(_load_cuda(), 'zernike_test')
    _zernike_test_kernel.compile()

    c_md, i_md, pxy_m = _zernike_populate_basis_map(indices)

    # Parse grid.
    (x_grid, y_grid) = _process_grid(grid)
    scale = Aperture.resolve(grid)._isotropic_scale()
    logger.debug("source Zernike scaling: %s", scale)
    x_grid = cp.array(x_grid, copy=True, dtype=np.float32)
    y_grid = cp.array(y_grid, copy=True, dtype=np.float32)

    x_grid *= scale
    y_grid *= scale

    (H, W) = x_grid.shape
    WH = int(W*H)
    (M, D) = c_md.shape

    out = cp.full((D,H,W), np.nan, dtype=np.float32)
    out.fill(-42)

    threads_per_block = int(_zernike_test_kernel.max_threads_per_block)
    blocks = (WH // threads_per_block) + 1

    # Call the RawKernel.
    _zernike_test_kernel(
        (blocks,),
        (threads_per_block,),
        (
            np.int32(WH), np.int32(D), np.int32(M),
            cp.array(c_md.ravel()),
            cp.array(i_md.ravel()),
            cp.array(pxy_m.ravel()),
            x_grid.ravel(),
            y_grid.ravel(),
            out.ravel()
        )
    )

    return out


# Polynomials.

def _cantor_pairing(xy):
    """
    Converts a 2D index to a unique 1D index according to the
    `Cantor pairing function <https://en.wikipedia.org/wiki/Pairing_function>`_.
    """
    xy = np.array(xy, dtype=int, copy=(False if np.__version__[0] == '1' else None)).reshape((-1, 2))
    return np.rint(.5 * (xy[:,0] + xy[:,1]) * (xy[:,0] + xy[:,1] + 1) + xy[:,1]).astype(int)


def _inverse_cantor_pairing(z):
    """
    Converts a 1D index to a unique 2D index according to the
    `Cantor pairing function <https://en.wikipedia.org/wiki/Pairing_function>`_.

    Returns shape ``(D, 2)``
    """
    z = np.array(z, dtype=int, copy=(False if np.__version__[0] == '1' else None))
    if z.ndim != 1:
        raise ValueError("Expected a list of shape (D,)")

    special = z < 0
    w = np.floor((np.sqrt(8*np.where(special, 0, z) + 1) - 1) // 2).astype(int)
    t = (w*w + w) // 2

    y = z-t
    x = w-y

    # Handle negative index case which is used for special indices.
    y[special] = 0
    x[special] = z[special]

    return np.vstack((x, y)).T


def _term_pathing(xy):
    """
    Returns the index for term sorting to minimize number of monomial multiplications when summing
    polynomials (with only one storage variable).

    It may be the case that division could yield a shorter path, but division is
    generally more expensive than multiplication so we omit this scenario.

    It may also be the case that optimizing for large-step multiplications can yield a
    speedup. (e.g. `x^5 = y * y * x` with `y = x * x` costs three multiplications instead
    of five) However, it is unlikely that users will need the very-high-order
    polynomials would experience an appreciable speedup.

    Parameters
    ----------
    xy : array_like
        Array of shape ``(M, 2)``.

    Returns
    -------
    I : numpy.ndarray
        Array of shape ``(M,)``. Best coefficient order.
    """
    # Prepare helper variables.
    xy = np.array(xy, dtype=int, copy=(False if np.__version__[0] == '1' else None))

    order = np.sum(xy, axis=1)
    delta = np.squeeze(np.diff(xy, axis=1))

    cantor = _cantor_pairing(xy)
    cantor_index = np.argsort(-cantor)

    # Prepare the output data structure.
    I = np.zeros_like(order, dtype=int)

    # Helper function to recurse through pathing options.
    def recurse(i0, j0):
        # Fill in the current values.
        I[j0] = cantor_index[i0]
        cantor[cantor_index[i0]] = -1

        if j0 == 0:
            return 0

        # Figure out the distance between the current index and all other indices.
        dd = delta - delta[cantor_index[i0]]
        do = order[cantor_index[i0]] - order

        # Find the best candidate for the next index in the thread.
        nearest = -cantor + np.where((np.abs(dd) > do) + (do <= 0) + (cantor < 0), np.inf, 0)
        i = np.argmin(nearest[cantor_index])

        # Either exit or continue this thread.
        if cantor[cantor_index[i]] != -1:
            return recurse(i, j0-1)
        else:
            return j0-1

    # Traverse backwards through the array,
    j = len(I)-1
    for i in range(len(order)):
        if cantor[cantor_index[i]] >= 0 and j >= 0:
            j = recurse(i, j)

    return I


def _parse_out(x_grid, out, stack=1):
    """
    Helper function to error check the shape and type of ``out``.
    """
    shape = tuple(np.concatenate(([stack], x_grid.shape)))

    if out is None:
        # Initialize out to zero.
        return get_array_module(x_grid).zeros(shape, x_grid.dtype)
    else:
        # Error check user-provided out.
        if out.size != np.prod(shape):
            raise ValueError("out must have same size as the stacked grid.")
        if out.dtype != x_grid.dtype:
            raise ValueError("out must have same type as grid.")
        if get_array_module(x_grid) is not get_array_module(out):
            raise ValueError("out and grid must both be cupy arrays if one is.")

        return out.reshape(shape)


def polynomial(grid, weights, terms=None, pathing=None, out=None):
    r"""
    Returns a summation of monomials. Specifically,

    .. math:: \phi(x, y) = \sum_{n,m \in T} w_{nm}x^ny^m

    where :math:`w_{nm}` are floating-point weights.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    weights : array_like of float
        Array of shape ``(D,)`` corresponding to the coefficient of each term.
        Can also be shape ``(D, N)`` if a stack of ``N`` polynomials is desired.
    terms : array_like of int OR None
        Array of shape ``(D, 2)`` corresponding to the :math:`x` and :math:`y` exponents
        for the ``D`` terms.
        Otherwise, array of shape ``(D,)`` corresponding to the Cantor indices of
        monomials.
        If ``None``, assumes the terms are Cantor indices of the range of ``weights``.
    pathing : array_like of int OR None
        Array of shape ``(D,)`` corresponding to an order that the terms should be
        calculated. If ``None``, chooses the path that reduces the number of
        multiplications when evaluating monomials.
    out : numpy.ndarray OR cupy.ndarray
        A location where the result is stored. Use this to avoid allocating new memory.

    Returns
    -------
    out : numpy.ndarray OR cupy.ndarray
        Result of the sum.
    """
    # Pre-parse weights.
    weights = np.array(weights)

    # Parse terms.
    if terms is None:
        D = weights.shape[0] if weights.ndim >= 1 else 1
        terms = _inverse_cantor_pairing(np.arange(D))
    else:
        terms = np.array(terms)

    if terms.ndim == 1:
        terms = _inverse_cantor_pairing(terms)

    if terms.shape[1] != 2:
        raise ValueError("Terms must be of shape (D, 2) or (D,). Found {}.".format(terms.shape))

    D = terms.shape[0]

    # Parse weights.
    if weights.ndim == 1:
        if len(weights) == D:
            weights = np.reshape(weights, (-1, 1))
        else:
            raise ValueError("Expected weights to have a common dimension with indices.")
    elif weights.ndim == 2:
        if weights.shape[0] != D:
            raise ValueError("Expected weights to have a common dimension with indices.")
    else:
        raise ValueError("Expected weights to be 1D or 2D.")

    (D, N) = weights.shape

    # Parse pathing.
    if pathing is False:
        pathing = np.arange(terms.shape[0])
    if pathing is None:
        pathing = _term_pathing(terms)

    # Prepare the grids and canvas.
    (x_grid, y_grid) = _process_grid(grid)
    out = _parse_out(x_grid, out, stack=N)

    out.fill(0)
    nx0 = ny0 = 0
    xp = get_array_module(x_grid)
    monomial = xp.ones_like(x_grid)

    # Force datatype for easier multiplication.
    weights = weights.astype(out.dtype)

    # Sum the result.
    for index in pathing:
        (nx, ny) = terms[index, :]

        if nx >= 0:                     # Usual case: monomial.
            # Reset if we're starting a new path.
            if nx - nx0 < 0 or ny - ny0 < 0:
                nx0 = ny0 = 0
                monomial.fill(1)

            # Traverse the path in +x or +y.
            for _ in range(nx - nx0):
                monomial *= x_grid
            for _ in range(ny - ny0):
                monomial *= y_grid

            # Update the current index.
            nx0, ny0 = nx, ny

            # We use a for loop here because the arrays will already be big (vectorization
            # overhead already amortized) and multiplying with zero or special indexing
            # can cost, esp. on GPU and scalar transfer is easier.
            for i in range(N):
                if weights[index, i] != 0:
                    out[i, ...] += weights[index, i] * monomial
        elif nx == -1 and ny == 0:      # Special case: vortex waveplate.
            if xp.iscomplexobj(x_grid):
                lg = xp.arctan2(xp.real(y_grid), xp.real(x_grid))
            else:
                lg = xp.arctan2(y_grid, x_grid)

            for i in range(N):
                if weights[index, i] != 0:
                    out[i, ...] += weights[index, i] * lg
        else:
            raise ValueError(f"Unrecognized terms {(nx, ny)} for index {index}.")

    return out