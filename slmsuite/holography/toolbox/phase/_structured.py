"""
Structured light.
"""
import os
import warnings
import time
import numpy as np
try:
    import cupy as cp   # type: ignore
except ImportError:
    cp = np
from scipy import special
from math import factorial
import matplotlib.pyplot as plt
from typing import Tuple, Union, Callable


from slmsuite.misc.math import REAL_TYPES
from slmsuite.holography.toolbox import _process_grid, imprint, format_2vectors
from slmsuite.holography.toolbox.phase._misc import _determine_source_radius


# Structured light.


def laguerre_gaussian(grid, l, p=0, w=None):
    r"""
    Returns the phase farfield for a
    `Laguerre-Gaussian <https://en.wikipedia.org/wiki/Gaussian_beam#Laguerre-Gaussian_modes>`_
    beam. Uses the formalism described by
    `this paper <https://doi.org/10.1364/JOSAA.25.001642>`_.

    Note
    ~~~~
    Without radial order (``p = 0``), this function distills to a
    `vortex waveplate <https://en.wikipedia.org/wiki/Optical_vortex>`_
    of given azimuthal order ``l``.

    Tip
    ~~~
    This function is especially useful to hone and validate SLM alignment. Perfect alignment will
    result in concentric and uniform fringes for higher order beams. Focusing issues, aberration,
    or pointing misalignment will mitigate this.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    l : int
        The azimuthal wavenumber, or orbital angular momentum. Can be negative.
    p : int
        The radial wavenumber. Should be non-negative.
    w : float OR None
        See :meth:`~slmsuite.holography.toolbox._determine_source_radius()`.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)

    w = _determine_source_radius(grid, w)

    theta_grid = np.arctan2(y_grid, x_grid)
    rr_grid = y_grid * y_grid + x_grid * x_grid

    canvas = 0

    if l != 0:
        canvas += l * theta_grid
    if p != 0:
        canvas += np.pi * np.heaviside(-special.genlaguerre(p, np.abs(l))(16 * rr_grid / w / w), 0)

    return canvas


def hermite_gaussian(grid, n, m, w=None):
    r"""
    Returns the phase farfield for a
    `Hermite-Gaussian <https://en.wikipedia.org/wiki/Gaussian_beam#Hermite-Gaussian_modes>`_
    beam. Uses the formalism described by `this paper <https://doi.org/10.1364/AO.54.008444>`_.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    n, m : int
        The horizontal ``n`` and vertical ``m`` wavenumbers. ``n = m = 0`` yields a flat
        phase and a Gaussian beam.
    w : float
        See :meth:`~slmsuite.holography.toolbox._determine_source_radius()`.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)
    w = _determine_source_radius(grid, w)

    # factor = np.sqrt(2) / w
    factor = 4 / w

    # Generate the amplitude of a Hermite-Gaussian mode.
    phase = special.hermite(n)(factor * x_grid) * special.hermite(m)(factor * y_grid)

    # This is real, so the phase is just the sign of the mode. This produces a
    # checkerboard pattern. Probably could make this faster by bitflipping rows and columns.
    phase[phase < 0] = 0
    phase[phase > 0] = np.pi

    return phase


def _ince_polynomial(p, m, parity, ellipticity, z):
    r"""
    Evaluate an Ince polynomial :math:`C_p^m(z, \varepsilon)` (even, ``parity=1``)
    or :math:`S_p^m(z, \varepsilon)` (odd, ``parity=-1``) by constructing the
    tridiagonal eigenvalue problem for the Fourier coefficients and summing the
    resulting Fourier series. Follows the recurrence relations in
    `DLMF §28.31 <https://dlmf.nist.gov/28.31#ii>`_.

    Parameters
    ----------
    p : int
        Order of the Ince polynomial (:math:`p \ge 0`).
    m : int
        Degree of the Ince polynomial. Must have the same parity as ``p``.
        For even parity, :math:`0 \le m \le p`.
        For odd parity, :math:`1 \le m \le p`.
    parity : {1, -1}
        ``1`` for even Ince polynomial :math:`C_p^m`, ``-1`` for odd :math:`S_p^m`.
    ellipticity : float
        The ellipticity parameter :math:`\varepsilon`.
    z : array_like
        Points at which to evaluate the polynomial.

    Returns
    -------
    numpy.ndarray
        Values of the Ince polynomial at the given points.
    """
    eps = ellipticity
    z = np.asarray(z)
    p_even = (p % 2 == 0)

    if parity == 1:  # Even: C_p^m
        if p_even:
            # DLMF 28.31.6: expansion sum_{l=0}^{p/2} A_{2l} cos(2l z)
            # Row 0: (p+2)/2 * eps * A_2 = eta * A_0   (from: -2*eta*A_0 + (2+p)*eps*A_2 = 0)
            # Row 1: p*eps*A_0 + 4*A_2 + (p/2+2)*eps*A_4 = eta*A_2
            # Row l>=2: (p/2-l+1)*eps*A_{2l-2} + (2l)^2 * A_{2l} + (p/2+l+1)*eps*A_{2l+2} = eta*A_{2l}
            N = p // 2 + 1
            idx = m // 2

            A = np.zeros((N, N), dtype=float)
            for l in range(N):
                A[l, l] = (2 * l) ** 2
            if N > 1:
                A[0, 1] = (2 + p) * eps / 2.0
                A[1, 0] = p * eps
            if N > 2:
                A[1, 2] = (p / 2.0 + 2) * eps
            for l in range(2, N):
                A[l, l - 1] = (p / 2.0 - l + 1) * eps
                if l < N - 1:
                    A[l, l + 1] = (p / 2.0 + l + 1) * eps

        else:
            # DLMF 28.31.7: expansion sum_{l=0}^{(p-1)/2} A_{2l+1} cos((2l+1)z)
            # Row 0: (1 + eps/2)*A_1 + (p/2+3/2)*eps*A_3 = eta*A_1
            # Row l>=1: (p/2-l+1/2)*eps*A_{2l-1} + (2l+1)^2*A_{2l+1} + (p/2+l+3/2)*eps*A_{2l+3} = eta*A_{2l+1}
            N = (p + 1) // 2
            idx = m // 2

            A = np.zeros((N, N), dtype=float)
            for l in range(N):
                A[l, l] = (2 * l + 1) ** 2
            A[0, 0] += eps / 2.0
            if N > 1:
                A[0, 1] = (p / 2.0 + 3 / 2.0) * eps
            for l in range(1, N):
                A[l, l - 1] = (p / 2.0 - l + 1 / 2.0) * eps
                if l < N - 1:
                    A[l, l + 1] = (p / 2.0 + l + 3 / 2.0) * eps

    else:  # parity == -1, Odd: S_p^m
        if p_even:
            # DLMF 28.31.9: expansion sum_{l=1}^{p/2} B_{2l} sin(2l z)
            # General: (p/2-l+1)*eps*B_{2l-2} + (2l)^2*B_{2l} + (p/2+l+1)*eps*B_{2l+2} = eta*B_{2l}
            N = p // 2
            idx = m // 2 - 1

            A = np.zeros((N, N), dtype=float)
            for k in range(N):
                l = k + 1
                A[k, k] = (2 * l) ** 2
                if k > 0:
                    A[k, k - 1] = (p / 2.0 - l + 1) * eps
                if k < N - 1:
                    A[k, k + 1] = (p / 2.0 + l + 1) * eps

        else:
            # DLMF 28.31.8: expansion sum_{l=0}^{(p-1)/2} B_{2l+1} sin((2l+1)z)
            # Row 0: (1 - eps/2)*B_1 + (p/2+3/2)*eps*B_3 = eta*B_1
            # Row l>=1: (p/2-l+1/2)*eps*B_{2l-1} + (2l+1)^2*B_{2l+1} + (p/2+l+3/2)*eps*B_{2l+3} = eta*B_{2l+1}
            N = (p + 1) // 2
            idx = (m - 1) // 2

            A = np.zeros((N, N), dtype=float)
            for l in range(N):
                A[l, l] = (2 * l + 1) ** 2
            A[0, 0] -= eps / 2.0
            if N > 1:
                A[0, 1] = (p / 2.0 + 3 / 2.0) * eps
            for l in range(1, N):
                A[l, l - 1] = (p / 2.0 - l + 1 / 2.0) * eps
                if l < N - 1:
                    A[l, l + 1] = (p / 2.0 + l + 3 / 2.0) * eps

    # Solve the eigenvalue problem. The matrix is generally NOT symmetric,
    # so use a general eigensolver.
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # Sort by eigenvalue and pick the idx-th one.
    order = np.argsort(eigenvalues)
    coeffs = eigenvectors[:, order[idx]]

    # Normalize: (1/pi) int_0^{2pi} (C_p^m)^2 dz = 1  (DLMF 28.31.12)
    if parity == 1 and p_even:
        norm_sq = 2 * coeffs[0] ** 2 + np.sum(coeffs[1:] ** 2)
    else:
        norm_sq = np.sum(coeffs ** 2)
    norm = np.sqrt(norm_sq)
    if norm > 0:
        coeffs /= norm

    # Sign convention (DLMF 28.31.12): C_p^m(0) > 0 and (S_p^m)'(0) > 0.
    if parity == 1:
        val_at_0 = np.sum(coeffs)
        if val_at_0 < 0:
            coeffs = -coeffs
    else:
        if p_even:
            deriv_at_0 = sum((2 * (k + 1)) * coeffs[k] for k in range(N))
        else:
            deriv_at_0 = sum((2 * k + 1) * coeffs[k] for k in range(N))
        if deriv_at_0 < 0:
            coeffs = -coeffs

    # Evaluate the Fourier series.
    result = np.zeros_like(z, dtype=complex if np.iscomplexobj(z) else float)
    if parity == 1:
        if p_even:
            for k in range(N):
                result = result + coeffs[k] * np.cos(2 * k * z)
        else:
            for k in range(N):
                result = result + coeffs[k] * np.cos((2 * k + 1) * z)
    else:
        if p_even:
            for k in range(N):
                result = result + coeffs[k] * np.sin(2 * (k + 1) * z)
        else:
            for k in range(N):
                result = result + coeffs[k] * np.sin((2 * k + 1) * z)

    return result


def ince_gaussian(grid, p, m, parity=1, ellipticity=1, w=None):
    r"""
    Returns the phase farfield for an
    `Ince-Gaussian <https://en.wikipedia.org/wiki/Gaussian_beam#Ince-Gaussian_modes>`_
    beam.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    p : int
        Ince polynomial order.
    m : int
        Ince polynomial degree.
    parity : {1, -1, 0}
        Whether to produce an even (1), odd (-1), or helical (0) Ince polynomial. A helical
        polynomial is the linear combination of even and odd polynomials.

        .. math:: IG^h_{p,m} = IG^e_{p,m} + iIG^o_{p,m}

    ellipticity : float
        Ellipticity of the beam. The semifocal distance is equal to ``ellipticity * w``,
        where the foci are the points which define the elliptical coordinate system.
    w : float
        See :meth:`~slmsuite.holography.toolbox._determine_source_radius()`.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)
    w = _determine_source_radius(grid, w)

    if parity == 1:
        if not 0 <= m <= p:
            raise ValueError("{} is an invalid Ince polynomial.".format((p,m)))
    else:
        if not 1 <= m <= p:
            raise ValueError("{} is an invalid Ince polynomial.".format((p,m)))

    if p % 2 != m % 2:
        raise ValueError(
            "p and m must have the same parity, got p={}, m={}".format(p, m)
        )

    # Elliptic coordinates: x + iy = f0 * cosh(xi + i*eta)
    # where f0 = w * sqrt(eps/2) is the semifocal distance.
    complex_grid = x_grid + 1j * y_grid
    factor = 1 / (w * np.sqrt(ellipticity / 2))

    elliptic_grid = np.arccosh(complex_grid * factor)
    xi = elliptic_grid.real    # radial coordinate (>= 0)
    eta = elliptic_grid.imag   # angular coordinate [0, 2*pi)

    # IG beam at the waist:
    # IG^e_{p,m} ~ C_p^m(i*xi, eps) * C_p^m(eta, eps) * exp(-r^2/w^2)
    # IG^o_{p,m} ~ S_p^m(i*xi, eps) * S_p^m(eta, eps) * exp(-r^2/w^2)
    # For a phase-only pattern, extract the phase of the transverse mode
    # (the Gaussian envelope is real and positive, so it doesn't affect phase).
    #
    # Note: C_p^m(i*xi) is always real (cosine series → cosh terms).
    # S_p^m(i*xi) is always purely imaginary (sine series → i*sinh terms).
    # For the radial part of odd modes, we use the real amplitude
    # S_p^m(i*xi) / i to ensure proper helical combination.
    if parity == 1:  # Even
        angular = _ince_polynomial(p, m, 1, ellipticity, eta)
        radial = _ince_polynomial(p, m, 1, ellipticity, 1j * xi)
        amplitude = radial.real * angular
    elif parity == -1:  # Odd
        angular = _ince_polynomial(p, m, -1, ellipticity, eta)
        radial_imag = _ince_polynomial(p, m, -1, ellipticity, 1j * xi)
        # S_p^m(i*xi) is purely imaginary; extract real radial envelope.
        amplitude = (radial_imag / 1j).real * angular
    else:  # Helical: IG^h = IG^e + i*IG^o
        angular_e = _ince_polynomial(p, m, 1, ellipticity, eta)
        radial_e = _ince_polynomial(p, m, 1, ellipticity, 1j * xi).real
        angular_o = _ince_polynomial(p, m, -1, ellipticity, eta)
        radial_o = (_ince_polynomial(p, m, -1, ellipticity, 1j * xi) / 1j).real
        amplitude = radial_e * angular_e + 1j * radial_o * angular_o

    return np.angle(amplitude)


def mathieu_gaussian(grid, r, q, w=None):
    r"""
    Returns the phase farfield for a
    `Mathieu-Gaussian <https://doi.org/10.1364/AO.49.006903>`_ beam.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    r : int
        Order of the Mathieu function. Positive values use even Mathieu
        functions (:math:`ce_r, Je_r`), negative values use odd Mathieu
        functions (:math:`se_{|r|}, Jo_{|r|}`). Must be nonzero for odd modes
        (``r < 0`` requires ``|r| >= 1``).
    q : float
        Mathieu parameter controlling the ellipticity.
        ``q = 0`` gives circular symmetry (Bessel beams).
        The semifocal distance is :math:`h = w\sqrt{q/2}`.
    w : float
        See :meth:`~slmsuite.holography.toolbox._determine_source_radius()`.

    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)
    w = _determine_source_radius(grid, w)

    # Semifocal distance: h = w * sqrt(q/2)
    # This is analogous to the IG beam where f0 = w*sqrt(eps/2).
    h = w * np.sqrt(np.abs(q) / 2) if q != 0 else 1.0

    # Elliptic coordinates from x + iy = h * cosh(mu + i*nu)
    complex_grid = x_grid + 1j * y_grid
    elliptic = np.arccosh(complex_grid / h) if q != 0 else (
        np.sqrt(x_grid**2 + y_grid**2) / w + 1j * np.arctan2(y_grid, x_grid)
    )
    mu = np.abs(elliptic.real)     # radial coordinate (>= 0)
    nu = elliptic.imag             # angular coordinate

    # Convert angular coordinate to degrees for scipy's mathieu functions.
    nu_deg = np.degrees(nu)

    if r >= 0:
        # Even Mathieu-Gaussian: ce_r(nu, q) * Je_r(mu, q)
        angular_vals, _ = special.mathieu_cem(r, q, nu_deg)
        radial_vals, _ = special.mathieu_modcem1(r, q, mu)
    else:
        # Odd Mathieu-Gaussian: se_{|r|}(nu, q) * Jo_{|r|}(mu, q)
        order = abs(r)
        angular_vals, _ = special.mathieu_sem(order, q, nu_deg)
        radial_vals, _ = special.mathieu_modsem1(order, q, mu)

    amplitude = angular_vals * radial_vals

    return np.angle(amplitude)


def airy(grid, f=(np.inf, np.inf), w=None):
    r"""
    Returns the cubic phase farfield for an
    `Airy <http://dx.doi.org/10.1103/PhysRevLett.99.213901>`_ beam.

    Applies a cubic phase ramp :math:`\phi(x,y) = (x/f_x)^3 + (y/f_y)^3`
    which, after propagation, produces a non-diffracting accelerating Airy beam.
    The cubic phase is scaled by the source radius to match the beam size.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    f : (float, float)
        Cubic phase scaling in the ``(x, y)`` directions, in normalized units.
        Larger values produce weaker cubic phase (more gradual acceleration).
        ``np.inf`` disables the cubic phase in that direction.
    w : float
        See :meth:`~slmsuite.holography.toolbox._determine_source_radius()`.


    Returns
    -------
    numpy.ndarray
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)
    w = _determine_source_radius(grid, w)

    canvas = np.zeros_like(x_grid)

    fx, fy = f

    if np.isfinite(fx):
        canvas += (x_grid / (fx * w)) ** 3
    if np.isfinite(fy):
        canvas += (y_grid / (fy * w)) ** 3

    return canvas