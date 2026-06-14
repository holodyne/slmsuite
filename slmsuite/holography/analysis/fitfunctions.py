r"""
Common fit functions.
"""

import numpy as np
from scipy.special import factorial


# 1D

def linear(x, m, b):
    r"""
    For fitting a line.

    .. math:: y(x) = mx + b

    Parameters
    ----------
    x : numpy.ndarray
        Some independent variable.
    m : float
        Slope of the line.
    b : float
        :math:`y`-intercept of the line.

    Returns
    -------
    y : numpy.ndarray
        Line evaluated at all ``x``.
    """
    return m * x + b


def parabola(x, a, x0, y0):
    r"""
    For fitting a parabola.

    .. math:: y(x) = a(x - x_0)^2 + y_0

    Parameters
    ----------
    x : numpy.ndarray
        Some independent variable.
    a : float
        Strength of the parabola.
    x0 : float
        :math:`x` position of the vertex.
    y0 : float
        :math:`y` position of the vertex.

    Returns
    -------
    y : numpy.ndarray
        Parabola evaluated at all ``x``.
    """
    return a * np.square(x - x0) + y0


def hyperbola(z, w0, z0, zr):
    r"""
    For fitting a hyperbola.

    .. math:: w(z) = w_0 \sqrt{1 + \left[\frac{z - z_0}{z_R}\right]^2}.

    Parameters
    ----------
    z : numpy.ndarray
        Distance.
    w0 : float
        Beamradius at :math:`z = z_0`.
    z0 : float
        Plane of focus :math:`z_0`, the center of the hyperbola.
    zr : float
        Rayleigh length :math:`z_R`, the depth of focus.

    Returns
    -------
    w : numpy.ndarray
        Hyperbola evaluated at all ``z``.
    """
    return w0 * np.sqrt(1 + np.square((z - z0) / zr))


def cos(x, b, a, c, k=1):
    r"""
    For fitting an offset sinusoid.

    .. math:: y(x) = c + \frac{a}{2} \left[1+\cos(kx-b) \right].

    Parameters
    ----------
    x : numpy.ndarray
        Phase in radians.
    b : float
        Phase offset.
    a : float
        Peak amplitude.
    c : float
        Amplitude offset.
    k : float
        Phase scale factor. Default is 1.

    Returns
    -------
    y : numpy.ndarray
        Cosine fit evaluated at all ``x``.
    """
    return a * 0.5 * (1 + np.cos(k * x - b)) + c


def lorentzian(x, x0, a, c, w):
    r"""
    For fitting a resonance.

    .. math:: y(x) = c + \frac{a}{1 + \left[\frac{x - x_0}{w}\right]^2}.

    Parameters
    ----------
    x : numpy.ndarray
        Points to fit upon.
    x0 : float
        Center of the peak.
    a : float
        Amplitude.
    c : float
        Constant offset.
    w : float
        Half width at half maximum (HWHM).

    Returns
    -------
    y : numpy.ndarray
        Lorentzian fit evaluated at all ``x``.
    """
    return a / (1 + ((x - x0) / w) ** 2) + c


def gaussian(x, x0, a, c, w):
    r"""
    For fitting a 1D Gaussian.

    .. math:: y(x) = c + a \exp \left[-\frac{(x-x_0)^2}{2w^2}\right].

    Parameters
    ----------
    x : numpy.ndarray
        Points to fit upon.
    x0 : float
        Positional offset.
    a : float
        Amplitude.
    c : float
        Constant offset.
    w : float
        The standard deviation of the normal distribution.
        Equivalent to the :math:`1/e` radius.
        This is related to the full width at half maximum (FWHM)
        by a factor of :math:`2\sqrt{2\ln{2}}`.

    Returns
    -------
    y : numpy.ndarray
        Gaussian fit evaluated at all ``x``.
    """
    return c + a * np.exp(-.5 * np.square((x - x0) * (1/w)))


# 2D

def gaussian2d(xy, x0, y0, a, c, wx, wy, wxy=0):
    r"""
    For fitting a 2D Gaussian.
    There are two cases which depend on the shear variance ``wxy``
    (equivalent to :math:`M_{11}`;
    see :meth:`~slmsuite.holography.analysis.image_moment()`).

    When ``wxy`` is zero, then we have the usual form of a Gaussian:

    .. math:: z(x,y) = c + a \exp \left[
                                -\frac{(x-x_0)^2}{2w_x^2} -
                                \frac{(y-y_0)^2}{2w_y^2}
                                \right].

    When ``wxy`` is nonzero, this allows the 2D Gaussian to have second
    order central moments (equivalent to variance;
    see :meth:`~slmsuite.holography.analysis.image_variances()`) satisfying:

    .. math::   M =
                \begin{bmatrix}
                    M_{20} & M_{11} \\
                    M_{11} & M_{02}
                \end{bmatrix}
                =
                \begin{bmatrix}
                    w_x^2 & w_{xy} \\
                    w_{xy} & w_y^2
                \end{bmatrix}.

    The following satisfies the above condition:

    .. math:: z(x,y) = c + a \exp \left[
                                -\frac{1}{2}\left(
                                K_{00}(x-x_0)^2 +
                                2*K_{10}(x-x_0)(y-y_0) +
                                K_{11}(y-y_0)^2
                                \right)
                                \right].

    Where

    .. math:: K =
                \begin{bmatrix}
                    K_{00} & K_{10} \\
                    K_{10} & K_{11}
                \end{bmatrix}
                = M^{-1}.

    Caution
    ~~~~~~~
    The shear variance :math:`w_{xy}` does not have the same units as the widths
    :math:`w_x` and :math:`w_y`. The widths have units of space, whereas the shear
    variance has units of squared space.

    Note
    ~~~~
    The shear variance ``wxy`` is currently bounded to magnitudes below ``wx*wy``.
    Higher values lead to solutions which cannot be normalized.
    When ``wxy = wx*wy``, this distribution is a line (an ellipse with zeroed minor
    axis).

    Parameters
    ----------
    xy : numpy.ndarray
        Points to fit upon. ``(x, y)`` form.
    x0, y0 : float
        Vector offset.
    a : float
        Amplitude.
    c : float
        Constant offset.
    wx, wy : float
        The standard deviation of the normal distribution.
        Equivalent to the :math:`1/e` radius.
        This is related to the full width at half maximum (FWHM)
        by a factor of :math:`2\sqrt{2\ln{2}}`.
    wxy : float
        Shear variance. See above.

    Returns
    -------
    z : numpy.ndarray
        Gaussian fit evaluated at all ``(x,y)`` in ``xy``.
    """
    x = xy[0] - x0
    y = xy[1] - y0

    wxy = np.sign(wxy) * np.min([np.abs(wxy), wx*wy])

    try:
        M = np.linalg.inv([[wx*wx, wxy], [wxy, wy*wy]])
    except np.linalg.LinAlgError:
        M = np.array([[1/wx/wx, 0], [0, 1/wy/wy]])

    argument = np.square(x) * M[0,0] + np.square(y) * M[1,1] + 2 * x * y * M[1,0]

    return c + a * np.exp(-.5 * argument)


def tophat2d(xy, x0, y0, R, a=1, c=0):
    r"""
    For fitting a 2D tophat distribution.

    .. math:: z(x,y) =  \left\{
                            \begin{array}{ll}
                                a + c, & x^2 + y^2 < R^2 \\
                                c, & \text{ otherwise}.
                            \end{array}
                        \right.

    Parameters
    ----------
    xy : numpy.ndarray
        Points to fit upon (x, y).
    x0, y0 : float
        Vector offset.
    R : float
        Active radius of the tophat.
    a : float
        Amplitude.
    c : float
        Offset.

    Returns
    -------
    z : numpy.ndarray
        Tophat fit evaluated at all ``(x,y)`` in ``xy``.
    """
    x = xy[0] - x0
    y = xy[1] - y0
    return np.where(np.square(x) + np.square(y) <= R*R, a+c, c)


def sinc2d(xy, x0, y0, R, a=1, b=0, c=0, d=0, kx=0, ky=0):
    r"""
    For fitting a 2D rectangular :math:`\text{sinc}^2` distribution, potentially with a sinusoidal modulation.

    .. math:: z(x,y) =  d + \left(c + \frac{a}{2} \left[1+\cos(k_xx+k_yy-b) \right]\right) *
                        \text{sinc}^2(\pi (x-x_0) / R) * \text{sinc}^2(\pi (y-y_0) / R).

    where

    .. math:: \text{sinc}(x) = \frac{\sin(x)}{x}

    Parameters
    ----------
    xy : numpy.ndarray
        Points to fit upon (x, y).
    x0, y0 : float
        Vector offset.
    R : float
        Square radius of the sinc (radius of the first zero).
    a : float
        Peak amplitude.
    b : float
        Phase offset.
    c : float
        Sinusoidal amplitude offset.
    d : float
        Global offset.
    kx, ky : float
        Vector phase scale factor. Default is 0.

    Returns
    -------
    z : numpy.ndarray
        Rectangular sinc fit evaluated at all ``(x,y)`` in ``xy``.
    """
    x = xy[0] - x0
    y = xy[1] - y0

    return np.square(np.sinc((1 / R) * x) * np.sinc((1 / R) * y)) \
            * (a * 0.5 * (1 + np.cos(kx * x + ky * y - b)) + c) + d


# sinc variations

def _sinc2d_nomod(xy, x0, y0, R, a=1, d=0):
    r"""
    For fitting a 2D rectangular sinc distribution, without sinusoidal modulation.

    .. math:: z(x,y) =  d + a * \text{sinc}^2(\pi (x-x_0) / R) * \text{sinc}^2(\pi (y-y_0) / R).

    where

    .. math:: \text{sinc}(x) = \frac{\sin(x)}{x}

    Parameters
    ----------
    xy : numpy.ndarray
        Points to fit upon (x, y).
    x0, y0 : float
        Vector offset.
    R : float
        Square radius of the sinc (radius of the first zero).
    a : float
        Peak amplitude.
    d : float
        Global offset.

    Returns
    -------
    z : numpy.ndarray
        Rectangular sinc fit evaluated at all ``(x,y)`` in ``xy``.
    """
    return (
        a * np.square(np.sinc((1 / R) * (xy[0] - x0)) * np.sinc((1 / R) * (xy[1] - y0))) + d
    )

def _sinc2d_nomod_taylor(xy, x0, y0, R, a=1, d=0):
    r"""
    For fitting a 2D rectangular sinc distribution, without sinusoidal modulation.

    .. math:: z(x,y) =  d + a * \text{sinc}^2(\pi (x-x_0) / R) * \text{sinc}^2(\pi (y-y_0) / R).

    where

    .. math:: \text{sinc}(x) = \frac{\sin(x)}{x}

    Parameters
    ----------
    xy : numpy.ndarray
        Points to fit upon (x, y).
    x0, y0 : float
        Vector offset.
    R : float
        Square radius of the sinc (radius of the first zero).
    a : float
        Peak amplitude.
    d : float
        Global offset.

    Returns
    -------
    z : numpy.ndarray
        Rectangular sinc fit evaluated at all ``(x,y)`` in ``xy``.
    """
    return (
        a * np.square(_sinc_taylor((1 / R) * (xy[0] - x0)) * _sinc_taylor((1 / R) * (xy[1] - y0))) + d
    )

def _sinc2d_centered(xy, R, a=1, b=0, c=0, d=0, kx=0, ky=0):
    r"""
    For fitting a 2D rectangular sinc distribution, potentially with a sinusoidal modulation.

    .. math:: z(x,y) =  d + \left(c + \frac{a}{2} \left[1+\cos(k_xx+k_yy-b) \right]\right) *
                        \text{sinc}^2(\pi x / R) * \text{sinc}^2(\pi y / R).

    where

    .. math:: \text{sinc}(x) = \frac{\sin(x)}{x}

    Parameters
    ----------
    xy : numpy.ndarray
        Points to fit upon (x, y).
    R : float
        Square radius of the sinc (radius of the first zero).
    a : float
        Peak amplitude.
    b : float
        Phase offset.
    c : float
        Sinusoidal amplitude offset.
    d : float
        Global offset.
    kx, ky : float
        Vector phase scale factor. Default is 0.

    Returns
    -------
    z : numpy.ndarray
        Rectangular sinc fit evaluated at all ``(x,y)`` in ``xy``.
    """
    return (
        np.square(np.sinc((1 / R) * xy[0]) * np.sinc((1 / R) * xy[1]))
        * (a * 0.5 * (1 + np.cos(kx * xy[0] + ky * xy[1] - b)) + c) + d
    )

def _sinc2d_centered_taylor(xy, R, a=1, b=0, c=0, d=0, kx=0, ky=0):
    r"""
    For fitting a 2D rectangular sinc distribution, potentially with a sinusoidal modulation.

    .. math:: z(x,y) =  d + \left(c + \frac{a}{2} \left[1+\cos(k_xx+k_yy-b) \right]\right) *
                        \text{sinc}^2(\pi x / R) * \text{sinc}^2(\pi y / R).

    where

    .. math:: \text{sinc}(x) = \frac{\sin(x)}{x}

    Parameters
    ----------
    xy : numpy.ndarray
        Points to fit upon (x, y).
    R : float
        Square radius of the sinc (radius of the first zero).
    a : float
        Peak amplitude.
    b : float
        Phase offset.
    c : float
        Sinusoidal amplitude offset.
    d : float
        Global offset.
    kx, ky : float
        Vector phase scale factor. Default is 0.

    Returns
    -------
    z : numpy.ndarray
        Rectangular sinc fit evaluated at all ``(x,y)`` in ``xy``.
    """
    return (
        np.square(_sinc_taylor((1 / R) * xy[0]) * _sinc_taylor((1 / R) * xy[1]))
        * (a * 0.5 * (1 + np.cos(kx * xy[0] + ky * xy[1] - b)) + c) + d
    )

def _sinc_taylor(x, order=12):
    """
    Taylor series approximation for sinc. We use the numpy normalization.

    Parameters
    ----------
    x : numpy.ndarray
        Array to approximate sinc(x) upon.
    order : int
        Order of 12 approximates well up to the second zero.
    """
    squared = np.square(np.pi * x)
    monomial = squared.copy()
    result = 1

    for n in range(2, order+2, 2):
        if n != 2:
            monomial *= squared
        result += monomial * ((-1 if n % 4 == 2 else 1) / factorial(n+1))

    return result

def _sinc2d_centered_jacobian(xy, R, a=1, b=0, c=0, d=0, kx=0, ky=0):
    r"""
    Jacobian of :meth:`.sinc2d_centered()`.

    Returns
    -------
    z : numpy.ndarray
        Rectangular sinc fit evaluated at all ``(x,y)`` in ``xy``.
    """
    scx = np.sinc((1 / R) * xy[0])
    scy = np.sinc((1 / R) * xy[1])
    cx = np.cos((1 / R) * xy[0])
    cy = np.cos((1 / R) * xy[1])
    sinc_term = np.square(scx * scy)
    cos_term = 0.5 * (1 + np.cos(kx * xy[0] + ky * xy[1] - b))
    dcos_term = -0.5 * np.sin(kx * xy[0] + ky * xy[1] - b)
    return np.vstack((
        # R
        (2 / R) * scx * scy * (scx * (scy - cy) + scy * (scx - cx))
        * (a * cos_term + c),
        # a
        sinc_term * cos_term,
        # b
        -sinc_term * a * dcos_term,
        # c
        sinc_term,
        # d
        np.full_like(xy[0], 1),
        # kx
        xy[0] * sinc_term * a * dcos_term,
        # ky
        xy[1] * sinc_term * a * dcos_term,
    )).T

