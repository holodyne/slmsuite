"""
Grating phase patterns.
"""
import numpy as np
try:
    import cupy as cp   # type: ignore
except ImportError:
    cp = np
from typing import Tuple, Union, Callable
from slmsuite.holography.toolbox import _process_grid, imprint, format_2vectors

# Basic gratings.

def blaze(
    grid: Union[Tuple[np.ndarray, np.ndarray], object],
    vector: Tuple[float, float] = (0, 0),
):
    r"""
    Returns a simple `blazed grating <https://en.wikipedia.org/wiki/Blazed_grating>`_,
    a linear phase ramp, toward a given vector in :math:`k`-space.

    .. math:: \phi(\vec{x}) = 2\pi \cdot \vec{k} \cdot \vec{x}


    :param grid:
        :math:`\vec{x}`. Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    :param vector:
        :math:`\vec{k}`. Blaze vector in normalized :math:`\frac{k_x}{k}` units.
        See :meth:`~slmsuite.holography.toolbox.convert_vector()`.
        If a 3-dimensional vector is passed, a normalized focusing term is added.
    :return:
        The phase for this function.
    """
    (x_grid, y_grid) = _process_grid(grid)

    # Optimize phase construction based on context.
    if vector[0] == 0 and vector[1] == 0:
        result = np.zeros_like(x_grid)
    elif vector[1] == 0:
        result = (2 * np.pi * vector[0]) * x_grid
    elif vector[0] == 0:
        result = (2 * np.pi * vector[1]) * y_grid
    else:
        result = (2 * np.pi * vector[0]) * x_grid + (2 * np.pi * vector[1]) * y_grid

    if len(vector) > 2:
        result += (np.pi * vector[2]) * (np.square(x_grid) + np.square(y_grid))

    return result


def sinusoid(
    grid: Union[Tuple[np.ndarray, np.ndarray], object],
    vector: Union[Tuple[float, float], Tuple[int, int]] = (0, 0),
    shift: float = 0,
    a: float = np.pi,
    b: float = 0,
):
    r"""
    Returns a simple `holographic grating
    <https://en.wikipedia.org/wiki/Diffraction_grating#SR_(Surface_Relief)_gratings>`_,
    a sinusoidal grating, toward a given vector in :math:`k`-space.

    .. math:: \phi(\vec{x}) = \frac{a-b}{2} [1 + \cos(2\pi \cdot \vec{k} \cdot \vec{x} + s)] + b

    Important
    ---------
    Unlike a blazed grating :meth:`.blaze()`, power will efficiently be deflected toward
    the mirror -1st order at :math:`-\vec{k}` in addition to the 1st order, by symmetry.


    :param grid:
        :math:`\vec{x}`. Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    :param vector:
        :math:`\vec{k}`. Blaze vector in normalized :math:`\frac{k_x}{k}` units.
        See :meth:`~slmsuite.holography.toolbox.convert_vector()`.
    :param shift:
        Radians to laterally shift the period of the grating by.
    :param a:
        Value at one extreme of the sinusoid.
        Ignoring crosstalk,
        the 0th order will be minimized when ``|a-b|`` is equal to :math:`\pi`.
    :param b:
        Value at the other extreme of the sinusoid.
        Defaults to zero, in which case ``a`` is the amplitude.
    :return:
        The phase for this function.
    """
    if vector[0] == 0 and vector[1] == 0:
        (x_grid, _) = _process_grid(grid)
        result = np.full_like(x_grid, (a-b)/2 * (1 + np.cos(shift)))
    else:
        result = (a-b)/2 * (1 + np.cos(blaze(grid, vector) + shift))

    # Add offset if provided.
    if b != 0:
        result += b

    return result


def binary(
    grid: Union[Tuple[np.ndarray, np.ndarray], object],
    vector: Union[Tuple[float, float], Tuple[int, int]] = (0, 0),
    shift: float = 0,
    a: float = np.pi,
    b: float = 0,
    duty_cycle: float = .5
) -> np.ndarray:
    r"""
    Returns a simple binary grating toward a given vector in :math:`k`-space.

    .. math:: \phi(\vec{x}) =
        \left\{
            \begin{array}{ll}
                a, & (
                    [2\pi \cdot \vec{k} \cdot \vec{x} + s] \,\,\,\,\text{mod}\,\,\,\, 2\pi
                    ) < 2\pi*d \\
                b, & \text{ otherwise}.
            \end{array}
        \right.

    To realize a binary grating with a given pixel period, either use
    :meth:`~slmsuite.holography.toolbox.convert_vector` with ``"freq"`` units
    or pass a vector with a coordinate larger than 1:

    .. highlight:: python
    .. code-block:: python

        n_x = 4     # Period in pixels

        # Option 1: convert
        binary_integer_period = toolbox.phase.binary(
            grid=slm,
            vector=toolbox.convert_vector(
                (1./n_x, 0),
                from_units="freq",
                to_units="kxy",
                hardware=slm
            )
        )

        # Option 2: pass directly
        binary_integer_period = toolbox.phase.binary(
            grid=slm,
            vector=(n_x, 0)
        )

    Note
    ~~~~
    When parameters are chosen to produce an integer period,
    this function uses speed optimizations **(implementation incomplete)**.
    Otherwise, this function uses ``np.mod`` on top of
    :meth:`~slmsuite.holography.toolbox.phase.blaze()` to compute gratings.


    :param grid:
        :math:`\vec{x}`. Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    :param vector:
        :math:`\vec{k}`. Blaze vector in normalized :math:`\frac{k_x}{k}` units.
        See :meth:`~slmsuite.holography.toolbox.convert_vector()`.

        If the user passes data greater than 1, this is interpreted
        as requesting a binary grating with the given period. This feature
        ignores whatever transformations might have been applied to ``grid``.
    :param shift:
        Radians to laterally shift the period of the grating by.
    :param a:
        Value at one extreme of the binary grating.
    :param b:
        Value at the other extreme of the binary grating.
        Defaults to zero, in which case ``a`` is the amplitude.
    :param duty_cycle:
        The grating value is ``a`` for ``duty_cycle * period``.
        Then the grating value is ``b`` for ``(1 - duty_cycle) * period``.
    :return:
        The phase for this function.
    """
    grid = (x_grid, y_grid) = _process_grid(grid)
    dtype = x_grid.dtype
    duty_cycle = np.clip(float(duty_cycle), 0, 1)

    # Check if we're in pixel period mode.
    if np.any(np.abs(vector) > 1):
        # This is not computationally efficient.
        grid = (x_grid, y_grid) = np.meshgrid(
            np.arange(x_grid.shape[1]).astype(float),
            np.arange(x_grid.shape[0]).astype(float)
        )
        vector = (
            0 if vector[0] == 0 else 1. / vector[0],
            0 if vector[1] == 0 else 1. / vector[1]
        )

    # Check if we're in an orthogonal case.
    if vector[0] == 0 and vector[1] == 0:
        phase = b
        if shift != 0:
            if np.mod(shift, 2*np.pi) >= (2 * np.pi * duty_cycle):
                phase = a
        return np.full(x_grid.shape, phase, dtype=dtype)
    elif vector[0] != 0 and vector[1] != 0:
        pass    # xor the next case.
    elif vector[0] == 0 or vector[1] == 0:
        period = 1/np.sum(vector)   # Relative to the grid
        duty = period*duty_cycle

        period_int = np.rint(period)
        duty_int = np.rint(duty)

        if np.all(np.isclose(period, period_int)) and np.all(np.isclose(duty, duty_int)):
            pass    # Future: speed optimization.

    # If we have not returned, then we have to use the slow np.mod option.
    decision = np.mod(blaze(grid, vector) + shift, 2*np.pi)
    decision[np.isclose(decision, 2*np.pi)] = 0   # Handle edge case
    decision -= (2 * np.pi * duty_cycle)
    return np.where((decision < 0) & ~np.isclose(decision, 0), a, b)


def _quadrants(
    grid: Union[Tuple[np.ndarray, np.ndarray], object],
    vectors: np.ndarray,
    grating: Callable = blaze,
) -> np.ndarray:
    """
    Given four 2-vectors in top-right bottom-right top-left bottom-left order,
    fill the quadrants with gratings in the chosen directions.
    """
    # Parse vectors
    vectors = format_2vectors(vectors)
    if vectors.shape != (2,4):
        raise ValueError("Expected four 2-vectors (2,4). Found {}.".format(vectors.shape))

    # Parse grid.
    grid = (x_grid, y_grid) = _process_grid(grid)
    canvas = np.zeros_like(x_grid)

    # Fill the quadrants.
    for i, vector in enumerate(vectors.T):
        # Future: center this on the (0,0) point of the current grid?
        imprint(
            matrix=canvas,
            window=[
                (canvas.shape[1] // 2) * ((3-i) // 2),      # x
                (canvas.shape[1] // 2),                     # w
                (canvas.shape[0] // 2) * (i % 2),           # y
                (canvas.shape[0] // 2),                     # h
            ],
            function=grating,
            grid=grid,
            vector=vector,      # Passed to function=grating
        )

    return canvas


def bahtinov(
    grid: Union[Tuple[np.ndarray, np.ndarray], object],
    radius: float = .001,
    angle: float = 10*np.pi/180,
    grating: Callable = binary,
) -> np.ndarray:
    r"""
    Returns a `Bahtinov mask <https://en.wikipedia.org/wiki/Bahtinov_mask>`_,
    commonly used for focusing telescopes.
    When the farfield pattern resulting from this mask is symmetric, the system is in focus.


    :param grid:
        :math:`\vec{x}`. Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    :param radius:
        Radius of the diffraction pattern in normalized :math:`\frac{k_x}{k}` units.
        See :meth:`~slmsuite.holography.toolbox.convert_radius()`.
        Defaults to a milliradian.
    :param angle:
        Angle of the right two quadrants from the left two quadrants in radians.
        Defaults to 10 degrees.
    :param grating:
        Type of grating to use for the mask. Must have a ``vector=`` argument.
        Defaults to :meth:`~slmsuite.holography.toolbox.phase.binary()`.
    :return:
        The phase for this function.
    """
    s = np.sin(angle)
    c = np.cos(angle)

    vectors = format_2vectors(
        radius * np.array([
            (s, c),
            (s, -c),
            (0, 1),
            (0, 1),
        ]).T
    )

    return _quadrants(
        grid=grid,
        vectors=vectors,
        grating=grating,
    )


def quadrants(
    grid: Union[Tuple[np.ndarray, np.ndarray], object],
    radius: float = .001,
    center: Tuple[float, float] = (0, 0),
) -> np.ndarray:
    r"""
    Returns a quadrant-based alignment mask similar to
    :meth:`~slmsuite.holography.toolbox.phase.bahtinov()`.
    In this case, each quadrant is filled with a blazed grating pointing in the
    direction of the quadrant. When the source is centered on the SLM, the four
    resulting spots will have the same intensity (to first order).
    The position of the spots on the camera can align
    the SLM to the optical axis of the system.

    :param grid:
        :math:`\vec{x}`. Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    :param radius:
        Radius of the diffraction pattern in normalized :math:`\frac{k_x}{k}` units.
        See :meth:`~slmsuite.holography.toolbox.convert_radius()`.
        Defaults to a milliradian.
    :param center:
        Center of the diffraction pattern in normalized :math:`\frac{k_x}{k}` units.
        Defaults to the origin.
    :return:
        The phase for this function.
    """
    vectors = format_2vectors(
        (radius / np.sqrt(2)) * np.array([
            (1, -1),
            (1, 1),
            (-1, -1),
            (-1, 1),
        ]).T
    ) + format_2vectors(center)

    return _quadrants(
        grid=grid,
        vectors=vectors,
        grating=blaze,
    )

