import numpy as np
from slmsuite.holography.toolbox import _process_grid

def _determine_source_radius(grid, w=None):
    r"""
    Helper function to determine the assumed Gaussian source :math:`1/e` amplitude
    radius (:math:`1/e^2` power radius) for various
    structured light conversion functions. This is important because structured light
    conversions need knowledge of the size of the incident Gaussian beam.
    For example, see the ``w`` parameter in
    :meth:`~slmsuite.holography.toolbox.phase.laguerre_gaussian()`.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    w : float OR None
        The radius of the phase pattern in normalized :math:`\frac{x}{\lambda}` units.
        To produce perfect structured beams, this radius is equal to the radius of
        the Gaussian profile of the source (ideally not clipped by the SLM).
        If an SLM was passed as grid, retrieves the data from
        :attr:`slmsuite.hardware.slms.slm.SLM.source` and
        :meth:`slmsuite.hardware.slms.slm.SLM.fit_source_amplitude()`.
        If ``w`` is left as ``None``, ``w`` is set to a quarter of the smallest normalized screen dimension.

    Returns
    -------
    w : float
        Determined radius. In normalized units.
    """
    if w is not None:
        return w

    if hasattr(grid, "slm") and hasattr(grid, "cam"):
        grid = grid.slm
    if hasattr(grid, "get_source_radius"):
        return grid.get_source_radius()

    (x_grid, y_grid) = _process_grid(grid)
    return np.min([np.amax(x_grid), np.amax(y_grid)]) / 4