"""
Unified description of the SLM's aperture, used to crop, scale, and shift
the SLM's source and applied functions (e.g., lenses, Zernike polynomials).
"""
import numpy as np


class Aperture:
    r"""
    A region of a coordinate grid, used to scale, crop, and shift functions defined on a
    canonical unit disk (e.g. lenses, axicons, vortices, or Zernike polynomials) and to
    describe the illuminated region of an SLM.
    
    Note
    ~~~~
    `Aperture` currently focuses on circular and elliptical shapes; future versions
    will broaden support to more general aperture shapes and shift the `Aperture` class
    into a broader `Source` class describing the SLM's working source. 

    Many useful phase functions are most naturally defined on a normalized unit disk
    rather than in raw grid units. An :class:`Aperture` maps normalized grid coordinates
    onto this unit disk via an (anisotropic) lateral scaling, optionally offset by a
    :attr:`center`. The edge of the disk corresponds to where
    :math:`(s_x (x - c_x))^2 + (s_y (y - c_y))^2 = 1`. Evaluating a unit-disk function on
    the :meth:`transform`\ ed coordinates therefore places, sizes, and positions that
    function on the grid; :meth:`mask` selects the pixels that fall inside the aperture.

    Tip
    ~~~
    Some unit-disk functions, such as Zernike polynomials, are canonically defined on a
    circular aperture but are often applied on a non-circular one (e.g. a rectangular
    SLM). Cropping a circular function to such a grid is fine for many applications, but
    note that for Zernike polynomials specifically it breaks the orthogonality and
    normalization of the set (acceptable for, e.g., aberration correction).

    Caution
    ~~~~~~~
    Anisotropic scaling can lead to unexpected behavior. For instance, an isotropic
    quadratic phase is a circular lens, but under anisotropic scaling it becomes an
    elliptical lens on the SLM which may not behave as expected. (The same applies to the
    :math:`Z_4 = Z_2^0 = 1 - 2x^2 - 2y^2` Zernike focusing term.)

    Attributes
    ----------
    spec : {"circular", "elliptical", "cropped"} OR float OR (float, float)
        How the unit disk is scaled relative to the grid. This is relative to the
        :math:`r = 1` edge of the canonical unit disk.

        - ``"circular"``
          Scaled isotropically until the pupil edge touches one set of opposite grid
          edges.

        - ``"elliptical"``
          Scaled anisotropically until each pupil edge touches a grid edge. Generally
          produces an ellipse.

        - ``"cropped"``
          Scaled isotropically until the rectangle of the grid is circumscribed by the
          circle. This is the default.

        - ``float`` OR ``(float, float)``
          Custom scaling multiplied directly into ``x_grid`` and ``y_grid``. A scalar
          assumes isotropic scaling.
    center : numpy.ndarray OR None
        The ``(x, y)`` coordinate of the aperture center **in the grid's normalized
        units** (the same units as :attr:`~slmsuite.hardware.slms.slm.SLM.grid`).
        ``None`` corresponds to the grid origin. Note that on an SLM the working
        :attr:`~slmsuite.hardware.slms.slm.SLM.grid` is already shifted onto this center,
        so :meth:`scaling` and :meth:`mask` (which evaluate at the grid origin) land on
        the aperture without a double-shift; use :meth:`transform` to apply the center to
        a raw, unshifted grid.
    """

    def __init__(self, spec="cropped", center=None):
        self._spec = spec
        self._center = None if center is None else np.array(center, dtype=float).ravel()
        self._mask_cache = None
        self._mask_cache_key = None

    @property
    def spec(self):
        return self._spec

    @property
    def center(self):
        return self._center

    def scaling(self, grid):
        """
        Compute the lateral scaling ``(x_scale, y_scale)`` mapping the grid onto the
        canonical unit disk.

        Parameters
        ----------
        grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
            Meshgrids of normalized coordinates, or an object exposing them.

        Returns
        -------
        (float, float)
        """
        from slmsuite.holography.toolbox import _process_grid

        (x_grid, y_grid) = _process_grid(grid)
        spec = self._spec

        if isinstance(spec, str):
            # Calculate the half-extent (radius) of the grid, which is shift-invariant.
            rx = (np.nanmax(x_grid) - np.nanmin(x_grid)) / 2
            ry = (np.nanmax(y_grid) - np.nanmin(y_grid)) / 2
            if spec == "elliptical":
                x_scale = 1 / rx
                y_scale = 1 / ry
            elif spec == "circular":
                x_scale = y_scale = 1 / np.amin([rx, ry])
            elif spec == "cropped":
                x_scale = y_scale = 1 / np.sqrt(rx**2 + ry**2)
            else:
                raise ValueError(f"Aperture spec '{spec}' is not implemented.")
        elif np.isscalar(spec):
            x_scale = y_scale = spec
        elif isinstance(spec, (list, tuple, np.ndarray)) and len(spec) == 2:
            x_scale = spec[0]
            y_scale = spec[1]
        else:
            raise ValueError("Aperture spec type {} not recognized.".format(type(spec)))

        return (x_scale, y_scale)

    def mask(self, grid):
        r"""
        The boolean mask of pixels inside the aperture, evaluated **at the grid origin**
        (i.e. :attr:`center` is *not* applied; the grid is assumed already centered on
        the aperture, as :attr:`~slmsuite.hardware.slms.slm.SLM.grid` is). For a raw,
        uncentered grid use :meth:`transform` instead.

        Returns the array ``(x s_x)**2 + (y s_y)**2 <= 1``; this is identical to the mask
        built inside :meth:`~slmsuite.holography.toolbox.phase.zernike_sum`.

        Parameters
        ----------
        grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`

        Returns
        -------
        numpy.ndarray OR cupy.ndarray
            Boolean mask of the grid's shape.
        """
        from slmsuite.holography.toolbox import _process_grid

        (x_grid, y_grid) = _process_grid(grid)

        # Cache keyed on the identity/shape of the grid; SLM rebuilds its grid array
        # (new id) whenever the aperture center changes, so this invalidates correctly.
        key = (id(x_grid), x_grid.shape)
        if self._mask_cache is not None and self._mask_cache_key == key:
            return self._mask_cache

        (x_scale, y_scale) = self.scaling(grid)
        mask = (x_grid * x_scale) ** 2 + (y_grid * y_scale) ** 2 <= 1

        self._mask_cache = mask
        self._mask_cache_key = key
        return mask

    def transform(self, grid):
        r"""
        Map a raw (uncentered) grid onto the unit disk, applying :attr:`center` then
        :meth:`scaling`. This lets an :class:`Aperture` place a unit-disk function (e.g.
        a lens or Zernike polynomial) or mask on an arbitrary grid on its own, without
        relying on the grid already being centered.

        Parameters
        ----------
        grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`

        Returns
        -------
        (array_like, array_like)
            The unit-disk coordinates ``(u, v)``; the aperture is the region
            ``u**2 + v**2 <= 1``.
        """
        from slmsuite.holography.toolbox import _process_grid

        (x_grid, y_grid) = _process_grid(grid)
        (x_scale, y_scale) = self.scaling(grid)

        if self._center is not None:
            x_grid = x_grid - self._center[0]
            y_grid = y_grid - self._center[1]

        return (x_grid * x_scale, y_grid * y_scale)

    @classmethod
    def resolve(cls, grid, aperture=None):
        """
        Resolve an ``aperture`` argument into an :class:`Aperture` instance.

        -   If ``aperture`` is already an :class:`Aperture`, it is returned unchanged.
        -   If ``aperture`` is ``None`` and ``grid`` resolves to an SLM (or cameraSLM)
            carrying an :class:`Aperture`, that aperture is returned (the source of
            truth).
        -   Otherwise an :class:`Aperture` is constructed from the legacy ``aperture``
            spec (defaulting to ``"cropped"``).

        Parameters
        ----------
        grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        aperture : Aperture OR spec OR None

        Returns
        -------
        Aperture
        """
        if isinstance(aperture, cls):
            return aperture

        if aperture is None:
            slm = getattr(grid, "slm", grid)
            existing = getattr(slm, "aperture", None)
            if isinstance(existing, cls):
                return existing
            return cls("cropped")

        return cls(aperture)

    def pickle(self, attributes=True, metadata=False):
        """
        Return an h5-serializable dict describing this aperture. Compatible with the
        :class:`~slmsuite.hardware._pickle._Picklable` recursion used by the SLM.
        """
        return {"spec": self._spec, "center": self._center}

    def __repr__(self):
        return "Aperture(spec={!r}, center={!r})".format(self._spec, self._center)
