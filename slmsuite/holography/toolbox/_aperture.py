"""
Unified description of the SLM's aperture, used to crop, scale, and shift
the SLM's source and applied functions (e.g., lenses, Zernike polynomials).
"""
from functools import cached_property

import numpy as np
from slmsuite.misc.xp import get_array_module

# The string ``spec`` keywords understood by :class:`Aperture`.
_STRING_SPECS = ("circular", "elliptical", "cropped")


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

    Note
    ~~~~
    **Centering is owned by the bound grid, and applied exactly once.** Both :meth:`mask`
    and :meth:`transform` subtract :attr:`center` from the bound grid before scaling, so an
    :class:`Aperture` must be bound to the **raw, unshifted** grid. An aperture used on the
    *derived* center-shifted :attr:`~slmsuite.hardware.slms.slm.SLM.grid` must instead be
    center-free, which :meth:`resolve` enforces.

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
        ``None`` corresponds to the grid origin. See the centering note above for how this
        interacts with an SLM's derived working grid.
    """

    def __init__(self, grid, spec="cropped", center=None):
        self._validate_spec(spec)
        self._grid = grid
        self._spec = spec
        self._center = None if center is None else np.array(center, dtype=float).ravel()

    @property
    def spec(self):
        return self._spec

    @property
    def center(self):
        return self._center

    @property
    def grid(self):
        """
        The grid this aperture is bound to. For an aperture held by an SLM this is
        the raw, unshifted grid; see the centering note on the class docstring.
        """
        return self._grid

    @staticmethod
    def _validate_spec(spec):
        """
        Validate an aperture ``spec``, raising a clear error for unsupported values.
        """
        if isinstance(spec, str):
            if spec not in _STRING_SPECS:
                raise ValueError(f"Aperture spec '{spec}' is not implemented.")
        elif np.isscalar(spec):
            pass
        elif isinstance(spec, (list, tuple, np.ndarray)) and len(spec) == 2:
            pass
        else:
            raise ValueError("Aperture spec type {} not recognized.".format(type(spec)))

    @property
    def is_isotropic(self):
        """
        Whether the aperture scaling is isotropic (circular), i.e.
        ``x_scale == y_scale``.
        """
        (x_scale, y_scale) = self.scale
        return bool(np.isclose(x_scale, y_scale))

    def _isotropic_scale(self):
        """
        The single lateral scale of an isotropic aperture. Raises :class:`ValueError` for
        a genuinely anisotropic (elliptical) aperture, which a lone scalar cannot
        represent; use :attr:`scale` for the full ``(x_scale, y_scale)`` ellipse instead.
        """
        if not self.is_isotropic:
            (x_scale, y_scale) = self.scale
            raise ValueError(
                "This operation requires an isotropic (circular) aperture, but the "
                "scaling is anisotropic: (x_scale, y_scale) = "
                f"({x_scale}, {y_scale}). Use Aperture.scale for the full ellipse."
            )
        return float(self.scale[0])

    @property
    def crops(self):
        """
        Whether the aperture actually crops the grid (masks any pixels off). ``False``
        only for the default centered ``"cropped"`` spec, which circumscribes the whole
        grid and so masks nothing; ``True`` otherwise. Lets callers skip masking work in
        the common no-aperture case without materializing :attr:`mask`.
        """
        # A str test, not ==: an ndarray spec compares elementwise and has no truth value.
        cropped = isinstance(self._spec, str) and self._spec == "cropped"
        return not cropped or self._center is not None

    @cached_property
    def scale(self):
        """
        The lateral scaling ``(x_scale, y_scale)`` mapping the grid onto the canonical
        unit disk. This is the general scale for *any* unit-disk function (lenses, axicons,
        vortices, Zernike polynomials, ...); it is not specific to Zernike despite the
        :attr:`~slmsuite.hardware.slms.slm.SLM.zernike_scaling` alias on the SLM.

        Returns
        -------
        (float, float)
        """
        from slmsuite.holography.toolbox import _process_grid

        (x_grid, y_grid) = _process_grid(self._grid)
        spec = self._spec

        if isinstance(spec, str):
            # Calculate the half-extent (radius) of the grid, which is shift-invariant.
            xp = get_array_module(x_grid)
            rx = float((xp.nanmax(x_grid) - xp.nanmin(x_grid)) / 2)
            ry = float((xp.nanmax(y_grid) - xp.nanmin(y_grid)) / 2)
            if spec == "elliptical":
                x_scale = 1 / rx
                y_scale = 1 / ry
            elif spec == "circular":
                x_scale = y_scale = 1 / min(rx, ry)
            elif spec == "cropped":
                x_scale = y_scale = 1 / np.sqrt(rx**2 + ry**2)
            else:
                raise ValueError(f"Aperture spec '{spec}' is not implemented.")
        elif np.isscalar(spec):
            x_scale = y_scale = spec
        else:
            # A length-2 spec (validated in __init__); custom (x_scale, y_scale).
            x_scale = spec[0]
            y_scale = spec[1]

        return (float(x_scale), float(y_scale))

    @cached_property
    def mask(self):
        r"""
        The boolean mask of pixels inside the aperture, evaluated on the aperture's bound
        grid with :attr:`center` **applied** (subtracted before scaling, identically to
        :meth:`transform`). The bound grid must therefore be the raw, unshifted grid; see
        the centering note on the class docstring.

        Returns the array ``u**2 + v**2 <= 1`` where ``(u, v) = transform()``; this is
        identical to the mask built inside
        :meth:`~slmsuite.holography.toolbox.phase.zernike_sum`.

        Returns
        -------
        numpy.ndarray OR cupy.ndarray
            Boolean mask of the grid's shape.
        """
        (u, v) = self.transform()
        return u**2 + v**2 <= 1

    def transform(self, grid=None):
        r"""
        Map a raw (uncentered) grid onto the unit disk, applying :attr:`center` then
        :attr:`scale`. This lets an :class:`Aperture` place a unit-disk function (e.g.
        a lens or Zernike polynomial) or mask on an arbitrary grid on its own, without
        relying on the grid already being centered.

        Parameters
        ----------
        grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM` OR None
            If None, defaults to the bound grid.

        Returns
        -------
        (array_like, array_like)
            The unit-disk coordinates ``(u, v)``; the aperture is the region
            ``u**2 + v**2 <= 1``.
        """
        if grid is None:
            grid = self._grid
        from slmsuite.holography.toolbox import _process_grid

        (x_grid, y_grid) = _process_grid(grid)
        (x_scale, y_scale) = self.scale

        if self._center is not None:
            x_grid = x_grid - self._center[0]
            y_grid = y_grid - self._center[1]

        return (x_grid * x_scale, y_grid * y_scale)

    @classmethod
    def resolve(cls, grid, aperture=None):
        """
        Resolve an ``aperture`` argument into an :class:`Aperture` instance bound to
        ``grid``. 

        Centering is owned by the bound grid (see the class centering note). When ``grid``
        resolves to an SLM, its working :attr:`~slmsuite.hardware.slms.slm.SLM.grid` is
        already shifted by the SLM aperture's center, so:

        -   If ``aperture`` is ``None``, the SLM's own (raw-grid-bound) aperture is returned
            unchanged. It is the single source of truth for the offset and its
            :attr:`mask` is cached. Its mask tests ``raw_grid - center``, identical pixel
            for pixel to the ``slm.grid`` a phase function evaluates on -- so the two stay
            consistent without re-subtracting the center.
        -   If an explicit :class:`Aperture` is passed, only its shape/scaling **spec** is
            adopted; it is bound center-free to the SLM working grid (a custom center is
            meaningless against an already-centered grid and would double-subtract).

        For a plain grid (no SLM), an :class:`Aperture` is returned as-is if already bound
        to ``grid``, otherwise rebound carrying its own ``spec`` and ``center``; a ``None``
        aperture defaults to ``"cropped"``; and a bare spec is wrapped directly.

        Parameters
        ----------
        grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        aperture : Aperture OR spec OR None

        Returns
        -------
        Aperture
        """
        # Identify an SLM (or cameraSLM) behind ``grid``, with its own Aperture.
        slm = getattr(grid, "slm", grid)
        existing = getattr(slm, "aperture", None)
        slm_aperture = existing if isinstance(existing, cls) else None

        if aperture is None:
            if slm_aperture is not None:
                return slm_aperture
            return cls(grid, "cropped")

        if isinstance(aperture, cls):
            if slm_aperture is not None:
                # Centering is owned by slm.grid; take only the scaling/shape spec.
                return cls(grid, aperture.spec)
            if aperture._grid is grid:
                return aperture
            return cls(grid, aperture.spec, center=aperture.center)

        return cls(grid, aperture)

    def pickle(self, attributes=True, metadata=False):
        """
        Return an h5-serializable dict describing this aperture. Compatible with the
        :class:`~slmsuite._pickling._Picklable` recursion used by the SLM.
        """
        return {"spec": self._spec, "center": self._center}

    def __repr__(self):
        return "Aperture(spec={!r}, center={!r})".format(self._spec, self._center)
