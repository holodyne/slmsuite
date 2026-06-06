r"""
Helper functions for manipulating phase patterns.
"""

import numpy as np
from scipy.spatial import distance
from scipy.spatial import Voronoi, voronoi_plot_2d
import cv2
import matplotlib.pyplot as plt
import warnings

from slmsuite.misc.math import INTEGER_TYPES, REAL_TYPES


# Unit definitions.


LENGTH_FACTORS = {
    "m": 1e6,
    "cm": 1e4,
    "mm": 1e3,
    "um": 1,
    "nm": 1e-3,
}
LENGTH_LABELS = {k : k for k in LENGTH_FACTORS.keys()}
LENGTH_LABELS["um"] = r"$\mu$m"

CAMERA_UNITS = ["ij"]

BLAZE_LABELS = {
    "rad":  (r"$\theta_x$ [rad]", r"$\theta_y$ [rad]"),
    "mrad": (r"$\theta_x$ [mrad]", r"$\theta_y$ [mrad]"),
    "deg":  (r"$\theta_x$ [$^\circ$]", r"$\theta_y$ [$^\circ$]"),
    "norm": (r"$k_x/k$", r"$k_y/k$"),
    "kxy":  (r"$k_x/k$", r"$k_y/k$"),
    "knm":  (r"$k_n$ [pix]", r"$k_m$ [pix]"),
    "freq": (r"$f_x$ [1/pix]", r"$f_y$ [1/pix]"),
    "lpmm": (r"$k_x/2\pi$ [1/mm]", r"$k_y/2\pi$ [1/mm]"),
    "zernike": (
        r"$x = Z_2 = Z_1^1$ [Zernike rad]",
        r"$y = Z_1 = Z_1^{-1}$ [Zernike rad]"
    ),
    "ij":   (r"Camera $i$ [pix]", r"Camera $j$ [pix]"),
}
for prefix, name in zip(["", "mag_"], ["Camera", "Experiment"]):
    for k in LENGTH_FACTORS.keys():
        u = LENGTH_LABELS[k]
        BLAZE_LABELS[prefix+k] = (f"{name} $x$ [{u}]", f"{name} $y$ [{u}]"),
        CAMERA_UNITS.append(prefix+k)

BLAZE_UNITS = list(BLAZE_LABELS.keys())


# Unit helper functions.


def convert_blaze_vector(*args, **kwargs):
    """
    Alias for :meth:`~slmsuite.holography.toolbox.convert_vector()`
    for backwards compatibility.
    """
    warnings.warn(
        "The backwards-compatible alias convert_blaze_vector will be depreciated "
        "in favor of convert_vector in a future release."
    )

    if "slm" in kwargs.keys():
        kwargs["hardware"] = kwargs.pop("slm")
        warnings.warn("convert_vector(slm=) was renamed convert_vector(hardware=).")

    return convert_vector(*args, **kwargs)


def convert_blaze_radius(*args, **kwargs):
    """
    Alias for :meth:`~slmsuite.holography.toolbox.convert_radius()`
    for backwards compatibility.
    """
    warnings.warn(
        "The backwards-compatible alias convert_blaze_radius will be depreciated "
        "in favor of convert_radius in a future release."
    )

    if "slm" in kwargs.keys():
        kwargs["hardware"] = kwargs.pop("slm")
        warnings.warn("convert_vector(slm=) was renamed convert_vector(hardware=).")

    return convert_radius(*args, **kwargs)


def convert_vector(vector, from_units="norm", to_units="norm", hardware=None, shape=None):
    r"""
    Helper function for vector unit conversions in the :math:`k`-space of the SLM.

    Currently supported units:

    -  ``"rad"``, ``"mrad"``, ``"deg"``
        Angle at which light is blazed in various units.
        The small angle approximation is assumed.

    -  ``"norm"``, ``"kxy"``
        Blaze :math:`k_x` normalized to wavenumber :math:`k = 2\pi/\lambda`, i.e. :math:`\frac{k_x}{k}`.
        Equivalent to radians ``"rad"`` in the small angle approximation.
        **This is the default** :mod:`slmsuite` **unit.**

    -  ``"knm"``
        Computational blaze units for a given Fourier domain ``shape``.
        This corresponds to integer points on the grid of this
        (potentially padded) SLM's Fourier transform.
        See :class:`~slmsuite.holography.algorithms.Hologram`.
        The ``"knm"`` basis is centered at ``shape/2``, unlike all of the other units.

    -  ``"freq"``
        Pixel frequency of a grating producing the blaze.
        e.g. 1/16 is a grating with a period of 16 pixels.
        Under these units, the edge of :math:`k`-space is at :math:`\pm 0.5`,
        which makes it useful to program in fractions of the SLM's space.

    -  ``"lpmm"``
        Line pairs per mm or lines per mm of a grating producing the blaze.
        This unit is commonly used to define static diffraction gratings used in spectrometers.

    -  ``"zernike"``
        The phase coefficients **in radians** of the tilt zernike terms
        :math:`x = Z_2 = Z_1^1` and
        :math:`y = Z_1 = Z_1^{-1}` necessary to produce a given blaze.
        The coefficients of these terms are used as weights multiplied directly with
        the normalized Zernike functions :meth:`~slmsuite.holography.toolbox.phase.zernike_sum()`.
        For instance, a weight coefficient of :math:`\pi` would
        produce a wavefront offset of :math:`\pm\pi` across the unit disk.

        These functions are defined within the unit disk, and canonically have amplitude
        of :math:`\pm 1` at the edges.
        The size of the disk when scaled onto the SLM is pulled from radial fits
        derived from the amplitude distribution of the SLM.
        See :meth:`~slmsuite.hardware.slms.slm.SLM.get_source_zernike_scaling()` and
        :meth:`~slmsuite.hardware.slms.slm.SLM.fit_source_amplitude()`, especially
        the ``extent_threshold`` keyword which determines the size of the disk.
        Requires a :class:`~slmsuite.hardware.slms.slm.SLM` or
        :class:`~slmsuite.hardware.cameraslms.FourierSLM` to be passed to ``hardware``.

    -  ``"ij"``
        Camera pixel units, relative to the origin of the camera.
        Requires a :class:`~slmsuite.hardware.cameraslms.FourierSLM` to be passed to ``hardware``.
        See :meth:`~slmsuite.hardware.cameraslms.FourierSLM.kxyslm_to_ijcam`
        and :meth:`~slmsuite.hardware.cameraslms.FourierSLM.ijcam_to_kxyslm`.

    -  ``"m"``, ``"cm"``, ``"mm"``, ``"um"``, ``"nm"``
        Camera position in metric length units, relative to the origin of the camera.
        Requires a :class:`~slmsuite.hardware.cameraslms.FourierSLM` to be passed to ``hardware``,
        along with knowledge of the camera pixel size ``pitch_um``.

    -  ``"mag_m"``, ``"mag_cm"``, ``"mag_mm"``, ``"mag_um"``, ``"mag_nm"``
        Scales the corresponding metric length unit according to the value stored in
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.mag` to match the true
        dimensions of the experiment plane, as opposed to the camera plane.
        Requires a :class:`~slmsuite.hardware.cameraslms.FourierSLM` to be passed to ``hardware``,
        along with knowledge of the camera pixel size ``pitch_um``.

    3D Vectors
    ~~~~~~~~~~

    If an array of 3D vectors is given, then the depth (:math:`z`) direction is handled
    differently than the field (:math:`xy`).
    Most units use the **normalized focal power** :math:`\frac{\lambda}{f}` on the SLM
    necessary to produce a spot at the defined depth relative to the focal plane.
    There are a few units where this differs:

    -  ``"zernike"``
        The phase coefficient of the :math:`2(x^2 + y^2) - 1 = Z_4 = Z_2^0`
        zernike focus term necessary to focus at the given depth.

    -  ``"ij"``
        True cartesian distance relative to the **camera plane** in pixels.

    -  ``"m"``, ``"cm"``, ``"mm"``, ``"um"``, ``"nm"``
        True cartesian distance relative to the **camera plane** in metric units.

    -  ``"mag_m"``, ``"mag_cm"``, ``"mag_mm"``, ``"mag_um"``, ``"mag_nm"``
        True cartesian distance relative to the **experiment plane** in metric units.
        Importantly, :math:`x` and :math:`y` are divided by
        :attr:`~slmsuite.hardware.cameraslms.FourierSLM.mag`,
        while :math:`z` is multiplied by it.

    Some of these units will not make sense in a system with anisotropic focusing, for
    instance due to cylindrical lenses in the optical train.

    Warning
    ~~~~~~~
    The units ``"freq"``, ``"knm"``, and ``"lpmm"`` depend on SLM pixel size,
    so an SLM should be passed to ``hardware``
    (otherwise returns an array of ``nan`` values).
    The unit ``"zernike"`` also requires an SLM.
    The unit ``"knm"`` additionally requires the ``shape`` of the computational space.
    If not included when ``hardware`` is passed, ``shape=slm.shape`` is assumed.
    The units ``"ij"``, ``"um"``, ``"mag_um"``, and related refer to distance on the
    camera and require calibration data stored in a
    :class:`~slmsuite.hardware.cameraslms.FourierSLM`,
    so this must be passed to ``hardware``.

    Parameters
    ----------
    vector : array_like
        Vectors for which we want to convert units, from ``from_units`` to ``to_units``.
        Processed according to :meth:`format_2vectors()`.
        Can be shape ``(2, N)`` or ``(3, N)``.
    from_units, to_units : str
        Units which we are converting between. See the listed units above for options.
        Defaults to ``"norm"``.
    hardware : :class:`~slmsuite.hardware.slms.slm.SLM` OR :class:`~slmsuite.hardware.cameraslms.FourierSLM` OR None
        Relevant hardware to pull calibration data from in the case of
        ``"freq"``, ``"knm"``, ``"lpmm"``, or ``"zernike"``.
        If :class:`~slmsuite.hardware.cameraslms.FourierSLM`, the unit ``"ij"`` and other
        length units can be processed too.
    shape : (int, int) OR None
        Shape of the computational SLM space. Needed for ``"knm"``.
        Defaults to ``slm.shape`` if ``hardware`` is not ``None``.

    Returns
    --------
    vector_converted : numpy.ndarray
        Result of the unit conversion, in the cleaned format of :meth:`format_2vectors()`.
    """
    # Parse units.
    if not (from_units in BLAZE_UNITS):
        raise ValueError(f"From unit '{from_units}' not recognized \
                         as a valid unit. Options: {BLAZE_UNITS}")
    if not (to_units in BLAZE_UNITS):
        raise ValueError(f"To unit '{to_units}' not recognized \
                         as a valid unit. Options: {BLAZE_UNITS}")

    # Parse vectors.
    vector_parsed = format_vectors(
        vector,
        expected_dimension=2,
        handle_dimension="pass"
    ).astype(float)

    if from_units == to_units:
        return vector_parsed

    vector_xy = vector_parsed[:2, :]
    if vector_parsed.shape[0] > 2:
        vector_z =  vector_parsed[[2], :]
    else:
        vector_z = None

    # Determine whether a CameraSLM was passed (to enable "ij" units and related).
    if hasattr(hardware, "slm") and hasattr(hardware, "cam"):
        cameraslm = hardware
        slm = hardware.slm
    else:
        cameraslm = None
        slm = hardware

    if from_units in CAMERA_UNITS or to_units in CAMERA_UNITS:
        if cameraslm is None or not "fourier" in cameraslm.calibrations:
            warnings.warn(
                f"CameraSLM must be passed as slm for conversion '{from_units}' to '{to_units}'"
            )
            return np.full_like(vector_parsed, np.nan)

        cam_pitch_um = cameraslm.cam.pitch_um

        if cam_pitch_um is None:
            # Don't error if ij.
            if from_units in CAMERA_UNITS[1:] or to_units in CAMERA_UNITS[1:]:
                warnings.warn(
                    f"Camera must have filled attribute pitch_um "
                    "for conversion '{from_units}' to '{to_units}'"
                )
                return np.full_like(vector_parsed, np.nan)
        else:
            cam_pitch_um = format_2vectors(cam_pitch_um)

    # Generate conversion factors for various units.
    if from_units == "freq" or to_units == "freq":
        if slm is None:
            warnings.warn("slm is required for unit 'freq'")
            pitch_um = np.nan
            wav_um = np.nan
        else:
            pitch_um = format_2vectors(slm.pitch_um)
            wav_um = slm.wav_um

    if from_units == "lpmm" or to_units == "lpmm":
        if slm is None:
            warnings.warn("slm is required for units 'lpmm'")
            wav_um = np.nan
        else:
            wav_um = slm.wav_um

    if from_units == "knm" or to_units == "knm":
        if slm is None:
            pitch = np.nan
        else:
            pitch = format_2vectors(slm.pitch)

        if shape is None:
            if slm is None:
                warnings.warn("shape or slm is required for unit 'knm'")
                shape = (np.nan, np.nan)
            else:
                shape = np.array(slm.shape, dtype=float)
        else:
            shape = np.array(format_shape(shape), dtype=float)

        shape = format_2vectors(np.flip(np.squeeze(shape)))

        knm_conv = pitch * shape

    if from_units == "zernike" or to_units == "zernike":
        if slm is None:
            zernike_scale = np.nan
        else:
            zernike_scale = 2 * np.pi * np.reciprocal(slm.get_source_zernike_scaling())

    # XY

    # Convert the xy input to normalized "kxy" units.
    if from_units == "norm" or from_units == "kxy" or from_units == "rad":
        rad = vector_xy
    elif from_units == "mrad":
        rad = vector_xy / 1000
    elif from_units == "deg":
        rad = vector_xy * np.pi / 180
    elif from_units == "knm":
        rad = (vector_xy - shape / 2.0) / knm_conv
    elif from_units == "freq":
        rad = vector_xy * wav_um / pitch_um
    elif from_units == "lpmm":
        rad = vector_xy * wav_um / 1000
    elif from_units == "zernike":
        rad = vector_xy / zernike_scale
    elif from_units == "ij":
        rad = cameraslm.ijcam_to_kxyslm(vector_xy)
    elif from_units in CAMERA_UNITS:
        unit = from_units.split("_")[-1]
        if "mag_" in from_units: vector_xy *= cameraslm.mag
        rad = cameraslm.ijcam_to_kxyslm(vector_xy * LENGTH_FACTORS[unit] / cam_pitch_um)

    # Convert from normalized "kxy" units to the desired xy output units.
    if to_units == "norm" or to_units == "kxy" or to_units == "rad":
        vector_xy = rad
    elif to_units == "mrad":
        vector_xy = rad * 1000
    elif to_units == "deg":
        vector_xy = rad * 180 / np.pi
    elif to_units == "knm":
        vector_xy = rad * knm_conv + shape / 2.0
    elif to_units == "freq":
        vector_xy = rad * pitch_um / wav_um
    elif to_units == "lpmm":
        vector_xy = rad * 1000 / wav_um
    elif to_units == "zernike":
        vector_xy = rad * zernike_scale
    elif to_units == "ij":
        vector_xy = cameraslm.kxyslm_to_ijcam(rad)
    elif to_units in CAMERA_UNITS:
        unit = to_units.split("_")[-1]
        vector_xy = cameraslm.kxyslm_to_ijcam(rad) * cam_pitch_um / LENGTH_FACTORS[unit]
        if "mag_" in to_units: vector_xy /= cameraslm.mag

    # Z

    if vector_z is not None:
        # Convert the z input to normalized "focal power" units.
        if from_units in CAMERA_UNITS:
            if from_units != "ij":
                unit = from_units.split("_")[-1]
                vector_z *= LENGTH_FACTORS[unit] / np.mean(cam_pitch_um)
                if "mag_" in from_units: vector_z /= cameraslm.mag

            focal_power = cameraslm._ijcam_to_kxyslm_depth(vector_z)

        elif from_units == "zernike":
            focal_power = vector_z * ((8 * np.pi) / (zernike_scale * zernike_scale))
        else:
            focal_power = vector_z

        # Convert the normalized "focal power" units to the desired z output units.
        if to_units in CAMERA_UNITS:
            vector_z = cameraslm._kxyslm_to_ijcam_depth(focal_power)

            if to_units != "ij":
                unit = to_units.split("_")[-1]
                vector_z *= np.mean(cam_pitch_um) / LENGTH_FACTORS[unit]
                if "mag_" in to_units: vector_z *= cameraslm.mag

        elif to_units == "zernike":
            vector_z = focal_power * ((zernike_scale * zernike_scale) / (8 * np.pi))
        else:
            vector_z = focal_power

        return np.vstack((vector_xy, vector_z))
    else:
        return vector_xy


def print_blaze_conversions(vector, from_units="norm", **kwargs):
    """
    Helper function to understand unit conversions.
    Prints all the supported unit conversions for a given vector.
    See :meth:`convert_vector()`.

    Parameters
    ----------
    vector : array_like
        Vector to convert. See :meth:`format_2vectors()` for format.
    from_units : str
        Units of ``vector``, i.e. units to convert from.
    **kwargs
        Passed to :meth:`convert_vector()`.
    """
    for unit in BLAZE_UNITS:
        result = convert_vector(vector, from_units=from_units, to_units=unit, **kwargs)

        print("'{}' : {}".format(unit, result.T[0, :]))


def convert_radius(radius, from_units="norm", to_units="norm", hardware=None, shape=None):
    """
    Helper function for scalar unit conversions.
    Uses :meth:`convert_vector` to deduce the (average, in the case of an
    anisotropic transformation) scalar radius when going between sets of units.

    Tip
    ~~~
    In the future, we might create a similar function to handle anisotropic
    conversions better by converting a 2x2 matrix representing a sheared parallelogram.

    Parameters
    ----------
    radius : float
        The scalar radius to convert.
    from_units, to_units : str
        Passed to :meth:`convert_vector`.
    hardware : :class:`~slmsuite.hardware.slms.slm.SLM` OR :class:`~slmsuite.hardware.cameraslms.FourierSLM` OR None
        Passed to :meth:`convert_vector`.
    shape : (int, int) OR None
        Passed to :meth:`convert_vector`.

    Returns
    -------
    radius : float
        New scalar radius.
    """
    v0 = convert_vector(
        (0, 0), from_units=from_units, to_units=to_units, hardware=hardware, shape=shape
    )
    vx = convert_vector(
        (radius, 0), from_units=from_units, to_units=to_units, hardware=hardware, shape=shape
    )
    vy = convert_vector(
        (0, radius), from_units=from_units, to_units=to_units, hardware=hardware, shape=shape
    )
    return np.mean([np.linalg.norm(vx - v0), np.linalg.norm(vy - v0)])


# Windows creation functions. Windows are views into 2D arrays.


def window_slice(window, shape=None, centered=False, circular=False):
    """
    Parses the slices that describe the window's view into the larger 2D array.

    Parameters
    ----------
    window : (int, int, int, int) OR (array_like, array_like) OR array_like OR None
        A number of formats are accepted:

        - List in ``(x, w, y, h)`` format, where ``w`` and ``h`` are the width and height of
          the region and  ``(x,y)`` is the upper-left coordinate.

            - If ``centered``, then ``(x,y)`` is instead the center of the region to imprint.
            - If ``circular``, then an elliptical region circumscribed by the rectangular region is returned.

        - Tuple containing arrays of identical length corresponding to y and x indices of points in the window.
          ``centered`` and ``circular`` are ignored.
        - Boolean array of same ``shape`` as ``matrix``; the window is defined where ``True`` pixels are.
          ``centered`` and ``circular`` are ignored.
        - If ``None``, the window is the full slice of the 2D array.

    shape : (int, int) OR None
        The (height, width) of the array that the window is a view into.
        If not ``None``, indices beyond those allowed by ``shape`` will be clipped.
    centered : bool
        See ``window``.
    circular : bool
        See ``window``.

    Returns
    -------
    slice_ : (slice, slice) OR (array_like, array_like) OR (array_like)
        The slice for the window.
    """
    if shape is not None:
        shape = format_shape(shape)

    # Case 0: No window, so return the full slice.
    if window is None:
        slice_ = (slice(None), slice(None))
    # Case 1: (x, w, y, h) format
    elif len(window) == 4:
        # Prepare helper vars
        xi = int(window[0] - ((window[1] - 1) / 2 if centered else 0))
        xf = xi + int(window[1])
        yi = int(window[2] - ((window[3] - 1) / 2 if centered else 0))
        yf = yi + int(window[3])

        xc = xi + int((window[1] - 1) / 2)
        yc = yi + int((window[3] - 1) / 2)

        if shape is not None:
            [xi, xf] = np.clip([xi, xf], 0, shape[1])
            [yi, yf] = np.clip([yi, yf], 0, shape[0])

        if circular:  # If a circular window is desired, compute this.
            x_list = np.arange(xi, xf)
            y_list = np.arange(yi, yf)
            x_grid, y_grid = np.meshgrid(x_list, y_list)

            rr_grid = (
                (window[3] ** 2) * np.square(x_grid.astype(float) - xc) +
                (window[1] ** 2) * np.square(y_grid.astype(float) - yc)
            )

            mask_grid = rr_grid <= (window[1] ** 2) * (window[3] ** 2) / 4.0

            # Pass things back through window_slice to crop the circle, should the user
            # have given values that are out of bounds.
            return window_slice((y_grid[mask_grid], x_grid[mask_grid]), shape=shape)
        else:  # Otherwise, return square slices in the python style.
            slice_ = (slice(yi, yf), slice(xi, xf))
    # Case 2: (y_ind, x_ind) format
    elif len(window) == 2:
        # Prepare the lists
        y_ind = np.ravel(window[0])
        x_ind = np.ravel(window[1])
        if shape is not None:
            x_ind = np.clip(x_ind, 0, shape[1] - 1)
            y_ind = np.clip(y_ind, 0, shape[0] - 1)
        slice_ = (y_ind, x_ind)
    # Case 3: Boolean numpy array.
    elif np.ndim(window) == 2:
        slice_ = window
    else:
        raise ValueError("Unrecognized format for `window`.")

    return slice_


def window_extent(window, padding_frac=0, padding_pix=0):
    """
    Find a rectangle that covers the active region of the 2D boolean mask ``window``.

    Parameters
    ----------
    window : numpy.ndarray<bool> (height, width)
        Boolean mask.
    padding_frac : float
        If this default window has width ``w`` and height ``h``,
        ``padding_frac`` proportionally changes these dimensions all sides.
        For instance, ``padding_frac=.5`` would modify the dimensions to be
        ``w = 1.5w`` and ``h = 1.5h`` and adjust the position of the window
        accordingly to keep it centered on the same region.
    padding_pix : float
        Additional padding to add, in pixels.
        This is applied after ``padding_frac``.

    Returns
    -------
    window_extent : (int, int, int, int)
        A rectangle centered on the active region of ``window``
        in the format ``(x, w, y, h)`` where
        ``(x, y)`` is the upper left coordinate, and
        ``(w, h)`` define the extent.
        This result is clipped to be within ``shape`` of the window.
    """
    limits = []

    # For each axis...
    for a in [0, 1]:
        if len(window) == 4:  # Handle the (x, w, y, h) case
            b = 2*(1-a)
            limit = np.array([window[b], window[b] + window[b + 1]])
        elif len(window) == 2:  # Handle two list case
            limit = np.array([np.amin(window[a]), np.amax(window[a]) + 1])
        elif np.ndim(window) == 2:  # Handle the boolean array case
            collapsed = np.where(np.any(window, axis=a))  # Collapse the other axis
            limit = np.array([np.amin(collapsed), np.amax(collapsed) + 1])
        else:
            raise ValueError("Unrecognized format for `window`.")

        # Add padding if desired.
        padding_ = int((np.floor(np.diff(limit) * padding_frac) + padding_pix).item())
        limit += np.array([-padding_, padding_])

        # Clip the padding to shape.
        if np.ndim(window) == 2:
            limit = np.clip(limit, 0, window.shape[1 - a])

        limits.append(tuple(limit))

    # Return desired format.
    return (limits[0][0], limits[0][1] - limits[0][0], limits[1][0], limits[1][1] - limits[1][0])


def voronoi_windows(grid, vectors, radius=None, plot=False):
    r"""
    Returns boolean array windows corresponding to the Voronoi cells for a set of vectors.
    These boolean array windows are in the style of :meth:`~slmsuite.holography.toolbox.imprint()`.
    The ith window corresponds to the Voronoi cell centered around the ith vector.

    Note
    ~~~~
    The :meth:`cv2.fillConvexPoly()` function used to fill each window dilates
    slightly outside the window bounds. To avoid pixels belonging to multiple windows
    simultaneously, we crop away previously-assigned pixels from new windows while these are
    being iteratively generated. As a result, windows earlier in the list will be slightly
    larger than windows later in the list.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM` OR (int, int)
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
        If an ``(int, int)`` is passed, this is assumed to be the shape of the device, and
        ``vectors`` are **assumed to be in pixel units instead of normalized units**.
    vectors : array_like
        Points to Voronoi-ify.
        Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
    radius : float
        Cells on the edge of the set of cells might be very large. This parameter bounds
        the cells with a boolean and to the aperture of the given ``radius``.
    plot : bool
        Whether to plot the resulting Voronoi diagram with :meth:`scipy.spatial.voronoi_plot_2d()`.

    Returns
    -------
    list of numpy.ndarray
        The resulting windows.
    """
    vectors = format_2vectors(vectors)

    if (
        isinstance(grid, (list, tuple))
        and isinstance(grid[0], (int))
        and isinstance(grid[1], (int))
    ):
        shape = grid
    else:
        (x_grid, y_grid) = _process_grid(grid)

        shape = x_grid.shape

        x_list = x_grid[0, :]
        y_list = y_grid[:, 0]

        vectors = np.vstack(
            (
                np.interp(vectors[0, :], x_list, np.arange(shape[1])),
                np.interp(vectors[1, :], y_list, np.arange(shape[0])),
            )
        )

    # Half shape data.
    hsx = shape[1] / 2
    hsy = shape[0] / 2

    # Add additional points in a diamond outside the shape of interest to cause all
    # windows of interest to be finite.
    vectors_voronoi = np.concatenate(
        (
            vectors.T,
            np.array([[hsx, -3 * hsy], [hsx, 5 * hsy], [-3 * hsx, hsy], [5 * hsx, hsy]]),
        )
    )

    vor = Voronoi(vectors_voronoi, furthest_site=False)

    if plot:
        sx = shape[1]
        sy = shape[0]

        # Use the built-in scipy function to plot a visualization of the windows.
        fig = voronoi_plot_2d(vor)

        # Plot a bounding box corresponding to the grid.
        plt.plot(np.array([0, sx, sx, 0, 0]), np.array([0, 0, sy, sy, 0]), "r")

        # Format and show the plot.
        plt.xlim(-0.05 * sx, 1.05 * sx)
        plt.ylim(1.05 * sy, -0.05 * sy)
        plt.gca().set_aspect("equal")
        plt.title("Voronoi Cells")
        plt.show()

    # Gather data from scipy Voronoi and return as a list of boolean windows.
    N = np.shape(vectors)[1]
    filled_regions = []
    already_filled = np.zeros(shape, dtype=np.uint8)

    for x in range(N):
        point = tuple(np.rint(vor.points[x]).astype(np.int32))
        region = vor.regions[vor.point_region[x]]
        pts = np.rint(vor.vertices[region]).astype(np.int32)

        canvas1 = np.zeros(shape, dtype=np.uint8)
        cv2.fillConvexPoly(canvas1, pts, 255, cv2.LINE_4)

        # Crop the window to with a given radius, if desired.
        if radius is not None and radius > 0:
            canvas2 = np.zeros(shape, dtype=np.uint8)
            cv2.circle(canvas2, point, int(np.ceil(radius)), 255, -1)

            filled_regions.append((canvas1 > 0) & (canvas2 > 0) & np.logical_not(already_filled))
        else:
            filled_regions.append((canvas1 > 0) & np.logical_not(already_filled))

        already_filled |= filled_regions[-1]

    return filled_regions


# Phase pattern collation and manipulation. Uses windows.


def imprint(
    matrix,
    window,
    function,
    grid=None,
    imprint_operation="replace",
    centered=False,
    circular=False,
    clip=True,
    transform=0,
    shift=(0, 0),
    **kwargs,
):
    r"""
    Imprints a region (defined by ``window``) of a ``matrix`` with a ``function`` in-place.
    This ``function`` must be in the style of :mod:`~slmsuite.holography.toolbox.phase`
    phase helper functions, which expect a ``grid`` parameter to define the coordinate basis
    (see :meth:`~slmsuite.holography.toolbox.phase.blaze()` or
    :meth:`~slmsuite.holography.toolbox.phase.lens()`).

    For instance, we can imprint a blaze on a 200 by 200 pixel region
    of the SLM with:

    .. highlight:: python
    .. code-block:: python

        canvas = np.zeros(shape=slm.shape)  # Matrix to imprint onto.
        window = [200, 200, 200, 200]       # Region of the matrix to imprint [x, w, y, h].
        toolbox.imprint(canvas, window=window, function=toolbox.phase.blaze, grid=slm, vector=(.001, .001))

    See also :ref:`examples`.

    Parameters
    ----------
    matrix : numpy.ndarray
        The data to imprint a ``function`` onto.
        This operation occurs in-place (original array is modified).
    window
        Passed to :meth:`~slmsuite.holography.toolbox.window_slice()`.
        See :meth:`~slmsuite.holography.toolbox.window_slice()` for various options.
    function : function OR float
        A function in the style of :mod:`~slmsuite.holography.toolbox` helper functions,
        which accept ``grid`` as the first argument.

        Note
        ~~~~
        2D functions in :mod:`~slmsuite.holography.analysis.fitfunctions`
        are also of this style.

        Note
        ~~~~
        Also accepts floating point values, in which case the value is simply added.
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM` OR None
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly if a
        :mod:`~slmsuite.holography.toolbox` function is used.
        ``None`` can only be passed if a float is passed as ``function``.
    imprint_operation : {"replace" OR "add"}
        Decides how the ``function`` is imparted to the ``matrix``.

        - If ``"replace"``, then the values of ``matrix``
          inside ``window`` are replaced with ``function``.
        - If ``"add"``, then these are instead added together
          (useful, for instance, for adding global blazes).

    centered
        See :meth:`~slmsuite.holography.toolbox.window_slice()`.
    circular
        See :meth:`~slmsuite.holography.toolbox.window_slice()`.
    clip : bool
        Whether to clip the imprint region if it exceeds the size of ``matrix``.
        If ``False``, then an error is raised when the size is exceeded.
        If ``True``, then the out-of-range pixels are instead filled with ``numpy.nan``.
    transform : float or ((float, float), (float, float))
       Passed to :meth:`transform_grid`, operating on the cropped imprint grid.
       This is left as an option such that the user does not have to transform the
       entire ``grid`` to satisfy a tiny imprinted patch.
       See :meth:`transform_grid` for more details.
    shift : (float, float) OR None OR True
       Passed to :meth:`transform_grid`, operating on the cropped imprint grid.
       This is left as an option such that the user does not have to transform the
       entire ``grid`` to satisfy a tiny imprinted patch.
       If ``True``, the grid is centered on the region.
       See :meth:`transform_grid` for more details.
    **kwargs :
        For passing additional arguments accepted by ``function``.

    Returns
    -------
    matrix : numpy.ndarray
        The modified image. Note that the matrix is modified in place, and this return
        is merely a copy of the user's pointer to the data.

    Raises
    ------
    ValueError
        If invalid ``window`` or ``imprint_operation`` are provided.
    """
    # Format the grid.
    if grid is not None:
        (x_grid, y_grid) = _process_grid(grid)

    # Get slices for the window in the matrix.
    shape = matrix.shape if clip else None
    slice_ = window_slice(window, shape=shape, centered=centered, circular=circular)

    # Decide whether to treat function as a float.
    is_float = isinstance(function, REAL_TYPES)

    if not is_float:
        if grid is None:
            raise ValueError(
                "grid cannot be None if a function is given; None is a float-only option."
            )

    # Modify the matrix.
    if imprint_operation == "replace":
        if is_float:
            matrix[slice_] = function
        else:
            matrix[slice_] = function(
                transform_grid((x_grid[slice_], y_grid[slice_]), transform, shift), **kwargs
            )
    elif imprint_operation == "add":
        if is_float:
            matrix[slice_] += function
        else:
            matrix[slice_] += function(
                transform_grid((x_grid[slice_], y_grid[slice_]), transform, shift), **kwargs
            )
    else:
        raise ValueError("Unrecognized imprint operation {}.".format(imprint_operation))

    return matrix


# Vector helper functions.


def format_vectors(vectors, expected_dimension=2, handle_dimension="pass"):
    """
    Validates that an array of M-dimensional vectors is a ``numpy.ndarray`` of shape ``(M, N)``.
    Handles shaping and transposing if, for instance, tuples or row vectors are passed.

    Parameters
    ----------
    vectors : array_like
        M-vector or array of M-vectors to process. Desires shape of ``(M, N)``.
    expected_dimension : int
        Dimension of the system, i.e. ``M``.
    handle_dimension : {"error", "crop", "pass"}
        If an array of vectors with larger dimensionality than
        ``expected_dimension = M`` is provided, decides how to handle these:

        - ``"error"`` Raises an error if not ``(M, N)``.

        - ``"crop"`` Crops the higher dimensions and returns ``(M, N)``.

        - ``"pass"`` Returns ``(K, N)`` if ``K`` is greater than or equal to ``M``.

        If the array has smaller dimensionality than expected, an error is always raised.

    Returns
    -------
    vectors : numpy.ndarray
        Cleaned column vector(s). Shape of ``(M, N)``.

    Raises
    ------
    ValueError
        If the vector input was inappropriate.
    """
    # Parse expected_dimension
    expected_dimension = int(expected_dimension)

    # Parse handle_dimension
    options_dimension = ["error", "crop", "pass"]
    if not (handle_dimension in options_dimension):
        raise ValueError(
            f"handle_dimension option '{handle_dimension}' not recognized. "
            f"Must be one of '{options_dimension}'."
        )

    # Convert to np.array and squeeze
    vectors = np.squeeze(vectors)

    # Handle the singleton cases.
    if len(vectors.shape) == 1:
        vectors = vectors[:, np.newaxis]
    elif len(vectors.shape) == 2 and vectors.shape[0] == 1:
        vectors = vectors.T

    # Make sure that we are an array of N M-vectors.
    if len(vectors.shape) != 2:
        raise ValueError(f"Wrong dimension {vectors.shape} for vectors.")

    if vectors.shape[0] == expected_dimension:
        pass
    elif vectors.shape[0] > expected_dimension:     # Handle unexpected case.
        if handle_dimension == "pass":
            pass
        elif handle_dimension == "crop":
            if vectors.shape[0] > expected_dimension:
                vectors = vectors[:expected_dimension,:]
            else:
                raise ValueError(f"{vectors.shape[0]}-vectors too small to crop to {expected_dimension}-vectors.")
        elif handle_dimension == "error":
            raise ValueError(f"Expected {expected_dimension}-vectors. Found {vectors.shape[0]}-vectors.")
    else:
        raise ValueError(f"Expected {expected_dimension}-vectors. Found {vectors.shape[0]}-vectors.")


    return vectors


def format_2vectors(vectors):
    """
    Validates that an array of 2-dimensional vectors is a ``numpy.ndarray`` of shape ``(2, N)``.
    Handles shaping and transposing if, for instance, tuples or row vectors are passed.
    This is a wrapper of :meth:`format_vectors` for backwards compatibility.

    Parameters
    ----------
    vectors : array_like
        2-vector or array of 2-vectors to process. Desires shape of ``(2, N)``.
        Uses the ``"crop"`` keyword of :meth:`format_vectors`.

    Returns
    -------
    vectors : numpy.ndarray
        Cleaned column vector(s). Shape of ``(2, N)``.

    Raises
    ------
    ValueError
        If the vector input was inappropriate.
    """
    return format_vectors(vectors, expected_dimension=2, handle_dimension="crop")


def fit_3pt(y0, y1, y2, N=None, x0=(0, 0), x1=(1, 0), x2=(0, 1), orientation_check=False):
    r"""
    Fits three points to an affine transformation. This transformation is given by:

    .. math:: \vec{y} = M \cdot \vec{x} + \vec{b}

    At base, this function finds and optionally uses affine transformations:

    .. highlight:: python
    .. code-block:: python

        y0 = (1.,1.)    # Origin
        y1 = (2.,2.)    # First point in x direction
        y2 = (1.,2.)    # first point in y direction

        # If N is None, return a dict with keys "M", and "b"
        affine_dict =   fit_3pt(y0, y1, y2, N=None)

        # If N is provided, evaluates the transformation on indices with the given shape
        # In this case, the requested 5x5 indices results in an array with shape (2,25)
        vector_array =  fit_3pt(y0, y1, y2, N=(5,5))

    However, ``fit_3pt`` is more powerful than this, and can fit an affine
    transformation to semi-arbitrary sets of points with known indices
    in the coordinate system of the dependent variable :math:`\vec{x}`,
    as long as the passed indices ``x0``, ``x1``, ``x2`` are not collinear.

    .. highlight:: python
    .. code-block:: python

        # y11 is at x index (1,1), etc
        fit_3pt(y11, y34, y78, N=(5,5), x0=(1,1), x1=(3,4), x2=(7,8))

        # These indices don't have to be integers
        fit_3pt(a, b, c, N=(5,5), x0=(np.pi,1.5), x1=(20.5,np.sqrt(2)), x2=(7.7,42.0))

    Optionally, basis vectors can be passed directly instead of adding these
    vectors to the origin, by making use of passing ``None`` for ``x1`` or ``x2``:

    .. highlight:: python
    .. code-block:: python

        origin =    (1.,1.)     # Origin
        dv1 =       (1.,1.)     # Basis vector in x direction
        dv2 =       (1.,0.)     # Basis vector in y direction

        # The following are equivalent:
        option1 = fit_3pt(origin, np.add(origin, dv1), np.add(origin, dv2), N=(5,5))
        option2 = fit_3pt(origin, dv1, dv2, N=(5,5), x1=None, x2=None)

    Parameters
    ----------
    y0, y1 : array_like
        See ``y2``.
    y2 : array_like
        2-vectors defining the affine transformation. These vectors correspond to
        positions which we will fit our transformation to. These vectors have
        corresponding indices ``x0``, ``x1``, ``x2``; see these variables for more
        information. With the default values for the indices, ``y0`` is base/origin
        and ``y1`` and ``y2`` are the positions of the first point in
        the ``x`` and ``y`` directions of index-space, respectively.
        Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
    N : int OR (int, int) OR numpy.ndarray OR None
        Size of the grid of vectors to return ``(N1, N2)``.
        If a scalar is passed, then the grid is assumed square.
        If ``None`` or any non-positive integer is passed, then a dictionary
        with the affine transformation is instead returned.
        Defaults to ``None``.
    x0, x1 : array_like OR None
        See ``x2``.
    x2 : array_like OR None
        Should not be collinear.
        If ``x0`` is ``None``, defaults to the origin ``(0,0)``.
        If ``x1`` or ``x2`` are ``None``, ``y1`` or ``y2`` are interpreted as
        **differences** between ``(0,0)`` and ``(1,0)`` or ``(0,0)`` and ``(0,1)``,
        respectively, instead of as positions.
        Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
    orientation_check : bool
        If ``True``, removes the last two points in the affine grid.
        If ``False``, does nothing.

    Returns
    -------
    numpy.ndarray OR dict
        2-vector or array of 2-vectors ``(2, N)`` in slm coordinates.
        If ``N`` is ``None`` or non-positive, then returns a dictionary with keys
        ``"M"`` and ``"b"`` (transformation matrix and shift, respectively).
    """
    # Parse vectors
    y0 = format_2vectors(y0)
    y1 = format_2vectors(y1)
    y2 = format_2vectors(y2)

    # Parse index vectors
    if x0 is None:
        x0 = (0, 0)
    x0 = format_2vectors(x0)

    if x1 is None:
        x1 = x0 + format_2vectors((1, 0))
    else:
        x1 = format_2vectors(x1)
        y1 = y1 - y0

    if x2 is None:
        x2 = x0 + format_2vectors((0, 1))
    else:
        x2 = format_2vectors(x2)
        y2 = y2 - y0

    dx1 = x1 - x0
    dx2 = x2 - x0

    # Invert the index matrix.
    colinear = np.abs(np.sum(dx1 * dx2)) == np.sqrt(np.sum(dx1 * dx1) * np.sum(dx2 * dx2))
    if colinear:
        raise ValueError("Indices must not be colinear.")

    J = np.linalg.inv(np.squeeze(np.array([[dx1[0], dx2[0]], [dx1[1], dx2[1]]])))

    # Construct the matrix.
    M = np.matmul(np.squeeze(np.array([[y1[0,0], y2[0,0]], [y1[1,0], y2[1,0]]])), J)
    b = y0 - np.matmul(M, x0)

    # Deal with N and make indices.
    indices = None
    affine_return = False

    if N is None:
        affine_return = True
    elif isinstance(N, INTEGER_TYPES):
        if N <= 0:
            affine_return = True
        else:
            N = (N, N)
    elif (
        not np.isscalar(N)
        and len(N) == 2
        and isinstance(N[0], INTEGER_TYPES)
        and isinstance(N[1], INTEGER_TYPES)
    ):
        if N[0] <= 0 or N[1] <= 0:
            affine_return = True
    elif isinstance(N, np.ndarray):
        indices = format_2vectors(N)
    else:
        raise ValueError("N={} not recognized.".format(N))

    if affine_return:
        return {"M": M, "b": b}
    else:
        if indices is None:
            x_list = np.arange(N[0])
            y_list = np.arange(N[1])

            x_grid, y_grid = np.meshgrid(x_list, y_list)
            indices = np.vstack((x_grid.ravel(), y_grid.ravel()))
        if orientation_check:
            indices = indices[:, 0:-2]

        return np.array(np.matmul(M, indices) + b)


def smallest_distance(vectors, metric="chebyshev"):
    r"""
    Returns the smallest distance between pairs of points under a given ``metric``.

    Tip
    ~~~
    This function supports a :math:`\mathcal{O}(N\log(N))` divide and conquer algorithm
    and can handle large pointsets.
    An :math:`\mathcal{O}(N^2)` brute force approach is implemented as a backup.

    Caution
    ~~~~~~~
    Vectors using unsigned datatypes can lead to unexpected results when
    evaluating a distance metric. Be sure that your vectors are signed.

    Parameters
    ----------
    vectors : array_like
        Points to compare.
        Cleaned with :meth:`~slmsuite.holography.toolbox.format_2vectors()`.
    metric : str OR function
        Function to use to compare.
        Defaults to ``"chebyshev"`` which corresponds to
        :meth:`scipy.spatial.distance.chebyshev()`.
        The :math:`\mathcal{O}(N\log(N))` divide and conquer algorithm is only
        compatible with string inputs allowed by :meth:`scipy.spatial.distance.pdist`.
        Function arguments will fallback to the brute force approach.

    Returns
    -------
    float
        Minimum distance between any pair of points under the given metric.
        If fewer than two points are given, then ``np.inf`` is returned.
    """

    def _divide_and_conquer_recursive(v, metric, axis=0, min_div=200):
        # Expects sorted v.
        N = v.shape[0]

        if N > min_div:
            M = int(N/2)

            # Divide the problem recursively.
            d1 = _divide_and_conquer_recursive(v[:M, :], metric, axis)
            d2 = _divide_and_conquer_recursive(v[M:, :], metric, axis)

            # Conquer.
            d = min(d1, d2)

            # Leave if we don't need to merge.
            if (v[M, axis] - v[M+1, axis]) > d:
                return d

            # Merge around average x0 between two sections.
            x0 = (v[M, axis] + v[M+1, axis]) / 2
            mask = np.abs(v[:, axis] - x0) < d
            subset = v[mask, :]

            return min(d, distance.pdist(subset, metric=metric).min())
        else:
            # Use pdist as a fast low-level distance calculator.
            return  distance.pdist(v, metric=metric).min()

    vectors = format_2vectors(vectors)
    N = vectors.shape[1]

    if N <= 1:
        return np.inf

    if isinstance(metric, str):     # Divide and conquer.
        if not metric in distance._METRIC_ALIAS:
            raise RuntimeError("Distance metric '{metric}' not recognized by scipy.")

        axis = 0
        min_div = 200

        # pdist needs transpose.
        vectors = vectors.T

        if N < 2*min_div:
            return distance.pdist(vectors, metric=metric).min()
        else:
            centroid = np.max(vectors, axis=axis, keepdims=True)

            # Slightly inefficient use of cdist.
            xorder = distance.cdist(vectors[:,[axis]], centroid[:,[axis]], metric=metric)

            I = np.argsort(np.squeeze(xorder))
            vsort = vectors[I, :]

            return _divide_and_conquer_recursive(vsort, metric, axis=axis, min_div=min_div)
    else:                           # Fallback to brute force.
        minimum = np.inf

        for x in range(N - 1):
            for y in range(x + 1, N):
                result = metric(vectors[:, x], vectors[:, y])
                if result < minimum:
                    minimum = result

        return minimum


def lloyds_algorithm(grid, vectors, iterations=10, plot=False):
    r"""
    Implements `Lloyd's Algorithm <https://en.wikipedia.org/wiki/Lloyd's_algorithm>`_
    on a set of seed ``vectors`` to promote even vector spacing.
    Vectors are bound within the grid.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM` OR (int, int)
        Extract the rectangular bounds from a grid, or from a shape.
        See :meth:`~slmsuite.holography.toolbox.voronoi_windows()`.
    vectors : array_like
        Initial seed points for Lloyd's Algorithm.
        See :meth:`~slmsuite.holography.toolbox.voronoi_windows()`.
    iterations : int
        Number of iterations to apply Lloyd's Algorithm.
    plot : bool
        Whether to plot each iteration of the algorithm.

    Returns
    -------
    numpy.ndarray
        The result of Lloyd's Algorithm.
    """
    result = np.copy(format_2vectors(vectors)).astype(float)

    # Parse grid
    if isinstance(grid, (tuple, list)) and all(isinstance(x, int) for x in grid):
        shape = grid
    else:
        x_grid, y_grid = _process_grid(grid)
        shape = x_grid.shape
    H, W = shape

    def polygon_centroid(polygon):
        x = polygon[:, 0]
        y = polygon[:, 1]
        x_shift = np.roll(x, -1)
        y_shift = np.roll(y, -1)

        cross = x * y_shift - x_shift * y
        area = 0.5 * np.sum(cross)

        if np.isclose(area, 0):
            return np.mean(polygon, axis=0)

        cx = np.sum((x + x_shift) * cross) / (6 * area)
        cy = np.sum((y + y_shift) * cross) / (6 * area)

        return np.array([cx, cy])

    def clip_polygon(polygon, edge_fn, intersect_fn):
        clipped = []
        prev = polygon[-1]

        for curr in polygon:
            if edge_fn(curr):
                if not edge_fn(prev):
                    clipped.append(intersect_fn(prev, curr))
                clipped.append(curr)
            elif edge_fn(prev):
                clipped.append(intersect_fn(prev, curr))
            prev = curr

        return clipped

    def clip_to_box(polygon):
        def intersect(p1, p2, edge):
            x1, y1 = p1
            x2, y2 = p2
            if edge == "left":
                x = 0
                y = y1 + (y2 - y1) * (0 - x1) / (x2 - x1)
            elif edge == "right":
                x = W
                y = y1 + (y2 - y1) * (W - x1) / (x2 - x1)
            elif edge == "bottom":
                y = 0
                x = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
            elif edge == "top":
                y = H
                x = x1 + (x2 - x1) * (H - y1) / (y2 - y1)
            return [x, y]

        for edge, edge_fn, intersect_fn in [
            ("left", lambda p: p[0] >= 0, lambda p1, p2: intersect(p1, p2, "left")),
            ("right", lambda p: p[0] <= W, lambda p1, p2: intersect(p1, p2, "right")),
            ("bottom", lambda p: p[1] >= 0, lambda p1, p2: intersect(p1, p2, "bottom")),
            ("top", lambda p: p[1] <= H, lambda p1, p2: intersect(p1, p2, "top")),
        ]:
            polygon = clip_polygon(polygon, edge_fn, intersect_fn)
            if not polygon:
                break
        return np.array(polygon)

    for _ in range(iterations):
        # Add points outside the shape to ensure bounded Voronoi cells
        hsx = W / 2
        hsy = H / 2
        vectors_ext = np.concatenate(
            (
                result.T,
                np.array([[hsx, -3 * hsy], [hsx, 5 * hsy], [-3 * hsx, hsy], [5 * hsx, hsy]]),
            )
        )

        # Recomputing this each time isn't too inefficient.
        vor = Voronoi(vectors_ext)

        if plot:
            sx = shape[1]
            sy = shape[0]

            # Use the built-in scipy function to plot a visualization of the windows.
            fig = voronoi_plot_2d(vor)

            # Plot a bounding box corresponding to the grid.
            plt.plot(np.array([0, sx, sx, 0, 0]), np.array([0, 0, sy, sy, 0]), "r")

            # Format and show the plot.
            plt.xlim(-0.05 * sx, 1.05 * sx)
            plt.ylim(1.05 * sy, -0.05 * sy)
            plt.gca().set_aspect("equal")
            plt.title("Voronoi Cells")
            plt.show()

        for i in range(result.shape[1]):
            # Don't move points that don't make sense.
            region_index = vor.point_region[i]
            region = vor.regions[region_index]

            if -1 in region or len(region) == 0:
                continue

            polygon = vor.vertices[region]
            polygon = clip_to_box(polygon)
            if len(polygon) < 3:
                continue

            # Otherwise, update the point to the centroid of its cell.
            centroid = polygon_centroid(polygon)

            result[0, i] = centroid[0]
            result[1, i] = centroid[1]

    return result


def lloyds_points(grid, n_points, iterations=10, plot=False):
    r"""
    Implements `Lloyd's Algorithm <https://en.wikipedia.org/wiki/Lloyd's_algorithm>`_
    without seed ``vectors``; instead, autogenerates the seed ``vectors`` randomly.
    See :meth:`~slmsuite.holography.toolbox.lloyds_algorithm()`.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM` OR (int, int)
        See :meth:`~slmsuite.holography.toolbox.voronoi_windows()`.
    n_points : int
        Number of points to generate inside a space.
    iterations : int
        Number of iterations to apply Lloyd's Algorithm.
    plot : bool
        Whether to plot each iteration of the algorithm.

    Returns
    -------
    numpy.ndarray
        The result of Lloyd's Algorithm.
    """
    if (
        isinstance(grid, (list, tuple))
        and isinstance(grid[0], INTEGER_TYPES)
        and isinstance(grid[1], INTEGER_TYPES)
    ):
        shape = grid
    else:
        (x_grid, y_grid) = _process_grid(grid)
        shape = x_grid.shape

    vectors = np.vstack(
        (np.random.randint(0, shape[1], n_points), np.random.randint(0, shape[0], n_points))
    )

    # Regenerate until no overlaps (improve for performance?)
    while smallest_distance(vectors) < 1:
        vectors = np.vstack(
            (np.random.randint(0, shape[1], n_points), np.random.randint(0, shape[0], n_points))
        )

    grid2 = np.meshgrid(range(shape[1]), range(shape[0]))

    result = lloyds_algorithm(grid2, vectors, iterations, plot)

    if isinstance(grid, (list, tuple)):
        return result
    else:
        result = np.rint(result).astype(int)
        return np.vstack((x_grid[result[1], result[0]], y_grid[result[1], result[0]]))


def assign_vectors(vectors, assignment_options):
    r"""
    Assigns each vector in ``vectors`` to the closest counterpart ``assignment_options``.
    Uses Euclidean distance.

    Note
    ~~~~
    An :math:`\mathcal{O}(N^2)` brute force approach is currently implemented,
    though it is vectorized. This could be sped up significantly.

    Parameters
    ----------
    vectors : array_like
        Array of M-vectors of shape ``(M, vector_count)``
    assignment_options : array_like
        Array of M-vectors of shape ``(M, option_count)``

    Returns
    -------
    numpy.ndarray
        For each vector, the index of the closest ``assignment_options``.
        Of shape ``(vector_count,)``.
    """
    vectors = format_vectors(vectors)[:, np.newaxis, :]
    assignment_options = format_vectors(assignment_options)[:, :, np.newaxis]

    distance = np.sum(np.square(vectors - assignment_options), axis=0)

    return np.argmin(distance, axis=0)


# Grid functions.


def _process_grid(grid):
    r"""
    Functions in :mod:`.toolbox` make use of normalized meshgrids containing the normalized
    coordinate of each corresponding pixel. This helper function interprets what the user passes.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.

    Returns
    -------
    (array_like, array_like)
        The grids in ``(x_grid, y_grid)`` form.
    """
    # Check if it's a cameraSLM, then default to the SLM.
    if hasattr(grid, "slm"):
        grid = grid.slm

    # See if grid is an object with grid fields.
    if hasattr(grid, "grid"):
        grid = grid.grid
    elif hasattr(grid, "x_grid") and hasattr(grid, "y_grid"):
        return (grid.x_grid, grid.y_grid)

    # Otherwise, assume it's a tuple
    if len(grid) != 2:
        raise ValueError("Expected a 2-tuple with x and y meshgrids.")
    if np.any(np.shape(grid[0]) != np.shape(grid[1])):
        raise ValueError("Expected a 2-tuple with x and y meshgrids.")

    return grid


def transform_grid(grid, transform=None, shift=None, direction="fwd"):
    r"""
    Returns a copy of a coordinate basis ``grid`` with a given ``shift`` and
    ``transformation``. These can be the :math:`\vec{b}` and :math:`M` of a standard
    affine transformation as used elsewhere in the package.
    Such grids are used as arguments for phase patterns, such as those in
    :mod:`slmsuite.holography.toolbox.phase`.

    Parameters
    ----------
    grid : (array_like, array_like) OR :class:`~slmsuite.hardware.slms.slm.SLM`
        Meshgrids of normalized :math:`\frac{x}{\lambda}` coordinates
        corresponding to SLM pixels, in ``(x_grid, y_grid)`` form.
        These are precalculated and stored in any :class:`~slmsuite.hardware.slms.slm.SLM`, so
        such a class can be passed instead of the grids directly.
    transform : float OR ((float, float), (float, float)) OR None
        If a scalar is passed, this is the angle to rotate the basis of the lens by.
        Defaults to zero if ``None``.
        If a 2x2 matrix is passed, transforms the :math:`x` and :math:`y` grids
        according to :math:`x' = M_{00}x + M_{01}y`,  :math:`y' = M_{10}x + M_{11}y`.
    shift : (float, float) OR None OR True
        Translational shift of the grid in normalized :math:`\frac{x}{\lambda}` coordinates
        ("fwd" direction). Defaults to no shift if ``None``.
        If ``True``, shifts the grid to be centered upon itself.
    direction : str in ``{"fwd", "rev"}``
        Defines the direction of the transform: forward (``"fwd"``) transforms then shifts;
        reverse (``"rev"``) undoes the shift then applies the inverse transform.
        For standard affine transforms, forward (reverse) mode transforms ``"kxy"`` to ``"ij"``
        (``"ij"`` to ``"kxy"``).

    Returns
    -------
    grid : (array_like, array_like)
        The shifted grid. In the case where the transform is the identity, a copy is
        returned not the original grids.
    """
    # Parse grid.
    (x_grid, y_grid) = _process_grid(grid)

    # Parse transform.
    if transform is None:
        transform = 0
    if not np.isscalar(transform):
        transform = np.squeeze(transform)
        if transform.shape != (2,2):
            raise ValueError("Expected transform to be None, scalar, or a 2x2 matrix.")

    # Parse shift.
    if shift is None:
        shift = (0, 0)
    if shift is True:
        shift = (-np.mean(x_grid), -np.mean(y_grid))
    shift = np.squeeze(shift)

    # Return the transformed grids.
    if np.isscalar(transform) and transform == 0:  # The trivial case
        if direction == "fwd":
            return (
                x_grid.copy() if shift[0] == 0 else (x_grid + shift[0]),
                y_grid.copy() if shift[1] == 0 else (y_grid + shift[1]),
            )
        elif direction == "rev":
            return (
                x_grid.copy() if shift[0] == 0 else (x_grid - shift[0]),
                y_grid.copy() if shift[1] == 0 else (y_grid - shift[1]),
            )
    else:  # transform is not trivial.
        # Interpret angular transform as a matrix.
        if np.isscalar(transform):
            s = np.sin(transform)
            c = np.cos(transform)
            transform = np.array([[c, -s], [s, c]])

        # Use the matrix to transform the grid.
        if direction == "fwd":
            return (
                transform[0, 0] * x_grid + transform[0, 1] * y_grid + shift[0],
                transform[1, 0] * x_grid + transform[1, 1] * y_grid + shift[1],
            )
        elif direction == "rev":
            transform = np.linalg.inv(transform)
            return (
                transform[0, 0] * (x_grid - shift[0]) + transform[0, 1] * (y_grid - shift[1]),
                transform[1, 0] * (x_grid - shift[0]) + transform[1, 1] * (y_grid - shift[1]),
            )


# Shape and padding functions.


def format_shape(shape, expected_dimension=2):
    """
    Helper function to format a shape tuple.

    Parameters
    ----------
    shape : (int, int) OR tuple of int
        The shape to format.
    expected_dimension : int OR None
        The expected number of dimensions. Ignored if ``None``.

    Returns
    -------
    tuple
        The formatted shape.
    """
    shape = tuple(np.squeeze(shape))

    if expected_dimension is not None:
        if len(shape) != expected_dimension:
            raise ValueError(f"Expected shape with {expected_dimension} dimensions, got {len(shape)}")

    for dim in shape:
        if not isinstance(dim, INTEGER_TYPES) or dim <= 0:
            raise ValueError(f"Expected positive integer dimensions, got {shape}")

    return tuple([int(d) for d in shape])


def pad(matrix, shape):
    """
    Helper function to pad data with zeros. The padding is centered.
    This is used to get higher resolution in the :math:`k`-space upon Fourier transform.

    Parameters
    ----------
    matrix : numpy.ndarray
        Data to pad.
    shape : (int, int) OR None
        The desired shape of the ``matrix`` in :mod:`numpy` ``(h, w)`` form.
        If ``None``, the ``matrix`` is returned unpadded.

    Returns
    -------
    numpy.ndarray
        Padded ``matrix``.
    """
    if shape is None:
        return matrix

    shape = format_shape(shape)

    deltashape = (
        (shape[0] - matrix.shape[0]) / 2.0,
        (shape[1] - matrix.shape[1]) / 2.0,
    )

    if not (deltashape[0] >= 0 and deltashape[1] >= 0):
        raise ValueError(f"Shape {tuple(matrix.shape)} is too large to pad to shape {shape}")

    pad_b = int(np.floor(deltashape[0]))
    pad_t = int(np.ceil(deltashape[0]))
    pad_l = int(np.floor(deltashape[1]))
    pad_r = int(np.ceil(deltashape[1]))

    padded = np.pad(matrix, [(pad_b, pad_t), (pad_l, pad_r)], mode="constant", constant_values=0)

    if not padded.shape == shape:
        raise RuntimeError("Padded result should have desired shape.")

    return padded


def unpad(matrix, shape):
    """
    Helper function to unpad data. The padding is assumed to be centered.
    This is used to get higher resolution in the :math:`k`-space upon Fourier transform.

    Parameters
    ----------
    matrix : numpy.ndarray OR (int, int)
        Data to unpad, if this is a matrix.
        If this is a shape in :mod:`numpy` ``(h, w)`` form,
        returns the four slicing integers used to unpad that shape ``[pad_b:pad_t, pad_l:pad_r]``.
    shape : (int, int) OR None
        The desired shape of the ``matrix`` in :mod:`numpy` ``(h, w)`` form.
        If ``None``, the ``matrix`` is returned unchanged.

    Returns
    -------
    numpy.ndarray OR (int, int, int, int)
        Either the unpadded ``matrix`` or the four slicing integers used to unpad such a matrix,
        depending on what is passed as ``matrix``.
    """
    mshape = np.shape(matrix)
    return_args = False
    if len(mshape) == 1 or np.prod(mshape) == 2:
        # Assume a shape was provided instead of a matrix.
        mshape = format_shape(matrix)
        return_args = True

    if shape is None:
        if return_args:
            return (0, mshape[0], 0, mshape[1])
        else:
            return matrix

    shape = format_shape(shape)

    deltashape = ((shape[0] - mshape[0]) / 2.0, (shape[1] - mshape[1]) / 2.0)

    if not (deltashape[0] <= 0 and deltashape[1] <= 0):
        raise ValueError(f"Shape {tuple(mshape)} is too small to unpad to shape {shape}")

    pad_b = int(np.floor(-deltashape[0]))
    pad_t = int(mshape[0] - np.ceil(-deltashape[0]))
    pad_l = int(np.floor(-deltashape[1]))
    pad_r = int(mshape[1] - np.ceil(-deltashape[1]))

    if return_args:
        return (pad_b, pad_t, pad_l, pad_r)

    unpadded = matrix[pad_b:pad_t, pad_l:pad_r]

    if not unpadded.shape == shape:
        raise RuntimeError("Unpadded result should have desired shape.")

    return unpadded
