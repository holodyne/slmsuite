r"""
Helper functions for processing images.
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import reduce
from scipy.optimize import curve_fit, minimize
from scipy.ndimage import binary_erosion

import warnings
try:
    import cupy as cp   # type: ignore
except ImportError:
    cp = np

from slmsuite.holography.toolbox import format_2vectors, _process_grid
from slmsuite.holography.toolbox.phase import zernike_sum, laguerre_gaussian
from slmsuite.misc.math import REAL_TYPES
from slmsuite.holography.analysis.fitfunctions import gaussian2d

# Take and associated functions.

def _center(width, integer=False):
    """
    Center of an index range with length ``width``.
    """
    if integer:
        return int((width - 1) / 2 if width % 2 else width / 2)
    else:
        return float(width - 1) / 2


def _coordinates(width, centered=False):
    """
    Coordinate indices of length ``width``.
    """
    xs = np.arange(width).astype(np.float64)
    if centered:
        center = np.float64(_center(width))
        xs -= center
    return xs


def _generate_grid(w_x, w_y, centered=False, integer=False):
    """

    """
    xs = np.reshape(np.arange(w_x, dtype=float), (1, 1, w_x))
    ys = np.reshape(np.arange(w_y, dtype=float), (1, w_y, 1))
    if centered:
        xs -= _center(w_x, integer=integer)
        ys -= _center(w_y, integer=integer)
    grid = np.meshgrid(xs, ys)
    return grid


def take(
        images,
        vectors,
        size,
        centered=True,
        integrate=False,
        clip=False,
        return_mask=False,
        plot=False,
        xp=None
    ):
    """
    Crop integration regions around an array of ``vectors``, yielding an array of images.

    Each integration region is a rectangle of the same ``size``. Similar to but more
    general than :meth:`numpy.take`. Useful for gathering data from spots in spot
    arrays. Operates with some speed due to the vectorized nature of implemented slicing.

    Parameters
    ----------
    images : array_like
        2D image or array of 2D images.
    vectors : array_like of floats
        2-vector (or 2-vector array). Location(s) of integration region anchor(s) in pixels,
        see ``centered``.
        See :meth:`~slmsuite.holography.toolbox.format_2vectors`.
    size : int or (int, int)
        Size of the rectangular integration region in ``(w, h)`` format in pixels.
        If a scalar is given, assume square ``(w, w)``.
    centered : bool
        Whether to center the integration region on the ``vectors``.
        If ``False``, ``vectors`` indicates the upper-left corner of the integration region.
        Defaults to ``True``.
    integrate : bool
        If ``True``, the spatial dimension are integrated (summed), yielding a result of the
        same length as the number of vectors. Forces floating point datatype before the
        summation is done, as integer data (especially for cameras near saturation) can overflow.
        Defaults to ``False``.
    clip : bool
        Whether to allow out-of-range integration regions. ``True`` allows regions outside
        the valid area, setting the invalid region to ``np.nan``
        (or zero if the array datatype does not support ``np.nan``).
        ``False`` throws an error upon out of range. Defaults to ``False``.
    return_mask : bool
        If ``True``, returns a boolean mask corresponding to the regions which are taken
        from. Defaults to ``False``. The average user will ignore this.
    plot : bool
        Calls :meth:`take_plot()` to visualize the images regions.
    xp : module OR None
        If ``images`` are :mod:`cupy` objects, then :mod:`cupy` must be passed as
        ``xp``. Very useful to minimize the cost of moving data between the GPU and CPU.
        If ``None``, defaults to :mod:`numpy`.
        Indexing variables inside :meth:`take` still use :mod:`numpy` for speed, no
        matter what module is used.

    Returns
    -------
    numpy.ndarray OR cupy.ndarray
        If ``integrate`` is ``False``, returns an array containing the images cropped
        from the regions of size ``(image_count, h, w)``.
        If ``integrate`` is ``True``, instead returns an array of floats of size ``(image_count,)``
        where each float corresponds to the :meth:`numpy.sum` of a cropped image.
        If ``xp`` is :mod:`cupy`, then a ``cupy.ndarray`` is returned.
    """
    # Clean variables.
    if np.isscalar(size):
        size = int(size)
        size = (size, size)
    else:
        s = np.asarray(size).ravel()
        size = (int(s[0]), int(s[1]))

    vectors = np.floor(format_2vectors(vectors)).astype(int)

    if xp is None:
        xp = np

    # Prepare helper variables. Future: consider caching for speed, if not negligible.
    edge_x = np.floor(_coordinates(size[0], centered)).astype(int)
    edge_y = np.floor(_coordinates(size[1], centered)).astype(int)

    region_x, region_y = np.meshgrid(edge_x, edge_y)

    # Get the lists for the integration regions.
    integration_x = np.add(region_x.ravel()[:, np.newaxis].T, vectors[:][0][:, np.newaxis])
    integration_y = np.add(region_y.ravel()[:, np.newaxis].T, vectors[:][1][:, np.newaxis])

    images = xp.array(images, copy=(False if np.__version__[0] == '1' else None))
    shape = xp.shape(images)

    if clip:  # Prevent out-of-range errors by clipping.
        mask = (
            (integration_x < 0) | (integration_x >= shape[-1]) |
            (integration_y < 0) | (integration_y >= shape[-2])
        )

        if np.any(mask):
            # Clip these indices to prevent errors.
            np.clip(integration_x, 0, shape[-1] - 1, out=integration_x)
            np.clip(integration_y, 0, shape[-2] - 1, out=integration_y)
        else:
            # No clipping needed.
            clip = False
    else:
        pass  # Don't prevent out-of-range errors.

    if return_mask:
        if return_mask == 2:
            canvas = np.full(images.shape, np.nan, dtype=float)
            canvas[integration_y, integration_x] = images[integration_y, integration_x]
        else:
            canvas = np.zeros(shape[:2], dtype=bool)
            canvas[integration_y, integration_x] = True

        if plot:
            plt.imshow(canvas)
            plt.show()

        return canvas
    else:
        # Take the data, depending on the shape of the images.
        if len(shape) == 2:
            result = images[np.newaxis, integration_y, integration_x]
        elif len(shape) == 3:
            result = images[:, integration_y, integration_x]
        else:
            raise RuntimeError("Unexpected shape for images: {}".format(shape))

        if clip:  # Set values that were out of range to nan instead of erroring.
            try:  # If the datatype of result is incompatible with nan, set to zero instead.
                result[:, mask] = np.nan
            except:
                result[:, mask] = 0
        else:
            pass

        # Plot if desired
        if plot:
            take_plot(xp.reshape(result, (vectors.shape[1], size[1], size[0])), separate_axes=False)

        if integrate:  # Sum over the integration axis.
            return xp.squeeze(xp.sum(result.astype(float), axis=-1))
        else:  # Reshape the integration axis.
            return xp.reshape(result, (vectors.shape[1], size[1], size[0]))


def take_plot(images, shape=None, separate_axes=False, cbar=True):
    """
    Plots non-integrated results of :meth:`.take()` in a square array of subplots.

    Parameters
    ----------
    images : numpy.ndarray
        Stack of 2D images, usually a :meth:`take()` output.
    shape : (int, int) or None
        Shape of the subplots.
        If ``None``, the shape is determined by the number of images (smallest square).
    separate_axes : bool
        If ``True``, each image is plotted in a separate subplot.
        If ``False``, uses :meth:`take_tile()` to plot all images on a single axes.
    cbar : bool
        Whether to include a colorbar. Currently only applies if ``separate_axes`` is ``False``.
    """
    # Gather helper variables and set the min and max of all the subplots.
    (img_count, sy, sx) = np.shape(images)
    img_count, (M, N) = _take_parse_shape(images, shape)

    if separate_axes:
        sx = sx / 2.0 - 0.5
        sy = sy / 2.0 - 0.5
        extent = (-sx, sx, -sy, sy)

        vmin = np.nanmin(images)
        vmax = np.nanmax(images)

        # Make the figure and subplots.
        plt.figure(figsize=(12, 12))

        for x in range(img_count):
            ax = plt.subplot(M, M, x + 1)

            ax.imshow(
                images[x, :, :],
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                interpolation='none'
            )
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
    else:
        im = plt.imshow(
            take_tile(images, shape),
            interpolation='none'
        )
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        # Draw horizontal and vertical lines to separate the images.
        for x in range(1, N):
            ax.axvline(x=sx * x, color='r', linewidth=0.5)
        for y in range(1, M):
            ax.axhline(y=sy * y, color='r', linewidth=0.5)

        if cbar:
            cax = make_axes_locatable(ax).append_axes("right", size="2%", pad=0.05)
            plt.gcf().colorbar(im, cax=cax, orientation="vertical")

            # Return the current axes to the original one.
            plt.sca(ax)


def _take_parse_shape(images, shape=None):
    """
    Parses the shape of the images and returns the number of images and the shape.
    """
    (img_count, _, _) = np.shape(images)

    # Parse shape.
    if shape is None:
        M = N = int(np.ceil(np.sqrt(img_count)))
    else:
        (M, N) = shape

    if M*N < img_count:
        warnings.warn("Not enough space to fit all images. Truncating the image count.")
        img_count = M*N

    return img_count, (M, N)


def take_tile(images, shape=None):
    """
    Tiles a stack of images into a single image.
    The stack is arranged into a grid of shape ``shape``.

    Parameters
    ----------
    images : numpy.ndarray
        Stack of 2D images, usually a :meth:`take()` output.
    shape : (int, int) or None
        Shape of the tiled grid.
        If ``None``, the shape is determined by the number of images (smallest square).
    """
    (img_count, sy, sx) = np.shape(images)
    img_count, (M, N) = _take_parse_shape(images, shape)

    result = np.zeros((M*N, sy, sx), images.dtype)
    result[:img_count, :, :] = images[:, :, :]

    return result.reshape(M, N, sy, sx).transpose(0, 2, 1, 3).reshape(M*sy, N*sx)


def image_remove_field(images, deviations=1, out=None):
    r"""
    Zeros the field of a stack of images such that moment calculations will succeed.
    Consider, for example, a small spot on a field with strong background.
    Moment calculations in this situation will dominantly measure the moments
    of the background (i.e. the field). This function zeros the image below some threshold.
    This threshold is set to either the mean plus ``deviations`` standard deviations,
    computed uniquely for each image, or the median of each image if ``deviations``
    is ``None``. This is equivalent to background subtraction.

    Important
    ~~~~~~~~~
    If a stack of images is provided, field removal is done individually on each image.
    Field removal is not done in aggregate.

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the width and height of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed, though the returned image remains shape ``(h, w)`` in that case.
    deviations : int OR None
        Number of standard deviations above the mean to set the threshold.
        If ``None``, uses the median as the threshold instead.
        Defaults to ``None``.
    out : numpy.ndarray or None
        The array to place the output data into. Should be the same shape as ``images``.
        This function operates in-place if ``out`` equals ``images``.

    Returns
    -------
    out : numpy.ndarray
        ``images`` or a copy of ``images``, with each image background-subtracted.
    """
    # Parse images. Convert to float.
    images = np.array(images, copy=(False if np.__version__[0] == '1' else None))
    if not isinstance(images.dtype, np.floating):
        images = np.array(images, copy=(False if np.__version__[0] == '1' else None), dtype=float)  # Hack to prevent integer underflow.

    # Parse out.
    if out is None:
        out = np.copy(images)
    elif not (out is images):
        np.copyto(out, images)

    # Make sure that we're testing 3D images.
    single_image = len(images.shape) == 2
    if single_image:
        images_ = np.reshape(images, (1, images.shape[0], images.shape[1]))
    else:
        images_ = images
    img_count = images_.shape[0]

    # Generate the threshold.
    if deviations is None:  # Median case
        threshold = np.nanmedian(images_, axis=(1, 2))
    else:   # Mean + deviations * std case
        threshold = (
            np.nanmean(images_, axis=(1, 2))
            + deviations * np.nanstd(images_, axis=(1, 2))
        )
    if not single_image:
        threshold = np.reshape(threshold, (img_count, 1, 1))

    out_max = np.amax(out, axis=(1,2), keepdims=True)

    # Remove the field. This needs the float from before. Unsigned integer could underflow.
    out -= threshold.astype(out.dtype)
    out[out < 0] = 0
    out[out > out_max - threshold] = 0

    return out


def image_relative_strehl(images):
    r"""
    Computes a metric proportional to the Strehl ratio of a stack of images.

    .. math:: S = \frac{\max_{x,y} I}{\sum_{x,y} I}

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the width and height of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed.

    Returns
    -------
    numpy.ndarray
        The relative Strehl ratio evaluated for every image. This is of size ``(image_count,)``
        for provided ``images`` data of shape ``(image_count, h, w)``.
    """
    images = np.array(images, copy=(False if np.__version__[0] == '1' else None))
    if len(images.shape) == 2:
        images = np.reshape(images, (1, images.shape[0], images.shape[1]))

    return np.amax(images, axis=(1,2)) / np.sum(images, axis=(1,2))


def image_moment(images, moment=(1, 0), centers=(0, 0), grid=None, normalize=True, nansum=False):
    r"""
    Computes the given `moment <https://en.wikipedia.org/wiki/Moment_(mathematics)>`_
    :math:`M_{m_xm_y}` for a stack of images.
    This involves discretely integrating each image against polynomial trial functions:

    .. math:: M_{m_xm_y} = \frac{   \int_{-w_x/2}^{+w_x/2} dx \, (x-c_x)^{m_x}
                                    \int_{-w_y/2}^{+w_y/2} dy \, (y-c_y)^{m_y}
                                    P(x, y)
                                }{  \int_{-w_x/2}^{+w_x/2} dx \,
                                    \int_{-w_y/2}^{+w_y/2} dy \,
                                    P(x, y)},

    where :math:`P(x, y)` is a given 2D image of size :math:`w_x \times w_y`,
    and :math:`(c_x, c_y)` is a shift in the center of the trial functions.

    Caution
    ~~~~~~~
    This function does not check for or correct for negative values in ``images``.
    Negative values may produce unusual results.

    Warning
    ~~~~~~~
    Higher order even moments (e.g. 2) will potentially yield unexpected results if
    the images are not background-subtracted. For instance, a calculation on an image
    with large background will yield the moment of the window, rather than say anything
    about the image. Consider using :meth:`image_remove_field()` to background-subtract.

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the width and height of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed.
    moment : (int, int)
        The moments in the :math:`x` and :math:`y` directions: :math:`(m_x, m_y)`. For instance,

        -   :math:`M_{m_xm_y} = M_{10}` corresponds to the :math:`x` moment or
            the position in the :math:`x` dimension.
        -   :math:`M_{m_xm_y} = M_{11}` corresponds to :math:`xy` shear.
        -   :math:`M_{m_xm_y} = M_{02}` corresponds to the :math:`y^2` moment, or the variance
            (squared width for a Gaussian) in the :math:`y` direction,
            given a zero or zeroed (via ``centers``) :math:`M_{01}` moment.

    centers : (float, float) or array_like
        Perturbations to the center of the trial function, :math:`(c_x, c_y)`.
        Of shape ``(2, image_count)`` if there is a custom center for each image.
    grid : float OR (float, float) OR (array_like, array_like) OR None
        If ``None`` (the default), the moment is reported in pixels of the image.
        However, the user may specify other units:

        -   Providing the scaling factor between pixels and the desired units as a
            ``float`` or an anisotropic ``(float, float)``.
            This corresponds to the pixel's :math:`\Delta x`, :math:`\Delta y`.
        -   Providing lists of length ``w`` and ``h`` as a tuple as the grid dimension.
            **If the user pre-allocates and reuses these lists, this case has best performance.**
        -   Providing two full grids of shape ``(h, w)``, one for each direction.
            Note that this case is the most general, and can lead to a rotated grid if a
            transformed grid is provided.

    normalize : bool
        Whether to normalize ``images``.
        If ``False``, normalization is assumed to have been precomputed.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.
        :meth:`numpy.nansum()` treats ``nan`` values as zeros.
        This is useful in the case where ``clip=True`` is passed to :meth:`take()`
        (out of range is set to ``nan``).

    Returns
    -------
    numpy.ndarray
        The moment :math:`M_{m_xm_y}` evaluated for every image. This is of size ``(image_count,)``
        for provided ``images`` data of shape ``(image_count, h, w)``.
    """
    # Parse arguments.
    images = np.array(images, copy=(False if np.__version__[0] == '1' else None))
    if len(images.shape) == 2:
        images = np.reshape(images, (1, images.shape[0], images.shape[1]))
    (img_count, w_y, w_x) = images.shape

    moment = (int(moment[0]), int(moment[1]))

    if nansum:
        np_sum = np.nansum
    else:
        np_sum = np.sum

    # Handle normalization.
    if normalize:
        normalization = np_sum(images, axis=(1, 2), keepdims=False).reshape((img_count, 1, 1))
        reciprocal = np.reciprocal(
            normalization, where=normalization != 0, out=np.zeros((img_count,1,1))
        )
    else:
        reciprocal = 1

    if moment[0] == 0 and moment[1] == 0:  # 0,0 (norm) case
        if normalize:
            return np.ones((img_count,))
        else:
            return np_sum(images, axis=(1, 2), keepdims=False)
    else:
        if len(np.shape(centers)) == 2:
            c_x = np.reshape(centers[0], (img_count, 1, 1))
            c_y = np.reshape(centers[1], (img_count, 1, 1))
        elif len(np.shape(centers)) == 1:
            c_x = centers[0]
            c_y = centers[1]

        # Parse grid.
        if grid is None or np.isscalar(grid) or (np.isscalar(grid[0]) and np.isscalar(grid[1])):
            # Default to the pixel grid.
            if moment[0] != 0:
                x_grid = np.reshape(np.arange(w_x) - _center(w_x), (1, 1, w_x)) - c_x
                if moment[0] != 1:
                    x_grid = np.power(x_grid, moment[0], out=x_grid)
            else:
                x_grid = 0

            if moment[1] != 0:
                y_grid = np.reshape(np.arange(w_y) - _center(w_y), (1, w_y, 1)) - c_y
                if moment[1] != 1:
                    y_grid = np.power(y_grid, moment[1], out=y_grid)
            else:
                y_grid = 0

            # Handle the dx, dy option.
            if grid is not None:
                if np.isscalar(grid):
                    x_grid *= grid[0]
                    y_grid *= grid[1]
                else:
                    x_grid *= grid
                    y_grid *= grid
        else:
            x_grid, y_grid = grid

            if len(np.shape(x_grid)) == 2:                          # 2D grids.
                x_grid = np.reshape(x_grid, (1, w_y, w_x)) - c_x
                y_grid = np.reshape(y_grid, (1, w_y, w_x)) - c_y
            elif len(np.shape(x_grid)) == 1:                        # 1D grids.
                x_grid = np.reshape(x_grid, (1, 1, w_x)) - c_x
                y_grid = np.reshape(y_grid, (1, w_y, 1)) - c_y
            elif len(np.shape(x_grid)) == 3:
                pass
            else:
                raise ValueError(f"Could not parse grid of shape {x_grid.shape}")

            # Don't modify original memory.
            if moment[0] > 1: x_grid = np.power(x_grid, moment[0])
            if moment[1] > 1: y_grid = np.power(y_grid, moment[1])

        if moment[1] == 0:      # Only-x case.
            return np_sum(images * x_grid * reciprocal, axis=(1, 2), keepdims=False)
        elif moment[0] == 0:    # Only-y case.
            return np_sum(images * y_grid * reciprocal, axis=(1, 2), keepdims=False)
        else:                   # Shear case.
            return np_sum(images * x_grid * y_grid * reciprocal, axis=(1, 2), keepdims=False)


def image_normalization(images, nansum=False):
    """
    Computes the zeroth order moments, equivalent to spot mass or normalization,
    for a stack of images.

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the width and height of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.

    Returns
    -------
    numpy.ndarray
        The normalization factor :math:`M_{11}` in an array of shape ``(image_count,)``.
    """
    return image_moment(images, (0, 0), normalize=False, nansum=nansum)


def image_normalize(images, nansum=False, remove_field=False):
    """
    Normalizes of a stack of images via the the zeroth order moments
    such that each image sums to one.

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the width and height of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed, though the returned image remains shape ``(h, w)`` in that case.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.
    remove_field : bool
        Whether to apply :meth:`.image_remove_field()` to avoid background-dominated moments.

    Returns
    -------
    images_normalized : numpy.ndarray
        A copy of ``images``, with each image normalized.
    """
    if remove_field:
        images = image_remove_field(images)
    else:
        images = np.array(images, copy=(False if np.__version__[0] == '1' else None), dtype=float)

    single_image = len(images.shape) == 2

    normalization = image_normalization(images, nansum=nansum)

    if single_image:
        normalization = float(normalization.item())
        if normalization == 0:
            return np.zeros_like(images)
        else:
            return images / normalization
    else:
        reciprocal = np.reciprocal(
            normalization, where=normalization != 0, out=np.zeros(len(normalization))
        )
        return images * np.reshape(reciprocal, (len(normalization), 1, 1))


def image_positions(images, grid=None, normalize=True, nansum=False):
    r"""
    Computes the two first order moments, equivalent to spot position
    :math:`\left<x\right>` relative to image center, for a stack of images.
    Specifically, returns :math:`M_{10}` and :math:`M_{01}`.

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the width and height of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed.
    grid : float OR (float, float) OR (array_like, array_like) OR None
        If ``None`` (the default), the moment is reported in pixels of the image.
        However, the user may specify other units:

        -   Providing the scaling factor between pixels and the desired units as a
            ``float`` or an anisotropic ``(float, float)``.
            This corresponds to the pixel's :math:`\Delta x`, :math:`\Delta y`.
        -   Providing lists of length ``w`` and ``h`` as a tuple as the grid dimension.
        -   Providing full grids of shape ``(w, h)`` in each direction. Note that this
            case is the most general, and can lead to a rotated grid if a transformed
            grid is provided.

    normalize : bool
        Whether to normalize ``images``.
        If ``False``, normalization is assumed to have been precomputed.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.

    Returns
    -------
    numpy.ndarray
        Stack of :math:`M_{10}`, :math:`M_{01}` in an array of shape ``(2, image_count)``.
    """
    if normalize:
        images = image_normalize(images, nansum=nansum)

    return np.vstack(
        (
            image_moment(images, (1, 0), grid=grid, normalize=False, nansum=nansum),
            image_moment(images, (0, 1), grid=grid, normalize=False, nansum=nansum),
        )
    )


def image_centroids(images, grid=None, normalize=True, nansum=False):
    """Alias for :meth:`image_positions()`"""
    return image_positions(images, grid, normalize, nansum)


def image_variances(images, centers=None, grid=None, normalize=True, nansum=False, exclude_shear=False):
    r"""
    Computes the three second order central moments, equivalent to variance, for a stack
    of images.
    Specifically, this function returns a stack of the moments :math:`M_{20}` and
    :math:`M_{02}`, along with :math:`M_{11}`, which are the variance in the :math:`x`
    and :math:`y` directions, along with the so-called shear variance.
    Recall that variance defined as

    .. math:: (\Delta x)^2 = \left<(x - \left<x\right>)^2\right>.

    This equation is made central by subtraction of :math:`\left<x\right>`.
    The user can of course use :meth:`take_moment` directly to access the
    non-central moments; this function is a helper to access useful quantities
    for analysis of spot size and skewness.

    Note
    ~~~~
    The moment :math:`M_{20} = (\Delta x)^2` is the variance in the
    :math:`x` direction, or the square of the standard deviation :math:`\Delta x`.
    The standard deviation :math:`\Delta x` is equal to the
    :math:`1/e` amplitude radius (:math:`1/e^2` power radius) of a Gaussian beam.

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the width and height of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed.
    centers : numpy.ndarray OR None
        If the user has already computed :math:`\left<x\right>`, for example via
        :meth:`image_positions()`, then this can be passed though ``centers``. The default
        ``None`` computes ``centers`` internally.
    grid : float OR (float, float) OR (array_like, array_like) OR None
        If ``None`` (the default), the moment is reported in pixels of the image.
        However, the user may specify other units:

        -   Providing the scaling factor between pixels and the desired units as a
            ``float`` or an anisotropic ``(float, float)``.
            This corresponds to the pixel's :math:`\Delta x`, :math:`\Delta y`.
        -   Providing lists of length ``w`` and ``h`` as a tuple as the grid dimension.
        -   Providing full grids of shape ``(w, h)`` in each direction. Note that this
            case is the most general, and can lead to a rotated grid if a transformed
            grid is provided.

    normalize : bool
        Whether to normalize ``images``.
        If ``False``, normalization is assumed to have been precomputed.
    nansum : bool
        Whether to use :meth:`numpy.nansum()` in place of :meth:`numpy.sum()`.
    exclude_shear : bool
        Whether to exclude calculation of the shear variance.
        The user can choose this for speed.

    Returns
    -------
    numpy.ndarray
        Stack of :math:`M_{20}`, :math:`M_{02}`, and :math:`M_{11}`
        in an array of shape ``(3, image_count)``.
        If ``exclude_shear``,
        Stack of :math:`M_{20}` and :math:`M_{02}`
        in an array of shape ``(2, image_count)``.
    """
    if normalize:
        images = image_normalize(images, nansum=nansum)

    if centers is None:
        centers = image_positions(images, normalize=False, nansum=nansum)

    m20 = image_moment(images, (2, 0), centers=centers, grid=grid, normalize=False, nansum=nansum)
    m02 = image_moment(images, (0, 2), centers=centers, grid=grid, normalize=False, nansum=nansum)

    if exclude_shear:
        return np.vstack((m20, m02))
    else:
        m11 = image_moment(images, (1, 1), centers=centers, grid=grid, normalize=False, nansum=nansum)

        return np.vstack((m20, m02, m11))


def image_std(images, centers=None, grid=None, normalize=True, nansum=False):
    """Near-alias of :meth:`image_variances()`. Excludes the shear variance."""
    return np.sqrt(image_variances(images, centers, grid, normalize, nansum, exclude_shear=True))


def image_ellipticity(variances):
    r"""
    Given the output of :meth:`image_variances()`,
    return a measure of spot ellipticity for each moment triplet.
    The output of :meth:`image_variances()` contains the moments :math:`M_{20}`,
    :math:`M_{02}`, and :math:`M_{11}`. These terms make up a :math:`2 \times 2` matrix,
    which is equivalent to a rotated elliptical scaling according to the eigenvalues
    :math:`\lambda_+` and :math:`\lambda_-` and some rotation matrix :math:`R(\phi)`.

    .. math::   \begin{bmatrix}
                    M_{20} & M_{11} \\
                    M_{11} & M_{02} \\
                \end{bmatrix}
                =
                R(-\phi)
                \begin{bmatrix}
                    \lambda_+ & 0 \\
                    0 & \lambda_- \\
                \end{bmatrix}
                R(\phi).

    We use this knowledge, along with tricks for eigenvalue calculations on
    :math:`2 \times 2` matrices, to build up a metric for ellipticity:

    .. math:: \mathcal{E} = 1 - \frac{\lambda_-}{\lambda_+}.

    Notice that

    - when :math:`\lambda_+ = \lambda_-` (isotropic scaling), the metric is zero and
    - when :math:`\lambda_- = 0` (flattened to a line), the metric is unity.

    Parameters
    ----------
    variances : numpy.ndarray
        The output of :meth:`image_variances()`. Shape ``(3, image_count)``.

    Returns
    -------
    numpy.ndarray
        Array of ellipticities for the given moments in an array of shape ``(image_count,)``.
    """
    m20 = variances[0, :]
    m02 = variances[1, :]
    m11 = variances[2, :]

    # We can use a trick for eigenvalue calculations of 2x2 matrices to avoid
    # more complicated calculations.
    half_trace = (m20 + m02) / 2
    determinant = m20 * m02 - m11 * m11

    eig_half_difference = np.sqrt(np.square(half_trace) - determinant)

    eig_plus = half_trace + eig_half_difference
    eig_minus = half_trace - eig_half_difference

    return 1 - (eig_minus / eig_plus)


def image_areas(variances):
    r"""
    Given the output of :meth:`image_variances()`,
    return a measure of spot area for each moment triplet.
    The output of :meth:`image_variances()` contains the moments :math:`M_{20}`,
    :math:`M_{02}`, and :math:`M_{11}`. We return the determinant :math:`|M|` which is a
    proxy for spot area.

    Parameters
    ----------
    variances : numpy.ndarray
        The output of :meth:`image_variances()`. Shape ``(3, image_count)``.

    Returns
    -------
    numpy.ndarray
        Array of areas for the given moments in an array of shape ``(image_count,)``.
    """
    m20 = variances[0, :]
    m02 = variances[1, :]
    m11 = variances[2, :]

    return m20 * m02 - m11 * m11


def image_ellipticity_angle(variances):
    r"""
    Given the output of :meth:`image_variances()`,
    return the rotation angle of the scaled basis for each moment triplet.
    This is the angle between the :math:`x` axis and the
    major axis (large eigenvalue axis).

    Parameters
    ----------
    variances : numpy.ndarray
        The output of :meth:`image_variances()`. Shape ``(3, image_count)``.

    Returns
    -------
    numpy.ndarray
        Array of angles for the given moments.
        For highly circular spots, this angle is not meaningful, and dominated by
        experimental noise.
        For perfectly circular spots, zero is returned.
        Shape ``(image_count,)``.
    """
    m20 = variances[0, :]
    m02 = variances[1, :]
    m11 = variances[2, :]

    # Some quick math (see image_ellipticity).
    half_trace = (m20 + m02) / 2
    determinant = m20 * m02 - m11 * m11

    eig_plus = half_trace + np.sqrt(np.square(half_trace) - determinant)

    # We know that M * v = eig_plus * v. This yields a system of equations:
    #   m20 * x + m11 * y = eig_plus * x
    #   m11 * x + m02 * y = eig_plus * y
    # We're trying to solve for angle, which is just atan(x/y). We can solve for x/y:
    #   m11 * x = (eig_plus - m02) * y        ==>         x/y = (eig_plus - m02) / m11
    return np.arctan2(eig_plus - m02, m11, where=m11 != 0, out=np.zeros_like(m11))


def image_fit(images, grid=None, function=gaussian2d, guess=None, plot=False):
    """
    Fit each image in a stack of images to a 2D ``function``.

    Parameters
    ----------
    images : numpy.ndarray (``image_count``, ``height``, ``width``)
        An image or array of images to fit. A single image is interpreted correctly as
        ``(1, h, w)`` even if ``(h, w)`` is passed.
    grid : (array_like, array_like) OR None
        Components of the meshgrid describing coordinates over the images.
        If ``None``, makes a grid with unit pitch centered on the images.
    function : lambda ((float, float), ... ) -> float
        Some fitfunction which accepts ``(x,y)`` coordinates as first argument.
        Defaults to :meth:`~slmsuite.misc.fitfunctions.gaussian2d()`.
    guess : None OR True OR numpy.ndarray (``image_count``, ``parameter_count``)
        - If ``guess`` is ``None`` or ``True``, will construct a guess based on the ``function`` passed.
          Functions for which guesses are implemented include:

          - :meth:`~slmsuite.misc.fitfunctions.gaussian2d()`

        - If ``guess`` is ``None`` and ``function`` does not have a guess
          implemented, no guess will be provided to the optimizer and the user will be warned.
        - If ``guess`` is ``True`` and ``function`` does not have a guess
          implemented, an error will be raised.
        - If ``guess`` is a ``numpy.ndarray``, a column of the array will be provided
          to the optimizer as a guess for the fit parameters for each image.
    plot : bool
        Whether to create a plot for each fit.
    show : bool
        Whether or not to call :meth:`matplotlib.pyplot.show` after generating
        the plot.

    Returns
    -------
    numpy.ndarray (``image_count``, ``result_count``)
        A matrix with the fit results. The first row
        contains the rsquared quality of each fit.
        The values in the remaining rows correspond to the parameters
        for the supplied fit function, then the errors for each of the parameters.
        Failed fits have an rsquared of ``numpy.nan`` and parameters
        are set to the provided or constructed guess or ``numpy.nan``
        if no guess was provided or constructed; errors are set to ``numpy.nan``.

    Raises
    ------
    NotImplementedError
        If the provided ``function`` does not have a guess implemented.
    """
    # Setup.
    if images.ndim == 2:
        images = images.reshape((1, *images.shape))
    (image_count, w_y, w_x) = images.shape
    img_shape = (w_y, w_x)

    if grid is None:
        grid = _generate_grid(w_x, w_y, centered=True)
    grid_ravel = (np.ravel(grid[0]), np.ravel(grid[1]))

    # Number of fit parameters the function accepts (minus 1 for xy).
    param_count =  function.__code__.co_argcount - 1

    # Number of parameters to return: fitted parameters, errors, and plus 1 for rsquared.
    result_count = 2 * param_count + 1
    result = np.full((image_count, result_count), np.nan)

    # Construct guesses.
    if guess is None or guess is True:
        if function is gaussian2d:
            images_normalized = image_normalize(images, remove_field=True)
            centers = image_positions(images_normalized, grid=grid, normalize=False)
            variances = image_variances(images_normalized, centers=centers, grid=grid, normalize=False)

            maxs = np.amax(images, axis=(1, 2))
            mins = np.amin(images, axis=(1, 2))
            guess = np.vstack((
                centers,
                maxs - mins,
                mins,
                np.sqrt(variances[:2, :]),
                variances[2, :]
            )).T
        else:
            message = f"Default guess for function {str(function)} not implemented."
            if guess is True:
                raise NotImplementedError(message)
            else:
                warnings.warn(message)

    # Fit and plot each image.
    for img_idx in range(image_count):
        img = images[img_idx, :, :].ravel()
        grid_ravel_ = grid_ravel

        # Deal with nans.
        undefined = np.isnan(img)
        if np.any(undefined):
            defined = np.logical_not(undefined)
            img = img[defined]
            grid_ravel_ = (grid_ravel[0][defined], grid_ravel[1][defined])

        # Get guess.
        p0 = None if guess is None else guess[img_idx]

        # Attempt fit.
        fit_succeeded = True
        popt = None
        perr = None

        try:
            popt, pcov = curve_fit(function, grid_ravel_, img, ftol=1e-5, p0=p0,)
            perr = np.sqrt(np.diag(pcov))
        except RuntimeError:    # The fit failed if scipy says so.
            fit_succeeded = False
        else:                   # The fit failed if any of the parameters aren't finite.
            if np.any(np.logical_not(np.isfinite(popt))):
                fit_succeeded = False

        if fit_succeeded:   # Calculate r2.
            ss_res = np.sum(np.square(img - function(grid_ravel_, *popt)))
            ss_tot = np.sum(np.square(img - np.mean(img)))
            r2 = 1 - (ss_res / ss_tot)
        else:               # r2 is nan and the fit parameters are the guess or nan.
            popt = p0 if p0 is not None else np.full(param_count, np.nan)
            r2 = np.nan
            perr = np.nan

        # Populate results.
        result[img_idx, 0] = r2
        result[img_idx, 1:(param_count+1)] = popt
        result[img_idx, (param_count+1):] = perr

        # Plot.
        if plot:
            # Data.
            data = np.reshape(img, img_shape)
            if p0 is not None:
                guess_ = np.reshape(function(grid_ravel, *p0), img_shape)
            else:
                guess_ = np.zeros(img_shape)
            result_ = np.reshape(function(grid_ravel, *popt), img_shape)
            vmin = np.min((
                np.min(data),
                np.min(guess_) if p0 is not None else np.inf,
                np.min(result_)
            ))
            vmax = np.max((
                np.max(data),
                np.max(guess_) if p0 is not None else -np.inf,
                np.max(result_)
            ))

            # Plot.
            fig, axs = plt.subplots(1, 3, figsize=(3 * 6.4, 4.8))
            fig.suptitle("Image {}".format(img_idx))
            ax0, ax1, ax2 = axs
            ax0.imshow(data, vmin=vmin, vmax=vmax)
            ax0.set_title("Data")
            ax1.imshow(guess_, vmin=vmin, vmax=vmax)
            ax1.set_title("Guess")
            ax2.imshow(result_, vmin=vmin, vmax=vmax)
            ax2.set_title("Result")

            plt.show()

    return result


# Helpers for phase images.

def image_zernike_fit(
    phase_images,
    grid,
    order=10,
    iterations=2,
    leastsquares=True,
    unwrap=True,
    **kwargs
):
    """
    Fits sets of Zernike polynomials to a stack of ``phase_images``, up to a desired ``order``.
    This is done in two steps:

    -   First, an iterative approach is used to subtract Zernike orders from each image.
        If the Zernike aperture is not cropped or occluded, the orthogonality of the Zernike
        basis makes this a good and exact approach apart from sampling error.
        However, if the polynomials lose orthogonality, then this process produces a
        good guess at best.
    -   Thus, the second step is to refine the guess with a least squares optimization.
        This can be time consuming.

    Note
    ~~~~
    The piston term (Zernike ANSI index 0) is omitted from the fit return.

    Note
    ~~~~
    In the future, we might also fit to the derivatives.

    Parameters
    ----------
    phase_images : numpy.ndarray (``image_count``, ``height``, ``width``)
        An image or array of phase_images to fit. A single image is interpreted correctly as
        ``(1, h, w)`` even if ``(h, w)`` is passed.
    grid : (array_like, array_like) OR None
        Components of the meshgrid describing coordinates over the phase_images.
        If ``None``, makes a grid with unit pitch centered on the phase_images.
    order : int
        Maximal radial Zernike order for the fitting basis.
    iterations : int
        Number of times to iterate the subtractive approach.
    leastsquares : bool
        Whether to do the least squares optimization step.
    unwrap : bool
        Whether to unwrap the phase images before fitting.
    **kwargs
        Passed to :meth:`~slmsuite.holography.toolbox.phase.zernike_sum()`.
    """
    # Setup.
    if phase_images.ndim == 2:
        phase_images = phase_images.reshape((1, *phase_images.shape))
    image_count = phase_images.shape[0]

    # Unwrap
    if unwrap:
        # Adding temporary phase unwrapping solution for testing
        try:
            from skimage.restoration import unwrap_phase
        except ImportError:
            raise ImportError("Phase unwrapping requires scikit-image.")

        phase_images = [unwrap_phase(im) for im in phase_images]

    # Generate Zernike terms and norms.
    order = int(order + 1)
    indices_ansi = np.arange((order * (order + 1)) // 2)
    D = len(indices_ansi)
    phases = zernike_sum(
        grid,
        indices_ansi[np.newaxis, :],
        np.diag(np.ones((D,))),
        use_mask=True,
        **kwargs
    )
    norm = np.reciprocal(np.nansum(np.square(phases), (1,2), keepdims=False))

    # Preallocate the result.
    vectors_zernike = np.zeros((D, image_count))
    images_remainders = np.copy(phase_images)     # Copy the data

    # First, make a guess of the result based on iteratively subtracting Zernike terms.
    for _ in range(int(iterations)):
        for i in range(D):
            # Compute the weights of the given Zernike term in the images.
            overlap = np.nansum(images_remainders * phases[[i]] * norm[i], axis=(1,2))

            # Record this value in the result.
            vectors_zernike[i, :] += overlap

            # Subtract the power from the images.
            images_remainders -= overlap * phases[[i]]

    # Second, if desired, hone the guess via leastsquares.
    # This is especially important for a basis that is no longer orthonormal
    # due to incomplete data or a cropped aperture.
    if leastsquares:
        # Make grid.
        grid = _process_grid(grid)
        grid_ravel = (np.ravel(grid[0]), np.ravel(grid[1]))

        for j in range(image_count):
            # Lambda to build the function from test parameters.
            def zsum(grid, *p):
                p = np.reshape(p, vectors_zernike.shape)

                return zernike_sum(
                    grid,
                    indices_ansi[np.newaxis, :],
                    p,
                    use_mask=True,
                    **kwargs
                )

            # Try the fit.
            try:
                popt, _ = curve_fit(zsum, grid_ravel, phase_images[j].ravel(), ftol=1e-5, p0=vectors_zernike[:, j])
                vectors_zernike = popt.reshape(vectors_zernike.shape)
            except RuntimeError:    # The fit failed if scipy says so.
                pass

    # Return the fit with the piston term omitted.
    return vectors_zernike[1:, :]


def _get_module(matrix):
    if np == cp:
        return np
    else:
        return cp.get_array_module(matrix)


def image_vortices(phase_image):
    """
    Find the coordinates of phase vortices inside a phase image by computing the
    winding number directly. The coordinates are returned as an image.

    Parameters
    ----------
    phase_image : array_like
        Image to detect winding number upon.

    Returns
    -------
    winding_number
        Image with the integer winding number at each pixel.
    """
    xp = _get_module(phase_image)

    # Discrete derivatives, with appropriate wrapping.
    dd = [
        xp.mod(xp.diff(phase_image, axis=a, prepend=xp.nan) - xp.pi, 2*xp.pi) for a in range(2)
    ]

    # Sum to compute the winding.
    winding_number = -(
        dd[0] - dd[1] - xp.roll(dd[0], shift=1, axis=1) + xp.roll(dd[1], shift=1, axis=0)
    ) / (2 * xp.pi)

    # Get rid of the nans on the edges.
    winding_number[xp.isnan(winding_number)] = 0

    return xp.rint(winding_number)


def image_vortices_coordinates(phase_image, mask=None):
    """
    Find the coordinates of phase vortices inside a phase image by computing the
    winding number directly.

    Parameters
    ----------
    phase_image : array_like
        Image to detect winding number upon.
    mask : array_like OR None
        Boolean mask to determine coordinates at.

    Returns
    -------
    coordinates, weights
        The coordinates and winding number of each coordinate.
    """
    xp = _get_module(phase_image)

    winding_number = image_vortices(phase_image)

    if mask is not None:
        winding_number[xp.logical_not(mask)] = 0

    coordinates = xp.where(winding_number)
    weights = winding_number[coordinates[0], coordinates[1]]

    return coordinates, weights


def image_remove_vortices(phase_image, mask=None, return_vortices_negative=False):
    """
    Find and then remove all the phase vortices in a phase image.

    Parameters
    ----------
    phase_image : array_like
        Image to remove vortices upon.
    mask : array_like OR None
        Removes vortices whose coordinates are inside this mask.
        The full phase image is changed, even outside the mask.
    return_vortices_negative : bool
        If ``False``, the original image is modified in-place with vortices removed
        inside the mask and returned.
        If ``True``, what would be added to the original image is returned instead.

    Returns
    -------
    phase_image
        The image or vortices, depending upon ``return_vortices``
    """
    xp = _get_module(phase_image)

    if mask is not None:
        mask_eroded = binary_erosion(mask, np.ones((5,5)))
    else:
        mask_eroded = None

    coordinates, weights = image_vortices_coordinates(phase_image, mask=mask_eroded)
    grid = _generate_grid(phase_image.shape[1], phase_image.shape[0], integer=False)

    if return_vortices_negative:
        canvas = np.zeros_like(phase_image)
    else:
        canvas = phase_image

    for x, y, w in zip(coordinates[1], coordinates[0], weights):
        canvas -= w * xp.arctan2(grid[0] - x, grid[1] - y)

    return canvas


def image_blaze_remove(**kwargs):
    """
    Backwards compatible alias for :meth:`image_remove_blaze()`.
    """
    warnings.warn("image_blaze_remove is deprecated; use image_remove_blaze instead.", DeprecationWarning)
    return image_remove_blaze(**kwargs)


def image_remove_blaze(phase_image, mask=None, plot=False):
    """
    Remove a global blaze from a phase image.

    Parameters
    ----------
    phase_image : np.ndarray
        The phase image to have the blaze removed.
    mask : np.ndarray OR None
        If provided, the mask will weight the significance of the blaze.
        This can be the amplitude image, such that the blaze removal is
        more effective in brighter regions.
    plot : bool
        Whether to enable debug plots.

    Returns
    -------
    np.ndarray
        A phase image with the blaze removal applied.
    """
    phase = np.mod(phase_image, 2*np.pi)

    # Get the gradients across the image.
    dx = np.mod(np.gradient(phase, axis=1) + np.pi/2, np.pi) - np.pi/2
    dy = np.mod(np.gradient(phase, axis=0) + np.pi/2, np.pi) - np.pi/2

    # Find the mean gradient, potentially weighted.
    if mask is None:
        dx_mean = np.nanmean(dx)
        dy_mean = np.nanmean(dy)
    else:
        dx_mean = np.nansum(dx * mask) / np.nansum(mask)
        dy_mean = np.nansum(dy * mask) / np.nansum(mask)

    # Subtract the gradient and re-mod.
    x = np.arange(phase.shape[1])
    y = np.arange(phase.shape[0])
    X, Y = np.meshgrid(x, y)
    result = np.mod(phase - dx_mean * X - dy_mean * Y, 2 * np.pi)

    if plot:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 4, 1)
        plt.imshow(phase)
        plt.title('phase')

        plt.subplot(1, 4, 2)
        plt.imshow(dx)
        plt.title('dx')

        plt.subplot(1, 4, 3)
        plt.imshow(dy)
        plt.title('dy')

        plt.subplot(1, 4, 4)
        plt.imshow(result)
        plt.title('removed')
        plt.show()

    return result


def image_reduce_wraps(phase_image, mask=None, steps=10, plot=False):
    """
    Reduce the number of phase wraps in the image by adding a global offset.

    Parameters
    ----------
    phase_image : np.ndarray
        The phase image to be offset.
    mask : np.ndarray OR None
        If provided, the mask will weight the significance of the phase wraps.
        This can be the amplitude image, such that fewer wraps appear where there is
        light.
    steps : int
        The number of steps to try for the phase offset.
    plot : bool
        Whether to enable debug plots.

    Returns
    -------
    np.ndarray
        A phase image with the offset applied.
    """
    # Initialize such that the first iteration is always the best.
    fom_min = np.inf
    result = None

    # Loop through all offsets.
    for phi in range(steps):
        shift = phi * 2 * np.pi / steps
        phase_shifted = np.mod(phase_image + shift, 2 * np.pi)

        # Find where the phase wraps.
        wrapping = (
            np.abs(np.gradient(phase_shifted, axis=1)) +
            np.abs(np.gradient(phase_shifted, axis=0))
        ) > np.pi

        if mask is not None:
            wrapping = wrapping * mask

        # Sum the total cut area, potentially weighted by the mask.
        fom = np.sum(wrapping)

        if fom < fom_min:
            fom_min = fom
            result = phase_shifted

            # If the mask is smaller than the 0 -> 2pi range, shift to either end.
            min = np.nanmin(result)
            mean = np.nanmean(result)
            max = np.nanmax(result)

            if mean - min < max - mean:
                result -= min
            else:
                result -= max - 2 * np.pi

            result = np.mod(result, 2 * np.pi)

    return result


# Array fitting functions.

def fit_affine(x, y, guess_affine=None, plot=False):
    r"""
    For two sets of ordered points with equal length, find the best-fit affine
    transformation that transforms from the first basis to the second.
    Best fit is defined as minimization on the least squares euclidean norm.

    .. math:: \vec{y} = M \cdot \vec{x} + \vec{b}

    Parameters
    ----------
    x, y : array_like
        Array of vectors of shape ``(2, N)`` in the style of :meth:`format_2vectors()`.
    guess_affine : dict OR None
        The user may provide a guess to immediately proceed with least squares fitting.
        This guess must be in the form of a dictionary with fields ``"M"`` and ``"b"``.
        If ``None``, a guess is computed based on centroiding and moment matching.
    plot : bool
        Whether to produce a debug plot.

    Returns
    -------
    dict
        A dictionary with fields ``"M"`` and ``"b"``.
    """
    x = format_2vectors(x)
    y = format_2vectors(y)
    assert x.shape == y.shape

    # If the user does not provide a guess, compute one based on centroiding and moment matching.
    if guess_affine is None:
        # Calculate the centroids and the centered coordinates.
        xc = np.nanmean(x, axis=1)[:, np.newaxis]
        yc = np.nanmean(y, axis=1)[:, np.newaxis]

        if np.any(np.isnan(xc)) or np.any(np.isnan(yc)):
            raise ValueError("Vectors cannot contain a row of all-nan values")

        x_ = x - xc
        y_ = y - yc

        # Points very close to the centroid have disproportionate influence on the guess.
        # Ignore the points which are closer than a median-dependent threshold.
        threshold = np.median(np.sqrt(np.sum(np.square(x_), axis=0))) / 2

        # Generate a guess transformation.
        nan_list = np.full_like(y_[0,:], np.nan)

        # This could probably be vectorized more. Also not sure if all corner cases work.
        M_guess = np.array([
            [
                np.nanmean(np.divide(y_[0,:], x_[0,:], where=x_[0,:] > threshold, out=nan_list.copy())),
                np.nanmean(np.divide(y_[0,:], x_[1,:], where=x_[1,:] > threshold, out=nan_list.copy()))
            ],
            [
                np.nanmean(np.divide(y_[1,:], x_[0,:], where=x_[0,:] > threshold, out=nan_list.copy())),
                np.nanmean(np.divide(y_[1,:], x_[1,:], where=x_[1,:] > threshold, out=nan_list.copy()))
            ]
        ])

        # Fix nan instances. This means the matrix is no longer unique, so we choose the
        # case where the nans are mapped to zero.
        M_guess[np.isnan(M_guess)] = 0

        # Back compute the offset.
        b_guess = yc - np.matmul(M_guess, xc)
    else:
        if isinstance(guess_affine, dict) and "M" in guess_affine and "b" in guess_affine:
            M_guess = guess_affine["M"]
            b_guess = guess_affine["b"]
        else:
            raise ValueError("guess_affine must be a dictionary with 'M' and 'b' fields.")

    # Least squares fit.
    def err(p):
        M = np.array([[p[0], p[1]], [p[2], p[3]]])
        b = format_2vectors([p[4], p[5]])

        y_ = np.matmul(M, x) + b

        return np.nansum(np.square(y_ - y))

    guess = (M_guess[0,0], M_guess[0,1], M_guess[1,0], M_guess[1,1], b_guess[0,0], b_guess[1,0])

    try:        # Try with default scipy minimization. (Future: better opt than minimize?).
        m = minimize(err, x0=guess)
        p = [float(pp) for pp in m.x]

        M = np.array([[p[0], p[1]], [p[2], p[3]]])
        b = format_2vectors([p[4], p[5]])
    except:     # Fail elegantly (warn the user?).
        M = M_guess
        b = b_guess

    # Debug plot if desired.
    if plot and x.shape[0] == 2:
        plt.scatter(y[0,:], y[1,:], s=20, fc="b", ec="b")

        result_guess = np.matmul(M_guess, x) + b_guess
        plt.scatter(result_guess[0,:], result_guess[1,:], s=40, fc="none", ec="r")

        result = np.matmul(M, x) + b
        plt.scatter(result[0,:], result[1,:], s=60, fc="none", ec="g")

        plt.gca().set_aspect("equal")
        plt.show()

    # Return as a dictionary
    return {"M":M, "b":b}


def blob_detect(
    img,
    filter=None,
    plot=False,
    **kwargs
):
    """
    Detect blobs in an image.

    Wraps :class:`cv2.SimpleBlobDetector` (see
    `these <https://docs.opencv.org/3.4/d8/da7/structcv_1_1SimpleBlobDetector_1_1Params.html>`_
    `links <https://learnopencv.com/blob-detection-using-opencv-python-c/>`_)
    Default parameters are optimized for bright spot detection on dark background,
    but can be changed with ``**kwargs``.

    Parameters
    ----------
    img : numpy.ndarray
        The image to perform blob detection on.
    filter : {"dist_to_center", "max_amp"} OR None
        One of ``dist_to_center`` or ``max_amp``.
    plot : bool
        Whether to show a debug plot.
    title : str
        Plot title.
    fig : matplotlib.figure.Figure
        Figure for plotting.
    axs : list of matplotlib.axes.Axis or matplotlib.axes.Axis
        Axes for plotting.
    show : bool
        Whether or not to show the plot.
    **kwargs
       Extra arguments for :class:`cv2.SimpleBlobDetector`.

    Returns
    -------
    blobs : ndarray
        List of blobs found by  ``detector``.
    detector : :class:`cv2.SimpleBlobDetector`
        A blob detector with customized parameters.
    """
    # Need copy for some reason:
    # https://github.com/opencv/opencv/issues/18120
    img_8bit = _make_8bit(np.copy(img))
    params = cv2.SimpleBlobDetector_Params()

    # Configure default parameters
    params.blobColor = 255
    params.minThreshold = 10
    params.maxThreshold = 255
    params.thresholdStep = 10
    # params.minArea = 0
    params.filterByArea = False  # Can be changed to compute rel to diffraction limit
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    # Load in custom configuration
    for key, val in kwargs.items():
        setattr(params, key, val)

    # Create the detector and detect blobs
    detector = cv2.SimpleBlobDetector_create(params)
    blobs = detector.detect(img_8bit)

    if len(blobs) == 0:
        return [], detector
        #raise Exception("No blobs found! Try blurring image.")

    # Sort blobs according to `filter`.
    if filter == "dist_to_center":
        dist_to_center = [
            np.linalg.norm(np.array(blob.pt) - np.array(img.shape[::-1]) / 2)
            for blob in blobs
        ]
        blobs = [blobs[np.argmin(dist_to_center)]]
    elif filter == "max_amp":
        bin_size = int(np.mean([blob.size for blob in blobs]))
        for i, blob in enumerate(blobs):
            # Try fails when blob is on edge of camera.
            try:
                blobs[i].response = float(
                    img_8bit[
                        np.ix_(
                            int(blob.pt[1]) + np.arange(-bin_size, bin_size),
                            int(blob.pt[0]) + np.arange(-bin_size, bin_size),
                        )
                    ].sum()
                )
            except Exception:
                blobs[i].response = float(0)
        blobs = [blobs[np.argmax([blob.response for blob in blobs])]]

    if plot:
        # Get blob statistics.
        blob_count = len(blobs)
        blob_centers = np.zeros((2, blob_count))
        blob_diameters = np.zeros(blob_count)
        for (blob_idx, blob) in enumerate(blobs):
            blob_centers[:, blob_idx] = blob.pt
            blob_diameters[blob_idx] = blob.size
        blob_xs = blob_centers[0, :]
        blob_ys = blob_centers[1, :]
        blob_xmin = np.min(blob_xs)
        blob_xmax = np.max(blob_xs)
        blob_ymin = np.min(blob_ys)
        blob_ymax = np.max(blob_ys)
        zoom_padx = 2 * (blob_xmax - blob_xmin) / blob_count
        zoom_pady = 2 * (blob_ymax - blob_ymin) / blob_count

        # Full image.
        vmin = np.min(img_8bit)
        vmax = np.max(img_8bit)
        im = plt.imshow(img_8bit, vmin=vmin, vmax=vmax)
        ax = plt.gca()
        plt.colorbar()

        # Blob patches
        for blob_idx in range(blob_count):
                patch = matplotlib.patches.Circle(
                    (float(blob_centers[0, blob_idx]), float(blob_centers[1, blob_idx])),
                    radius=float(blob_diameters[blob_idx] / 2),
                    color="red",
                    linewidth=1,
                    fill=None
                )
                ax.add_patch(patch)

    return blobs, detector


def blob_array_detect(
    img,
    size,
    orientation=None,
    orientation_check=True,
    dft_threshold=100,
    dft_padding=0,
    k=8,
    tol=0.1,
    plot=False,
):
    r"""
    Detect an array of spots and return the orientation as an affine transformation.
    Primarily used for calibration.

    For a rectangular array of spots imaged in ``img``,
    find the variables :math:`\vec{M}` and :math:`\vec{b}` for the  affine transformation

    .. math:: \vec{y} = M \cdot \vec{x} + \vec{b}

    which converts spot indices :math:`\vec{x}` into camera pixel indices :math:`\vec{y}`.

    Parameters
    ----------
    img : numpy.ndarray
        The image in question.
    size : (int, int) OR int
        The size of the rectangular array in number of spots ``(Nx, Ny)``.
        If a single ``int`` size is given, then assume ``(N, N)``.
    orientation : dict or None
        Guess array orientation (same format as the returned) from previous known results.
        If ``None`` (the default), orientation is estimated from looking for peaks in the
        Fourier transform of the image.
    orientation_check : bool
        If enabled, looks for two missing spots at one corner as a parity check on rotation.
        Used by :meth:`~slmsuite.hardware.cameraslms.FourierSLM.fourier_calibrate()`.
        See :meth:`~slmsuite.hardware.cameraslms.FourierSLM.make_rectangular_array()`.
    dft_threshold : float in [0, 255]
        Minimum value of peak in blob detect of the DFT of ``img`` when ``orientation`` is ``None``.
        Passed as keyword argument to :meth:`blob_detect` with keyword ``minThreshold``.
    dft_padding : int
        Increases the dimensions of the padded ``img`` before the DFT is taken when ``orientation``
        is ``None``. Dimensions are increased by a factor of ``2 ** dft_padding``.
        Increasing this value increases the :math:`k`-space resolution of the DFT,
        and can improve orientation detection.
    k : int
        Number of nearest neighbors to use for each point when lattice matching.
        Defaults to 8.
    tol : float
        Difference in normalized displacement between reciprocal lattice points to
        be considered members of the same group when lattice fitting. Defaults to 10%.
    plot : bool
        Whether or not to plot debug plots. Default is ``False``.

    Returns
    --------
    dict
        Orientation dictionary with the following keys, corresponding to
        the affine transformation:

        - ``"M"`` : ``numpy.ndarray`` (2, 2).
        - ``"b"`` : ``numpy.ndarray`` (2, 1).
    """
    if len(np.shape(img)) != 2:
        raise RuntimeError(f"Cannot interpret image with shape {np.shape(img)}")

    img_8bit = _make_8bit(img)

    if np.amax(img_8bit) == 0:
        raise RuntimeError(
            "Cannot fit an image of all zeros. "
            "Check your camera to make sure it is snapping correctly."
        )

    # If an orientation was provided, use this as a guess.
    if orientation is not None:
        M = orientation["M"]

    # Otherwise, find a guess orientation.
    else:
        # 1) FFT to find array pitch and orientation.
        # Take the largest dimension rounded down to nearest power of 2.
        # FUTURE: clean this up to behave like other parts of the package.
        fft_size = int(2 ** (np.floor(np.log2(np.max(np.shape(img)))) + dft_padding))
        dft = np.abs(np.fft.fftshift(np.fft.fft2(img, s=[fft_size, fft_size])))

        # 2) Detect and plot FFT peaks
        # 2.1) Prepare some helper variables, mainly for filtering out the 0th order.
        fft_blur_size = int(np.clip(fft_size/200, 1, 5))*2 + 1
        downscaling = 1
        dft_amp = None
        zo_size = 8*fft_blur_size
        if fft_size <= zo_size*4:
            raise ValueError(f"Image of shape {img.shape} is too small to use with blob_array_detect.")
        zo_x, zo_y = np.meshgrid(
            np.linspace(-zo_size/2, zo_size/2, zo_size),
            np.linspace(-zo_size/2, zo_size/2, zo_size)
        )
        zo_filter = gaussian2d([zo_x, zo_y], 0, 0, -1, 1, fft_blur_size/2, fft_blur_size/2)
        points = []
        blobs = None
        i = 0

        # 2.2) Look for peaks with progressively greater downscaled blurring. This helps
        # to mitigate noise on the DFT peaks and enhance the most prominent peaks.
        while fft_size / downscaling > zo_size*4:
            dft_amp = cv2.GaussianBlur(dft, (fft_blur_size, fft_blur_size), fft_blur_size/4)

            # Filter 0 order (dominates in the presence of a slowly varying background)
            zo_i = int(fft_size/2/downscaling-zo_size/2)
            zo_j = zo_i+zo_size
            dft_amp[zo_i:zo_j, zo_i:zo_j] *= zo_filter

            thresholdStep = 10
            blobs, _ = blob_detect(
                dft_amp,
                minThreshold=dft_threshold,
                thresholdStep=thresholdStep,
            )
            points += [np.array(blob.pt) * downscaling for blob in blobs]

            # Exit if we've already got enough points.
            if len(points) > 4 * (i+1):
                break

            # Downscale so we can try to find peaks again with greater blurring.
            if fft_size / downscaling > zo_size*4:
                if not fft_size / (2*downscaling) > zo_size*4:
                    break
                dft = dft[0::2, 0::2] + dft[0::2, 1::2] + dft[1::2, 0::2] + dft[1::2, 1::2]
                downscaling *= 2
                i += 1

        if len(points) < 4:

            # Plot which diffraction orders we used
            if plot:
                plt.imshow(img)
                plt.title("Image")
                plt.show()

                plt.imshow(dft)
                plt.title("DFT")
                plt.show()

                plt.imshow(dft_amp)
                plt.title("Processed DFT")
                plt.show()

            raise RuntimeError(
                "Array fitting looks for prominent periodicity, "
                "but failed to find such in the given image. Try the following:\n"
                "- Verify that the camera is returning an image of the array. "
                "For instance, the SLM `settle_time_s` could be too fast for the "
                "camera to record the settled response, or "
                "the camera interface could be returning stale frames. "
                "Pass `plot>=1` to see debug plots. You can also view `cam.last_image`.\n"
                "- Increasing exposure time to enhance the prominence of spots,\n"
                "- Increasing the pitch of the array in the image, "
                "which can isolate spots from neighboring crosstalk,\n"
                "- Create an issue at https://github.com/holodyne/slmsuite/issues, "
                "attaching the image data that is resulting in failure. "
                "You can use `cam.save()` to record the last image alongside metadata."
            )

        # 3) Fit the primitive lattice vectors
        # 3.1) Make a list of displacements to each peak's k nearest neighbors
        def get_kNN(points, k):
            "Return list of k closest points for each x"
            dx = points[:,0][:,np.newaxis] - points[:,0][:,np.newaxis].T
            dy = points[:,1][:,np.newaxis] - points[:,1][:,np.newaxis].T
            d = np.sqrt(dx**2 + dy**2)
            inds = np.argsort(d, axis=0)
            kNN = points[inds[1:k+1,:]] - points
            kNN = kNN.reshape((points.shape[0]*k, 2))

            # Make inverted copies to avoid group separation based upon arb. branch cut.
            kNN = np.vstack((kNN, -kNN))

            return kNN

        # Get rid of the points closest to the center - these are noise on the 0th order.
        points = np.array(points)

        points_lengths = np.sqrt((points[:,0]-fft_size/2)**2 + (points[:,1]-fft_size/2)**2)
        points = points[points_lengths > .5*np.mean(points_lengths), :]
        points = np.concatenate((points, np.array([[fft_size/2, fft_size/2]])))

        # Now compute the differences of lattice vectors.
        k = min(k, len(points)-1)
        kNN = get_kNN(points, k)

        # 3.2) Cluster into lattice vectors.
        def cluster(points, k, tol=tol):
            "Cluster points from k nearest neighbors into groups and return the centers"

            # Find matrix of normalized displacements between points.
            dx = points[:,0][:,np.newaxis] - points[:,0][:,np.newaxis].T
            dy = points[:,1][:,np.newaxis] - points[:,1][:,np.newaxis].T
            dnorm = np.sqrt(dx**2 + dy**2) / np.linalg.norm(points, axis=1)

            # Normalized inverted displacements.
            dx = points[:,0][:,np.newaxis] + points[:,0][:,np.newaxis].T
            dy = points[:,1][:,np.newaxis] + points[:,1][:,np.newaxis].T
            inorm = np.sqrt(dx**2 + dy**2) / np.linalg.norm(points, axis=1)

            # Find groups of points separated by dnorm less than tol.
            group = 1
            tags = np.zeros(points.shape[0])
            for i in np.arange(points.shape[0]):
                # Assign if they are within tol and have not been assigned yet.
                new = ((dnorm[i,:] < tol) | (inorm[i,:] < tol)) & np.array(tags == 0) #  | (inorm[i,:] < tol)
                tags[new] = group
                if np.any(new): group += 1
                
            # Get the centerpoint of each group
            def mean_group(points):
                len0 = np.sum(np.square(points[0, :]))
                diff = np.sum(np.square(points - points[[0], :]), axis=1)
                points[diff > len0] = -points[diff > len0]

                final = np.mean(points, axis=0)

                if final[0] < 0:
                    final *= -1

                return final

            # Filter by the k most populated clusters.
            tag, count = np.unique(tags, return_counts=True)
            k = min(k, len(count))
            best_groups = np.argsort(-count)[:k]
            centers = np.array([
                mean_group(points[tags == tag[group]])
                for group in best_groups
            ])
            count = count[best_groups]

            # Order by closest point to center. Choose the closest as our base vector.
            distance_to_center = np.sqrt(np.square(centers[:, 0]) + np.square(centers[:, 1]))
            distance_to_center /= np.max(distance_to_center)    # normalize
            best_groups = np.argsort(distance_to_center)
            count = count[best_groups]
            centers = centers[best_groups, :]
            
            # Weight by orthogonality to the first vector.
            centers_length = np.sqrt(np.sum(np.square(centers), 1, keepdims=True))
            centers_norm = centers / centers_length
            cross_product = (
                centers_norm[:, 0] * centers_norm[0, 1] -
                centers_norm[:, 1] * centers_norm[0, 0]
            )
            cross_product[0] = 2

            # Prefer orthogal vectors most, then prefer distance to center.
            fom = 1e4 * (np.abs(cross_product)) - distance_to_center
            best_groups = np.argsort(-fom)
            count = count[best_groups]
            centers = centers[best_groups, :]

            return centers

        centers = cluster(kNN, k).T

        # 3.3) Primitive lattice vectors are the best two.
        lv = np.array([centers[:,0], centers[:,1]]).T

        if plot > 1:
            # Plot the points, kNN, and the chosen lattice vecs
            fig, ax = plt.subplots(constrained_layout=True)
            kNN_plt = ax.scatter(kNN[:,0], kNN[:,1], fc='none', ec='k', zorder=0)

            for center in centers.T:
                for s in [-1, 1]:
                    cir = matplotlib.patches.Circle(
                        (s*center[0], s*center[1]),
                        np.linalg.norm(center)*.1,
                        fill=False,
                        ec='r',
                        zorder=10,
                    )
                    circ = ax.add_patch(cir)

            for i in [0, 1]:
                lv_plt = ax.plot([-lv[0,i], lv[0,i]], [-lv[1,i], lv[1,i]], marker='.', c='r')
            lv_plt = ax.scatter(lv[0,:], lv[1,:], marker='.', c='r')

            ax.set_aspect('equal')
            ax.set_title('Reciprocal Lattice Vector Fitting')
            ax.legend([kNN_plt, circ, lv_plt], ['Peak Spacing', '$k$ Clusters', 'Lattice Vectors'])
            ax.grid()
            plt.show()

        # 3.4) Convert to image space (dx = 1/dk)
        M = fft_size*lv/(np.linalg.norm(lv, axis=0)**2)

        # Plot which diffraction orders we used
        if plot > 1:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')

            plt_img = _make_8bit(dft_amp.copy())

            # Determine the bounds of the zoom region, padded by zoom_pad
            zoom_pad = 50

            # x = np.array([blob.pt[0] for blob in blobs])
            x = points[:, 0] / downscaling
            xl = [
                np.clip(np.amin(x) - zoom_pad, 0, dft_amp.shape[1]),
                np.clip(np.amax(x) + zoom_pad, 0, dft_amp.shape[1]),
            ]

            # y = np.array([blob.pt[1] for blob in blobs])
            y = points[:, 1] / downscaling
            yl = [
                np.clip(np.amin(y) - zoom_pad, 0, dft_amp.shape[0]),
                np.clip(np.amax(y) + zoom_pad, 0, dft_amp.shape[0]),
            ]

            # Plot the unzoomed figure
            axs[0].imshow(plt_img)

            # Plot a red rectangle to show the extents of the zoom region
            rect = plt.Rectangle(
                (float(xl[0]), float(yl[0])),
                float(np.diff(xl).item()), float(np.diff(yl).item()),
                ec="r", fc="none"
            )
            axs[0].add_patch(rect)
            axs[0].set_title(f"DFT Result ({downscaling}x downscale) - Full")
            axs[0].set_xticks([])
            axs[0].set_yticks([])

            # Plot the zoomed figure
            axs[1].imshow(plt_img)
            axs[1].scatter(
                x,
                y,
                facecolors="none",
                edgecolors="r",
                marker="o",
                s=100,
                linewidths=0.5
            )
            c = fft_size/2/downscaling
            lv /= downscaling
            for i in [0, 1]:
                lv_plt = axs[1].plot([c-lv[0,i], c+lv[0,i]], [c-lv[1,i], c+lv[1,i]], marker='.', c='r')

            for spine in ["top", "bottom", "right", "left"]:
                axs[1].spines[spine].set_color("r")
                axs[1].spines[spine].set_linewidth(1.5)
            axs[1].set_title(f"DFT Result ({downscaling}x downscale) - Zoom")
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            axs[1].set_xlim(xl)
            axs[1].set_ylim(np.flip(yl))

            for ax in axs:
                ax.set_xlabel("Image Reciprocal $x$ [1/pix]")
                ax.set_ylabel("Image Reciprocal $y$ [1/pix]")
            fig.tight_layout(pad=4.0)

            plt.show()

    # 4) Make the array kernel for convolutional detection of the array center.
    # Make lists that we will use to make the kernel: the array...
    x_list = np.arange(-(size[0] - 1) / 2.0, (size[0] + 1) / 2.0)
    y_list = np.arange(-(size[1] - 1) / 2.0, (size[1] + 1) / 2.0)

    x_centergrid, y_centergrid = np.meshgrid(x_list, y_list)
    centers = np.vstack((x_centergrid.ravel(), y_centergrid.ravel()))

    # ...and the array padded by one (penalize the border to avoid off-by-one errors).
    pad = 1
    p = int(pad * 2)

    x_list_larger = np.arange(-(size[0] + p - 1) / 2.0, (size[0] + p + 1) / 2.0)
    y_list_larger = np.arange(-(size[1] + p - 1) / 2.0, (size[1] + p + 1) / 2.0)

    x_centergrid_larger, y_centergrid_larger = np.meshgrid(x_list_larger, y_list_larger)
    centers_larger = np.vstack(
        (x_centergrid_larger.ravel(), y_centergrid_larger.ravel())
    )

    # If we're not sure about how things are flipped, consider alternatives...
    if size[0] != size[1] and orientation is None:
        M_alternative = np.array([[M[0, 1], M[0, 0]], [M[1, 1], M[1, 0]]])
        M_options = [M, M_alternative]
    else:
        M_options = [M]
    results = []
    # Iterate through these alternatives.
    for M_trial in M_options:
        # Find the position of the centers for this trial matrix.
        rotated_centers = np.matmul(M_trial, centers)
        rotated_centers_larger = np.matmul(M_trial, centers_larger)

        # Make the kernel
        max_pitch = int(
            np.amax([np.linalg.norm(M_trial[:, 0]), np.linalg.norm(M_trial[:, 1])])
        )
        mask = np.zeros(
            (
                int(
                    np.amax(rotated_centers_larger[1, :])
                    - np.amin(rotated_centers_larger[1, :])
                    + max_pitch
                ),
                int(
                    np.amax(rotated_centers_larger[0, :])
                    - np.amin(rotated_centers_larger[0, :])
                    + max_pitch
                ),
            )
        )

        rotated_centers += np.flip(mask.shape)[:, np.newaxis] / 2
        rotated_centers_larger += np.flip(mask.shape)[:, np.newaxis] / 2

        # Pixels to use for the kernel.
        x_array = np.rint(rotated_centers[0, :]).astype(int)
        y_array = np.rint(rotated_centers[1, :]).astype(int)

        x_larger = np.rint(rotated_centers_larger[0, :]).astype(int)
        y_larger = np.rint(rotated_centers_larger[1, :]).astype(int)

        # Make a mask with negative power at the border, positive
        # at the array, with integrated intensity of 0.
        area = size[0] * size[1]
        perimeter = 2 * (size[0] + size[1]) + 4

        mask[y_larger, x_larger] = -area/perimeter
        mask[y_array, x_array] = 1

        mask = _make_8bit(mask)

        # 5) Do the autocorrelation
        try:
            res = cv2.matchTemplate(img_8bit, mask, cv2.TM_CCOEFF)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
        except:
            max_val = 0
            max_loc = [0, 0]

        b_fixed = (
            np.array(max_loc)[:, np.newaxis] + np.flip(mask.shape)[:, np.newaxis] / 2
        )

        # Parity check
        if orientation is None and orientation_check:
            try:
                cam_array_ind = np.ix_(
                    max_loc[1] + np.arange(mask.shape[0]),
                    max_loc[0] + np.arange(mask.shape[1]),
                )
                match = img_8bit[cam_array_ind]

                # TODO: replace with take
                wmask = 0.2
                w = np.max([1, int(wmask * max_pitch)])
                edge = np.arange(-w, w + 1)

                integration_x, integration_y = np.meshgrid(edge, edge)

                rotated_integration_x = np.rint(np.add(
                    integration_x.ravel()[:, np.newaxis].T,
                    rotated_centers[:][0][:, np.newaxis],
                )).astype(int)
                rotated_integration_y = np.rint(np.add(
                    integration_y.ravel()[:, np.newaxis].T,
                    rotated_centers[:][1][:, np.newaxis],
                )).astype(int)

                spotpowers = np.reshape(
                    np.sum(match[rotated_integration_y, rotated_integration_x], 1),
                    np.flip(size),
                )

                # Find the two dimmest pixels.
                spotbooleans = spotpowers <= np.sort(spotpowers.ravel())[1]

                assert np.sum(spotbooleans) == 2

                # Find whether the corners are dimmest.
                corners = spotbooleans[[-1, -1, 0, 0], [-1, 0, 0, -1]]

                assert np.sum(corners) == 1

                # We want a dim corner at -1, -1.
                rotation_parity = np.where(corners)[0][0]
                spotbooleans_rotated = np.rot90(spotbooleans, rotation_parity)

                theta = rotation_parity * np.pi / 2
                c = np.cos(theta)
                s = np.sin(theta)
                rotation = np.array([[c, -s], [s, c]])

                # Look for the second missing spot.
                flip_parity = (
                    int(spotbooleans_rotated[-1, -2]) -
                    int(spotbooleans_rotated[-2, -1])
                )

                assert abs(flip_parity) == 1

                if flip_parity == 1:
                    flip = np.array([[1, 0], [0, 1]])
                else:
                    flip = np.array([[0, 1], [1, 0]])

                M_fixed = np.matmul(M_trial, np.matmul(rotation, flip))
                parity_success = True
            except Exception as e:
                M_fixed = M_trial
                parity_success = False
        else:
            M_fixed = M_trial
            parity_success = True

        results.append((max_val, b_fixed, M_fixed, parity_success))

    if len(results) == 1:
        index = 0
    else:
        # Default to max_val if the parity check results in the same thing.
        if results[0][3] == results[1][3]:
            index = int(results[1][0] > results[0][0])
        else:
            # Index is the one that satisfied the parity check
            index = int(results[1][3])

    orientation = {"M": results[index][2], "b": results[index][1]}

    # Hone the center of our fit by averaging the positional deviations of spots.
    # We do this multiple times to allow outliers (1 std error above) to stabilize.
    # FUTURE: Use a more physics-based psf, optimize for speed, maybe remove_field.
    hone_count = 3
    region_fraction = 1
    for _ in range(hone_count):
        guess_positions = np.matmul(orientation["M"], centers) + orientation["b"]

        # Calculate a point spread function (psf) integration window size.
        psf = 2 * int(np.floor(np.amin(np.amax(np.abs(orientation["M"]), axis=0))) / 2) + 1
        psf = np.max([3, psf])

        # Grab windows (sized by psf) about the guess_positions.
        regions = take(
            img, guess_positions, psf, centered=True, integrate=False, clip=True
        )
        region_fraction = np.sum(regions) / np.sum(img)

        # Get the first order moment rint each of the guess windows.
        shift = image_positions(regions) - (guess_positions - np.rint(guess_positions))

        # Remove outliers.
        shift_error = np.sqrt(np.square(shift[0, :]) + np.square(shift[1, :]))
        thresh = np.mean(shift_error) + np.std(shift_error)
        shift[:, shift_error > thresh] = np.nan

        # Locally fit an affine based on the measured positions.
        true_positions = guess_positions + shift
        orientation = fit_affine(centers, true_positions, orientation)

    # Warn the user if the mask was >= (or close to) camera size.
    if np.any(mask.shape > 0.95 * np.array(img_8bit.shape)):
        warnings.warn(
            "The computed Fourier grid size exceeds or approaches the camera size; "
            "calibration results may be improperly centered as a result."
        )
    # Also warn if computed positions approach camera FOV boundary.
    elif np.any(np.nanmax(true_positions, axis=1) > 0.95 * np.array(img_8bit.shape)) or \
         np.any(np.nanmin(true_positions, axis=1) < 0.05 * np.array(img_8bit.shape)):
        warnings.warn(
            "The fitted spot array approaches or exceeds the camera FOV; "
            "calibration results may be improperly centered as a result."
        )
    # Warn if the array does not match the received pattern on the camera.
    if region_fraction < .5:
        warnings.warn(
            f"{(1-region_fraction)*100:.1f}% of the image's power outside the spot array. "
            "This might have caused the array fit to be poor."
        )

    if plot:
        array_center = orientation["b"]
        true_centers = np.matmul(orientation["M"], centers) + orientation["b"]

        showmatch = False

        fig, axs = plt.subplots(
            1, 2 + showmatch, figsize=(12, 12)
        )

        # Determine the bounds of the zoom region, padded by 50.
        x = true_centers[0, :]
        xl = [
            np.clip(np.amin(x) - max_pitch, 0, img.shape[1]),
            np.clip(np.amax(x) + max_pitch, 0, img.shape[1]),
        ]
        y = true_centers[1, :]
        yl = [
            np.clip(np.amin(y) - max_pitch, 0, img.shape[0]),
            np.clip(np.amax(y) + max_pitch, 0, img.shape[0]),
        ]

        # Plot the zoom window, with red axes.
        axs[1].imshow(img)
        axs[1].scatter(
            x[:-2], y[:-2], facecolors="none", edgecolors="r", marker="o", s=80, linewidths=0.5
        )

        # for i, ctr in enumerate(true_centers.T):
        #     if i < 2 or i > len(true_centers.T) - 3:
        #         axs[1].annotate(i, (ctr[0] + 4, ctr[1] - 4), c="r", size="x-small")

        axs[1].scatter(array_center[0], array_center[1], c="r", marker="x", s=10)
        axs[1].set_title("Result - Zoom")
        axs[1].set_xlim(xl)
        axs[1].set_ylim(np.flip(yl))

        for spine in ["top", "bottom", "right", "left"]:
            axs[1].spines[spine].set_color("r")
            axs[1].spines[spine].set_linewidth(1.5)

        # Plot the non-zoom axes.
        axs[0].imshow(img_8bit)
        axs[0].scatter(array_center[0], array_center[1], c="r", marker="x", s=10)

        # Plot a red rectangle to show the extents of the zoom region
        rect = plt.Rectangle(
            (float(xl[0]), float(yl[0])),
            float(np.diff(xl).item()), float(np.diff(yl).item()),
            ec="r", fc="none"
        )
        axs[0].add_patch(rect)
        axs[0].set_title("Result - Full")

        if showmatch:
            axs[2].imshow(match)
            axs[2].set_title("Result - Match")

        # Handle xy labels.
        for ax in axs[:2]:
            ax.set_xlabel("Image $x$ [pix]")
            ax.set_ylabel("Image $y$ [pix]")
        fig.tight_layout(pad=4.0)

        plt.show()

    return orientation


# Other image helper functions.

def _make_8bit(img):
    """
    Convert an image to ``numpy.uint8``, scaling to the limits.

    This function is useful to convert float or larger bitdepth images to
    8-bit, which :mod:`cv2` accepts and can speedily process.

    Parameters
    ----------
    img : numpy.ndarray
        The image in question.

    Returns
    -------
    ndarray
        img as an 8-bit image.
    """
    img = img.astype(float)

    img -= np.amin(img)
    max = np.amax(img)
    if max > 0: img = img / max * (2 ** 8 - 1)

    return img.astype(np.uint8)


def get_orientation_transformation(rot="0", fliplr=False, flipud=False):
    """
    Compile a transformation lambda from simple rotates and flips.

    Useful to turn an image to an orientation which is user-friendly.
    Used by :class:`~slmsuite.hardware.cameras.camera.Camera` and subclasses.

    Parameters
    ----------
    rot : str OR int
        Rotates returned image by the corresponding degrees in ``["90", "180", "270"]``
        or :meth:`numpy.rot90` code in ``[1, 2, 3]``. Defaults to no rotation otherwise.
    fliplr : bool
        Flips returned image left right.
    flipud : bool
        Flips returned image up down.

    Returns
    -------
    function (array_like) -> numpy.ndarray
        Compiled image transformation.
    """
    transforms = list()

    if fliplr == True:
        transforms.append(np.fliplr)
    if flipud == True:
        transforms.append(np.flipud)

    if rot == "90" or rot == 1:
        transforms.append(lambda img: np.rot90(img, 1))
    elif rot == "180" or rot == 2:
        transforms.append(lambda img: np.rot90(img, 2))
    elif rot == "270" or rot == 3:
        transforms.append(lambda img: np.rot90(img, 3))

    return reduce(lambda f, g: lambda x: f(g(x)), transforms, lambda x: x)
