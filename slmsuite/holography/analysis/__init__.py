r"""
Helper functions for processing images.
"""

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from slmsuite._plotting import _slmsuite_plt_show
from mpl_toolkits.axes_grid1 import make_axes_locatable
import enum
from scipy.optimize import curve_fit, minimize
from scipy.ndimage import binary_erosion, maximum_filter
from scipy.fft import next_fast_len

import warnings
try:
    import cupy as cp   # type: ignore
except ImportError:
    cp = np

from slmsuite.holography.toolbox import format_2vectors, _process_grid
from slmsuite.holography.toolbox.phase import (
    zernike_sum, laguerre_gaussian, ZernikeBasis, _zernike_get_basis,
)
from slmsuite.misc.math import REAL_TYPES
from slmsuite.holography.analysis.fitfunctions import gaussian2d
from slmsuite._logging import make_logger
logger = make_logger(__name__)

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
    return np.meshgrid(xs, ys)


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
    images : array_like OR (int, int)
        2D image of shape ``(h, w)`` or stack of 2D images of shape ``(image_count, h, w)``.
        If a shape ``(int, int)`` is given, then ``return_mask`` is forced to ``True``.
    vectors : array_like of floats
        2-vector (or 2-vector array).
        Location(s) of integration region anchor(s) in pixels, see ``centered``.
        See :meth:`~slmsuite.holography.toolbox.format_2vectors`.
    size : int or (int, int)
        Size of the rectangular integration region in ``(iw, ih)`` format in pixels.
        If a scalar is given, assume square ``(iw, iw)``.
    centered : bool
        Whether to center the integration region on the ``vectors``.
        If ``False``, ``vectors`` indicates the upper-left corner of the integration region.
        Defaults to ``True``.
    integrate : bool
        If ``True``, the spatial dimensions are integrated (summed), yielding a result of the
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
        from the regions of size ``(vector_count, h, w)``.
        If ``integrate`` is ``True``, instead returns an array of floats of size ``(vector_count,)``
        where each float corresponds to the :meth:`numpy.sum` of a cropped image.
        If ``xp`` is :mod:`cupy`, then a ``cupy.ndarray`` is returned.
        If a stack of ``image_count`` images is given, the output is of shape
        ``(image_count, vector_count, h, w)`` or ``(image_count, vector_count)`` depending on ``integrate``.
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

    # Parse images and shape.
    shape = xp.shape(images)

    if shape == (2,):
        shape = (int(images[0]), int(images[1]))
        return_mask = True

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
            canvas = np.full(shape[-2:], np.nan, dtype=float)
            canvas[integration_y, integration_x] = images[integration_y, integration_x]
        else:
            canvas = np.zeros(shape[-2:], dtype=bool)
            canvas[integration_y, integration_x] = True

        if plot:
            plt.imshow(canvas)
            _slmsuite_plt_show(name="take")

        return canvas
    else:
        images = xp.array(images, copy=(False if np.__version__[0] == '1' else None))

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
            except Exception:
                result[:, mask] = 0

        if len(shape) == 2:
            crop_shape = (vectors.shape[1], size[1], size[0])
        elif len(shape) == 3:
            crop_shape = (shape[0], vectors.shape[1], size[1], size[0])

        if plot:
            if len(shape) == 2:
                take_plot(xp.reshape(result, crop_shape), separate_axes=False)
            elif len(shape) == 3:
                take_plot(xp.reshape(result, crop_shape)[0], separate_axes=False)

        if integrate:   # Sum over the integration axis.
            if len(shape) == 2:
                final_shape = (vectors.shape[1],)
            elif len(shape) == 3:
                final_shape = (shape[0], vectors.shape[1],)
            return xp.sum(result.astype(float), axis=-1).reshape(final_shape)
        else:           # Reshape the integration axis.
            return xp.reshape(result, crop_shape)


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

    if cp is not np and isinstance(images, cp.ndarray):
        images = cp.asnumpy(images)

    if separate_axes:
        sx = sx / 2.0 - 0.5
        sy = sy / 2.0 - 0.5
        extent = (-sx, sx, -sy, sy)

        vmin = np.nanmin(images)
        vmax = np.nanmax(images)

        for x in range(img_count):
            ax = plt.subplot(M, N, x + 1)

            ax.imshow(
                images[x, :, :],
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                interpolation='none'
            )
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        _slmsuite_plt_show(name="take_plot")
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

        _slmsuite_plt_show(name="take_plot")


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
        logger.warning("Not enough space to fit all images. Try truncating the image count.")
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
        Defaults to ``1``.
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
    if not np.issubdtype(images.dtype, np.floating):
        images = np.array(images, copy=(False if np.__version__[0] == '1' else None), dtype=float)  # Hack to prevent integer underflow.

    # Parse out.
    if out is None:
        out = np.copy(images)
    elif not (out is images):
        np.copyto(out, images)

    # Make sure that we're testing 3D images. out_ is a view of out, so in-place
    # operations on out_ also update out (which is what we return).
    single_image = len(images.shape) == 2
    if single_image:
        images_ = np.reshape(images, (1, images.shape[0], images.shape[1]))
        out_ = np.reshape(out, (1, out.shape[0], out.shape[1]))
    else:
        images_ = images
        out_ = out
    img_count = images_.shape[0]

    # Generate the threshold.
    if deviations is None:  # Median case
        threshold = np.nanmedian(images_, axis=(1, 2))
    else:   # Mean + deviations * std case
        threshold = (
            np.nanmean(images_, axis=(1, 2))
            + deviations * np.nanstd(images_, axis=(1, 2))
        )
    threshold = np.reshape(threshold, (img_count, 1, 1))

    out_max = np.amax(out_, axis=(1, 2), keepdims=True)

    # Remove the field. This needs the float from before. Unsigned integer could underflow.
    out_ -= threshold.astype(out_.dtype)
    out_[out_ < 0] = 0
    out_[out_ > out_max - threshold] = 0

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
        normalization = np_sum(
            images, axis=(1, 2), keepdims=False
        ).reshape((img_count, 1, 1)).astype(float)
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
            if grid is None:
                scale_x = scale_y = 1
            elif np.isscalar(grid):
                scale_x = scale_y = grid
            else:
                scale_x, scale_y = grid[0], grid[1]

            if moment[0] != 0:
                x_grid = np.reshape(np.arange(w_x) - _center(w_x), (1, 1, w_x)) * scale_x - c_x
                if moment[0] != 1:
                    x_grid = np.power(x_grid, moment[0], out=x_grid)
            else:
                x_grid = 0

            if moment[1] != 0:
                y_grid = np.reshape(np.arange(w_y) - _center(w_y), (1, w_y, 1)) * scale_y - c_y
                if moment[1] != 1:
                    y_grid = np.power(y_grid, moment[1], out=y_grid)
            else:
                y_grid = 0
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
        The normalization factor :math:`M_{00}` in an array of shape ``(image_count,)``.
    """
    return image_moment(images, (0, 0), normalize=False, nansum=nansum)


def image_normalize(images, nansum=False, remove_field=False):
    """
    Normalizes a stack of images via the zeroth order moments
    such that each image sums to one.

    Parameters
    ----------
    images : numpy.ndarray
        A matrix in the style of the output of :meth:`take()`, with shape ``(image_count, h, w)``, where
        ``(h, w)`` is the height and width of the 2D images and ``image_count`` is the number of
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
        ``(h, w)`` is the height and width of the 2D images and ``image_count`` is the number of
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
    Recall that variance is defined as

    .. math:: (\Delta x)^2 = \left<(x - \left<x\right>)^2\right>.

    This equation is made central by subtraction of :math:`\left<x\right>`.
    The user can of course use :meth:`image_moment` directly to access the
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
        ``(h, w)`` is the height and width of the 2D images and ``image_count`` is the number of
        images. A single image is interpreted correctly as ``(1, h, w)`` even if
        ``(h, w)`` is passed.
    centers : numpy.ndarray OR None
        If the user has already computed :math:`\left<x\right>`, for example via
        :meth:`image_positions()`, then this can be passed through ``centers``. The default
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
        centers = image_positions(images, grid=grid, normalize=False, nansum=nansum)

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

    eig_half_difference = np.sqrt(np.maximum(np.square(half_trace) - determinant, 0))

    eig_plus = half_trace + eig_half_difference
    eig_minus = half_trace - eig_half_difference

    # eig_plus == 0 for an all-zero / uniform image; return nan there instead of dividing.
    ratio = np.divide(
        eig_minus, eig_plus,
        out=np.full_like(eig_plus, np.nan, dtype=float),
        where=eig_plus != 0,
    )
    return 1 - ratio


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

    # Major-axis orientation of the [[m20, m11], [m11, m02]] covariance.
    return 0.5 * np.arctan2(2 * m11, m20 - m02)


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
        Defaults to :meth:`~slmsuite.holography.analysis.fitfunctions.gaussian2d()`.
    guess : None OR True OR numpy.ndarray (``image_count``, ``parameter_count``)
        - If ``guess`` is ``None`` or ``True``, will construct a guess based on the ``function`` passed.
          Functions for which guesses are implemented include:

          - :meth:`~slmsuite.holography.analysis.fitfunctions.gaussian2d()`

        - If ``guess`` is ``None`` and ``function`` does not have a guess
          implemented, no guess will be provided to the optimizer and the user will be warned.
        - If ``guess`` is ``True`` and ``function`` does not have a guess
          implemented, an error will be raised.
        - If ``guess`` is a ``numpy.ndarray``, a row of the array will be provided
          to the optimizer as a guess for the fit parameters for each image.
    plot : bool
        Whether to create a plot for each fit.

    Returns
    -------
    numpy.ndarray (``image_count``, ``result_count``)
        A matrix with the fit results. The first column
        contains the rsquared quality of each fit.
        The values in the remaining columns correspond to the parameters
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
                logger.warning(message)

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
            r2 = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)
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
            data = images[img_idx, :, :]
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

            _slmsuite_plt_show(name="image_fit")

    return result


# Helpers for phase images.

def _wrapped_gradient(phase_images, xp):
    r"""
    Wrap-aware central differences of stacked ``phase_images``.

    Returns ``(dx, dy)``, the raw two-pixel central differences along the last
    and second-to-last axes, cropped to the interior (shapes ``(..., h, w-2)``
    and ``(..., h-2, w)``). Each is reduced modulo :math:`2\pi` and centred on
    zero, which folds out :math:`2\pi` phase wraps: where the true phase change
    over two pixels stays within :math:`(-\pi, \pi)` the result equals the true
    (unwrapped) central difference, so it is correct everywhere except across
    vortices.
    """
    dx = phase_images[..., :, 2:] - phase_images[..., :, :-2]
    dy = phase_images[..., 2:, :] - phase_images[..., :-2, :]
    dx = xp.mod(dx + np.pi, 2 * np.pi) - np.pi
    dy = xp.mod(dy + np.pi, 2 * np.pi) - np.pi
    return dx, dy


def image_zernike_fit(
    phase_images,
    grid,
    order=10,
    leastsquares=True,
    aperture=None,
    use_mask=True,
    gradient=True,
):
    r"""
    Fits sets of Zernike polynomials to a stack of ``phase_images``.

    When ``gradient=True``, the fit is performed against the *gradient* of the
    phase rather than the phase itself. The gradient of a :math:`2\pi`-wrapped
    phase equals the gradient of the true unwrapped phase everywhere except at
    vortices, so this recovers accurate coefficients from a phase-wrapped image
    without any expensive and fragile 2D unwrapping. In this case,
    fitting Zernike coefficients is a linear least-squares problem
    (at the cost of the omission of the piston term).
    This runs on the GPU when ``phase_images`` (or ``grid``) are :mod:`cupy` arrays.

    When ``gradient=False``, imperfect unwrapping breaks the assumption of linearity,
    so large phase wraps can lead to inaccurate fits.

    Note
    ~~~~
    The piston term (Zernike ANSI index 0) is omitted from the fit return.

    Parameters
    ----------
    phase_images : numpy.ndarray OR cupy.ndarray (``image_count``, ``height``, ``width``)
        An image or array of phase_images to fit. A single image is interpreted
        correctly as ``(1, h, w)`` even if ``(h, w)`` is passed.
    grid : (array_like, array_like) OR :class:`~slmsuite.holography.toolbox.phase.ZernikeBasis` OR None
        Components of the meshgrid describing coordinates over the phase_images.
        If ``None``, makes a grid with unit pitch centered on the phase_images.

        Repeated calls against the same grid transparently reuse a cached
        :class:`~slmsuite.holography.toolbox.phase.ZernikeBasis`, so the basis is
        built only once. An explicit basis may also be passed, in which case
        ``order``, ``aperture``, and ``use_mask`` are ignored (they are fixed when
        the basis is built).
    order : int OR list of int
        Maximal radial Zernike order for the fitting basis. If a list of int is
        provided, these are the ANSI indices of the Zernike polynomials to fit.
        Ignored if ``grid`` is a :class:`ZernikeBasis`.
    leastsquares : bool
        If ``True`` (default), solve the exact least-squares fit via the normal
        equations. If ``False``, return the cheaper per-mode basis projection
        :math:`\langle\phi, Z_i\rangle / \langle Z_i, Z_i\rangle`, which ignores
        cross-terms between modes and is exact only when the basis is orthonormal
        over the sampled aperture.
    aperture : :class:`~slmsuite.holography.toolbox.Aperture` OR spec OR None
        The aperture defining the lateral scaling of the Zernike polynomials. Resolved
        with :meth:`~slmsuite.holography.toolbox.Aperture.resolve`. Ignored if ``grid``
        is a :class:`ZernikeBasis`.
    use_mask : bool
        Whether to mask the region outside the Zernike pupil. Ignored if ``grid``
        is a :class:`ZernikeBasis`.
    gradient : bool
        If ``True``, fit the wrap-aware phase gradient against the gradient
        Zernike basis instead of fitting the phase directly. This recovers
        accurate coefficients from a phase-wrapped ``phase_images`` without
        unwrapping. See the note above.

    Returns
    -------
    numpy.ndarray OR cupy.ndarray
        The fit Zernike coefficients, of shape ``(D, image_count)``, piston omitted.
    """
    # Setup.
    if phase_images.ndim == 2:
        phase_images = phase_images.reshape((1, *phase_images.shape))
    image_count = phase_images.shape[0]

    # Build the Zernike basis, or reuse a cached one for this grid.
    if isinstance(grid, ZernikeBasis):
        basis = grid
    else:
        if np.isscalar(order):
            order = int(order + 1)
            indices_ansi = np.arange((order * (order + 1)) // 2)[1:]    # Omit piston.
        else:
            indices_ansi = np.array(order, dtype=int)
            if 0 in indices_ansi:
                indices_ansi = indices_ansi[indices_ansi != 0]
                warnings.warn("Piston term (Zernike ANSI index 0) is omitted from the fit.")
        basis = _zernike_get_basis(grid, indices_ansi, aperture=aperture, use_mask=use_mask)

    # Work on the basis's array module (GPU when the basis is GPU-resident).
    xp = _get_module(basis.basis_flat)
    phase_images = xp.asarray(phase_images)

    if gradient:
        dx, dy = _wrapped_gradient(phase_images, xp)
        # Integer indexing avoids GPU-CPU synchronization from boolean masking in cupy.
        dx_flat = dx.reshape(image_count, -1)
        dy_flat = dy.reshape(image_count, -1)
        stacked = xp.concatenate(
            [dx_flat[:, basis.grad_idx_x].T, dy_flat[:, basis.grad_idx_y].T], axis=0
        )                                                               # (2P, image_count)

        # Project the gradient onto the gradient basis.
        b = basis.grad_basis_flat @ stacked                             # (D, image_count)

        if leastsquares:
            # Exact least-squares fit: solve via the precomputed inverse of the Gram matrix.
            # Avoids xp.linalg.solve (cuSOLVER) kernel launch overhead on GPU.
            vectors_zernike = basis.grad_gram_inv @ b
        else:
            # Cheaper per-mode projection (the gradient basis is not orthonormal).
            vectors_zernike = b / basis.grad_norm[:, np.newaxis]

        return vectors_zernike

    # Mask the phase images to the pupil and flatten to match the basis.
    phase_flat = (phase_images * basis.mask).reshape(image_count, -1)   # (image_count, h*w)

    # Project the images onto the basis: b[d, n] = <phase_n, Z_d>.
    b = basis.basis_flat @ phase_flat.T                                 # (D, image_count)

    if leastsquares:
        # Least-squares fit: solve via the precomputed inverse of the Gram matrix.
        # Avoids xp.linalg.solve (cuSOLVER) kernel launch overhead on GPU.
        vectors_zernike = basis.gram_inv @ b
    else:
        # Cheaper per-mode projection (exact iff the basis is orthonormal).
        vectors_zernike = b / basis.norm[:, np.newaxis]

    return vectors_zernike


def _get_module(matrix):
    if np == cp:
        return np
    else:
        return cp.get_array_module(matrix)


def image_vortices(phase_image):
    """
    Compute the winding number of a phase image directly, at every pixel.

    Parameters
    ----------
    phase_image : array_like
        Image to detect winding number upon.

    Returns
    -------
    winding_number
        Image with the integer winding number at each pixel,
        negated relative to the conventional right-handed sense.
    """
    xp = _get_module(phase_image)

    # Discrete derivatives, with appropriate wrapping.
    dd = [
        (xp.diff(phase_image, axis=a, prepend=xp.nan) + xp.pi) % (2*xp.pi) - xp.pi for a in range(2)
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
        The image or vortices, depending upon ``return_vortices_negative``
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


def image_blaze_remove(*args, **kwargs):
    """
    Backwards compatible alias for :meth:`image_remove_blaze()`.
    """
    warnings.warn("image_blaze_remove is deprecated; use image_remove_blaze instead.", DeprecationWarning)
    return image_remove_blaze(*args, **kwargs)


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
        _slmsuite_plt_show(name="image_remove_blaze")

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
                np.nanmean(np.divide(y_[0,:], x_[0,:], where=np.abs(x_[0,:]) > threshold, out=nan_list.copy())),
                np.nanmean(np.divide(y_[0,:], x_[1,:], where=np.abs(x_[1,:]) > threshold, out=nan_list.copy()))
            ],
            [
                np.nanmean(np.divide(y_[1,:], x_[0,:], where=np.abs(x_[0,:]) > threshold, out=nan_list.copy())),
                np.nanmean(np.divide(y_[1,:], x_[1,:], where=np.abs(x_[1,:]) > threshold, out=nan_list.copy()))
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
    except Exception:     # Fail elegantly (warn the user?).
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
        _slmsuite_plt_show(name="fit_affine")

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
        Keeps only the blob nearest the center of ``img``, or only the blob
        collecting the most power. ``None`` keeps every blob.
    plot : bool
        Whether to show a debug plot.
    **kwargs
       Extra arguments for :class:`cv2.SimpleBlobDetector`.

    Returns
    -------
    blobs : ndarray
        List of blobs found by ``detector``.
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

    # Keep the single best blob according to `filter`.
    if filter is not None:
        centers = format_2vectors(np.vstack([blob.pt for blob in blobs]).T)

    if filter == "dist_to_center":
        distance = np.linalg.norm(
            centers - format_2vectors(np.flip(np.shape(img)) / 2), axis=0
        )
        blobs = [blobs[int(np.argmin(distance))]]
    elif filter == "max_amp":
        power = np.nan_to_num(take(
            img_8bit, centers, 2 * int(np.mean([blob.size for blob in blobs])),
            centered=True, integrate=True, clip=True,
        ))
        blobs = [blobs[int(np.argmax(power))]]

    if plot:
        plt.imshow(img_8bit)
        plt.colorbar()
        for blob in blobs:
            plt.gca().add_patch(matplotlib.patches.Circle(
                blob.pt, radius=blob.size / 2, color="red", linewidth=1, fill=None
            ))

    return blobs, detector


def image_lattice_detect(img, method="autocorrelation", plot=False, **kwargs):
    r"""
    Detect the primitive lattice vectors of a periodic array of spots in an image.

    This recovers the lattice (pitch, rotation, shear) but **not** the absolute
    position of the array. The two returned vectors are unordered and defined only
    up to sign: resolving that dihedral ambiguity requires a feature which breaks
    the lattice's symmetry, such as the fiducial spots that
    :meth:`blob_array_detect()` looks for.

    Parameters
    ----------
    method : {"autocorrelation", "fourier"}
        Whether to look for the lattice in the image's autocorrelation, whose 0th
        order is only as wide as a *spot*, or for the reciprocal lattice in its
        Fourier transform, whose 0th order is as wide as the whole *array* and so
        hides the lattice of a coarse one. Both survive the array being cropped.
    **kwargs
        Passed to the chosen method.

    Returns
    -------
    numpy.ndarray
        Lattice matrix of shape ``(2, 2)`` whose columns are the two primitive
        lattice vectors, in image coordinates (pixels).

    Raises
    ------
    RuntimeError
        If no prominent periodicity is found in the image.
    """
    if method == "autocorrelation":
        return _lattice_autocorrelation(img, plot=plot, **kwargs)
    elif method == "fourier":
        return _lattice_fourier(img, plot=plot, **kwargs)
    else:
        raise ValueError(f"Unrecognized lattice detection method '{method}'.")


def _lattice_autocorrelation(img, spot_size=2, threshold=0.2, plot=False):
    """
    Lattice of a spot array from the peaks of the image's autocorrelation.
    See :meth:`image_lattice_detect()`.

    Parameters
    ----------
    spot_size : float
        Size of a spot in pixels, used to mask the autocorrelation's central peak.
        Overestimating this hides fine lattices; underestimating it lets the central
        peak masquerade as one.
    threshold : float
        Peaks below this fraction of the brightest off-center peak are ignored.
    """
    if np.ndim(img) != 2:
        raise RuntimeError(f"Cannot interpret image with shape {np.shape(img)}")

    # Autocorrelation by the Wiener-Khinchin theorem, zero displacement centered.
    # Transformed at an FFT-friendly size, which is faster than an awkward camera shape
    # and, padding rather than cropping, leaves the autocorrelation linear not circular.
    img = np.asarray(img, dtype=float)
    size = [next_fast_len(n, real=True) for n in np.shape(img)]
    correlation = np.fft.fftshift(
        np.fft.ifft2(np.abs(np.fft.fft2(img - img.mean(), s=size)) ** 2).real
    )
    center = np.array([correlation.shape[1] // 2, correlation.shape[0] // 2])

    # Mask the central (zero-displacement) peak, then find local maxima. This is
    # done over the whole plane so that a peak straddling two pixels suppresses its
    # own shoulder, and only afterwards restricted to one half plane, the
    # autocorrelation being symmetric.
    (yy, xx) = np.indices(correlation.shape)
    masked = np.where(
        np.hypot(xx - center[0], yy - center[1]) < max(2.0, spot_size),
        -np.inf,
        correlation,
    )

    peaks = (
        (masked == maximum_filter(masked, size=3))
        & np.isfinite(masked)
        & ((yy > center[1]) | ((yy == center[1]) & (xx > center[0])))
    )
    (py, px) = np.nonzero(peaks)
    if len(px) < 2:
        raise RuntimeError(
            "Lattice fitting looks for prominent periodicity, but the image's "
            "autocorrelation has no clear peaks. The array may be too dim, too "
            "cropped, or too finely spaced to resolve."
        )

    values = masked[py, px]
    keep = values >= threshold * values.max()

    # A real lattice peak stands above the valley between it and the origin, whereas
    # the central peak's shoulder is still descending. This also drops peaks at even
    # multiples of the lattice, whose midpoint is itself a peak, which costs some
    # leverage in the fit below but no accuracy.
    keep &= values > correlation[
        np.rint(0.5 * (py + center[1])).astype(int),
        np.rint(0.5 * (px + center[0])).astype(int),
    ]

    vectors = np.vstack((px[keep] - center[0], py[keep] - center[1])).astype(float)

    if vectors.shape[1] < 2:
        raise RuntimeError(
            "Lattice fitting looks for prominent periodicity, but the image's "
            "autocorrelation has no clear peaks. The array may be too dim, too "
            "cropped, or too finely spaced to resolve."
        )

    # The two shortest independent peaks are the primitive vectors.
    order = np.argsort(np.linalg.norm(vectors, axis=0))
    first = vectors[:, order[0]]
    lattice = None
    for index in order[1:]:
        second = vectors[:, index]
        cross = np.abs(first[0] * second[1] - first[1] * second[0])
        if cross > 0.3 * np.linalg.norm(first) * np.linalg.norm(second):
            lattice = np.vstack((first, second)).T
            break

    if lattice is None:
        raise RuntimeError(
            "Lattice fitting found periodicity in only one direction. The array may "
            "be cropped to a single row or column of spots."
        )

    # The peaks above are quantized to whole pixels, which is a large error for a
    # fine lattice. Refine by fitting every peak to its (integer) lattice index;
    # distant peaks have the most leverage. Repeated, since a quantized guess can
    # mis-index the most distant (and most informative) peaks.
    for _ in range(3):
        indices = np.rint(np.linalg.solve(lattice, vectors))
        residual = vectors - lattice @ indices
        inliers = np.linalg.norm(residual, axis=0) < 0.25 * np.min(
            np.linalg.norm(lattice, axis=0)
        )
        if np.sum(inliers) < 3:
            break
        (design, target) = (indices[:, inliers], vectors[:, inliers])
        lattice = np.linalg.solve(
            design @ design.T, design @ target.T
        ).T

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor="white")

        plt_img = _make_8bit(np.where(np.isfinite(masked), masked, 0))

        # The fit uses one half plane; the plot shows both, the correlation being even.
        mirrored = np.hstack((vectors, -vectors))
        (x, y) = (mirrored[0] + center[0], mirrored[1] + center[1])

        # Determine the bounds of the zoom region, padded by zoom_pad
        zoom_pad = 50
        xl = [
            np.clip(np.amin(x) - zoom_pad, 0, plt_img.shape[1]),
            np.clip(np.amax(x) + zoom_pad, 0, plt_img.shape[1]),
        ]
        yl = [
            np.clip(np.amin(y) - zoom_pad, 0, plt_img.shape[0]),
            np.clip(np.amax(y) + zoom_pad, 0, plt_img.shape[0]),
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
        axs[0].set_title("Autocorrelation - Full")
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        # Plot the zoomed figure
        axs[1].imshow(plt_img)
        axs[1].scatter(
            x, y, facecolors="none", edgecolors="r", marker="o", s=100, linewidths=0.5
        )
        for i in [0, 1]:
            axs[1].plot(
                [center[0] - lattice[0, i], center[0] + lattice[0, i]],
                [center[1] - lattice[1, i], center[1] + lattice[1, i]],
                marker=".", c="r",
            )

        for spine in ["top", "bottom", "right", "left"]:
            axs[1].spines[spine].set_color("r")
            axs[1].spines[spine].set_linewidth(1.5)
        axs[1].set_title("Autocorrelation - Zoom")
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_xlim(xl)
        axs[1].set_ylim(np.flip(yl))

        for ax in axs:
            ax.set_xlabel("Image Displacement $x$ [pix]")
            ax.set_ylabel("Image Displacement $y$ [pix]")
        fig.tight_layout(pad=4.0)

        _slmsuite_plt_show(name="image_lattice_detect")

    return lattice


def _array_indices(size, pad=0):
    """
    Centered index offsets of a ``size`` spot array grown by ``pad`` on every side,
    ordered as :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
    orders its spots; at ``pad=0`` the last two are the withheld fiducials.
    """
    (x, y) = np.meshgrid(*[
        np.arange(size[axis] + 2 * pad) - (size[axis] + 2 * pad - 1) / 2.0
        for axis in range(2)
    ])
    return np.vstack((x.ravel(), y.ravel()))


def _score_array_orientation(img, M, b, size, psf, threshold=0.2):
    """
    Scores each orientation of a rectangular spot array against an image.

    An array is unchanged by a rotation or flip of its axes, and by an offset of a
    whole pitch, so its lattice alone leaves eight possible affines. The two spots
    that :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`
    withholds break both ties: only at the true orientation do they land on darkness.

    Parameters
    ----------
    img : numpy.ndarray
        Image of the array, with its background already removed.
    M, b : array_like
        Affine mapping array indices to image coordinates.
    size : (int, int)
        Size of the array in number of spots.
    psf : int
        Width of the window that each spot's power is collected over.
    threshold : float
        Fraction of a typical spot's power above which a spot counts as lit.

    Returns
    -------
    (OrientationTransform.D_4, float, float, float) OR None
        The best orientation, the fraction of its spots that are lit, the power at
        its withheld pair relative to a spot, and the median distance from the spots
        to their predictions. ``None`` if too little of the array is in view.
    """
    centers = _array_indices(size)
    b = format_2vectors(b)
    (h, w) = np.shape(img)
    best = None

    for code in OrientationTransform.D_4:
        predicted = (
            np.matmul(M, np.matmul(OrientationTransform.from_code(code).M(), centers)) + b
        )

        # Only spots that are actually in view can be checked.
        visible = (
            (predicted[0] > 1) & (predicted[0] < w - 2)
            & (predicted[1] > 1) & (predicted[1] < h - 2)
        )
        (fiducial, spots) = (visible.copy(), visible.copy())
        fiducial[:-2] = False
        spots[-2:] = False
        if np.sum(spots) < 4:
            continue

        windows = np.nan_to_num(take(
            img, predicted, psf, centered=True, integrate=False, clip=True
        ))
        power = np.sum(windows, axis=(1, 2))
        reference = np.median(power[spots])
        if reference <= 0:
            continue

        lit = np.mean(power[spots] > threshold * reference)
        # A pair pushed off the camera cannot be checked, so it scores as though it
        # were lit rather than passing for free.
        dark = np.median(power[fiducial]) / reference if np.any(fiducial) else 1.0
        # take() floors its positions; undo that to measure from the prediction.
        shift = image_positions(windows) - (predicted - np.floor(predicted))
        residual = float(np.median(np.linalg.norm(shift[:, spots], axis=0)))

        if best is None or lit - dark > best[1] - best[2]:
            best = (code, lit, dark, residual)

    return best


def _lattice_fourier(img, dft_threshold=100, dft_padding=0, k=8, tol=0.1, plot=False):
    """
    Lattice of a spot array from the peaks of the image's Fourier transform, which
    are the reciprocal lattice. See :meth:`image_lattice_detect()`.
    """
    # 1) FFT to find array pitch and orientation.
    # 1.1) Subtract the background field before the FFT.
    # A noisy camera background can dominate the FFT's 0th order.
    img_centered = image_remove_field([img], deviations=None)[0]

    # 1.2) Take the largest dimension rounded down to nearest power of 2.
    # FUTURE: clean this up to behave like other parts of the package.
    fft_size = int(2 ** (np.floor(np.log2(np.max(np.shape(img)))) + dft_padding))

    # 1.3) Actually FFT.
    dft = np.abs(np.fft.fftshift(np.fft.fft2(img_centered, s=[fft_size, fft_size])))

    # 2) Detect and plot FFT peaks
    # 2.1) Prepare some helper variables, mainly for filtering out the 0th order.
    fft_blur_size = int(np.clip(fft_size/200, 1, 5))*2 + 1
    downscaling = 1
    dft_amp = None
    zo_size = 8*fft_blur_size
    if fft_size <= zo_size*4:
        raise ValueError(f"Image of shape {img.shape} is too small to detect a lattice by Fourier transform.")
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
            _slmsuite_plt_show(name="blob_array_detect_img")

            plt.imshow(dft)
            plt.title("DFT")
            _slmsuite_plt_show(name="blob_array_detect_dft")

            plt.imshow(dft_amp)
            plt.title("Processed DFT")
            _slmsuite_plt_show(name="blob_array_detect_processed_dft")

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
        _slmsuite_plt_show(name="blob_array_detect_lattice")

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

        _slmsuite_plt_show(name="blob_array_detect_reciprocal_lattice")

    return M


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
    method="fourier",
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
        See :meth:`~slmsuite.holography.algorithms.SpotHologram.make_rectangular_array()`.
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
    method : str
        How to find the array's lattice, when ``orientation`` is not given.
        Passed to :meth:`image_lattice_detect()`.

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

    # Parse size.
    if np.isscalar(size):
        size = (int(size), int(size))

    img_8bit = _make_8bit(img)

    if np.amax(img_8bit) == 0:
        raise RuntimeError(
            "Cannot fit an image of all zeros. "
            "Check your camera to make sure it is snapping correctly."
        )

    # If an orientation was provided, use this as a guess. Otherwise, measure
    # the lattice that the array's spots sit on.
    if orientation is not None:
        M = orientation["M"]
    else:
        M = image_lattice_detect(
            img,
            method=method,
            plot=plot,
            **(
                dict(dft_threshold=dft_threshold, dft_padding=dft_padding, k=k, tol=tol)
                if method == "fourier" else {}
            ),
        )

    # (4) [after orientation detection] 
    # Make the array kernel for convolutional detection of the array center:
    # the array, and the array padded by one to penalize the border against
    # off-by-one errors.
    centers = _array_indices(size)
    centers_larger = _array_indices(size, pad=1)

    # If we're not sure about how things are flipped, consider alternatives...
    # A guessed `orientation` fixes the lattice but not which vector is which:
    # the alternative is tried for a guess too, and scored by template match.
    if size[0] != size[1]:
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

        # 5) Locate the array. The template match locates the kernel, which
        # coincides with the array only when the array is symmetric about it, so a
        # guessed position is scored against it rather than either being trusted.
        candidates = []

        if orientation is not None and "b" in orientation:
            candidates.append(format_2vectors(orientation["b"]).astype(float))

        try:
            res = cv2.matchTemplate(img_8bit, mask, cv2.TM_CCOEFF)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            candidates.append(
                np.array(max_loc, dtype=float)[:, np.newaxis]
                + np.flip(mask.shape)[:, np.newaxis] / 2
            )
        except Exception:
            pass

        if len(candidates) == 0:
            candidates.append(np.flip(mask.shape)[:, np.newaxis] / 2.0)

        def spot_power(b):
            """Typical power collected where this position would place the spots."""
            return float(np.nanmedian(take(
                img_8bit, np.matmul(M_trial, centers) + b, 3,
                centered=True, integrate=True, clip=True,
            )))

        scores = [spot_power(b) for b in candidates]
        b_fixed = candidates[int(np.argmax(scores))]
        max_val = np.max(scores)

        # Parity check. A guess constrains the lattice vectors but not the
        # rotation and flip between them, which only the fiducial spots reveal.
        if orientation_check:
            try:
                # Parity is decided by the fiducials, so a cropped array cannot
                # decide it; the position is the one chosen above, not the template
                # match, which may have lost to the guess.
                corner = np.rint(
                    np.squeeze(b_fixed) - np.flip(mask.shape) / 2
                ).astype(int)
                if np.any(corner < 0) or np.any(
                    np.flip(corner) + mask.shape > img_8bit.shape
                ):
                    raise IndexError("Array is not wholly within the image.")

                best = _score_array_orientation(
                    image_remove_field(img, deviations=None),
                    M_trial,
                    b_fixed,
                    size,
                    2 * np.max([1, int(0.2 * max_pitch)]) + 1,
                )
                assert best is not None

                M_fixed = np.matmul(
                    M_trial, OrientationTransform.from_code(best[0]).M()
                )
                parity_success = True
            except Exception:
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
        region_fraction = np.nansum(regions) / np.nansum(img)

        # Get the first order moment rint each of the guess windows.
        shift = image_positions(regions) - (guess_positions - np.floor(guess_positions))

        # Remove outliers.
        shift_error = np.sqrt(np.square(shift[0, :]) + np.square(shift[1, :]))
        thresh = np.mean(shift_error) + np.std(shift_error)
        shift[:, shift_error > thresh] = np.nan

        # Locally fit an affine based on the measured positions.
        true_positions = guess_positions + shift
        orientation = fit_affine(centers, true_positions, orientation)

    # Warn the user if the mask was >= (or close to) camera size.
    if np.any(mask.shape > 0.95 * np.array(img_8bit.shape)):
        logger.warning(
            "The computed Fourier grid size exceeds or approaches the camera size; "
            "calibration results may be improperly centered as a result."
        )
    # Also warn if computed positions approach camera FOV boundary.
    elif np.any(np.nanmax(true_positions, axis=1) > 0.95 * np.flip(np.array(img_8bit.shape))) or \
         np.any(np.nanmin(true_positions, axis=1) < 0.05 * np.flip(np.array(img_8bit.shape))):
        logger.warning(
            "The fitted spot array approaches or exceeds the camera FOV; "
            "calibration results may be improperly centered as a result."
        )
    # Warn if the array does not match the received pattern on the camera.
    if region_fraction < .5:
        logger.warning(
            "%.1f%% of the image's power outside the spot array. "
            "This might have caused the array fit to be poor.", (1 - region_fraction) * 100
        )

    if plot:
        array_center = orientation["b"]
        true_centers = np.matmul(orientation["M"], centers) + orientation["b"]

        fig, axs = plt.subplots(1, 2, figsize=(12, 12))

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

        # Handle xy labels.
        for ax in axs[:2]:
            ax.set_xlabel("Image $x$ [pix]")
            ax.set_ylabel("Image $y$ [pix]")
        fig.tight_layout(pad=4.0)

        _slmsuite_plt_show(name="blob_array_detect")

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


# Transformations

class Affine(object):
    """
    Implements 2D affine transformation.
    """
    def __init__(self, M, b, a=None):
        self.M = np.array(M).astype(float)
        self.b = format_2vectors(b).astype(float)
        if a is not None:
            self.b -= np.matmul(self.M, format_2vectors(a).astype(float))

    def __mul__(self, x):
        return self.__matmul__(x)

    def __matmul__(self, x):
        if isinstance(x, Affine):
            M_new = np.matmul(self.M, x.M)
            b_new = np.matmul(self.M, x.b) + self.b
            return Affine(M_new, b_new)
        elif isinstance(x, np.ndarray):
            return np.matmul(self.M, x) + self.b
        else:
            raise TypeError(
                "Unsupported operand type(s) for @: "
                f"'Affine' and '{type(x).__name__}'"
            )

    @property
    def inv(self):
        """The inverse affine transformation."""
        M_inv = np.linalg.inv(self.M)
        b_inv = -np.matmul(M_inv, self.b)
        return Affine(M_inv, b_inv)

    def det(self):
        return np.linalg.det(self.M)

    def to_dict(self):
        return {"M": self.M, "b": self.b, "a": 0 * self.b}


class OrientationTransform:
    """
    Camera-specific orientation transform.

    The group elements are combinations of 90 deg rotations and flips.
    All image transforms are zero-copy numpy view operations.

    Parameters
    ----------
    rot : str or int
        Rotation in CCW degrees: ``"0"``, ``"90"``, ``"180"``, ``"270"`` or equivalently
        the :func:`numpy.rot90` step counts ``0``, ``1``, ``2``, ``3``.
    fliplr : bool
        Apply a left-right flip *before* rotation. ``fliplr=True, flipud=True`` is
        equivalent to a 180° rotation and will be collapsed accordingly.
    flipud : bool
        Apply an up-down flip *before* rotation.
    """
    # Dihedral group elements.
    class D_4(enum.IntEnum):
        """Enums for image orientations."""
        IDENTITY    = 0  # no transform
        ROT90       = 1  # 90° CCW
        ROT180      = 2  # 180°
        ROT270      = 3  # 270° CCW  (= 90° CW)
        FLIP        = 4  # left-right flip            (fliplr)
        FLIP_ROT90  = 5  # left-right flip then 90° CCW  (= transpose)
        FLIP_ROT180 = 6  # left-right flip then 180°     (= up-down flip)
        FLIP_ROT270 = 7  # left-right flip then 270° CCW (= anti-transpose)

    # Inverse elements
    _INVERSE_D_4 = [0, 3, 2, 1, 4, 5, 6, 7]

    # 2x2 matrix
    _MATRICES = (
        ((1, 0), (0, 1)),  ((0, 1), (-1, 0)), ((-1, 0), (0, -1)), ((0, -1), (1, 0)),
        ((-1, 0), (0, 1)), ((0, 1), (1, 0)),  ((1, 0), (0, -1)),  ((0, -1), (-1, 0)),
    )

    _COMPOSITION_TABLE = [
        [0, 1, 2, 3, 4, 5, 6, 7],   # IDENTITY
        [1, 2, 3, 0, 5, 6, 7, 4],   # ROT90
        [2, 3, 0, 1, 6, 7, 4, 5],   # ROT180
        [3, 0, 1, 2, 7, 4, 5, 6],   # ROT270
        [4, 7, 6, 5, 0, 3, 2, 1],   # FLIP
        [5, 4, 7, 6, 1, 0, 3, 2],   # FLIP_ROT90
        [6, 5, 4, 7, 2, 1, 0, 3],   # FLIP_ROT180
        [7, 6, 5, 4, 3, 2, 1, 0],   # FLIP_ROT270
    ]

    def __init__(self, rot="0", fliplr=False, flipud=False):
        if rot in ("90", 1):
            rot_steps = 1
        elif rot in ("180", 2):
            rot_steps = 2
        elif rot in ("270", 3):
            rot_steps = 3
        else:
            rot_steps = 0

        # Canonicalize: flipud ≡ fliplr + rot180; both flips ≡ rot180
        if fliplr and flipud:
            has_flip, extra_rot = False, 2
        elif flipud:
            has_flip, extra_rot = True, 2
        else:
            has_flip, extra_rot = bool(fliplr), 0

        self.code = self.D_4((rot_steps + extra_rot) % 4 + (4 if has_flip else 0))

    @classmethod
    def from_code(cls, code):
        """Construct directly from a :class:`D_4` value or integer (0-7)."""
        obj = cls.__new__(cls)
        obj.code = cls.D_4(int(code))
        return obj

    def __mul__(self, other):
        return self.__matmul__(other)

    def __matmul__(self, other):
        """Return the composition of this transform with another (``self * other``)."""
        if not isinstance(other, OrientationTransform):
            raise TypeError(
                "Unsupported operand type(s) for *: "
                f"'OrientationTransform' and '{type(other).__name__}'"
            )
        new_code = self._COMPOSITION_TABLE[self.code.value][other.code.value]
        return OrientationTransform.from_code(new_code)

    # Properties

    @property
    def swaps_xy(self):
        """bool : True if x and y axes are exchanged (image shape changes for non-square inputs)."""
        return self.code in (
            self.D_4.ROT90, self.D_4.ROT270,
            self.D_4.FLIP_ROT90, self.D_4.FLIP_ROT270,
        )

    @property
    def inverse(self):
        """OrientationTransform : The inverse transformation."""
        return self.from_code(self._INVERSE_D_4[self.code.value])

    def M(self):
        """
        numpy.ndarray : 2x2 matrix transformation
        """
        return np.array(self._MATRICES[self.code.value], dtype=float)

    def b(self, shape):
        """
        numpy.ndarray : 2-vector translation.
        """
        H, W = shape
        M = self.M()
        return np.array([
            max(0., -M[0, 0]) * (W - 1) + max(0., -M[0, 1]) * (H - 1),
            max(0., -M[1, 0]) * (W - 1) + max(0., -M[1, 1]) * (H - 1),
        ])

    def affine(self, shape):
        """
        Return an :class:`Affine` for the pixel-space transform of a ``(H, W)`` image.

        Parameters
        ----------
        shape : (int, int)
            Image shape ``(H, W)`` in the original (untransformed) space.

        Returns
        -------
        Affine
        """
        return Affine(self.M(), self.b(shape))

    # Actual transform

    def __call__(self, img):
        """
        Apply the orientation transform to an image or stack of images.

        Parameters
        ----------
        img : array_like
            Image or image stack. The last two axes are treated as ``(H, W)``.

        Returns
        -------
        numpy.ndarray
            Transformed array. Shape is ``(..., H, W)`` for transforms that preserve
            axis order or ``(..., W, H)`` for transforms that swap the spatial axes
            (90° / 270° rotations and their flip variants).
        """
        img = np.asarray(img)
        c = self.code
        C = self.D_4
        # Operates on the last two axes, so 2D and ND (batched) inputs share one path.
        if   c == C.IDENTITY:
            return img
        elif c == C.ROT90:
            return np.rot90(img, 1, axes=(-2, -1))
        elif c == C.ROT180:
            return img[..., ::-1, ::-1]
        elif c == C.ROT270:
            return np.rot90(img, 3, axes=(-2, -1))
        elif c == C.FLIP:
            return img[..., ::-1]
        elif c == C.FLIP_ROT90:
            return img.swapaxes(-2, -1)
        elif c == C.FLIP_ROT180:
            return img[..., ::-1, :]
        else:
            return img[..., ::-1, ::-1].swapaxes(-2, -1)  # FLIP_ROT270

    def transform_shape(self, shape):
        """
        Return the image shape after applying this transformation.

        Parameters
        ----------
        shape : (int, int)
            Image shape ``(H, W)`` before transformation.

        Returns
        -------
        (int, int)
            Shape ``(H', W')`` after transformation.
        """
        H, W = shape
        return (W, H) if self.swaps_xy else (H, W)

    # ---- WOI transform ----------------------------------------------------

    def transform_woi(self, woi, shape, binning_in=1, binning_out=1):
        """
        Transform a window of interest (WOI) from the original image space into the
        transformed image space.

        Parameters
        ----------
        woi : (int, int, int, int)
            Window of interest ``(x0, w, y0, h)`` in the original space, where
            ``x0``, ``y0`` are the column/row of the top-left corner and ``w``, ``h``
            are the width and height. Must satisfy ``x0 + w <= W`` and
            ``y0 + h <= H`` (at ``binning_in`` scale).
        shape : (int, int)
            Full image shape ``(H, W)`` in the original space at ``binning_in`` scale.
        binning_in : int or (int, int)
            Pixel binning of the input coordinate space: scalar or ``(bx, by)``.
            A value of 1 (default) means native (un-binned) pixels.
        binning_out : int or (int, int)
            Pixel binning of the output (transformed) coordinate space.

        Returns
        -------
        (int, int, int, int)
            Transformed WOI ``(x0', w', y0', h')`` in the transformed space at
            ``binning_out`` scale.
        """
        bxi, byi = (binning_in, binning_in) if np.isscalar(binning_in) else tuple(binning_in)
        bxo, byo = (binning_out, binning_out) if np.isscalar(binning_out) else tuple(binning_out)

        x0, w, y0, h = woi
        isfloat = any(isinstance(v, float) for v in (x0, w, y0, h))
        H, W = shape

        # Scale to native pixels (binning_in = 1)
        x0n, wn = x0 * bxi, w * bxi
        y0n, hn = y0 * byi, h * byi
        Wn, Hn = W * bxi, H * byi

        # Apply orientation transform in native coords.
        (a, b), (c_, d) = self._MATRICES[self.code.value]
        ox = max(0, -a) * Wn + max(0, -b) * Hn
        oy = max(0, -c_) * Wn + max(0, -d) * Hn
        xs = (a * x0n + b * y0n + ox, a * (x0n + wn) + b * (y0n + hn) + ox)
        ys = (c_ * x0n + d * y0n + oy, c_ * (x0n + wn) + d * (y0n + hn) + oy)
        x0t, wt = min(xs), abs(a) * wn + abs(b) * hn
        y0t, ht = min(ys), abs(c_) * wn + abs(d) * hn

        if isfloat:
            return (x0t / bxo, wt / bxo, y0t / byo, ht / byo)
        else:
            return (x0t // bxo, wt // bxo, y0t // byo, ht // byo)

    def inverse_woi(self, woi, transformed_shape, binning_in=1, binning_out=1):
        """
        Inverse of :meth:`transform_woi`: map a WOI from the transformed space back to
        the original image space.
        """
        return self.inverse.transform_woi(
            woi, transformed_shape, binning_in=binning_out, binning_out=binning_in
        )

    def __repr__(self):
        return f"OrientationTransform({self.code.name})"

    def __eq__(self, other):
        if isinstance(other, OrientationTransform):
            return self.code == other.code
        return NotImplemented

    def __hash__(self):
        return hash(self.code)


def get_orientation_transformation(rot="0", fliplr=False, flipud=False):
    """
    Compile an image orientation transformation from rotate/flip parameters.

    Parameters
    ----------
    rot : str or int
        Rotation in degrees: ``"0"``, ``"90"``, ``"180"``, ``"270"`` or
        :func:`numpy.rot90` step counts ``0``-``3``.
    fliplr : bool
        Flip left-right before rotation.
    flipud : bool
        Flip up-down before rotation.

    Returns
    -------
    OrientationTransform
        Callable transformation object.
    """
    return OrientationTransform(rot=rot, fliplr=fliplr, flipud=flipud)
