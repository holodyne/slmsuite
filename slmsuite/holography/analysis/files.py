"""
Utilities for interfacing with files.
This includes helper functions for naming directories without conflicts, and convenience
wrappers for file writing. :mod:`slmsuite` uses the
`HDF5 filetype <https://hdfgroup.org/solutions/hdf5>`_
(.h5) by default, as it is fast, compact, and
`widely supported by programming languages for scientific computing
<https://en.wikipedia.org/wiki/Hierarchical_Data_Format#Interfaces>`_.
This uses the :mod:`h5py` `module <https://h5py.org>`_.
"""

import os
import re
import warnings

import cv2
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

from slmsuite.holography.toolbox import pad


def _max_numeric_id(path, name, extension=None, kind="file", digit_count=5):
    """
    Obtains the maximum numeric identifier ``id`` for the
    directory or file like ``path/name_id.extension``.

    Parameters
    ----------
    path
        See :meth:`generate_path`.
    name
        See :meth:`generate_path`.
    extension
        See :meth:`generate_path`.
    kind
        See :meth:`generate_path`.
    digit_count
        See :meth:`generate_path`.

    Returns
    -------
    max_numeric_id : int
         The maximum numeric identifier of the specified object or -1 if no object with
         ``name`` in ``path`` could be found.
    """
    # Search all objects in path for conflicts.
    conflict_regex = "{}_{}{}{}".format(name, r"\d{", digit_count, r"}")
    if extension is not None and kind == "file":
        conflict_regex = f"{conflict_regex}.{extension}"
    max_numeric_id = -1
    for name_ in os.listdir(path):
        # Check current object for conflict.
        conflict = re.search(conflict_regex, name_) is not None
        # Get numeric identifier from conflicting object.
        if conflict:
            suffix = name_.split(f"{name}_")[1]
            numeric_id = int(suffix[:digit_count])
            max_numeric_id = max(numeric_id, max_numeric_id)

    return max_numeric_id


def generate_path(path, name, extension=None, kind="file", digit_count=5, path_count=1):
    """
    Generate a file path like ``path/name_id.extension``
    or a directory path like ``path/name_id``
    where ``id`` is a unique numeric identifier.

    Parameters
    ----------
    path : str
        Top level directory to create the object in.
        If ``path`` does not exist, it and nonexistent
        parent directories will be created.
    name : str
        Identifier for the object. This should not contain underscores.
    extension : str or None
        The extension to append to the file name, not including the ``.`` separator.
        If ``None``, no extension will be added.
    kind : {"file", "dir"}
        String that identifies what type of object to create.
        If ``kind="dir"`` the directory will be created.
    digit_count : int
        The number of digits to use in the numeric identifier.
    path_count : int
        The number of paths to create. Currently only applicable to file creation.

    Returns
    -------
    file_path : str or list of str
        The full path or paths requested.

    Notes
    -----
    This function is not thread safe.
    """
    path = os.path.abspath(path)
    # Ensure path exists.
    os.makedirs(path, exist_ok=True)

    # Create the name
    max_numeric_id = _max_numeric_id(
        path, name, extension=extension, kind=kind, digit_count=digit_count
    )
    name_format = f"{{}}_{{:0{digit_count}d}}"
    name_augmented = name_format.format(name, max_numeric_id + 1)
    if extension is not None and kind == "file":
        name_augmented = f"{name_augmented}.{extension}"
    name_augmented = os.path.join(path, name_augmented)

    # If it's a directory, create it.
    if kind == "dir":
        os.makedirs(name_augmented)

    ret = None
    if path_count == 1:
        ret = name_augmented
    else:
        ret = []
        for path_idx in range(path_count):
            name_augmented = name_format.format(name, max_numeric_id + 1 + path_idx)
            if extension is not None and kind == "file":
                name_augmented = f"{name_augmented}.{extension}"
            name_augmented = os.path.join(path, name_augmented)
            ret.append(name_augmented)
        # ENDFOR
    # ENDIF

    return ret


def latest_path(path, name, extension=None, kind="file", digit_count=5):
    """
    Obtains the path for the file or directory in ``path`` like ``path/name_id``
    where ``id`` is the greatest identifier in ``path`` for the given ``name``.

    Parameters
    ----------
    path
        See :meth:`generate_path`.
    name
        See :meth:`generate_path`.
    extension
        See :meth:`generate_path`.
    kind
        See :meth:`generate_path`.
    digit_count
        See :meth:`generate_path`.

    Returns
    -------
    file_path : str or None
        The path requested. ``None`` if no file could be found.
    """
    ret = None
    max_numeric_id = _max_numeric_id(
        path, name, extension=extension, kind=kind, digit_count=digit_count
    )
    if max_numeric_id != -1:
        name_format = f"{{}}_{{:0{digit_count}d}}"
        name_augmented = name_format.format(name, max_numeric_id)
        if extension is not None and kind == "file":
            name_augmented = f"{name_augmented}.{extension}"
        ret = os.path.join(path, name_augmented)

    return ret


def read_h5(file_path, decode_bytes=True):
    """Backwards-compatible alias of :meth:`load_h5`"""
    return load_h5(file_path, decode_bytes)


def load_h5(file_path, decode_bytes=True):
    """
    Read data from an h5 file into a dictionary.
    In the case of more complicated h5 hierarchy, a dictionary of dictionaries is returned.

    Parameters
    ----------
    file_path : str
        Full path to the file to read the data from.
    decode_bytes : bool
        Whether or not objects with type ``bytes`` should be decoded.
        By default HDF5 writes strings as bytes objects; this functionality
        will make strings read back from the file ``str`` type.

    Returns
    -------
    data : dict
        Dictionary of the data stored in the file.
    """

    def recurse(group):
        data = {}

        for key in group:
            if isinstance(group[key], h5py.Group):
                data[key] = recurse(group[key])
            else:
                data_ = group[key][()]
                if decode_bytes:
                    if isinstance(data_, bytes):
                        data_ = bytes.decode(data_)
                    elif np.isscalar(data_):
                        pass
                    elif (
                        isinstance(data_, np.ndarray)
                        and len(data_) > 0
                        and isinstance(data_[0], bytes)
                    ):
                        data_ = np.vectorize(bytes.decode)(data_)
                data[key] = data_

        return data

    with h5py.File(file_path, "r") as file_:
        data = recurse(file_)

    return data


def write_h5(file_path, data, mode="w"):
    """Backwards-compatible alias of :meth:`save_h5`"""
    return save_h5(file_path, data, mode)


def save_h5(file_path, data, mode="w"):
    """
    Write data in a dictionary to an `h5 file
    <https://docs.h5py.org/en/stable/high/file.html#opening-creating-files>`_.

    Note
    ~~~~
    There are some limitations to what the h5 file standard can store, along with
    limitations on what is currently implemented in this function.

    Supported types:

    - Nested dictionaries which are written as h5 group hierarchy,
    - ``None`` (though this is written as ``False``),
    - Uniform arrays of numeric or string data.

    Example unsupported types:

    - Staggered arrays (an array consisting of arrays of different sizes),
    - Non-numeric or non-string data (e.g. object),

    Parameters
    ----------
    file_path : str
        Full path to the file to save the data in.
    data : dict
        Dictionary of data to save in the file.
    mode : str
        The mode to open the file with.
    """

    def recurse(group, data):
        for key in data:
            if isinstance(data[key], dict):
                new_group = group.create_group(key)
                recurse(new_group, data[key])
            elif isinstance(data[key], str):
                group[key] = bytes(data[key], "utf-8")
            elif data[key] is None:
                group[key] = False
            else:
                try:
                    array = np.array(data[key])
                except ValueError as e:
                    raise ValueError(
                        f"save_h5() does not support saving staggered arrays such as {data[key]!s}. "
                        f"Arrays must be uniform. {e!s}"
                      ) from e
                except Exception as e:
                    raise e

                if array.dtype.char == "U":
                    array = np.vectorize(str.encode)(array)

                group[key] = array

    with h5py.File(file_path, mode) as file_:
        recurse(file_, data)


def _load_image(path, shape, target_shape=None, angle=0, shift=(-225, -170)):
    """Helper function for examples."""
    # Load the image.
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Image not found at path '{path}'")

    # Invert if necessary such that the majority of the image is dark.
    if np.mean(img) > np.mean(cv2.bitwise_not(img)):
        img = cv2.bitwise_not(img)

    if angle != 0:
        img = ndimage.rotate(img, angle)

    if target_shape is not None:
        zoom_x = target_shape[0] / img.shape[0]
        zoom_y = target_shape[1] / img.shape[1]
        img = ndimage.zoom(img, min(zoom_x, zoom_y))

    # sqrt to get the amplitude.
    target_ij = pad(np.sqrt(img), shape)

    # Shift to the desired center.
    target_ij = np.roll(target_ij, shift, axis=(0, 1))

    return target_ij


def _gray2rgb(images, cmap=False, lut=None, normalize=True, border=None):
    """
    Currently-hidden function to convert a stack of
    grayscale images to color with a colormap.

    Returns
    -------
    numpy.ndarray
        The converted images. This is of size ``(image_count, h, w, 4)``,
        where the last axis is RGBA color, 8 bits per channel.
    """
    # Parse images.
    images = np.array(images, copy=(False if np.__version__[0] == "1" else None))
    if len(images.shape) == 2:
        images = np.reshape(images, (1, images.shape[0], images.shape[1]))
    elif len(images.shape) >= 3 and images.shape[-1] in [3, 4]:  # Already RGB or RGBA
        return images
    elif len(images.shape) > 3:
        raise RuntimeError(f"Images shape {images.shape} could not be parsed.")

    isfloat = np.issubdtype(images.dtype, np.floating)

    # Parse cmap.
    if cmap == "default":
        cmap = True
    if cmap == "grayscale":
        cmap = False

    if not isinstance(cmap, str) and not hasattr(cmap, "N"):
        if cmap is True:
            cmap = mpl.rcParams["image.cmap"]
        else:
            # Grayscale is forced to have an lut smaller than 256.
            if lut is None or lut > 256:
                lut = 256

    # Parse lut.
    if lut is None:
        if isfloat:
            lut = mpl.rcParams["image.lut"] - 1
        else:
            lut = np.nanmax(images)
    # lut = np.clip(lut, 0, np.max(images))
    lut = np.array([lut]).astype(images.dtype)[0]

    # Check for nan.
    nanmask = np.isnan(images)
    hasnan = np.any(nanmask)
    if hasnan:
        images[nanmask] = 0

    # Convert images to integers scaled to the lut size.
    if normalize:
        images = np.rint(images * ((float(lut) - 1) / np.max(images))).astype(int)
        images = np.clip(images, 0, int(lut))
    elif isfloat:
        images = np.rint(images * (float(lut) - 1)).astype(int)
        images = np.clip(images, 0, int(lut))

    # Convert images to RGB.
    if isinstance(cmap, str) or hasattr(cmap, "N"):
        if isinstance(cmap, str):
            cm = plt.get_cmap(cmap, int(lut) + 1)
        else:
            cm = cmap

        if hasattr(cm, "colors"):
            c = cm.colors
        else:
            c = cm(np.arange(0, cm.N))

        images = 255 * c[images]
        if hasnan:
            images[nanmask, 3] = 0

    images = images.astype(np.uint8)

    # Add a border if desired.
    if border is not None:
        # If border is a single numeric value, convert it to a list
        if np.isscalar(border):
            border = [border]
        images[:, 0, :, : len(border)] = border
        images[:, -1, :, : len(border)] = border
        images[:, :, 0, : len(border)] = border
        images[:, :, -1, : len(border)] = border

    return images


def save_image(file_path, images, cmap=False, lut=None, normalize=True, border=None, **kwargs):
    """
    Save a grayscale image or stacks of grayscale images
    as a filetype supported by :mod:`imageio`.
    Handles :mod:`matplotlib` colormapping.
    Negative values are truncated to zero.
    ``np.nan`` values are set to transparent.

    Parameters
    ----------
    file_path : str
        Full path to the file to save the data in.
    images : numpy.ndarray
        A 2D matrix (image formats) or stack of 2D matrices (video formats).

    cmap : str OR bool OR None
        If ``str``, the :mod:`matplotlib` colormap under this name is used.
        If ``None`` or ``False``, the images are directly saved as grayscale 8-bit images.
        If ``True``, the default colormap is used.
    lut : int OR None
        Size of the lookup table for the colormap. This determines the number of colors
        the resulting image has. This can be larger than 256 values because RGB data can
        realize more colors than grayscale.
        If ``None`, Defaults to ``mpl.rcParams['image.lut']`` (if the image is floating
        point) or the maximum of the image (if the image )
    normalize : bool
        If ``True``, the maximum of the image is taken as the image maximum.
        If ``False`` and using integer data, the data is unchanged.
        If ``False`` and using floating point data, 1 is taken to be the maximum, and
        this is scaled to the ``lut``.
    **kwargs
        Passed to ``imageio.imsave()`` or ``imageio.mimsave()``. Useful for choosing a ``plugin`` or ``format``.
    """
    images = _gray2rgb(images, cmap=cmap, lut=lut, normalize=normalize, border=border)

    # Determine the file format
    extension = file_path.split(".")[-1]

    # Check that imageio is there and write the data
    try:
        from imageio import imsave, mimsave
    except Exception:
        raise ValueError("imageio is required for save_image().") from None

    if images.shape[0] == 1:
        imsave(file_path, images[0], **kwargs)
    else:
        mimsave(file_path, images, **kwargs)

    # Optimize .gif if pygifsicle is installed
    if extension == "gif":
        try:
            from pygifsicle import optimize

            optimize(file_path)
        except ImportError:
            warnings.warn("pip install pygifsicle to optimize .gif file size.")
        except Exception as e:
            warnings.warn(f"pygifsicle optimization failed: {e}")
