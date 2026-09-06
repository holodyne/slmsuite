"""
Unit tests for slmsuite.holography.analysis module.
"""
import contextlib
import logging

import pytest
import numpy as np
import matplotlib.pyplot as plt

from slmsuite.holography import analysis
from slmsuite.holography.analysis.fitfunctions import gaussian2d


@contextlib.contextmanager
def _shows():
    """Intercept slmsuite's internal show hook, yielding the plot names it is called with."""
    names = []
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            analysis, "_slmsuite_plt_show", lambda name=None, *args, **kwargs: names.append(name)
        )
        yield names
    plt.close("all")


def _raise_runtime_error(*args, **kwargs):
    """Stand-in for a solver that gives up."""
    raise RuntimeError("forced failure")


def test_center(subtests):
    """Test _center() index center of a range."""
    with subtests.test("an even width centers between the two middle indices"):
        assert analysis._center(10) == 4.5

    with subtests.test("an odd width centers on the middle index"):
        assert analysis._center(11) == 5.0

    with subtests.test("integer=True rounds the center up to a pixel"):
        assert analysis._center(10, integer=True) == 5
        assert analysis._center(11, integer=True) == 5


def test_coordinates(subtests):
    """Test _coordinates() index range."""
    with subtests.test("centered coordinates are symmetric about zero"):
        np.testing.assert_array_equal(analysis._coordinates(5, centered=True), [-2, -1, 0, 1, 2])
        np.testing.assert_array_equal(
            analysis._coordinates(4, centered=True), [-1.5, -0.5, 0.5, 1.5]
        )

    with subtests.test("uncentered coordinates are the pixel indices"):
        np.testing.assert_array_equal(analysis._coordinates(5), np.arange(5))


def test_generate_grid(subtests):
    """Test _generate_grid() meshgrid generation."""
    with subtests.test("a centered grid is symmetric about zero in each axis"):
        grid_x, grid_y = analysis._generate_grid(3, 4, centered=True)
        np.testing.assert_array_equal(grid_x, np.tile([-1.0, 0.0, 1.0], (4, 1)))
        np.testing.assert_array_equal(grid_y, np.tile([[-1.5], [-0.5], [0.5], [1.5]], (1, 3)))

    with subtests.test("integer=True places the center on a pixel"):
        grid_x, _ = analysis._generate_grid(4, 4, centered=True, integer=True)
        np.testing.assert_array_equal(grid_x[0], [-2.0, -1.0, 0.0, 1.0])

    with subtests.test("an uncentered grid is the pixel indices"):
        grid_x, grid_y = analysis._generate_grid(3, 3)
        np.testing.assert_array_equal(grid_x, np.tile(np.arange(3.0), (3, 1)))
        np.testing.assert_array_equal(grid_y, np.tile(np.arange(3.0)[:, np.newaxis], (1, 3)))


def test_take(subtests, benchmark):
    """Test take() region extraction."""
    image = np.arange(100 * 100, dtype=float).reshape(100, 100)

    with subtests.test("benchmark"):
        rng = np.random.default_rng(42)
        big = rng.random((512, 512)).astype(np.float32)
        vectors = np.stack([rng.integers(20, 492, 50), rng.integers(20, 492, 50)])
        benchmark(analysis.take, big, vectors=vectors, size=20, centered=True)

    with subtests.test("a centered region is the slice around the vector"):
        result = analysis.take(image, vectors=[50, 40], size=10, centered=True)
        np.testing.assert_array_equal(result[0], image[35:45, 45:55])

    with subtests.test("centered=False anchors the region at the vector"):
        result = analysis.take(image, vectors=[20, 10], size=10, centered=False)
        np.testing.assert_array_equal(result[0], image[10:20, 20:30])

    with subtests.test("size is (width, height)"):
        result = analysis.take(image, vectors=[50, 40], size=(8, 12), centered=False)
        assert result.shape == (1, 12, 8)
        np.testing.assert_array_equal(result[0], image[40:52, 50:58])

    with subtests.test("float vectors floor to the pixel below"):
        np.testing.assert_array_equal(
            analysis.take(image, vectors=[50.7, 40.3], size=10, centered=True),
            analysis.take(image, vectors=[50, 40], size=10, centered=True),
        )

    with subtests.test("integrate sums each region"):
        np.testing.assert_array_equal(
            analysis.take(np.ones((100, 100)), vectors=[50, 50], size=10,
                          centered=True, integrate=True),
            [100.0],
        )

    with subtests.test("output shape tracks the stack, the vector count, and integrate"):
        stack = np.ones((4, 100, 100))
        vectors = np.array([[25, 50, 75], [50, 50, 50]])
        for (images, integrate, shape) in (
            (image, False, (3, 10, 10)),
            (stack, False, (4, 3, 10, 10)),
            (image, True, (3,)),
            (stack, True, (4, 3)),
        ):
            result = analysis.take(
                images, vectors=vectors, size=10, centered=True, integrate=integrate
            )
            assert result.shape == shape

    with subtests.test("clip=True blanks the out-of-range pixels with nan"):
        result = analysis.take(image, vectors=[0, 0], size=20, centered=True, clip=True)
        assert np.sum(np.isnan(result)) == 20 * 20 - 10 * 10
        np.testing.assert_array_equal(result[0][10:, 10:], image[0:10, 0:10])

    with subtests.test("clip=True blanks with zero when nan does not fit the dtype"):
        integers = np.full((50, 50), 42, dtype=np.uint8)
        result = analysis.take(integers, vectors=[0, 0], size=20, centered=True, clip=True)
        assert np.sum(result == 0) == 20 * 20 - 10 * 10

    with subtests.test("clip=True is a no-op for a region that is in range"):
        np.testing.assert_array_equal(
            analysis.take(image, vectors=[50, 40], size=10, centered=True, clip=True),
            analysis.take(image, vectors=[50, 40], size=10, centered=True),
        )

    with subtests.test("return_mask marks exactly the pixels that would be taken"):
        canvas = analysis.take(image, vectors=[50, 40], size=10, centered=True, return_mask=True)
        assert canvas.dtype == bool and canvas.shape == image.shape
        assert np.sum(canvas) == 100
        assert canvas[35:45, 45:55].all()

    with subtests.test("return_mask=2 keeps the taken pixels and nans the rest"):
        canvas = analysis.take(image, vectors=[50, 40], size=10, centered=True, return_mask=2)
        assert np.sum(np.isnan(canvas)) == canvas.size - 100
        np.testing.assert_array_equal(canvas[35:45, 45:55], image[35:45, 45:55])

    with subtests.test("a shape tuple in place of images returns the mask alone"):
        canvas = analysis.take((80, 60), vectors=[30, 20], size=10, centered=True)
        assert canvas.dtype == bool and canvas.shape == (80, 60)
        assert np.sum(canvas) == 100

    with subtests.test("an out-of-range region without clip raises"):
        for vector in ([-1, 25], [25, -1], [54, 25], [25, 54], [100, 100]):
            with pytest.raises(IndexError):
                analysis.take(np.zeros((50, 50)), vectors=vector, size=10, centered=True)

    with subtests.test("...and the error localizes the regions it could not take"):
        # A frame mismatch is only diagnosable if the message says where the regions landed.
        with pytest.raises(IndexError) as excinfo:
            analysis.take(np.zeros((50, 50)), vectors=[120, 130], size=10, centered=True)
        message = str(excinfo.value)
        assert "(115, 124)" in message and "(125, 134)" in message
        assert "(50, 50)" in message

    with subtests.test("clip=True integrates only the pixels it measured"):
        # A window hanging off the frame must not poison the whole spot with nan.
        result = analysis.take(
            np.ones((50, 50)), vectors=[0, 0], size=3, centered=True,
            integrate=True, clip=True,
        )
        assert result[0] == pytest.approx(4)     # the 2x2 corner that exists

    with subtests.test("clip=True is a no-op for an in-range region, integrated too"):
        # The in-range fast path must agree exactly with clip=False.
        np.testing.assert_array_equal(
            analysis.take(image, vectors=[50, 40], size=10, centered=True,
                          integrate=True, clip=True),
            analysis.take(image, vectors=[50, 40], size=10, centered=True, integrate=True),
        )

    with subtests.test("more than three image dimensions raises"):
        with pytest.raises(RuntimeError):
            analysis.take(np.zeros((2, 3, 50, 50)), vectors=[25, 25], size=10)

    with subtests.test("plot renders the region, or the mask for return_mask"):
        with _shows() as shown:
            analysis.take(image, vectors=[30, 30], size=10, centered=True, plot=True)
            analysis.take(image, vectors=[30, 30], size=10, centered=True,
                          return_mask=True, plot=True)
        assert shown == ["take_plot", "take"]


def test_take_plot(subtests):
    """Test take_plot() rendering of a stack of images."""
    images = np.random.rand(3, 10, 10)

    for separate_axes in (False, True):
        with subtests.test(f"separate_axes={separate_axes} renders one figure"):
            with _shows() as shown:
                analysis.take_plot(images, shape=(2, 2), separate_axes=separate_axes)
            assert shown == ["take_plot"]


def test_take_parse_shape(subtests, caplog):
    """Test _take_parse_shape() tiling shape selection."""
    images = np.empty((3, 10, 10))

    with subtests.test("shape=None is the smallest square that holds every image"):
        assert analysis._take_parse_shape(images, shape=None) == (3, (2, 2))

    with subtests.test("an explicit shape is passed through"):
        assert analysis._take_parse_shape(images, shape=(1, 4)) == (3, (1, 4))

    with subtests.test("too small a shape truncates the image count and warns"):
        with caplog.at_level(logging.WARNING, logger="slmsuite"):
            caplog.clear()
            assert analysis._take_parse_shape(images, shape=(1, 2)) == (2, (1, 2))
        assert any("Not enough space" in record.getMessage() for record in caplog.records)


def test_take_tile(subtests):
    """Test take_tile() tiling of a stack of images."""
    images = np.arange(3 * 4 * 5, dtype=float).reshape(3, 4, 5)

    with subtests.test("images tile in row-major order and empty cells are zero"):
        tiled = analysis.take_tile(images, shape=(2, 2))
        assert tiled.shape == (8, 10)
        np.testing.assert_array_equal(tiled[:4, :5], images[0])
        np.testing.assert_array_equal(tiled[:4, 5:], images[1])
        np.testing.assert_array_equal(tiled[4:, :5], images[2])
        np.testing.assert_array_equal(tiled[4:, 5:], 0)

    with subtests.test("shape=None is the smallest square that holds every image"):
        assert analysis.take_tile(images).shape == (8, 10)


def test_image_remove_field(subtests):
    """Test image_remove_field() background removal."""
    image = np.full((1, 100, 100), 10.0)
    image[0, 45:55, 45:55] = 100.0
    median_removed = np.zeros((1, 100, 100))
    median_removed[0, 45:55, 45:55] = 90.0

    with subtests.test("deviations=None subtracts the median"):
        np.testing.assert_array_equal(
            analysis.image_remove_field(image, deviations=None), median_removed
        )

    with subtests.test("deviations above the mean zeroes everything but the spot"):
        cleaned = analysis.image_remove_field(image, deviations=2)
        assert np.count_nonzero(cleaned) == 100

    with subtests.test("integer images are promoted to float rather than underflowing"):
        integers = np.full((1, 8, 8), 10, dtype=np.uint8)
        integers[0, 3:5, 3:5] = 200
        expected = np.zeros((1, 8, 8))
        expected[0, 3:5, 3:5] = 190

        result = analysis.image_remove_field(integers, deviations=None)

        np.testing.assert_array_equal(result, expected)
        assert np.issubdtype(result.dtype, np.floating)

    with subtests.test("out=images subtracts in place"):
        in_place = image.copy()
        result = analysis.image_remove_field(in_place, deviations=None, out=in_place)
        assert result is in_place
        np.testing.assert_array_equal(in_place, median_removed)

    with subtests.test("a separate out leaves the input untouched"):
        out = np.empty_like(image)
        result = analysis.image_remove_field(image, deviations=None, out=out)
        assert result is out
        np.testing.assert_array_equal(out, median_removed)
        assert image[0, 0, 0] == 10.0

    with subtests.test("an integer out raises"):
        integers = np.full((1, 8, 8), 10, dtype=np.uint8)
        with pytest.raises(ValueError, match="floating point"):
            analysis.image_remove_field(integers, deviations=None, out=integers)


def test_image_relative_strehl(subtests):
    """Test image_relative_strehl(), the peak fraction of the total intensity."""
    image = np.zeros((50, 50))
    image[24, 24] = 100

    with subtests.test("a uniform image spreads evenly over its pixels"):
        assert analysis.image_relative_strehl(np.ones((1, 10, 10)))[0] == pytest.approx(1 / 100)

    with subtests.test("a single lit pixel holds all of the intensity"):
        assert analysis.image_relative_strehl(image[np.newaxis])[0] == pytest.approx(1.0)

    with subtests.test("a 2D image is read as a stack of one"):
        np.testing.assert_array_equal(
            analysis.image_relative_strehl(image), analysis.image_relative_strehl(image[np.newaxis])
        )


def test_image_moment(subtests, benchmark):
    """Test image_moment() moment calculation."""
    # A 15 x 10 block of ones, whose central moments are those of a uniform distribution.
    block = np.zeros((1, 60, 60))
    block[0, 20:30, 20:35] = 1.0
    block_center = analysis.image_positions(block)

    # Two point sources at (x, y) = +-(10, 5) about the center of the image.
    points = np.zeros((1, 101, 101))
    points[0, 55, 60] = points[0, 45, 40] = 1.0

    with subtests.test("benchmark"):
        image = np.random.rand(1, 128, 128).astype(np.float32)
        benchmark(analysis.image_moment, image, moment=(1, 0))

    with subtests.test("the unnormalized zeroth moment is the total intensity"):
        uniform = np.full((1, 50, 50), 0.5)
        assert analysis.image_moment(uniform, (0, 0), normalize=False)[0] == pytest.approx(1250.0)

    with subtests.test("the normalized zeroth moment is unity"):
        assert analysis.image_moment(np.full((1, 50, 50), 0.5), (0, 0))[0] == 1

    with subtests.test("the first moments are the centroid in centered pixel units"):
        # The block spans columns 20-34 and rows 20-29 of an image centered at 29.5.
        assert analysis.image_moment(block, (1, 0))[0] == pytest.approx(27.0 - 29.5)
        assert analysis.image_moment(block, (0, 1))[0] == pytest.approx(24.5 - 29.5)

    with subtests.test("the second central moments of a block are the uniform (n^2-1)/12"):
        m20 = analysis.image_moment(block, (2, 0), centers=block_center)
        m02 = analysis.image_moment(block, (0, 2), centers=block_center)
        assert m20[0] == pytest.approx((15**2 - 1) / 12)
        assert m02[0] == pytest.approx((10**2 - 1) / 12)

    with subtests.test("the shear moment of a separable block vanishes"):
        m11 = analysis.image_moment(block, (1, 1), centers=block_center)
        assert m11[0] == pytest.approx(0, abs=1e-12)

    with subtests.test("the moments of two point sources are their offsets"):
        assert analysis.image_moment(points, (2, 0))[0] == pytest.approx(10.0**2)
        assert analysis.image_moment(points, (0, 2))[0] == pytest.approx(5.0**2)
        assert analysis.image_moment(points, (1, 1))[0] == pytest.approx(10.0 * 5.0)

    with subtests.test("a grid scale factors out of each moment"):
        scaled = analysis.image_moment(points, (1, 1), grid=(2.0, 3.0))
        assert scaled[0] == pytest.approx(2.0 * 3.0 * 10.0 * 5.0)
        np.testing.assert_allclose(
            analysis.image_moment(points, (2, 0), grid=2.5),
            analysis.image_moment(points, (2, 0), grid=(2.5, 2.5)),
        )

    with subtests.test("1D, 2D, and per-image grids agree with the equivalent scale"):
        xs = 2.0 * (np.arange(101.0) - 50.0)
        ys = 3.0 * (np.arange(101.0) - 50.0)
        grid_x, grid_y = np.meshgrid(xs, ys)
        expected = analysis.image_moment(points, (1, 1), grid=(2.0, 3.0))
        for grid in ((xs, ys), (grid_x, grid_y), (grid_x[np.newaxis], grid_y[np.newaxis])):
            np.testing.assert_allclose(analysis.image_moment(points, (1, 1), grid=grid), expected)

    with subtests.test("nansum treats nan pixels as zero"):
        image = np.ones((1, 30, 30))
        image[0, 5:10, 5:10] = np.nan
        moment = analysis.image_moment(image, (0, 0), normalize=False, nansum=True)
        assert moment[0] == pytest.approx(30 * 30 - 5 * 5)


def test_image_normalization(subtests):
    """Test image_normalization(), the zeroth moment of each image."""
    with subtests.test("the normalization is the total intensity"):
        assert analysis.image_normalization(np.full((1, 50, 50), 2.0))[0] == pytest.approx(5000.0)

    with subtests.test("nansum treats nan pixels as zero"):
        image = np.ones((1, 50, 50))
        image[0, 10:20, 10:20] = np.nan
        assert analysis.image_normalization(image, nansum=True)[0] == pytest.approx(2400.0)


def test_image_normalize(subtests):
    """Test image_normalize() rescaling to unit sum."""
    with subtests.test("every image in a stack sums to one"):
        images = np.random.rand(3, 50, 50) * 100 + 50
        np.testing.assert_allclose(np.sum(analysis.image_normalize(images), axis=(1, 2)), 1.0)

    with subtests.test("a 2D image keeps its shape"):
        result = analysis.image_normalize(np.full((30, 30), 4.0))
        assert result.shape == (30, 30)
        np.testing.assert_allclose(result, 1 / (30 * 30))

    with subtests.test("an all-zero image normalizes to zeros"):
        np.testing.assert_array_equal(analysis.image_normalize(np.zeros((30, 30))), 0)

    with subtests.test("remove_field subtracts the background before normalizing"):
        image = np.full((1, 50, 50), 10.0)
        image[0, 20:30, 20:30] = 100.0

        result = analysis.image_normalize(image, remove_field=True)

        assert np.count_nonzero(result) == 100
        assert np.sum(result) == pytest.approx(1.0)


def test_image_positions(subtests):
    """Test image_positions() first order moments."""
    # A block over rows 30-39 and columns 60-69 of an image centered at 49.5.
    image = np.zeros((1, 100, 100))
    image[0, 30:40, 60:70] = 1.0
    centered = np.zeros((1, 100, 100))
    centered[0, 45:55, 45:55] = 1.0

    with subtests.test("the position is the centroid relative to the image center"):
        np.testing.assert_allclose(analysis.image_positions(image), [[15.0], [-15.0]])

    with subtests.test("a centered spot sits at the origin"):
        np.testing.assert_allclose(analysis.image_positions(centered), 0, atol=1e-12)

    with subtests.test("a grid scale multiplies each position"):
        np.testing.assert_allclose(
            analysis.image_positions(image, grid=(2.0, 4.0)), [[30.0], [-60.0]]
        )

    with subtests.test("each image in a stack gets its own position"):
        stack = np.concatenate((image, centered))
        np.testing.assert_allclose(
            analysis.image_positions(stack), [[15.0, 0.0], [-15.0, 0.0]], atol=1e-12
        )


def test_image_centroids():
    """image_centroids() is an alias of image_positions(), positional arguments included."""
    images = np.random.rand(2, 40, 30)
    np.testing.assert_array_equal(
        analysis.image_centroids(images, 2.0, False, True),
        analysis.image_positions(images, grid=2.0, normalize=False, nansum=True),
    )


def test_image_variances(subtests):
    """Test image_variances() second order central moments."""
    # Two point sources at (x, y) = +-(10, 5) about the center of the image.
    points = np.zeros((1, 101, 101))
    points[0, 55, 60] = points[0, 45, 40] = 1.0

    # A 15 x 10 block of ones, whose central moments are those of a uniform distribution.
    block = np.zeros((1, 60, 60))
    block[0, 20:30, 20:35] = 1.0

    with subtests.test("the variances of two point sources are their squared offsets"):
        np.testing.assert_allclose(analysis.image_variances(points), [[100.0], [25.0], [50.0]])

    with subtests.test("a uniform block has variance (n^2-1)/12 and no shear"):
        np.testing.assert_allclose(
            analysis.image_variances(block),
            [[(15**2 - 1) / 12], [(10**2 - 1) / 12], [0.0]],
            atol=1e-12,
        )

    with subtests.test("moments are taken about the centroid, not the grid origin"):
        grid_x, grid_y = np.meshgrid(np.arange(101.0), np.arange(101.0))
        np.testing.assert_allclose(
            analysis.image_variances(points, grid=(grid_x, grid_y)), [[100.0], [25.0], [50.0]]
        )

    with subtests.test("a grid scale factors out of each variance"):
        np.testing.assert_allclose(
            analysis.image_variances(points, grid=(2.0, 3.0)),
            [[4 * 100.0], [9 * 25.0], [6 * 50.0]],
        )
        grid_x, grid_y = np.meshgrid(2.0 * np.arange(101.0), 3.0 * np.arange(101.0))
        np.testing.assert_allclose(
            analysis.image_variances(points, grid=(grid_x, grid_y)),
            [[4 * 100.0], [9 * 25.0], [6 * 50.0]],
        )
        np.testing.assert_allclose(
            analysis.image_variances(points, grid=2.5), analysis.image_variances(points, grid=(2.5, 2.5))
        )

    with subtests.test("given centers, the moment is not central but parallel-axis shifted"):
        # The block's centroid is 2.5 left of and 5 above the center of the image.
        raw = analysis.image_variances(block, centers=np.zeros((2, 1)))
        assert raw[0, 0] == pytest.approx((15**2 - 1) / 12 + 2.5**2)
        assert raw[1, 0] == pytest.approx((10**2 - 1) / 12 + 5.0**2)

    with subtests.test("exclude_shear drops the shear moment"):
        np.testing.assert_allclose(
            analysis.image_variances(points, exclude_shear=True), [[100.0], [25.0]]
        )


def test_image_std(subtests):
    """Test image_std(), the root of the x and y variances."""
    # Truncation and pixelation keep a sampled Gaussian from its analytic width.
    (y, x) = np.ogrid[:100, :100]
    isotropic = np.exp(-((x - 50) ** 2 + (y - 50) ** 2) / (2 * 10.0**2))[np.newaxis]
    elliptical = np.exp(-((x - 50) ** 2 / (2 * 5.0**2) + (y - 50) ** 2 / (2 * 15.0**2)))[np.newaxis]

    with subtests.test("an isotropic Gaussian returns its own width in both axes"):
        np.testing.assert_allclose(analysis.image_std(isotropic), 10.0, rtol=0.01)

    with subtests.test("an elliptical Gaussian returns each axis width"):
        np.testing.assert_allclose(analysis.image_std(elliptical), [[5.0], [15.0]], rtol=0.05)

    with subtests.test("the std is the root of the shear-free variances"):
        np.testing.assert_allclose(
            np.square(analysis.image_std(elliptical)),
            analysis.image_variances(elliptical, exclude_shear=True),
        )

    with subtests.test("a grid scale multiplies the std linearly"):
        np.testing.assert_allclose(
            analysis.image_std(elliptical, grid=4.0), 4.0 * analysis.image_std(elliptical)
        )


def test_image_ellipticity(subtests):
    """Test image_ellipticity(), one minus the ratio of the moment eigenvalues."""
    with subtests.test("each column is evaluated independently"):
        variances = np.array([[100.0, 200.0, 100.0], [100.0, 100.0, 25.0], [0.0, 0.0, 50.0]])
        np.testing.assert_allclose(analysis.image_ellipticity(variances), [0.0, 0.5, 1.0])

    with subtests.test("ellipticity is invariant under rotation of the moment matrix"):
        for theta in (0.0, 0.3, np.pi / 4, -0.7):
            rotation = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            moments = rotation @ np.diag([200.0, 100.0]) @ rotation.T
            variances = np.array([[moments[0, 0]], [moments[1, 1]], [moments[0, 1]]])
            assert analysis.image_ellipticity(variances)[0] == pytest.approx(0.5)

    with subtests.test("an image with no intensity has undefined ellipticity"):
        assert np.isnan(analysis.image_ellipticity(np.zeros((3, 1)))[0])


def test_image_areas(subtests):
    """Test image_areas(), the determinant of the moment matrix."""
    with subtests.test("the area is the determinant of the moment matrix"):
        variances = np.array([[200.0, 100.0, 100.0], [100.0, 100.0, 25.0], [0.0, 0.0, 50.0]])
        np.testing.assert_allclose(analysis.image_areas(variances), [20000.0, 10000.0, 0.0])

    with subtests.test("the area is invariant under rotation of the moment matrix"):
        for theta in (0.0, 0.3, np.pi / 4, -0.7):
            rotation = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            moments = rotation @ np.diag([200.0, 100.0]) @ rotation.T
            variances = np.array([[moments[0, 0]], [moments[1, 1]], [moments[0, 1]]])
            assert analysis.image_areas(variances)[0] == pytest.approx(20000.0)


def test_image_ellipticity_angle(subtests):
    """Test image_ellipticity_angle(), the major axis angle of the moment matrix."""
    with subtests.test("a circular spot returns zero"):
        assert analysis.image_ellipticity_angle(np.array([[100.0], [100.0], [0.0]]))[0] == 0

    with subtests.test("the angle is the rotation of the moment matrix eigenbasis"):
        for theta in (0.0, 0.3, np.pi / 4, -0.7):
            rotation = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            moments = rotation @ np.diag([200.0, 100.0]) @ rotation.T
            variances = np.array([[moments[0, 0]], [moments[1, 1]], [moments[0, 1]]])
            assert analysis.image_ellipticity_angle(variances)[0] == pytest.approx(theta)

    with subtests.test("two point sources are elongated along the line joining them"):
        points = np.zeros((1, 101, 101))
        points[0, 55, 60] = points[0, 45, 40] = 1.0
        angles = analysis.image_ellipticity_angle(analysis.image_variances(points))
        assert angles[0] == pytest.approx(np.arctan2(5.0, 10.0))


def test_image_strehl(subtests):
    """Test image_strehl() against a diffraction-limited reference."""
    (x, y) = np.meshgrid(np.arange(-16, 16), np.arange(-16, 16))
    r2 = x ** 2 + y ** 2
    narrow = np.exp(-r2 / (2 * 1.5 ** 2))
    broad = np.exp(-r2 / (2 * 3.0 ** 2))

    with subtests.test("an image against itself is unity"):
        np.testing.assert_allclose(analysis.image_strehl(narrow, narrow), 1)

    with subtests.test("insensitive to exposure and gain"):
        # Each image is normalized by its own total, so a scale factor cancels.
        np.testing.assert_allclose(
            analysis.image_strehl(1e4 * broad, narrow),
            analysis.image_strehl(broad, narrow),
        )

    with subtests.test("a broadened spot is below unity"):
        # A Gaussian's peak fraction goes as 1/width^2: twice the width, a quarter the Strehl.
        np.testing.assert_allclose(
            analysis.image_strehl(broad, narrow), (1.5 / 3.0) ** 2, rtol=1e-3
        )

    with subtests.test("a stack against one reference"):
        strehl = analysis.image_strehl(np.stack((narrow, broad)), narrow)
        assert strehl.shape == (2,)
        np.testing.assert_allclose(strehl, [1, (1.5 / 3.0) ** 2], rtol=1e-3)


def test_image_fit(subtests, benchmark, caplog):
    """Test image_fit() fitting of a stack of images."""
    x = np.linspace(-10, 10, 50)
    grid = np.meshgrid(x, x)
    truth = dict(x0=2, y0=-1, a=10, c=1, wx=2, wy=3)
    image = gaussian2d(grid, **truth)[np.newaxis]

    def linear(xy, a, b):
        return a * xy[0] + b * xy[1]

    with subtests.test("benchmark"):
        benchmark(analysis.image_fit, image, grid=grid, function=gaussian2d, plot=False)

    with subtests.test("a noiseless Gaussian recovers its own parameters with zero error"):
        result = analysis.image_fit(image, grid=grid, function=gaussian2d, plot=False)
        assert result.shape == (1, 15)
        assert result[0, 0] == pytest.approx(1.0)
        np.testing.assert_allclose(
            result[0, 1:8], [2, -1, 10, 1, 2, 3, 0], atol=1e-6
        )
        np.testing.assert_allclose(result[0, 8:], 0, atol=1e-6)

    with subtests.test("a 2D image is fitted as a stack of one"):
        np.testing.assert_allclose(
            analysis.image_fit(image[0], grid=grid, function=gaussian2d),
            analysis.image_fit(image, grid=grid, function=gaussian2d),
        )

    with subtests.test("grid=None fits on the centered pixel grid"):
        pixels = np.meshgrid(np.arange(31.0) - 15.0, np.arange(31.0) - 15.0)
        result = analysis.image_fit(
            gaussian2d(pixels, x0=3, y0=-4, a=5, c=0, wx=3, wy=3)[np.newaxis],
            function=gaussian2d,
        )
        np.testing.assert_allclose(result[0, 1:4], [3, -4, 5], atol=1e-6)

    with subtests.test("nan pixels are dropped from the fit"):
        punctured = gaussian2d(grid, **truth)
        punctured[10:15, 10:15] = np.nan
        guess = np.array([[1.5, -0.5, 8.0, 0.5, 2.5, 2.5, 0.0]])

        result = analysis.image_fit(
            punctured[np.newaxis], grid=grid, function=gaussian2d, guess=guess
        )

        np.testing.assert_allclose(result[0, 1:8], [2, -1, 10, 1, 2, 3, 0], atol=1e-6)

    with subtests.test("a function with no default guess warns and fits without one"):
        with caplog.at_level(logging.WARNING, logger="slmsuite"):
            caplog.clear()
            result = analysis.image_fit(np.random.rand(1, 20, 20), function=linear, guess=None)
        assert any("not implemented" in record.getMessage() for record in caplog.records)
        assert result.shape == (1, 5)

    with subtests.test("a function with no default guess raises for guess=True"):
        with pytest.raises(NotImplementedError, match="not implemented"):
            analysis.image_fit(np.random.rand(1, 20, 20), function=linear, guess=True)

    with subtests.test("a failed fit returns a nan r2 and the guess parameters"):
        guess = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(analysis, "curve_fit", _raise_runtime_error)
            result = analysis.image_fit(image, grid=grid, function=gaussian2d, guess=guess)

        assert np.isnan(result[0, 0])
        np.testing.assert_array_equal(result[0, 1:8], guess[0])
        assert np.all(np.isnan(result[0, 8:]))

    with subtests.test("plot renders one figure per image, with or without a guess"):
        with _shows() as shown:
            analysis.image_fit(image, grid=grid, function=gaussian2d, plot=True)
            analysis.image_fit(
                (2.0 * grid[0] + 3.0 * grid[1])[np.newaxis],
                grid=grid, function=linear, guess=None, plot=True,
            )
        assert shown == ["image_fit", "image_fit"]


def test_image_zernike_fit(subtests):
    """Test image_zernike_fit() Zernike decomposition."""
    x_small = np.linspace(-1, 1, 64)
    grid_small = np.meshgrid(x_small, x_small)

    with subtests.test("an omitted grid is built from the pixel indices"):
        indices = [1, 2]
        weights = np.array([0.3, -0.4])
        pixels = np.meshgrid(np.arange(64) - 31.5, np.arange(64) - 31.5)
        coeffs = analysis.image_zernike_fit(
            analysis.zernike_sum(pixels, indices, weights), order=indices
        )
        assert np.allclose(coeffs[:, 0], weights, atol=1e-6)

    with subtests.test("exact least-squares recovers a known combination"):
        indices = [1, 2, 3, 4, 5]
        weights = np.array([0.3, -0.4, 0.2, 0.5, -0.1])
        phase_image = analysis.zernike_sum(grid_small, indices, weights)
        coeffs = analysis.image_zernike_fit(phase_image, grid_small, order=indices)
        assert coeffs.shape == (len(indices), 1)
        assert np.allclose(coeffs[:, 0], weights, atol=1e-6)

    with subtests.test("a scalar order fits every mode up to that order, omitting piston"):
        phase_2d = 0.5 * grid_small[0] + 0.3 * grid_small[1]
        coeffs = analysis.image_zernike_fit(phase_2d, grid_small, order=3)
        assert coeffs.shape == ((3 + 1) * (3 + 2) // 2 - 1, 1)

    with subtests.test("leastsquares=False does a per-mode projection"):
        phase_2d = 0.5 * grid_small[0] + 0.3 * grid_small[1]
        coeffs = analysis.image_zernike_fit(
            phase_2d, grid_small, order=3, leastsquares=False, gradient=False
        )
        # The grid's circumscribing aperture has radius sqrt(2), which scales the tilt.
        assert np.allclose(coeffs[:2, 0], np.sqrt(2) * np.array([0.3, 0.5]), atol=1e-6)

    with subtests.test("leastsquares=False is rejected against the gradient basis"):
        with pytest.raises(ValueError):
            analysis.image_zernike_fit(
                0.5 * grid_small[0], grid_small, order=3, leastsquares=False, gradient=True
            )

    with subtests.test("a prebuilt ZernikeBasis is used without rebuilding"):
        indices = [1, 2, 4]
        weights = np.array([0.2, -0.3, 0.4])
        basis = analysis.ZernikeBasis(grid_small, indices)
        phase_image = analysis.zernike_sum(basis, None, weights)
        coeffs = analysis.image_zernike_fit(phase_image, basis)
        assert np.allclose(coeffs[:, 0], weights, atol=1e-6)

    with subtests.test("a stack of images is fitted at once"):
        indices = [1, 2, 4]
        basis = analysis.ZernikeBasis(grid_small, indices)
        weights_stack = np.array([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.1]])
        phase_stack = analysis.zernike_sum(basis, None, weights_stack)
        coeffs = analysis.image_zernike_fit(phase_stack, basis)
        assert coeffs.shape == (len(indices), 2)
        assert np.allclose(coeffs, weights_stack, atol=1e-6)

    with subtests.test("gradient mode recovers weights through phase wraps"):
        indices = [2, 1, 4, 3, 5]
        weights = np.array([1.5, -1.0, 3.0, 0.4, -0.6])
        basis = analysis.ZernikeBasis(grid_small, indices)
        phase_image = analysis.zernike_sum(basis, None, weights)
        wrapped = np.mod(phase_image, 2 * np.pi)
        # The synthesized phase must actually wrap for this test to mean anything.
        assert np.ptp(phase_image[basis.mask.astype(bool)]) > 2 * np.pi

        grad_coeffs = analysis.image_zernike_fit(wrapped, basis, gradient=True)
        plain_coeffs = analysis.image_zernike_fit(wrapped, basis, gradient=False)

        grad_err = np.max(np.abs(grad_coeffs[:, 0] - weights))
        plain_err = np.max(np.abs(plain_coeffs[:, 0] - weights))
        assert grad_err < 1e-2
        assert plain_err > 10 * grad_err

    with subtests.test("gradient mode builds a basis from a raw grid"):
        indices = [2, 1, 4, 3, 5]
        weights = np.array([1.5, -1.0, 3.0, 0.4, -0.6])
        phase_image = analysis.zernike_sum(grid_small, indices, weights)
        wrapped = np.angle(np.exp(1j * phase_image))
        grad_coeffs = analysis.image_zernike_fit(
            wrapped, grid_small, order=indices, gradient=True
        )
        assert np.allclose(grad_coeffs[:, 0], weights, atol=1e-2)


def test_image_vortices(subtests):
    """Test image_vortices() winding number map."""
    grid_x, grid_y = np.meshgrid(np.arange(128.0), np.arange(128.0))

    with subtests.test("a single vortex is the only nonzero winding"):
        winding = analysis.image_vortices(np.arctan2(grid_y - 64, grid_x - 64))
        assert winding.shape == grid_x.shape
        assert np.count_nonzero(winding) == 1
        assert winding[64, 64] == -1

    with subtests.test("a smooth ramp has no winding anywhere"):
        np.testing.assert_array_equal(
            analysis.image_vortices(np.mod(0.1 * grid_x + 0.2 * grid_y, 2 * np.pi)), 0
        )


def test_image_vortices_coordinates(subtests):
    """Test image_vortices_coordinates() vortex location and charge."""
    grid_x, grid_y = np.meshgrid(np.arange(128.0), np.arange(128.0))
    pair = np.arctan2(grid_y - 40, grid_x - 40) - np.arctan2(grid_y - 90, grid_x - 90)

    with subtests.test("a vortex pair is found with opposite unit charges"):
        (rows, cols), weights = analysis.image_vortices_coordinates(pair)
        # The winding is a plaquette sum, so a vortex lands within a pixel of its core.
        np.testing.assert_allclose(rows, [40, 90], atol=1)
        np.testing.assert_allclose(cols, [40, 90], atol=1)
        np.testing.assert_array_equal(weights, [-1, 1])

    with subtests.test("a mask discards the vortices outside it"):
        mask = np.zeros_like(pair, dtype=bool)
        mask[:20, :20] = True
        _, weights = analysis.image_vortices_coordinates(pair, mask=mask)
        assert len(weights) == 0


def test_image_remove_vortices(subtests):
    """Test image_remove_vortices() vortex cancellation."""
    grid_x, grid_y = np.meshgrid(np.arange(128.0), np.arange(128.0))
    phase = np.arctan2(grid_y - 64, grid_x - 64)

    with subtests.test("removal leaves no winding behind"):
        assert np.count_nonzero(analysis.image_vortices(phase)) == 1
        removed = analysis.image_remove_vortices(phase.copy())
        assert np.count_nonzero(analysis.image_vortices(removed)) == 0

    with subtests.test("return_vortices_negative gives the correction that is added in place"):
        correction = analysis.image_remove_vortices(phase.copy(), return_vortices_negative=True)
        np.testing.assert_array_equal(phase + correction, analysis.image_remove_vortices(phase.copy()))

    with subtests.test("a mask restricts removal to the vortices inside it"):
        mask = np.zeros_like(phase, dtype=bool)
        mask[:32, :32] = True
        np.testing.assert_array_equal(
            analysis.image_remove_vortices(phase.copy(), mask=mask), phase
        )
        removed = analysis.image_remove_vortices(
            phase.copy(), mask=np.ones_like(phase, dtype=bool)
        )
        assert np.count_nonzero(analysis.image_vortices(removed)) == 0


def test_image_remove_blaze(subtests):
    """Test image_remove_blaze() global ramp removal."""
    grid_x, grid_y = np.meshgrid(np.arange(96.0), np.arange(96.0))
    ramp = np.mod(0.15 * grid_x + 0.22 * grid_y + 0.5, 2 * np.pi)

    with subtests.test("a pure ramp is flattened to its own offset"):
        np.testing.assert_allclose(analysis.image_remove_blaze(ramp), 0.5, atol=1e-9)

    with subtests.test("a uniform mask matches no mask at all"):
        np.testing.assert_allclose(
            analysis.image_remove_blaze(ramp, mask=np.ones_like(ramp)),
            analysis.image_remove_blaze(ramp),
        )

    with subtests.test("a mask flattens its own region rather than the whole image"):
        mask = np.zeros_like(ramp)
        mask[20:80, 20:80] = 1.0
        mixed = np.where(mask > 0, ramp, np.mod(-0.4 * grid_x + 0.6 * grid_y, 2 * np.pi))
        interior = (slice(25, 75), slice(25, 75))

        masked = analysis.image_remove_blaze(mixed, mask=mask)

        assert np.ptp(masked[interior]) < 0.2 * np.ptp(analysis.image_remove_blaze(mixed)[interior])

    with subtests.test("a stack of phase images is rejected"):
        with pytest.raises(ValueError):
            analysis.image_remove_blaze(ramp[np.newaxis])

    with subtests.test("plot renders the phase, both gradients, and the result"):
        with _shows() as shown:
            analysis.image_remove_blaze(ramp, plot=True)
        assert shown == ["image_remove_blaze"]


def test_image_reduce_wraps(subtests):
    """Test image_reduce_wraps() phase offset optimization."""
    grid_x, grid_y = np.meshgrid(np.arange(128.0), np.arange(128.0))
    # A smooth phase straddling the branch cut, so wrapping it costs a whole cut line.
    smooth = 0.5 * np.sin(grid_x / 20) + 0.5 * np.cos(grid_y / 25)
    phase = np.mod(smooth, 2 * np.pi)

    def wraps(image):
        gradient = np.abs(np.gradient(image, axis=1)) + np.abs(np.gradient(image, axis=0))
        return np.count_nonzero(gradient > np.pi)

    with subtests.test("the branch cut through a smooth phase is offset away"):
        assert wraps(phase) > 0
        reduced = analysis.image_reduce_wraps(phase, steps=24)
        assert wraps(reduced) == 0

    with subtests.test("the result is a global offset of the input, inside [0, 2pi)"):
        reduced = analysis.image_reduce_wraps(phase, steps=24)
        offset = np.angle(np.exp(1j * (reduced - phase)))
        np.testing.assert_allclose(offset, offset.flat[0])
        assert np.nanmin(reduced) >= 0 and np.nanmax(reduced) <= 2 * np.pi

    with subtests.test("a mask weights which wraps matter, still by a global offset"):
        mask = np.zeros_like(phase)
        mask[32:96, 32:96] = 1.0
        reduced = analysis.image_reduce_wraps(phase, mask=mask, steps=12)
        offset = np.angle(np.exp(1j * (reduced - phase)))
        np.testing.assert_allclose(offset, offset.flat[0])


def test_fit_affine(subtests):
    """Test fit_affine() affine transformation fitting."""
    rng = np.random.default_rng(42)
    x = rng.uniform(-5, 5, size=(2, 50))
    theta = np.pi / 6
    cases = {
        "identity": (np.eye(2), np.zeros((2, 1))),
        "translation": (np.eye(2), np.array([[3.0], [-7.0]])),
        "scaling": (np.diag([2.0, 0.5]), np.zeros((2, 1))),
        "rotation": (
            np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
            np.zeros((2, 1)),
        ),
        "shear with offset": (np.array([[1.5, -0.3], [0.4, 2.0]]), np.array([[10.0], [-5.0]])),
    }

    for (name, (M, b)) in cases.items():
        with subtests.test(f"a noiseless {name} is recovered"):
            result = analysis.fit_affine(x, M @ x + b)
            assert set(result.keys()) == {"M", "b"}
            np.testing.assert_allclose(result["M"], M, atol=1e-3)
            np.testing.assert_allclose(result["b"], b, atol=1e-3)

    with subtests.test("a guess is refined rather than returned"):
        b = np.array([[2.0], [3.0]])
        guess = {"M": np.eye(2), "b": np.array([[1.0], [1.0]])}
        result = analysis.fit_affine(x, x + b, guess_affine=guess)
        np.testing.assert_allclose(result["M"], np.eye(2), atol=1e-3)
        np.testing.assert_allclose(result["b"], b, atol=1e-3)

    with subtests.test("noise perturbs the fit by less than its own scale"):
        M = np.array([[1.2, -0.1], [0.3, 0.9]])
        b = np.array([[1.0], [-2.0]])
        points = rng.uniform(-10, 10, size=(2, 200))
        result = analysis.fit_affine(points, M @ points + b + rng.normal(0, 0.05, size=(2, 200)))
        np.testing.assert_allclose(result["M"], M, atol=0.05)
        np.testing.assert_allclose(result["b"], b, atol=0.1)

    with subtests.test("nested lists are accepted"):
        points = [[1, 2, 3], [4, 5, 6]]
        result = analysis.fit_affine(points, [[2, 4, 6], [8, 10, 12]])
        np.testing.assert_allclose(
            result["M"] @ np.array(points) + result["b"], [[2, 4, 6], [8, 10, 12]], atol=1e-3
        )

    with subtests.test("an incomplete guess raises"):
        with pytest.raises(ValueError, match="guess_affine must be a dictionary"):
            analysis.fit_affine(x, x, guess_affine="bad")
        with pytest.raises(ValueError, match="guess_affine must be a dictionary"):
            analysis.fit_affine(x, x, guess_affine={"M": np.eye(2)})

    with subtests.test("an all-nan coordinate raises"):
        nans = np.vstack((np.full((1, 5), np.nan), rng.uniform(-1, 1, size=(1, 5))))
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            with pytest.raises(ValueError, match="all-nan"):
                analysis.fit_affine(nans, rng.uniform(-1, 1, size=(2, 5)))

    with subtests.test("mismatched point counts raise a ValueError, not an assertion"):
        # An assert vanishes under python -O, which would return the guess as a fit.
        with pytest.raises(ValueError):
            analysis.fit_affine(x[:, :5], x[:, :6])

    with subtests.test("a failed optimization falls back to the guess"):
        guess = {"M": np.array([[7.0, 8.0], [9.0, 10.0]]), "b": np.array([[11.0], [12.0]])}
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(analysis, "minimize", _raise_runtime_error)
            result = analysis.fit_affine(x, 2 * x, guess_affine=guess)

        np.testing.assert_array_equal(result["M"], guess["M"])
        np.testing.assert_array_equal(result["b"], guess["b"])

    with subtests.test("plot renders the fit"):
        with _shows() as shown:
            analysis.fit_affine(x, x + np.array([[0.5], [0.25]]), plot=True)
        assert shown == ["fit_affine"]


def _spot_array(lattice, count=(9, 9), shape=(160, 160), center=None, spot=1.2):
    """Render a spot array with the given (2, 2) lattice for lattice-detection tests."""
    if center is None:
        center = (shape[1] / 2, shape[0] / 2)
    (gx, gy) = np.meshgrid(
        np.arange(count[0]) - (count[0] - 1) / 2,
        np.arange(count[1]) - (count[1] - 1) / 2,
    )
    points = np.array(lattice) @ np.vstack((gx.ravel(), gy.ravel()))

    (yy, xx) = np.indices(shape)
    image = np.zeros(shape)
    for (px, py) in zip(points[0] + center[0], points[1] + center[1]):
        image += np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * spot**2))
    return image


def test_image_lattice_detect(subtests):
    """Test analysis.image_lattice_detect() over pitches, rotation, and shear."""
    # The lattice is only defined up to relabeling of its two vectors.
    dihedral = [
        np.diag([sx, sy]) @ permutation
        for permutation in (np.eye(2), np.array([[0.0, 1.0], [1.0, 0.0]]))
        for sx in (1, -1) for sy in (1, -1)
    ]

    def error(detected, truth):
        return min(
            np.linalg.norm(detected @ relabel - truth) / np.linalg.norm(truth)
            for relabel in dihedral
        )

    theta = np.radians(20)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    lattices = {
        "square": np.array([[8.0, 0.0], [0.0, 8.0]]),
        "coarse": np.array([[24.0, 0.0], [0.0, 24.0]]),
        "anisotropic": np.array([[18.0, 0.0], [0.0, 6.0]]),
        "rotated": rotation @ np.array([[10.0, 0.0], [0.0, 10.0]]),
        "sheared": np.array([[11.0, 2.0], [-1.5, 9.0]]),
    }

    # Past this pitch a lattice's reciprocal peaks collapse into the 0th order Fourier filters.
    fourier_limit = 10.0

    for (name, lattice) in lattices.items():
        for method in ("autocorrelation", "fourier"):
            if method == "fourier" and np.max(np.linalg.norm(lattice, axis=0)) > fourier_limit:
                continue
            with subtests.test(f"{name} ({method})"):
                detected = analysis.image_lattice_detect(_spot_array(lattice), method=method)
                assert error(detected, lattice) < 0.05, (
                    f"{name} ({method}): detected\n{detected}\nexpected\n{lattice}"
                )

    with subtests.test("survives cropping to a corner of the array"):
        # Cropping to a few periods weakens the autocorrelation peaks but does not move them.
        image = _spot_array(lattices["coarse"], count=(15, 15))[:130, :100]
        detected = analysis.image_lattice_detect(image, method="autocorrelation")
        assert error(detected, lattices["coarse"]) < 0.05

    with subtests.test("unknown method raises"):
        with pytest.raises(ValueError):
            analysis.image_lattice_detect(np.zeros((64, 64)), method="bogus")

    for method in ("autocorrelation", "fourier"):
        with subtests.test(f"blank image raises ({method})"):
            with pytest.raises(RuntimeError):
                analysis.image_lattice_detect(np.zeros((160, 160)), method=method)

    with subtests.test("single row of spots raises"):
        # The autocorrelation refuses rather than inventing a second lattice vector.
        with pytest.raises(RuntimeError):
            analysis.image_lattice_detect(
                _spot_array(lattices["square"], count=(9, 1)), method="autocorrelation"
            )


def test_make_8bit(subtests):
    """Test _make_8bit() conversion and scaling behavior."""
    with subtests.test("the dynamic range is stretched onto 0-255"):
        converted = analysis._make_8bit(np.array([[10.0, 20.0], [30.0, 40.0]]))
        assert converted.dtype == np.uint8
        np.testing.assert_array_equal(converted, [[0, 85], [170, 255]])

    with subtests.test("a constant image becomes zeros"):
        converted = analysis._make_8bit(np.full((4, 4), 7.3))
        assert converted.dtype == np.uint8
        np.testing.assert_array_equal(converted, 0)


def test_score_array_orientation(subtests):
    """Test _score_array_orientation()'s recovery of an array's orientation."""
    array_shape = (5, 5)
    M = np.array([[22.0, 0.0], [0.0, 22.0]])
    b = np.array([[150.0], [150.0]])
    codes = list(analysis.OrientationTransform.D_4)
    centers = analysis._array_indices(array_shape)

    def placement(code, shape=array_shape):
        return np.matmul(
            M,
            np.matmul(
                analysis.OrientationTransform.from_code(code).M(),
                analysis._array_indices(shape),
            ),
        ) + b

    def render(code, withhold=True, dark=(), sigma=0, blank=(), shape=array_shape):
        """Image of the array under ``code``, optionally dimming or keeping the fiducials."""
        placed = placement(code, shape)
        count = placed.shape[1] - (2 if withhold else 0)
        image = np.zeros((300, 300))
        for i in range(count):
            if i in blank:
                continue
            (x, y) = np.rint(placed[:, i]).astype(int)
            image[y - 1 : y + 2, x - 1 : x + 2] = 20.0 if i in dark else 1000.0
        if sigma:
            image = image + np.random.default_rng(0).normal(0, sigma, image.shape)
        return analysis.image_remove_field(image, deviations=None)

    with subtests.test("every orientation is recovered from its withheld pair"):
        for code in codes:
            best = analysis._score_array_orientation(render(code), M, b, array_shape, 5)
            assert best is not None and best[0] == code

    with subtests.test("a withheld pair reads dark, so the orientation is verified"):
        for sigma in (0, 1, 5):
            best = analysis._score_array_orientation(
                render(codes[0], sigma=sigma), M, b, array_shape, 5
            )
            assert best is not None and best[0] == codes[0] and best[2] < 0.5

    with subtests.test("a full array leaves no pair dark, so no orientation is verified"):
        for sigma in (0, 1, 5):
            best = analysis._score_array_orientation(
                render(codes[0], withhold=False, sigma=sigma), M, b, array_shape, 5
            )
            assert best is not None and best[2] >= 0.5

    with subtests.test("dimmed spots do not prevent recovery"):
        for code in codes:
            best = analysis._score_array_orientation(
                render(code, dark=(0, 4, 9)), M, b, array_shape, 5
            )
            assert best is not None and best[0] == code

    with subtests.test("a non-square array is placed by how many spots land lit"):
        # A rotation carries a non-square array off its own lattice, darkening its spots.
        wide = (6, 4)
        for code in codes:
            best = analysis._score_array_orientation(
                render(code, shape=wide), M, b, wide, 5
            )
            assert best is not None and best[0] == code

    with subtests.test("an orientation is picked even when a rival's pair is dark too"):
        # Blank the sites the flipped orientation withholds too, so both read dark.
        identity = placement(codes[0])
        flipped = placement(analysis.OrientationTransform.D_4.FLIP)
        blank = {
            int(np.argmin(np.linalg.norm(identity - site[:, np.newaxis], axis=0)))
            for site in flipped.T[-2:]
        }
        best = analysis._score_array_orientation(
            render(codes[0], blank=blank), M, b, array_shape, 5
        )
        assert best is not None and best[0] in (codes[0], analysis.OrientationTransform.D_4.FLIP)


def test_get_orientation_transformation(subtests):
    """Test get_orientation_transformation() composition of rotate/flip operations."""
    image = np.arange(9).reshape(3, 3)

    with subtests.test("no arguments is the identity"):
        np.testing.assert_array_equal(analysis.get_orientation_transformation()(image), image)

    with subtests.test("fliplr and flipud mirror the image"):
        np.testing.assert_array_equal(
            analysis.get_orientation_transformation(fliplr=True)(image), np.fliplr(image)
        )
        np.testing.assert_array_equal(
            analysis.get_orientation_transformation(flipud=True)(image), np.flipud(image)
        )

    with subtests.test("rot names and rot90 step counts agree"):
        for (name, steps) in (("90", 1), ("180", 2), ("270", 3)):
            np.testing.assert_array_equal(
                analysis.get_orientation_transformation(rot=name)(image), np.rot90(image, steps)
            )
            np.testing.assert_array_equal(
                analysis.get_orientation_transformation(rot=steps)(image), np.rot90(image, steps)
            )

    with subtests.test("flips are applied after the rotation, for every combination"):
        # Non-square, so an orientation that swaps the axes cannot hide.
        wide = np.arange(40).reshape(5, 8)
        for (rot, steps) in (("0", 0), ("90", 1), ("180", 2), ("270", 3)):
            for fliplr in (False, True):
                for flipud in (False, True):
                    expected = np.rot90(wide, steps)
                    if flipud:
                        expected = np.flipud(expected)
                    if fliplr:
                        expected = np.fliplr(expected)
                    np.testing.assert_array_equal(
                        analysis.get_orientation_transformation(rot, fliplr, flipud)(wide),
                        expected,
                    )


class TestAffine:
    """Test the Affine transformation class."""

    def test_matmul(self, subtests):
        """Affine @ x applies y = M(x - a) + b, and Affine @ Affine composes."""
        M = np.array([[2.0, 0.0], [0.0, 3.0]])
        b = np.array([1.0, -1.0])
        affine = analysis.Affine(M, b)

        with subtests.test("a stack of column vectors is transformed at once"):
            x = np.array([[1.0, 4.0], [3.0, 5.0]])
            np.testing.assert_allclose(affine @ x, M @ x + b[:, np.newaxis])

        with subtests.test("a shifts the origin of the transformation"):
            a = np.array([1.0, 2.0])
            x = np.array([[5.0], [6.0]])
            np.testing.assert_allclose(
                analysis.Affine(M, b, a) @ x, M @ (x - a[:, np.newaxis]) + b[:, np.newaxis]
            )

        with subtests.test("composition applies the right transformation first"):
            other = analysis.Affine(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0.5, 0.5]))
            x = np.array([[3.0], [7.0]])
            composed = affine @ other
            assert isinstance(composed, analysis.Affine)
            np.testing.assert_allclose(composed @ x, affine @ (other @ x))

        with subtests.test("an unsupported operand raises"):
            with pytest.raises(TypeError):
                affine @ "not a vector"

    def test_inv(self, subtests):
        """Affine.inv is a property that undoes the transformation."""
        affine = analysis.Affine(np.array([[3.0, 1.0], [0.0, 2.0]]), np.array([4.0, -2.0]))
        x = np.array([[5.0], [3.0]])

        with subtests.test("inv is a property returning an Affine"):
            assert isinstance(affine.inv, analysis.Affine)

        with subtests.test("the inverse round-trips in both directions"):
            np.testing.assert_allclose(affine.inv @ (affine @ x), x, atol=1e-10)
            np.testing.assert_allclose(affine @ (affine.inv @ x), x, atol=1e-10)

    def test_to_dict(self, subtests):
        """Affine.to_dict serializes to an equivalent Affine."""
        affine = analysis.Affine(
            np.array([[2.0, 1.0], [0.0, 3.0]]), np.array([5.0, -3.0]), np.array([1.0, 2.0])
        )
        serialized = affine.to_dict()

        with subtests.test("a is baked into b, leaving no residual origin"):
            np.testing.assert_allclose(serialized["a"], 0)

        with subtests.test("the reconstructed Affine transforms identically"):
            x = np.array([[4.0], [6.0]])
            reconstructed = analysis.Affine(serialized["M"], serialized["b"])
            np.testing.assert_allclose(reconstructed @ x, affine @ x, atol=1e-12)


class TestOrientationTransform:
    """Test the OrientationTransform D4 group of image orientations."""

    def test_M(self, subtests):
        """M() is an isometry for every element of the group."""
        for code in analysis.OrientationTransform.D_4:
            M = analysis.OrientationTransform.from_code(code).M()
            with subtests.test(f"{code.name} is orthogonal with unit determinant"):
                np.testing.assert_allclose(M.T @ M, np.eye(2), atol=1e-10)
                assert abs(np.linalg.det(M)) == pytest.approx(1.0)

    def test_affine(self, subtests):
        """affine(shape) sends each pixel where __call__ moves it."""
        (h, w) = (4, 6)  # Non-square, to expose an axis swap.
        image = np.arange(h * w, dtype=float).reshape(h, w)
        (xs, ys) = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
        source = np.vstack((xs.ravel(), ys.ravel()))

        for code in analysis.OrientationTransform.D_4:
            transform = analysis.OrientationTransform.from_code(code)
            transformed = transform(image)
            destination = transform.affine((h, w)) @ source

            with subtests.test(f"{code.name} maps every pixel onto its transformed position"):
                np.testing.assert_array_equal(destination, np.round(destination))
                (x_out, y_out) = np.round(destination).astype(int)
                assert transformed.shape == transform.transform_shape((h, w))
                assert (x_out.min(), y_out.min()) == (0, 0)
                assert (x_out.max(), y_out.max()) == (transformed.shape[1] - 1, transformed.shape[0] - 1)
                np.testing.assert_array_equal(transformed[y_out, x_out], image.ravel())


@pytest.mark.gpu
def test_take_gpu(benchmark, has_cupy):
    """GPU variant of take() using cupy arrays."""
    import cupy as cp

    rng = np.random.default_rng(42)
    image = cp.array(rng.random((512, 512)).astype(np.float32))
    vectors = np.stack([rng.integers(20, 492, 50), rng.integers(20, 492, 50)])

    result = benchmark(analysis.take, image, vectors=vectors, size=20, centered=True, xp=cp)
    assert result.shape == (50, 20, 20)
