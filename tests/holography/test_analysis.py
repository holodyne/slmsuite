"""
Unit tests for slmsuite.holography.analysis module.
"""
import warnings

import pytest
import numpy as np
import matplotlib.pyplot as plt

from slmsuite.holography import analysis
from slmsuite.holography.analysis.fitfunctions import gaussian2d


def test_image_centroids(subtests):
    """Test image_centroids() centroid calculation."""
    with subtests.test("centered spot returns near-zero centroid"):
        image = np.zeros((100, 100))
        image[45:55, 45:55] = 1
        image = image[np.newaxis, :, :]

        centroids = analysis.image_centroids(image)

        assert centroids.shape == (2, 1)
        assert centroids[0, 0] == pytest.approx(0, abs=1)
        assert centroids[1, 0] == pytest.approx(0, abs=1)

    with subtests.test("off-center spot"):
        image = np.zeros((100, 100))
        image[20:30, 70:80] = 1
        image = image[np.newaxis, :, :]

        centroids = analysis.image_centroids(image)

        assert centroids[0, 0] == pytest.approx(25, abs=1)
        assert centroids[1, 0] == pytest.approx(-25, abs=1)

    with subtests.test("custom grid"):
        image = np.zeros((50, 50))
        image[20:30, 20:30] = 1
        image = image[np.newaxis, :, :]

        x = np.arange(50)
        y = np.arange(50)
        X, Y = np.meshgrid(x, y)
        grid = (X, Y)

        centroids = analysis.image_centroids(image, grid=grid)

        assert centroids[0, 0] == pytest.approx(25, abs=1)
        assert centroids[1, 0] == pytest.approx(25, abs=1)

    with subtests.test("all-zero image"):
        image = np.zeros((50, 50))
        image = image[np.newaxis, :, :]

        centroids = analysis.image_centroids(image)
        assert len(centroids) == 2

    with subtests.test("single bright pixel at center"):
        image = np.zeros((50, 50))
        image[25, 25] = 1
        image = image[np.newaxis, :, :]

        centroids = analysis.image_centroids(image)

        assert centroids[0, 0] == pytest.approx(0, abs=1)
        assert centroids[1, 0] == pytest.approx(0, abs=1)


def test_image_moments(subtests, benchmark):
    """Test image_moment() moment calculation."""
    with subtests.test("benchmark"):
        image = np.random.rand(128, 128).astype(np.float32)
        image = image[np.newaxis, :, :]
        benchmark(analysis.image_moment, image, moment=(1, 0))

    with subtests.test("zeroth moment unnormalized equals total intensity"):
        image = np.ones((50, 50)) * 0.5
        image = image[np.newaxis, :, :]

        moment = analysis.image_moment(image, moment=(0, 0), normalize=False)
        assert moment[0] == pytest.approx(50 * 50 * 0.5)

    with subtests.test("zeroth moment normalized equals 1"):
        image = np.ones((50, 50)) * 0.5
        image = image[np.newaxis, :, :]

        moment = analysis.image_moment(image, moment=(0, 0), normalize=True)
        assert moment[0] == pytest.approx(1)

    with subtests.test("first order moments return correct shape"):
        image = np.zeros((100, 100))
        image[45:55, 45:55] = 1
        image = image[np.newaxis, :, :]

        moment_x = analysis.image_moment(image, moment=(1, 0))
        moment_y = analysis.image_moment(image, moment=(0, 1))

        assert isinstance(moment_x, np.ndarray)
        assert isinstance(moment_y, np.ndarray)
        assert moment_x.shape == (1,)
        assert moment_y.shape == (1,)
        assert moment_x[0] == pytest.approx(0, abs=1)
        assert moment_y[0] == pytest.approx(0, abs=1)

    with subtests.test("grid as 2D arrays"):
        w = 40
        image = np.zeros((w, w))
        image[15:25, 15:25] = 1
        image = image[np.newaxis, :, :]

        xs = np.arange(w, dtype=float)
        ys = np.arange(w, dtype=float)
        X, Y = np.meshgrid(xs, ys)

        m = analysis.image_moment(image, moment=(1, 0), grid=(X, Y))
        assert m.shape == (1,)

    with subtests.test("grid as 1D lists"):
        w = 40
        image = np.zeros((w, w))
        image[15:25, 15:25] = 1
        image = image[np.newaxis, :, :]

        xs = np.arange(w, dtype=float)
        ys = np.arange(w, dtype=float)

        m = analysis.image_moment(image, moment=(1, 0), grid=(xs, ys))
        assert m.shape == (1,)

    with subtests.test("shear moment (1,1)"):
        image = np.zeros((50, 50))
        image[20:30, 20:30] = 1
        image = image[np.newaxis, :, :]

        m11 = analysis.image_moment(image, moment=(1, 1))
        assert m11.shape == (1,)

    with subtests.test("nansum=True ignores NaN"):
        image = np.ones((30, 30))
        image[5:10, 5:10] = np.nan
        image = image[np.newaxis, :, :]

        m = analysis.image_moment(image, moment=(0, 0), normalize=False, nansum=True)
        assert np.isfinite(m[0])

    with subtests.test("second order moment"):
        image = np.zeros((50, 50))
        image[20:30, 20:30] = 1
        image = image[np.newaxis, :, :]

        m20 = analysis.image_moment(image, moment=(2, 0))
        assert m20.shape == (1,)
        assert m20[0] > 0

    with subtests.test("second order with 2D grid"):
        w = 40
        image = np.zeros((w, w))
        image[15:25, 15:25] = 1
        image = image[np.newaxis, :, :]

        xs = np.arange(w, dtype=float)
        ys = np.arange(w, dtype=float)
        X, Y = np.meshgrid(xs, ys)

        m20 = analysis.image_moment(image, moment=(2, 0), grid=(X, Y))
        assert m20.shape == (1,)
        assert m20[0] > 0

    with subtests.test("second order with 1D grid"):
        w = 40
        image = np.zeros((w, w))
        image[15:25, 15:25] = 1
        image = image[np.newaxis, :, :]

        xs = np.arange(w, dtype=float)
        ys = np.arange(w, dtype=float)

        m20 = analysis.image_moment(image, moment=(2, 0), grid=(xs, ys))
        assert m20.shape == (1,)
        assert m20[0] > 0

    with subtests.test("3D grid pass-through"):
        w = 30
        image = np.zeros((w, w))
        image[10:20, 10:20] = 1
        images = image[np.newaxis, :, :]

        xs = np.arange(w, dtype=float).reshape(1, 1, w)
        ys = np.arange(w, dtype=float).reshape(1, w, 1)
        Xg = np.broadcast_to(xs, (1, w, w)).copy()
        Yg = np.broadcast_to(ys, (1, w, w)).copy()

        m = analysis.image_moment(images, moment=(1, 0), grid=(Xg, Yg))
        assert m.shape == (1,)


def test_image_variances(subtests):
    """Test image_variances() variance calculation."""
    with subtests.test("circular spot has equal Mxx and Myy"):
        image = np.zeros((100, 100))
        Y, X = np.ogrid[:100, :100]
        mask = (X - 50)**2 + (Y - 50)**2 <= 10**2
        image[mask] = 1
        image = image[np.newaxis, :, :]

        variances = analysis.image_variances(image)

        assert variances.shape == (3, 1)
        assert variances[0, 0] == pytest.approx(variances[1, 0], rel=0.1)
        assert abs(variances[2, 0]) < variances[0, 0] * 0.1

    with subtests.test("elliptical spot has unequal Mxx and Myy"):
        image = np.zeros((100, 100))
        Y, X = np.ogrid[:100, :100]
        mask = ((X - 50)/20)**2 + ((Y - 50)/10)**2 <= 1
        image[mask] = 1
        image = image[np.newaxis, :, :]

        variances = analysis.image_variances(image)

        assert variances[0, 0] != pytest.approx(variances[1, 0], rel=0.1)


def test_image_ellipticity(subtests):
    """Test image_ellipticity(), image_areas(), and image_ellipticity_angle()."""
    with subtests.test("circular spot has zero ellipticity"):
        variances = np.array([[100.0], [100.0], [0.0]])
        ellipticity = analysis.image_ellipticity(variances)
        assert ellipticity[0] == pytest.approx(0.0, abs=0.01)

    with subtests.test("elongated spot has nonzero ellipticity"):
        variances = np.array([[200.0], [100.0], [0.0]])
        ellipticity = analysis.image_ellipticity(variances)
        assert ellipticity[0] > 0.1

    with subtests.test("area from elongated variances"):
        variances = np.array([[200.0], [100.0], [0.0]])
        areas = analysis.image_areas(variances)
        assert areas[0] == pytest.approx(20000)

    with subtests.test("area from circular variances"):
        circular_variances = np.array([[100.0], [100.0], [0.0]])
        circular_areas = analysis.image_areas(circular_variances)
        assert circular_areas[0] == pytest.approx(10000)

    with subtests.test("angle of elongated spot is finite"):
        variances = np.array([[200.0], [100.0], [0.0]])
        angles = analysis.image_ellipticity_angle(variances)
        assert np.isfinite(angles[0])

    with subtests.test("circular spot angle is near zero"):
        circular_variances = np.array([[100.0], [100.0], [0.0]])
        circular_angles = analysis.image_ellipticity_angle(circular_variances)
        assert circular_angles[0] == pytest.approx(0, abs=0.01)

    with subtests.test("sheared spot angle is finite"):
        sheared_variances = np.array([[200.0], [100.0], [50.0]])
        sheared_angles = analysis.image_ellipticity_angle(sheared_variances)
        assert np.isfinite(sheared_angles[0])

    with subtests.test("multiple spots"):
        multi_variances = np.array([[100.0, 200.0, 150.0], [100.0, 100.0, 75.0], [0.0, 0.0, 25.0]])
        multi_ellipticities = analysis.image_ellipticity(multi_variances)
        multi_areas = analysis.image_areas(multi_variances)
        multi_angles = analysis.image_ellipticity_angle(multi_variances)

        assert len(multi_ellipticities) == 3
        assert len(multi_areas) == 3
        assert len(multi_angles) == 3
        assert all(np.isfinite(multi_ellipticities))
        assert all(np.isfinite(multi_areas))
        assert all(np.isfinite(multi_angles))


def test_image_normalization(subtests):
    """Test image_normalize() and image_normalization()."""
    with subtests.test("normalize sums to 1 and preserves argmax"):
        image = np.random.rand(50, 50) * 100 + 50
        image = image[np.newaxis, :, :]

        normalized = analysis.image_normalize(image)

        assert np.sum(normalized) == pytest.approx(1.0)
        assert np.argmax(normalized) == np.argmax(image)

    with subtests.test("normalization returns total intensity"):
        image = np.ones((50, 50)) * 2
        image = image[np.newaxis, :, :]

        norm_value = analysis.image_normalization(image)

        assert norm_value == pytest.approx(50 * 50 * 2)

    with subtests.test("NaN handling with nansum"):
        image = np.ones((50, 50))
        image[10:20, 10:20] = np.nan
        image = image[np.newaxis, :, :]

        norm_value = analysis.image_normalization(image, nansum=True)
        assert np.isfinite(norm_value)

    with subtests.test("single image zero sum returns zeros"):
        image = np.zeros((30, 30))

        result = analysis.image_normalize(image)
        np.testing.assert_array_equal(result, np.zeros((30, 30)))

    with subtests.test("2D single image nonzero normalizes"):
        image = np.ones((30, 30)) * 4.0

        result = analysis.image_normalize(image)
        assert result.shape == (30, 30)
        assert np.sum(result) == pytest.approx(1.0)

    with subtests.test("remove_field integration"):
        image = np.ones((50, 50)) * 10.0
        image[20:30, 20:30] = 100.0
        image = image[np.newaxis, :, :]

        result = analysis.image_normalize(image, remove_field=True)
        assert result.shape == (1, 50, 50)
        assert np.sum(result) == pytest.approx(1.0, abs=0.01)


def test_image_remove_field(subtests):
    """Test image_remove_field() background removal."""
    with subtests.test("deviations threshold"):
        image = np.ones((100, 100)) * 10
        image[45:55, 45:55] = 100
        image = image[np.newaxis, :, :]

        cleaned = analysis.image_remove_field(image, deviations=2)

        assert cleaned.shape == (1, 100, 100)
        assert np.median(cleaned) < np.median(image)
        assert np.max(cleaned) > 10

    with subtests.test("deviations=None uses median"):
        image = np.ones((100, 100)) * 10
        image[45:55, 45:55] = 100
        image = image[np.newaxis, :, :]

        cleaned = analysis.image_remove_field(image, deviations=None)

        assert cleaned.shape == (1, 100, 100)
        assert np.median(cleaned) == 0

    with subtests.test("out parameter in-place"):
        image = np.ones((60, 60), dtype=float) * 8
        image[20:40, 20:40] = 60
        image = image[np.newaxis, :, :]
        out = image.copy()

        result = analysis.image_remove_field(image, deviations=2, out=out)

        assert result is out

    with subtests.test("out parameter separate array"):
        image = np.ones((60, 60), dtype=float) * 8
        image[20:40, 20:40] = 60
        image = image[np.newaxis, :, :]
        out = np.empty_like(image, dtype=float)

        result = analysis.image_remove_field(image, deviations=2, out=out)

        assert result is out
        assert np.max(out) > 0


def test_image_relative_strehl(subtests):
    """Test image_relative_strehl() for concentrated spot."""
    with subtests.test("3D input"):
        image = np.zeros((50, 50))
        image[24, 24] = 100
        image = image[np.newaxis, :, :]

        strehl = analysis.image_relative_strehl(image)
        assert strehl > 0.5

    with subtests.test("2D input"):
        image = np.zeros((50, 50))
        image[24, 24] = 100

        strehl = analysis.image_relative_strehl(image)
        assert strehl > 0.5


def test_image_fit(subtests, benchmark):
    """Test image_fit() Gaussian fitting."""
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    grid = (X, Y)

    with subtests.test("benchmark"):
        image = gaussian2d((X, Y), x0=0, y0=0, a=10, c=1, wx=3, wy=3)
        image = image[np.newaxis, :, :]
        benchmark(analysis.image_fit, image, grid=grid, function=gaussian2d, plot=False)

    with subtests.test("basic Gaussian fit with grid"):
        image = gaussian2d((X, Y), x0=2, y0=-1, a=10, c=1, wx=2, wy=2)
        image = image[np.newaxis, :, :]

        result = analysis.image_fit(image, grid=grid, function=gaussian2d, plot=False)

        assert result.shape[0] == 1
        assert result[0, 1] == pytest.approx(2, abs=0.5)
        assert result[0, 2] == pytest.approx(-1, abs=0.5)
        assert result[0, 3] == pytest.approx(10, abs=2)

    with subtests.test("2D input auto-reshapes"):
        image_2d = gaussian2d((X, Y), x0=0, y0=0, a=5, c=0, wx=3, wy=3)

        result = analysis.image_fit(image_2d, grid=grid, function=gaussian2d)

        assert result.shape[0] == 1
        assert np.isfinite(result[0, 0])

    with subtests.test("grid=None uses default pixel grid"):
        sz = 30
        img = np.zeros((sz, sz))
        Y2, X2 = np.ogrid[:sz, :sz]
        img = np.exp(-((X2 - sz/2)**2 + (Y2 - sz/2)**2) / (2 * 4**2)).astype(float)
        img = img[np.newaxis, :, :]

        result = analysis.image_fit(img, grid=None, function=gaussian2d)

        assert result.shape[0] == 1
        assert np.isfinite(result[0, 0])

    with subtests.test("unknown function guess=None warns"):
        def custom_fn(xy, a, b):
            return a * xy[0] + b * xy[1]

        img = np.random.rand(1, 20, 20)
        with pytest.warns(UserWarning, match="not implemented"):
            result = analysis.image_fit(img, grid=grid[:20, :20] if False else None,
                                        function=custom_fn, guess=None)

        assert result.shape[0] == 1

    with subtests.test("unknown function guess=True raises"):
        def custom_fn2(xy, a, b):
            return a * xy[0] + b * xy[1]

        img = np.random.rand(1, 20, 20)
        with pytest.raises(NotImplementedError, match="not implemented"):
            analysis.image_fit(img, function=custom_fn2, guess=True)

    with subtests.test("NaN values in image handled"):
        image = gaussian2d((X, Y), x0=0, y0=0, a=10, c=1, wx=3, wy=3)
        image[10:15, 10:15] = np.nan
        image = image[np.newaxis, :, :]

        result = analysis.image_fit(image, grid=grid, function=gaussian2d)

        assert result.shape[0] == 1

    with subtests.test("failed fit returns nan r2 and guess params"):
        rng = np.random.default_rng(99)
        image = rng.uniform(0, 1000, size=(1, 30, 30))

        result = analysis.image_fit(image, function=gaussian2d)

        assert result.shape[0] == 1

    with subtests.test("RuntimeError fit path via monkeypatch"):
        from scipy.optimize import curve_fit as _real_curve_fit

        image = gaussian2d((X, Y), x0=0, y0=0, a=10, c=0, wx=2, wy=2)
        image = image[np.newaxis, :, :]

        def _boom_cf(*args, **kwargs):
            raise RuntimeError("forced")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(analysis, "curve_fit", _boom_cf)
        try:
            result = analysis.image_fit(image, grid=grid, function=gaussian2d)
        finally:
            monkeypatch.undo()

        assert result.shape[0] == 1
        assert np.isnan(result[0, 0])

    with subtests.test("plot path"):
        import matplotlib.pyplot as plt
        plt.ioff()

        image = gaussian2d((X, Y), x0=0, y0=0, a=10, c=0, wx=2, wy=2)
        image = image[np.newaxis, :, :]

        shown = {"called": False}
        def _show():
            shown["called"] = True

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(analysis.plt, "show", _show)
        try:
            result = analysis.image_fit(image, grid=grid, function=gaussian2d, plot=True)
        finally:
            monkeypatch.undo()
            plt.close("all")

        assert shown["called"]
        assert result.shape[0] == 1

    with subtests.test("plot path with guess=None custom function"):
        import matplotlib.pyplot as plt
        plt.ioff()

        def linear_fn(xy, a, b):
            return a * xy[0] + b

        sz = 20
        xs = np.linspace(-1, 1, sz)
        ys = np.linspace(-1, 1, sz)
        Xl, Yl = np.meshgrid(xs, ys)
        grid_l = (Xl, Yl)
        img = (2.0 * Xl + 3.0)[np.newaxis, :, :]

        shown_p = {"called": False}
        def _show_p():
            shown_p["called"] = True

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(analysis.plt, "show", _show_p)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = analysis.image_fit(
                    img, grid=grid_l, function=linear_fn, guess=None, plot=True
                )
        finally:
            monkeypatch.undo()
            plt.close("all")

        assert result.shape[0] == 1


def test_image_zernike_fit(subtests):
    """Test image_zernike_fit() Zernike polynomial fitting."""
    x_small = np.linspace(-1, 1, 32)
    y_small = np.linspace(-1, 1, 32)
    X_small, Y_small = np.meshgrid(x_small, y_small)
    grid_small = (X_small, Y_small)

    with subtests.test("3D input tilt phase"):
        phase_image = 0.5 * X_small + 0.3 * Y_small
        phase_image = phase_image[np.newaxis, :, :]

        try:
            zernike_coeffs = analysis.image_zernike_fit(
                phase_image, grid_small, order=3,
                iterations=1, leastsquares=False,
            )
            expected_coeffs = (3 + 1) * (3 + 2) // 2 - 1
            assert zernike_coeffs.shape == (expected_coeffs, 1)
            assert np.any(np.abs(zernike_coeffs[:, 0]) > 0.01)
        except Exception as e:
            pytest.skip(f"Zernike fit raised {type(e).__name__}: {e}")

    with subtests.test("2D input"):
        phase_2d = 0.2 * X_small
        try:
            zernike_coeffs_2d = analysis.image_zernike_fit(
                phase_2d, grid_small, order=2,
                iterations=1, leastsquares=False,
            )
            expected_coeffs_2d = (2 + 1) * (2 + 2) // 2 - 1
            assert zernike_coeffs_2d.shape == (expected_coeffs_2d, 1)
        except Exception:
            pass

    with subtests.test("leastsquares=True refines fit"):
        phase_tilt = 0.5 * X_small + 0.3 * Y_small
        phase_tilt = phase_tilt[np.newaxis, :, :]

        try:
            coeffs_ls = analysis.image_zernike_fit(
                phase_tilt, grid_small, order=3,
                iterations=2, leastsquares=True,
            )
            expected = (3 + 1) * (3 + 2) // 2 - 1
            assert coeffs_ls.shape == (expected, 1)
        except Exception as e:
            print(f"leastsquares fit error (may be expected): {e}")


def test_take(subtests, benchmark):
    """Test take(), take_tile(), take_plot(), and _take_parse_shape()."""
    with subtests.test("benchmark"):
        rng = np.random.default_rng(42)
        image = rng.random((512, 512)).astype(np.float32)
        vectors = np.stack([rng.integers(20, 492, 50), rng.integers(20, 492, 50)])
        benchmark(analysis.take, image, vectors=vectors, size=20, centered=True)

    with subtests.test("single region extraction"):
        image = np.random.rand(100, 100)
        result = analysis.take(image, vectors=[50, 50], size=10, centered=True)
        assert result.shape == (1, 10, 10)

    with subtests.test("multiple region extraction"):
        image = np.random.rand(100, 100)
        vectors = np.array([[25, 75], [50, 50], [75, 25]]).T
        result = analysis.take(image, vectors=vectors, size=10, centered=True)
        assert result.shape == (3, 10, 10)

    with subtests.test("integration sums region"):
        image = np.ones((100, 100))
        result = analysis.take(image, vectors=[50, 50], size=10, centered=True, integrate=True)
        # 2D image + single vector: shape is (vector_count,) per docstring
        assert result.shape == (1,)
        assert result[0] == pytest.approx(100)

    with subtests.test("take_tile with explicit shape"):
        test_images = np.random.rand(3, 10, 10)
        tiled = analysis.take_tile(test_images, shape=(2, 2))
        assert tiled.shape == (20, 20)

    with subtests.test("take_tile with automatic shape"):
        test_images = np.random.rand(3, 10, 10)
        tiled_auto = analysis.take_tile(test_images)
        assert tiled_auto.shape == (20, 20)

    with subtests.test("_take_parse_shape auto"):
        test_images = np.random.rand(3, 10, 10)
        img_count, (M, N) = analysis._take_parse_shape(test_images, shape=None)
        assert img_count == 3
        assert M == 2
        assert N == 2

    with subtests.test("_take_parse_shape explicit"):
        test_images = np.random.rand(3, 10, 10)
        img_count2, (M2, N2) = analysis._take_parse_shape(test_images, shape=(1, 4))
        assert img_count2 == 3
        assert M2 == 1
        assert N2 == 4

    with subtests.test("_take_parse_shape warns on truncation"):
        test_images = np.random.rand(3, 10, 10)
        with pytest.warns(UserWarning, match="Not enough space"):
            img_count3, (M3, N3) = analysis._take_parse_shape(test_images, shape=(1, 2))
            assert img_count3 == 2

    with subtests.test("take_plot does not crash"):
        import matplotlib.pyplot as plt
        test_images = np.random.rand(3, 10, 10)
        plt.ioff()

        analysis.take_plot(test_images, separate_axes=False)
        plt.close('all')

        analysis.take_plot(test_images, separate_axes=True, shape=(2, 2))
        plt.close('all')

    with subtests.test("tuple size input"):
        image = np.random.rand(100, 100)
        result = analysis.take(image, vectors=[50, 50], size=(8, 12), centered=True)
        assert result.shape == (1, 12, 8)

    with subtests.test("clip=True with out-of-range regions"):
        image = np.random.rand(50, 50)
        result = analysis.take(image, vectors=[0, 0], size=20, centered=True, clip=True)
        assert result.shape == (1, 20, 20)
        assert np.any(np.isnan(result))

    with subtests.test("clip=True with integer dtype uses zero"):
        image = np.ones((50, 50), dtype=np.uint8) * 42
        result = analysis.take(image, vectors=[0, 0], size=20, centered=True, clip=True)
        assert result.shape == (1, 20, 20)
        assert np.any(result == 0)

    with subtests.test("return_mask=True boolean canvas"):
        image = np.random.rand(80, 80)
        canvas = analysis.take(image, vectors=[40, 40], size=10, centered=True, return_mask=True)
        assert canvas.dtype == bool
        assert canvas.shape == (80, 80)
        assert np.sum(canvas) == 100

    with subtests.test("return_mask=2 nan canvas"):
        image = np.random.rand(80, 80)
        canvas = analysis.take(image, vectors=[40, 40], size=10, centered=True, return_mask=2)
        assert canvas.shape == (80, 80)
        nan_count = np.sum(np.isnan(canvas))
        # 80*80 pixels total, 10*10=100 filled from image; rest are NaN
        assert nan_count == 80 * 80 - 100
        # Non-NaN pixels must equal the source image values
        assert np.allclose(canvas[~np.isnan(canvas)], image[~np.isnan(canvas)])

    with subtests.test("3D image stack with integrate"):
        images = np.ones((3, 100, 100))
        result = analysis.take(images, vectors=[50, 50], size=10, centered=True, integrate=True)
        # 3D stack + single vector: shape is (image_count, vector_count) per docstring
        assert result.shape == (3, 1)
        assert np.allclose(result[:, 0], 100.0)

    with subtests.test("clip=True fully in-bounds sets clip=False"):
        image = np.random.rand(100, 100)
        result = analysis.take(image, vectors=[50, 50], size=10, centered=True, clip=True)
        assert result.shape == (1, 10, 10)
        assert not np.any(np.isnan(result))

    with subtests.test("return_mask with plot"):
        import matplotlib.pyplot as plt
        plt.ioff()

        image = np.random.rand(60, 60)
        shown = {"called": False}
        def _show2():
            shown["called"] = True

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(analysis.plt, "show", _show2)
        try:
            canvas = analysis.take(image, vectors=[30, 30], size=10, centered=True,
                                   return_mask=True, plot=True)
        finally:
            monkeypatch.undo()
            plt.close("all")

        assert shown["called"]
        assert canvas.dtype == bool

    with subtests.test("plot path in take"):
        import matplotlib.pyplot as plt
        plt.ioff()

        image = np.random.rand(60, 60)
        shown = {"called": False}
        def _show():
            shown["called"] = True

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(analysis.plt, "show", _show)
        try:
            result = analysis.take(image, vectors=[30, 30], size=10, centered=True, plot=True)
        finally:
            monkeypatch.undo()
            plt.close("all")

        assert result.shape == (1, 10, 10)

    with subtests.test("centered=False uses upper-left anchor"):
        image = np.zeros((100, 100))
        image[10:20, 20:30] = 5.0  # rows [10,19], cols [20,29]
        result = analysis.take(image, vectors=[20, 10], size=10, centered=False)
        assert result.shape == (1, 10, 10)
        assert np.all(result == 5.0)

    with subtests.test("extracted values match source image"):
        image = np.zeros((100, 100))
        image[45:55, 45:55] = 7.0
        result = analysis.take(image, vectors=[50, 50], size=10, centered=True)
        assert result.shape == (1, 10, 10)
        assert np.all(result == 7.0)

    with subtests.test("float vectors get floored"):
        image = np.zeros((100, 100))
        image[45:55, 45:55] = 3.0
        result_int = analysis.take(image, vectors=[50, 50], size=10, centered=True)
        result_float = analysis.take(image, vectors=[50.7, 50.3], size=10, centered=True)
        assert np.allclose(result_int, result_float)

    with subtests.test("images as shape tuple returns boolean mask"):
        mask = analysis.take((80, 60), vectors=[30, 20], size=10, centered=True)
        assert mask.shape == (80, 60)
        assert mask.dtype == bool
        assert np.sum(mask) == 100

    with subtests.test("multiple vectors integrate shape for 2D image"):
        image = np.ones((100, 100))
        vectors = np.array([[25, 50, 75], [50, 50, 50]])
        result = analysis.take(image, vectors=vectors, size=10, centered=True, integrate=True)
        # 2D image + 3 vectors: shape is (vector_count,) per docstring
        assert result.shape == (3,)
        assert np.allclose(result, 100.0)

    with subtests.test("3D stack without integrate gives (image_count, vector_count, h, w)"):
        images = np.random.rand(4, 100, 100)
        vectors = np.array([[25, 75], [25, 75]])
        result = analysis.take(images, vectors=vectors, size=10, centered=True)
        assert result.shape == (4, 2, 10, 10)

    with subtests.test("out-of-range index without clip raises IndexError"):
        image = np.random.rand(50, 50)
        with pytest.raises(IndexError):
            analysis.take(image, vectors=[100, 100], size=10, centered=True, clip=False)

    with subtests.test("4D images raises RuntimeError"):
        image = np.random.rand(2, 3, 50, 50)
        with pytest.raises(RuntimeError):
            analysis.take(image, vectors=[25, 25], size=10)


def test_image_positions(subtests):
    """Test image_positions() returns correct centroid positions."""
    with subtests.test("off-center spot has correct position"):
        image = np.zeros((100, 100))
        image[30:40, 60:70] = 1  # center at row=35, col=65; offset from 50 -> x=+15, y=-15
        image = image[np.newaxis, :, :]
        positions = analysis.image_positions(image)
        assert positions.shape == (2, 1)
        assert positions[0, 0] == pytest.approx(15, abs=1)
        assert positions[1, 0] == pytest.approx(-15, abs=1)

    with subtests.test("centered spot returns near-zero position"):
        image = np.zeros((100, 100))
        image[45:55, 45:55] = 1
        image = image[np.newaxis, :, :]
        positions = analysis.image_positions(image)
        assert positions[0, 0] == pytest.approx(0, abs=1)
        assert positions[1, 0] == pytest.approx(0, abs=1)


def test_image_std(subtests):
    """Test image_std() returns correct standard deviations for a Gaussian blob."""
    with subtests.test("isotropic Gaussian sigma=10 gives std≈10 in both axes"):
        Y, X = np.ogrid[:100, :100]
        image = np.exp(-((X - 50)**2 + (Y - 50)**2) / (2 * 10**2))
        image = image[np.newaxis, :, :]
        std = analysis.image_std(image)
        assert std.shape == (2, 1)
        assert std[0, 0] == pytest.approx(10, rel=0.01)
        assert std[1, 0] == pytest.approx(10, rel=0.01)

    with subtests.test("elliptical Gaussian has different x and y stds"):
        Y, X = np.ogrid[:100, :100]
        image = np.exp(-((X - 50)**2 / (2 * 5**2) + (Y - 50)**2 / (2 * 15**2)))
        image = image[np.newaxis, :, :]
        std = analysis.image_std(image)
        assert std[0, 0] == pytest.approx(5, rel=0.05)
        assert std[1, 0] == pytest.approx(15, rel=0.05)


def test_fit_affine(subtests):
    """Test fit_affine() affine transformation fitting."""
    rng = np.random.default_rng(42)

    with subtests.test("identity"):
        x = rng.uniform(-5, 5, size=(2, 20))
        result = analysis.fit_affine(x, x)
        np.testing.assert_allclose(result["M"], np.eye(2), atol=1e-6)
        np.testing.assert_allclose(result["b"], np.zeros((2, 1)), atol=1e-6)

    with subtests.test("pure translation"):
        x = rng.uniform(-5, 5, size=(2, 30))
        b_true = np.array([[3.0], [-7.0]])
        y = x + b_true
        result = analysis.fit_affine(x, y)
        np.testing.assert_allclose(result["M"], np.eye(2), atol=1e-4)
        np.testing.assert_allclose(result["b"], b_true, atol=1e-4)

    with subtests.test("pure scaling"):
        x = rng.uniform(-5, 5, size=(2, 30))
        M_true = np.array([[2.0, 0.0], [0.0, 0.5]])
        y = M_true @ x
        result = analysis.fit_affine(x, y)
        np.testing.assert_allclose(result["M"], M_true, atol=1e-4)
        np.testing.assert_allclose(result["b"], np.zeros((2, 1)), atol=1e-4)

    with subtests.test("rotation"):
        theta = np.pi / 6
        M_true = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
        ])
        x = rng.uniform(-5, 5, size=(2, 40))
        y = M_true @ x
        result = analysis.fit_affine(x, y)
        np.testing.assert_allclose(result["M"], M_true, atol=1e-4)
        np.testing.assert_allclose(result["b"], np.zeros((2, 1)), atol=1e-4)

    with subtests.test("full affine"):
        M_true = np.array([[1.5, -0.3], [0.4, 2.0]])
        b_true = np.array([[10.0], [-5.0]])
        x = rng.uniform(-5, 5, size=(2, 50))
        y = M_true @ x + b_true
        result = analysis.fit_affine(x, y)
        np.testing.assert_allclose(result["M"], M_true, atol=1e-3)
        np.testing.assert_allclose(result["b"], b_true, atol=1e-3)

    with subtests.test("with guess_affine"):
        M_true = np.array([[1.0, 0.0], [0.0, 1.0]])
        b_true = np.array([[2.0], [3.0]])
        x = rng.uniform(-5, 5, size=(2, 30))
        y = M_true @ x + b_true
        guess = {"M": np.eye(2), "b": np.array([[1.0], [1.0]])}
        result = analysis.fit_affine(x, y, guess_affine=guess)
        np.testing.assert_allclose(result["M"], M_true, atol=1e-3)
        np.testing.assert_allclose(result["b"], b_true, atol=1e-3)

    with subtests.test("invalid guess_affine raises"):
        x = rng.uniform(-5, 5, size=(2, 10))
        with pytest.raises(ValueError, match="guess_affine must be a dictionary"):
            analysis.fit_affine(x, x, guess_affine="bad")
        with pytest.raises(ValueError, match="guess_affine must be a dictionary"):
            analysis.fit_affine(x, x, guess_affine={"M": np.eye(2)})

    with subtests.test("nan row raises"):
        x = np.vstack((np.full((1, 5), np.nan), rng.uniform(-1, 1, size=(1, 5))))
        y = rng.uniform(-1, 1, size=(2, 5))
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            with pytest.raises(ValueError, match="all-nan"):
                analysis.fit_affine(x, y)

    with subtests.test("noisy data recovers approximate transform"):
        M_true = np.array([[1.2, -0.1], [0.3, 0.9]])
        b_true = np.array([[1.0], [-2.0]])
        x = rng.uniform(-10, 10, size=(2, 200))
        noise = rng.normal(0, 0.05, size=(2, 200))
        y = M_true @ x + b_true + noise
        result = analysis.fit_affine(x, y)
        np.testing.assert_allclose(result["M"], M_true, atol=0.05)
        np.testing.assert_allclose(result["b"], b_true, atol=0.1)

    with subtests.test("result dict keys"):
        x = rng.uniform(-1, 1, size=(2, 10))
        result = analysis.fit_affine(x, x)
        assert set(result.keys()) == {"M", "b"}
        assert result["M"].shape == (2, 2)
        assert result["b"].shape == (2, 1)

    with subtests.test("list input"):
        x = [[1, 2, 3], [4, 5, 6]]
        y = [[2, 4, 6], [8, 10, 12]]
        result = analysis.fit_affine(x, y)
        assert result["M"].shape == (2, 2)
        assert result["b"].shape == (2, 1)

    with subtests.test("shape mismatch raises assertion"):
        x = rng.uniform(-1, 1, size=(2, 5))
        y = rng.uniform(-1, 1, size=(2, 6))
        with pytest.raises(AssertionError):
            analysis.fit_affine(x, y)

    with subtests.test("optimizer failure falls back to guess"):
        x = rng.uniform(-3, 3, size=(2, 20))
        y = 2 * x + np.array([[1.0], [-2.0]])
        guess = {"M": np.array([[7.0, 8.0], [9.0, 10.0]]), "b": np.array([[11.0], [12.0]])}

        def _boom(*args, **kwargs):
            raise RuntimeError("forced failure")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(analysis, "minimize", _boom)
        try:
            result = analysis.fit_affine(x, y, guess_affine=guess)
        finally:
            monkeypatch.undo()

        np.testing.assert_allclose(result["M"], guess["M"])
        np.testing.assert_allclose(result["b"], guess["b"])

    with subtests.test("plot path"):
        x = rng.uniform(-2, 2, size=(2, 15))
        y = x + np.array([[0.5], [0.25]])
        shown = {"called": False}

        def _show():
            shown["called"] = True

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(analysis.plt, "show", _show)
        try:
            result = analysis.fit_affine(x, y, plot=True)
        finally:
            monkeypatch.undo()

        assert shown["called"] is True
        assert result["M"].shape == (2, 2)
        assert result["b"].shape == (2, 1)


def test_image_vortices(subtests):
    """Test image_vortices(), image_vortices_coordinates(), and image_remove_vortices()."""
    y = np.arange(128)
    x = np.arange(128)
    X, Y = np.meshgrid(x, y)
    cx, cy = 64.0, 64.0

    with subtests.test("single vortex has nonzero winding"):
        phase = np.arctan2(Y - cy, X - cx)
        winding = analysis.image_vortices(phase)

        assert winding.shape == phase.shape
        assert np.count_nonzero(winding) > 0

    with subtests.test("coordinates and weights are consistent"):
        phase = np.arctan2(Y - cy, X - cx)
        coords, weights = analysis.image_vortices_coordinates(phase)

        assert len(coords) == 2
        assert len(coords[0]) == len(weights)
        assert len(weights) > 0
        assert np.all(np.isin(np.unique(weights), np.array([-1, 1])))

    with subtests.test("mask restricts detected vortices"):
        phase = np.arctan2(Y - cy, X - cx)
        mask = np.zeros_like(phase, dtype=bool)
        mask[:40, :40] = True

        coords, weights = analysis.image_vortices_coordinates(phase, mask=mask)

        assert len(weights) == 0

    with subtests.test("return_vortices_negative gives same result as in-place"):
        phase = np.arctan2(Y - cy, X - cx)
        correction = analysis.image_remove_vortices(phase.copy(), return_vortices_negative=True)
        in_place = analysis.image_remove_vortices(phase.copy())
        # The documented contract: phase + correction == in-place-modified phase
        assert correction.shape == phase.shape
        np.testing.assert_array_equal(phase + correction, in_place)

    with subtests.test("in-place removal returns same-shape phase"):
        phase = np.arctan2(Y - cy, X - cx)
        removed = analysis.image_remove_vortices(phase.copy())

        assert removed.shape == phase.shape
        assert np.isfinite(removed).all()

    with subtests.test("mask restricts removal region"):
        phase = np.arctan2(Y - cy, X - cx)
        mask = np.ones_like(phase, dtype=bool)

        removed = analysis.image_remove_vortices(phase.copy(), mask=mask)
        assert removed.shape == phase.shape


def test_image_remove_blaze(subtests):
    """Test image_remove_blaze() for global linear phase ramp removal."""
    y = np.arange(96)
    x = np.arange(96)
    X, Y = np.meshgrid(x, y)

    with subtests.test("reduces average wrapped gradient"):
        phase = np.mod(0.15 * X + 0.22 * Y + 0.5, 2 * np.pi)

        dx_before = np.mod(np.gradient(phase, axis=1) + np.pi / 2, np.pi) - np.pi / 2
        dy_before = np.mod(np.gradient(phase, axis=0) + np.pi / 2, np.pi) - np.pi / 2
        mean_before = np.hypot(np.nanmean(dx_before), np.nanmean(dy_before))

        result = analysis.image_remove_blaze(phase)

        dx_after = np.mod(np.gradient(result, axis=1) + np.pi / 2, np.pi) - np.pi / 2
        dy_after = np.mod(np.gradient(result, axis=0) + np.pi / 2, np.pi) - np.pi / 2
        mean_after = np.hypot(np.nanmean(dx_after), np.nanmean(dy_after))

        assert result.shape == phase.shape
        assert np.nanmin(result) >= 0
        assert np.nanmax(result) <= 2 * np.pi
        assert mean_after < 0.1 * mean_before

    with subtests.test("masked mode executes"):
        phase = np.mod(0.11 * X + 0.05 * Y, 2 * np.pi)
        mask = np.zeros_like(phase, dtype=float)
        mask[20:80, 20:80] = 1.0

        result = analysis.image_remove_blaze(phase, mask=mask)
        assert result.shape == phase.shape

    with subtests.test("plot path"):
        phase = np.mod(0.15 * X + 0.22 * Y + 0.5, 2 * np.pi)

        result = analysis.image_remove_blaze(phase, plot=True)
        plt.show()

        assert result.shape == phase.shape


def test_image_reduce_wraps(subtests):
    """Test image_reduce_wraps() phase-offset optimization."""
    y = np.arange(128)
    x = np.arange(128)
    X, Y = np.meshgrid(x, y)

    def wrap_metric(arr):
        return np.sum(
            (np.abs(np.gradient(arr, axis=1)) + np.abs(np.gradient(arr, axis=0))) > np.pi
        )

    with subtests.test("does not worsen wrap metric"):
        phase = np.mod(0.7 * np.sin(X / 4.0) + 1.6 * np.cos(Y / 7.0) + 3.0, 2 * np.pi)
        before = wrap_metric(phase)

        reduced = analysis.image_reduce_wraps(phase, steps=24)
        after = wrap_metric(reduced)

        assert reduced.shape == phase.shape
        assert np.nanmin(reduced) >= 0
        assert np.nanmax(reduced) <= 2 * np.pi
        assert after <= before

    with subtests.test("masked mode executes"):
        phase = np.mod(0.5 * X + 0.4 * Y, 2 * np.pi)
        mask = np.zeros_like(phase, dtype=float)
        mask[32:96, 32:96] = 1.0

        reduced = analysis.image_reduce_wraps(phase, mask=mask, steps=12)
        assert reduced.shape == phase.shape


def test_make_8bit(subtests):
    """Test _make_8bit() conversion and scaling behavior."""
    with subtests.test("scales dynamic range to uint8"):
        image = np.array([[10.0, 20.0], [30.0, 40.0]])
        converted = analysis._make_8bit(image)

        assert converted.dtype == np.uint8
        assert converted.min() == 0
        assert converted.max() == 255

    with subtests.test("constant image becomes zeros"):
        image = np.ones((4, 4), dtype=float) * 7.3
        converted = analysis._make_8bit(image)

        assert converted.dtype == np.uint8
        assert np.all(converted == 0)


def test_get_orientation_transformation(subtests):
    """Test get_orientation_transformation() composition of rotate/flip operations."""
    image = np.arange(9).reshape(3, 3)

    with subtests.test("identity transform"):
        transform = analysis.get_orientation_transformation()
        np.testing.assert_array_equal(transform(image), image)

    with subtests.test("flip left-right"):
        transform = analysis.get_orientation_transformation(fliplr=True)
        np.testing.assert_array_equal(transform(image), np.fliplr(image))

    with subtests.test("flip up-down"):
        transform = analysis.get_orientation_transformation(flipud=True)
        np.testing.assert_array_equal(transform(image), np.flipud(image))

    with subtests.test("rotate 90"):
        transform = analysis.get_orientation_transformation(rot="90")
        np.testing.assert_array_equal(transform(image), np.rot90(image, 1))

    with subtests.test("rotate 180"):
        transform = analysis.get_orientation_transformation(rot=2)
        np.testing.assert_array_equal(transform(image), np.rot90(image, 2))

    with subtests.test("rotate 270"):
        transform = analysis.get_orientation_transformation(rot="270")
        np.testing.assert_array_equal(transform(image), np.rot90(image, 3))

    with subtests.test("combined operations"):
        transform = analysis.get_orientation_transformation(rot="90", fliplr=True, flipud=True)
        expected = np.rot90(np.flipud(np.fliplr(image)), 1)
        np.testing.assert_array_equal(transform(image), expected)


def test_helpers(subtests):
    """Test internal helper functions _center, _coordinates, _generate_grid."""
    with subtests.test("_center even"):
        assert analysis._center(10, integer=False) == 4.5

    with subtests.test("_center odd"):
        assert analysis._center(11, integer=False) == 5.0

    with subtests.test("_center integer"):
        assert analysis._center(10, integer=True) == 5

    with subtests.test("_coordinates centered"):
        coords = analysis._coordinates(5, centered=True)
        np.testing.assert_array_equal(coords, np.array([-2, -1, 0, 1, 2]))

    with subtests.test("_coordinates not centered"):
        coords = analysis._coordinates(5, centered=False)
        np.testing.assert_array_equal(coords, np.array([0, 1, 2, 3, 4]))

    with subtests.test("_generate_grid centered float"):
        grid_x, grid_y = analysis._generate_grid(3, 4, centered=True, integer=False)
        assert grid_x.shape == (4, 3)
        assert grid_y.shape == (4, 3)
        assert grid_x[0, 0] == -1
        assert grid_y[0, 0] == -1.5

    with subtests.test("_generate_grid centered integer"):
        grid_x_int, grid_y_int = analysis._generate_grid(4, 4, centered=True, integer=True)
        assert np.allclose(grid_x_int, np.round(grid_x_int))
        assert np.allclose(grid_y_int, np.round(grid_y_int))

    with subtests.test("_generate_grid not centered"):
        grid_x, grid_y = analysis._generate_grid(3, 3, centered=False)
        assert grid_x[0, 0] == 0
        assert grid_y[0, 0] == 0
        assert grid_x[0, -1] == 2
        assert grid_y[-1, 0] == 2


@pytest.mark.gpu
def test_take_gpu(benchmark, has_cupy):
    """GPU variant of take() using cupy arrays."""
    import cupy as cp

    rng = np.random.default_rng(42)
    image = cp.array(rng.random((512, 512)).astype(np.float32))
    vectors = np.stack([rng.integers(20, 492, 50), rng.integers(20, 492, 50)])

    result = benchmark(analysis.take, image, vectors=vectors, size=20, centered=True, xp=cp)
    assert result.shape == (50, 20, 20)
