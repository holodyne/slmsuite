"""
Unit tests for slmsuite.misc modules.
"""
import pytest
import numpy as np
import tempfile
import os

from slmsuite.misc.math import *
from slmsuite.misc.fitfunctions import *
from slmsuite.holography.analysis.files import generate_path, latest_path, save_h5, load_h5


# Test misc.math functions

def test_iseven(subtests):
    with subtests.test("scalar integers"):
        assert iseven(0) == True
        assert iseven(1) == False
        assert iseven(2) == True
        assert iseven(3) == False
        assert iseven(4) == True
        assert iseven(-1) == False
        assert iseven(-2) == True

    with subtests.test("arrays"):
        x = np.array([0, 1, 2, 3, 4, 5])
        result = iseven(x)
        expected = np.array([True, False, True, False, True, False])
        np.testing.assert_array_equal(result, expected)

    with subtests.test("float inputs"):
        assert iseven(2.1) == True   # rounds to 2
        assert iseven(2.9) == False  # rounds to 3
        assert iseven(3.1) == False  # rounds to 3
        assert iseven(3.9) == True   # rounds to 4

    with subtests.test("banker rounding"):
        # np.around uses banker's rounding: half-integers always round to nearest even
        assert iseven(0.5) == True   # rounds to 0
        assert iseven(1.5) == True   # rounds to 2
        assert iseven(2.5) == True   # rounds to 2
        assert iseven(3.5) == True   # rounds to 4


def test_type_tuples(subtests):
    with subtests.test("basic types"):
        assert int in INTEGER_TYPES
        assert isinstance(np.int32(1), INTEGER_TYPES)
        assert float in FLOAT_TYPES
        assert isinstance(np.float64(1.0), FLOAT_TYPES)
        assert int in REAL_TYPES
        assert float in REAL_TYPES
        assert complex in SCALAR_TYPES

    with subtests.test("numpy integer types"):
        assert isinstance(np.int8(1), INTEGER_TYPES)
        assert isinstance(np.int16(1), INTEGER_TYPES)
        assert isinstance(np.int32(1), INTEGER_TYPES)
        assert isinstance(np.int64(1), INTEGER_TYPES)
        assert isinstance(np.uint8(1), INTEGER_TYPES)
        assert isinstance(np.uint16(1), INTEGER_TYPES)
        assert isinstance(np.uint32(1), INTEGER_TYPES)
        assert isinstance(np.uint64(1), INTEGER_TYPES)

    with subtests.test("numpy float types"):
        assert isinstance(np.float32(1.0), FLOAT_TYPES)
        assert isinstance(np.float64(1.0), FLOAT_TYPES)

    with subtests.test("complex types"):
        assert isinstance(1+1j, SCALAR_TYPES)
        assert isinstance(np.complex64(1+1j), SCALAR_TYPES)
        assert isinstance(np.complex128(1+1j), SCALAR_TYPES)


# Test 1D fit functions

def test_linear(subtests):
    with subtests.test("basic"):
        x = np.linspace(0, 10, 100)
        y = linear(x, m=2, b=3)
        assert y[0] == pytest.approx(3.0)
        assert y[-1] == pytest.approx(23.0)
        assert np.all(np.diff(y) > 0)  # Increasing

    with subtests.test("horizontal line"):
        x = np.array([0, 1, -1, 10, -10])
        y = linear(x, m=0, b=5)
        expected = np.array([5, 5, 5, 5, 5])
        np.testing.assert_array_equal(y, expected)

    with subtests.test("through origin"):
        x = np.array([0, 1, -1, 10, -10])
        y = linear(x, m=2, b=0)
        expected = 2 * x
        np.testing.assert_array_equal(y, expected)

    with subtests.test("negative slope"):
        x = np.array([0, 1, -1, 10, -10])
        y = linear(x, m=-1, b=0)
        expected = -x
        np.testing.assert_array_equal(y, expected)


def test_parabola(subtests):
    with subtests.test("vertex value"):
        assert parabola(np.array([0.0]), a=3, x0=0, y0=7)[0] == pytest.approx(7.0)
        assert parabola(np.array([2.0]), a=3, x0=2, y0=7)[0] == pytest.approx(7.0)
        assert parabola(np.array([-5.0]), a=1, x0=-5, y0=0)[0] == pytest.approx(0.0)

    with subtests.test("formula correctness"):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = parabola(x, a=2, x0=1, y0=3)
        expected = 2 * (x - 1) ** 2 + 3
        np.testing.assert_array_almost_equal(y, expected, decimal=10)

    with subtests.test("symmetry about x0"):
        x_sym = np.array([0.0, 2.0])
        y_sym = parabola(x_sym, a=1, x0=1, y0=0)
        assert y_sym[0] == pytest.approx(y_sym[1])

    with subtests.test("negative a (downward)"):
        x = np.linspace(-5, 5, 11)
        y = parabola(x, a=-1, x0=0, y0=5)
        assert y[5] == pytest.approx(5.0, abs=0.001)  # Maximum at vertex
        assert y[6] < y[5]  # Values decrease away from vertex
        assert y[4] < y[5]


def test_hyperbola(subtests):
    with subtests.test("value at z0 equals w0"):
        assert hyperbola(np.array([2.0]), w0=3, z0=2, zr=5)[0] == pytest.approx(3.0)
        assert hyperbola(np.array([0.0]), w0=7, z0=0, zr=2)[0] == pytest.approx(7.0)

    with subtests.test("sqrt2 times w0 at z0 plus or minus zr"):
        # w(z0 +/- zr) = w0 * sqrt(1 + 1) = w0 * sqrt(2)
        w_at_zr = hyperbola(np.array([1.0, 3.0]), w0=1, z0=2, zr=1)
        np.testing.assert_array_almost_equal(w_at_zr, [np.sqrt(2), np.sqrt(2)], decimal=10)

    with subtests.test("symmetry about z0"):
        dz = np.array([1.0, 2.0, 3.0, 5.0])
        z0 = 7.0
        w_left = hyperbola(z0 - dz, w0=2, z0=z0, zr=3)
        w_right = hyperbola(z0 + dz, w0=2, z0=z0, zr=3)
        np.testing.assert_array_almost_equal(w_left, w_right, decimal=10)

    with subtests.test("minimum at z0"):
        z = np.linspace(-10, 10, 1001)
        w = hyperbola(z, w0=1, z0=0, zr=1)
        min_idx = np.argmin(w)
        assert z[min_idx] == pytest.approx(0.0, abs=0.02)
        assert w[min_idx] == pytest.approx(1.0, abs=0.001)


def test_cos(subtests):
    with subtests.test("formula y = c + a/2 * (1 + cos(kx - b))"):
        x = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        y = cos(x, b=0, a=2, c=1)
        expected = np.array([3.0, 2.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(y, expected, decimal=10)

    with subtests.test("phase offset"):
        x = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        y = cos(x, b=np.pi / 2, a=2, c=1)
        expected = np.array([2.0, 3.0, 2.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(y, expected, decimal=10)

    with subtests.test("k parameter doubles frequency"):
        x = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        y = cos(x, b=0, a=2, c=1, k=2)
        expected = np.array([3.0, 1.0, 3.0, 1.0, 3.0])
        np.testing.assert_array_almost_equal(y, expected, decimal=10)

    with subtests.test("amplitude range"):
        x = np.linspace(0, 2 * np.pi, 1000)
        y = cos(x, b=0, a=2, c=1, k=1)
        assert np.max(y) == pytest.approx(3.0, abs=0.001)
        assert np.min(y) == pytest.approx(1.0, abs=0.001)


def test_lorentzian(subtests):
    with subtests.test("peak value at x0"):
        y = lorentzian(np.array([1000.0]), x0=1000, a=10, c=1, w=1)
        assert y[0] == pytest.approx(11.0)

    with subtests.test("half amplitude at x0 plus or minus w"):
        # Formula: a / (1 + ((x-x0)/w)^2) + c; at |x-x0|=w the a-term is a/2
        y = lorentzian(np.array([999.0, 1001.0]), x0=1000, a=10, c=0, w=1)
        np.testing.assert_array_almost_equal(y, [5.0, 5.0], decimal=10)

    with subtests.test("offset c"):
        y = lorentzian(np.array([1000.0]), x0=1000, a=10, c=7, w=1)
        assert y[0] == pytest.approx(17.0)

    with subtests.test("symmetry about x0"):
        dx = np.array([1.0, 2.0, 5.0, 10.0])
        x0 = 500.0
        y_left = lorentzian(x0 - dx, x0=x0, a=10, c=1, w=3)
        y_right = lorentzian(x0 + dx, x0=x0, a=10, c=1, w=3)
        np.testing.assert_array_almost_equal(y_left, y_right, decimal=10)

    with subtests.test("narrower w gives sharper peak"):
        x = np.linspace(990, 1010, 1000)
        y_narrow = lorentzian(x, x0=1000, a=10, c=1, w=1)
        y_broad = lorentzian(x, x0=1000, a=10, c=1, w=10)
        assert y_narrow[0] < y_broad[0]


def test_gaussian(subtests):
    with subtests.test("peak at x0"):
        x = np.linspace(-10, 10, 1001)
        y = gaussian(x, x0=0, a=10, c=1, w=2)
        max_idx = np.argmax(y)
        assert x[max_idx] == pytest.approx(0.0, abs=0.02)
        assert y[max_idx] == pytest.approx(11.0, abs=0.001)

    with subtests.test("1/e amplitude at x0 plus or minus w*sqrt(2)"):
        # exp(-0.5 * (w*sqrt(2) / w)^2) = exp(-1) = 1/e
        w = 3.0
        x_1e = np.array([w * np.sqrt(2), -w * np.sqrt(2)])
        y_1e = gaussian(x_1e, x0=0, a=1, c=0, w=w)
        np.testing.assert_array_almost_equal(y_1e, [np.exp(-1), np.exp(-1)], decimal=10)

    with subtests.test("half maximum at x0 plus or minus w*sqrt(2*ln2)"):
        w = 2.0
        x_half = w * np.sqrt(2 * np.log(2))
        y_half = gaussian(np.array([x_half, -x_half]), x0=0, a=1, c=0, w=w)
        np.testing.assert_array_almost_equal(y_half, [0.5, 0.5], decimal=10)

    with subtests.test("offset peak"):
        x = np.linspace(-10, 10, 1001)
        y = gaussian(x, x0=3, a=10, c=1, w=1)
        max_idx = np.argmax(y)
        assert x[max_idx] == pytest.approx(3.0, abs=0.02)

    with subtests.test("narrow vs broad: narrow falls off faster"):
        x = np.linspace(-10, 10, 1001)
        y_narrow = gaussian(x, x0=0, a=10, c=0, w=0.5)
        y_broad = gaussian(x, x0=0, a=10, c=0, w=5)
        far_idx = np.searchsorted(x, 3.0)
        assert y_narrow[far_idx] < y_broad[far_idx]
        assert y_narrow[500] == pytest.approx(y_broad[500], rel=0.001)


# Test 2D fit functions

def test_gaussian2d(subtests):
    with subtests.test("peak at center"):
        point = np.array([[0.0], [0.0]])
        z = gaussian2d(point, x0=0, y0=0, a=10, c=1, wx=2, wy=2)
        assert z[0] == pytest.approx(11.0)

    with subtests.test("matches product of 1D gaussians without shear"):
        x = np.linspace(-5, 5, 51)
        y = np.linspace(-5, 5, 51)
        X, Y = np.meshgrid(x, y)
        xy = np.array([X, Y])
        z = gaussian2d(xy, x0=0, y0=0, a=1, c=0, wx=2, wy=3)
        expected = np.exp(-0.5 * (X / 2) ** 2) * np.exp(-0.5 * (Y / 3) ** 2)
        np.testing.assert_array_almost_equal(z, expected, decimal=10)

    with subtests.test("value at (wx, 0) is exp(-0.5) times a"):
        # exp(-0.5 * (wx/wx)^2) * exp(0) = exp(-0.5)
        point = np.array([[2.0], [0.0]])
        z = gaussian2d(point, x0=0, y0=0, a=1, c=0, wx=2, wy=3)
        assert z[0] == pytest.approx(np.exp(-0.5), rel=1e-10)

    with subtests.test("offset center"):
        x = np.linspace(-10, 10, 101)
        y = np.linspace(-10, 10, 101)
        X, Y = np.meshgrid(x, y)
        xy = np.array([X, Y])
        z = gaussian2d(xy, x0=2, y0=-3, a=1, c=0, wx=1, wy=1)
        max_idx = np.unravel_index(np.argmax(z), z.shape)
        assert x[max_idx[1]] == pytest.approx(2.0, abs=0.2)
        assert y[max_idx[0]] == pytest.approx(-3.0, abs=0.2)


def test_tophat2d(subtests):
    x = np.linspace(-10, 10, 1001)
    y = np.linspace(-10, 10, 1001)
    X, Y = np.meshgrid(x, y)
    xy = np.array([X, Y])
    z = tophat2d(xy, x0=0, y0=0, R=5, a=10, c=1)

    with subtests.test("center value"):
        assert z[500, 500] == pytest.approx(11.0)

    with subtests.test("outside corner value"):
        assert z[0, 0] == pytest.approx(1.0)

    with subtests.test("boundary is inclusive"):
        # At exactly (R, 0): distance == R, condition is <=, so inside
        point = np.array([[5.0], [0.0]])
        z_bound = tophat2d(point, x0=0, y0=0, R=5, a=10, c=1)
        assert z_bound[0] == pytest.approx(11.0)

    with subtests.test("just outside boundary"):
        point = np.array([[5.01], [0.0]])
        z_out = tophat2d(point, x0=0, y0=0, R=5, a=10, c=1)
        assert z_out[0] == pytest.approx(1.0)

    with subtests.test("uses euclidean not rectangular distance"):
        # (4, 4): satisfies |x| < R=5 and |y| < R=5 but sqrt(16+16) > 5
        point = np.array([[4.0], [4.0]])
        z_rect = tophat2d(point, x0=0, y0=0, R=5, a=10, c=1)
        assert z_rect[0] == pytest.approx(1.0)


def test_sinc2d(subtests):
    x = np.linspace(-10, 10, 1001)
    y = np.linspace(-10, 10, 1001)
    X, Y = np.meshgrid(x, y)
    xy = np.array([X, Y])

    with subtests.test("peak at center"):
        # sinc(0)=1; cos term at b=0,kx=0,ky=0 is a+c; plus d
        z = sinc2d(xy, x0=0, y0=0, R=2, a=10, c=0, d=1)
        assert z[500, 500] == pytest.approx(11.0, abs=0.01)

    with subtests.test("zeros at multiples of R along axes"):
        # np.sinc(1) = sin(pi)/pi = 0, so sinc^2(x/R)*sinc^2(0)=0 at x=R
        R = 2
        z = sinc2d(xy, x0=0, y0=0, R=R, a=10, c=0, d=0)
        r_idx = np.argmin(np.abs(x - R))
        assert z[500, r_idx] == pytest.approx(0.0, abs=1e-10)
        assert z[r_idx, 500] == pytest.approx(0.0, abs=1e-10)

    with subtests.test("symmetry about center"):
        z = sinc2d(xy, x0=0, y0=0, R=2, a=10, c=0, d=1)
        assert z[500, 600] == pytest.approx(z[500, 400], abs=1e-10)
        assert z[600, 500] == pytest.approx(z[400, 500], abs=1e-10)

    with subtests.test("global offset d is additive"):
        z_no_d = sinc2d(xy, x0=0, y0=0, R=2, a=10, c=0, d=0)
        z_with_d = sinc2d(xy, x0=0, y0=0, R=2, a=10, c=0, d=5)
        np.testing.assert_array_almost_equal(z_with_d, z_no_d + 5, decimal=10)

