"""
Unit tests for slmsuite.misc modules.
"""
import pytest
import numpy as np

from slmsuite.misc.math import *
from slmsuite.misc.fitfunctions import *


# slmsuite.misc.math

def test_iseven(subtests):
    """Test iseven() parity."""
    with subtests.test("parity of scalar and array integers, including negatives"):
        assert iseven(0) == True
        assert iseven(-1) == False
        x = np.array([0, 1, 2, 3, 4, -1, -2])
        expected = np.array([True, False, True, False, True, False, True])
        np.testing.assert_array_equal(iseven(x), expected)

    with subtests.test("non-integers round before parity is tested, ties to even"):
        # np.around uses banker's rounding: 0.5, 1.5, 2.5, 3.5 round to 0, 2, 2, 4
        x = np.array([2.1, 2.9, 3.1, 3.9, 0.5, 1.5, 2.5, 3.5])
        expected = np.array([True, False, False, True, True, True, True, True])
        np.testing.assert_array_equal(iseven(x), expected)


def test_type_tuples(subtests):
    """Test the INTEGER_TYPES/FLOAT_TYPES/REAL_TYPES/SCALAR_TYPES membership tuples."""
    with subtests.test("REAL_TYPES is the union of INTEGER_TYPES and FLOAT_TYPES"):
        assert int in INTEGER_TYPES and int in REAL_TYPES and int not in FLOAT_TYPES
        assert float in FLOAT_TYPES and float in REAL_TYPES and float not in INTEGER_TYPES

    with subtests.test("SCALAR_TYPES adds complex to REAL_TYPES"):
        assert complex in SCALAR_TYPES and complex not in REAL_TYPES
        for t in REAL_TYPES:
            assert t in SCALAR_TYPES

    with subtests.test("numpy scalar dtypes are members of the matching tuple"):
        for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
            assert isinstance(dtype(1), INTEGER_TYPES)
        for dtype in (np.float32, np.float64):
            assert isinstance(dtype(1.0), FLOAT_TYPES)
        for dtype in (np.complex64, np.complex128):
            assert isinstance(dtype(1 + 1j), SCALAR_TYPES)


# slmsuite.misc.fitfunctions -- 1D

def test_linear(subtests):
    """Test linear() line fit function."""
    with subtests.test("value at x=0 is b; slope between two points is m"):
        y = linear(np.array([-2.0, 0.0, 1.0, 5.0]), m=2, b=3)
        assert y[1] == pytest.approx(3.0)
        assert y[0] == pytest.approx(-1.0)
        assert (y[3] - y[1]) / 5.0 == pytest.approx(2.0)

    with subtests.test("m=0 is a horizontal line at y=b"):
        y = linear(np.array([0, 1, -1, 10, -10]), m=0, b=5)
        np.testing.assert_array_equal(y, np.full(5, 5))


def test_parabola(subtests):
    """Test parabola() fit function."""
    with subtests.test("value at the vertex is y0, and a scales the rise from it"):
        assert parabola(np.array([0.0]), a=3, x0=0, y0=7)[0] == pytest.approx(7.0)
        assert parabola(np.array([1.0]), a=2, x0=0, y0=3)[0] == pytest.approx(5.0)

    with subtests.test("symmetric about x0"):
        y = parabola(np.array([0.0, 2.0]), a=1, x0=1, y0=0)
        assert y[0] == pytest.approx(y[1])

    with subtests.test("doubling the offset from x0 quadruples the rise above y0"):
        y0, a = 3.0, 2.0
        rise_d = parabola(np.array([1.0]), a=a, x0=0, y0=y0)[0] - y0
        rise_2d = parabola(np.array([2.0]), a=a, x0=0, y0=y0)[0] - y0
        assert rise_2d == pytest.approx(4 * rise_d)

    with subtests.test("negative a opens downward"):
        y = parabola(np.linspace(-5, 5, 11), a=-1, x0=0, y0=5)
        assert np.argmax(y) == 5


def test_hyperbola(subtests):
    """Test hyperbola() beam-radius fit function."""
    with subtests.test("value at z0 equals w0"):
        assert hyperbola(np.array([2.0]), w0=3, z0=2, zr=5)[0] == pytest.approx(3.0)

    with subtests.test("value at z0 plus or minus zr is w0*sqrt(2)"):
        w = hyperbola(np.array([1.0, 3.0]), w0=1, z0=2, zr=1)
        np.testing.assert_array_almost_equal(w, [np.sqrt(2), np.sqrt(2)], decimal=10)

    with subtests.test("symmetric about z0"):
        dz = np.array([1.0, 2.0, 3.0, 5.0])
        z0 = 7.0
        w_left = hyperbola(z0 - dz, w0=2, z0=z0, zr=3)
        w_right = hyperbola(z0 + dz, w0=2, z0=z0, zr=3)
        np.testing.assert_array_almost_equal(w_left, w_right, decimal=10)

    with subtests.test("global minimum is at z0"):
        z = np.linspace(-10, 10, 1001)
        w = hyperbola(z, w0=1, z0=0, zr=1)
        assert np.argmin(w) == 500


def test_cos(subtests):
    """Test cos() offset-sinusoid fit function."""
    x = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])

    with subtests.test("y = c + a/2 * (1 + cos(kx - b))"):
        y = cos(x, b=0, a=2, c=1)
        np.testing.assert_array_almost_equal(y, [3.0, 2.0, 1.0, 2.0, 3.0], decimal=10)

    with subtests.test("b shifts the phase"):
        y = cos(x, b=np.pi / 2, a=2, c=1)
        np.testing.assert_array_almost_equal(y, [2.0, 3.0, 2.0, 1.0, 2.0], decimal=10)

    with subtests.test("k scales the frequency"):
        y = cos(x, b=0, a=2, c=1, k=2)
        np.testing.assert_array_almost_equal(y, [3.0, 1.0, 3.0, 1.0, 3.0], decimal=10)

    with subtests.test("range is [c, c+a] regardless of phase"):
        y = cos(np.linspace(0, 2 * np.pi, 1000), b=0, a=2, c=1, k=1)
        assert np.max(y) == pytest.approx(3.0, abs=0.001)
        assert np.min(y) == pytest.approx(1.0, abs=0.001)


def test_lorentzian(subtests):
    """Test lorentzian() resonance fit function."""
    with subtests.test("peak value at x0 is a + c"):
        assert lorentzian(np.array([1000.0]), x0=1000, a=10, c=1, w=1)[0] == pytest.approx(11.0)
        assert lorentzian(np.array([1000.0]), x0=1000, a=10, c=7, w=1)[0] == pytest.approx(17.0)

    with subtests.test("half amplitude at x0 plus or minus w (HWHM)"):
        y = lorentzian(np.array([999.0, 1001.0]), x0=1000, a=10, c=0, w=1)
        np.testing.assert_array_almost_equal(y, [5.0, 5.0], decimal=10)

    with subtests.test("symmetric about x0"):
        dx = np.array([1.0, 2.0, 5.0, 10.0])
        x0 = 500.0
        y_left = lorentzian(x0 - dx, x0=x0, a=10, c=1, w=3)
        y_right = lorentzian(x0 + dx, x0=x0, a=10, c=1, w=3)
        np.testing.assert_array_almost_equal(y_left, y_right, decimal=10)

    with subtests.test("narrower w gives a sharper peak"):
        x = np.linspace(990, 1010, 1000)
        y_narrow = lorentzian(x, x0=1000, a=10, c=1, w=1)
        y_broad = lorentzian(x, x0=1000, a=10, c=1, w=10)
        assert y_narrow[0] < y_broad[0]


def test_gaussian(subtests):
    """Test gaussian() 1D fit function."""
    with subtests.test("value at x0 equals a + c, at center or offset"):
        assert gaussian(np.array([0.0]), x0=0, a=10, c=1, w=2)[0] == pytest.approx(11.0)
        assert gaussian(np.array([3.0]), x0=3, a=10, c=1, w=1)[0] == pytest.approx(11.0)

    with subtests.test("1/e amplitude at x0 plus or minus w*sqrt(2)"):
        w = 3.0
        x_1e = np.array([w * np.sqrt(2), -w * np.sqrt(2)])
        y_1e = gaussian(x_1e, x0=0, a=1, c=0, w=w)
        np.testing.assert_array_almost_equal(y_1e, [np.exp(-1), np.exp(-1)], decimal=10)

    with subtests.test("FWHM is 2*w*sqrt(2*ln2)"):
        w = 2.0
        half_width = w * np.sqrt(2 * np.log(2))
        y_half = gaussian(np.array([half_width, -half_width]), x0=0, a=1, c=0, w=w)
        np.testing.assert_array_almost_equal(y_half, [0.5, 0.5], decimal=10)

    with subtests.test("narrower w falls off faster away from the peak"):
        x = np.linspace(-10, 10, 1001)
        y_narrow = gaussian(x, x0=0, a=10, c=0, w=0.5)
        y_broad = gaussian(x, x0=0, a=10, c=0, w=5)
        far_idx = np.searchsorted(x, 3.0)
        assert y_narrow[far_idx] < y_broad[far_idx]


# slmsuite.misc.fitfunctions -- 2D

def test_gaussian2d(subtests):
    """Test gaussian2d() 2D fit function."""
    with subtests.test("value at (x0, y0) is a + c, at center or offset"):
        assert gaussian2d(np.array([[0.0], [0.0]]), x0=0, y0=0, a=10, c=1, wx=2, wy=2)[0] == pytest.approx(11.0)
        assert gaussian2d(np.array([[2.0], [-3.0]]), x0=2, y0=-3, a=1, c=0, wx=1, wy=1)[0] == pytest.approx(1.0)

    with subtests.test("factors into the product of 1D gaussians when wxy=0"):
        x = np.linspace(-5, 5, 51)
        y = np.linspace(-5, 5, 51)
        X, Y = np.meshgrid(x, y)
        z = gaussian2d(np.array([X, Y]), x0=0, y0=0, a=1, c=0, wx=2, wy=3)
        expected = np.exp(-0.5 * (X / 2) ** 2) * np.exp(-0.5 * (Y / 3) ** 2)
        np.testing.assert_array_almost_equal(z, expected, decimal=10)

    with subtests.test("value at (wx, 0) is a * exp(-0.5)"):
        z = gaussian2d(np.array([[2.0], [0.0]]), x0=0, y0=0, a=1, c=0, wx=2, wy=3)
        assert z[0] == pytest.approx(np.exp(-0.5), rel=1e-10)


def test_tophat2d(subtests):
    """Test tophat2d() fit function."""
    with subtests.test("inside the disk is a + c, outside is c"):
        assert tophat2d(np.array([[0.0], [0.0]]), x0=0, y0=0, R=5, a=10, c=1)[0] == pytest.approx(11.0)
        assert tophat2d(np.array([[20.0], [20.0]]), x0=0, y0=0, R=5, a=10, c=1)[0] == pytest.approx(1.0)

    with subtests.test("boundary r=R is inside (inclusive)"):
        z = tophat2d(np.array([[5.0], [0.0]]), x0=0, y0=0, R=5, a=10, c=1)
        assert z[0] == pytest.approx(11.0)

    with subtests.test("just outside r=R is outside"):
        z = tophat2d(np.array([[5.01], [0.0]]), x0=0, y0=0, R=5, a=10, c=1)
        assert z[0] == pytest.approx(1.0)

    with subtests.test("uses euclidean not Chebyshev distance"):
        # |x|,|y| < R=5 but sqrt(4^2+4^2) > 5
        z = tophat2d(np.array([[4.0], [4.0]]), x0=0, y0=0, R=5, a=10, c=1)
        assert z[0] == pytest.approx(1.0)


def test_sinc2d(subtests):
    """Test sinc2d() 2D fit function."""
    with subtests.test("value at center is a + c + d"):
        z = sinc2d(np.array([[0.0], [0.0]]), x0=0, y0=0, R=2, a=10, c=0, d=1)
        assert z[0] == pytest.approx(11.0)

    with subtests.test("zero at the first sinc null, x=R or y=R"):
        z_x = sinc2d(np.array([[2.0], [0.0]]), x0=0, y0=0, R=2, a=10, c=0, d=0)
        z_y = sinc2d(np.array([[0.0], [2.0]]), x0=0, y0=0, R=2, a=10, c=0, d=0)
        assert z_x[0] == pytest.approx(0.0, abs=1e-10)
        assert z_y[0] == pytest.approx(0.0, abs=1e-10)

    with subtests.test("symmetric about the center along x and y"):
        for point in (np.array([[1.0], [0.0]]), np.array([[0.0], [1.0]])):
            z_pos = sinc2d(point, x0=0, y0=0, R=2, a=10, c=0, d=1)
            z_neg = sinc2d(-point, x0=0, y0=0, R=2, a=10, c=0, d=1)
            assert z_pos[0] == pytest.approx(z_neg[0], abs=1e-10)

    with subtests.test("d is an additive offset"):
        xy = np.array([[0.0, 1.0, 3.0], [0.0, -2.0, 4.0]])
        z_no_d = sinc2d(xy, x0=0, y0=0, R=2, a=10, c=0, d=0)
        z_with_d = sinc2d(xy, x0=0, y0=0, R=2, a=10, c=0, d=5)
        np.testing.assert_array_almost_equal(z_with_d, z_no_d + 5, decimal=10)
