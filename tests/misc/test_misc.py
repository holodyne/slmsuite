"""
Unit tests for slmsuite.misc modules.
"""

import numpy as np
import pytest

from slmsuite.misc.fitfunctions import *
from slmsuite.misc.math import *

# Test misc.math functions


def test_iseven(subtests):
    """Test iseven function with various inputs."""
    with subtests.test("scalar integers"):
        assert iseven(0)
        assert not iseven(1)
        assert iseven(2)
        assert not iseven(3)
        assert iseven(4)
        assert not iseven(-1)
        assert iseven(-2)

    with subtests.test("arrays"):
        x = np.array([0, 1, 2, 3, 4, 5])
        result = iseven(x)
        expected = np.array([True, False, True, False, True, False])
        np.testing.assert_array_equal(result, expected)

    with subtests.test("float inputs"):
        assert iseven(2.1)  # rounds to 2
        assert not iseven(2.9)  # rounds to 3
        assert not iseven(3.1)  # rounds to 3
        assert iseven(3.9)  # rounds to 4


def test_type_tuples(subtests):
    """Test that type tuples are correctly defined."""
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
        assert isinstance(1 + 1j, SCALAR_TYPES)
        assert isinstance(np.complex64(1 + 1j), SCALAR_TYPES)
        assert isinstance(np.complex128(1 + 1j), SCALAR_TYPES)


# Test 1D fit functions


def test_linear(subtests):
    """Test linear function."""
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
    """Test parabola function."""
    with subtests.test("basic"):
        x = np.linspace(-5, 5, 100)
        y = parabola(x, a=1, x0=0, y0=1)
        # Minimum at x0
        assert y[50] == pytest.approx(1.0, abs=0.1)
        # Symmetric
        assert y[0] == pytest.approx(y[-1], abs=0.1)

    with subtests.test("vertex at origin"):
        x = np.linspace(-5, 5, 11)
        y = parabola(x, a=1, x0=0, y0=0)
        assert y[5] == pytest.approx(0.0, abs=0.001)  # At x=0
        assert y[6] == pytest.approx(1.0, abs=0.001)  # At x=1
        assert y[4] == pytest.approx(1.0, abs=0.001)  # At x=-1

    with subtests.test("offset vertex"):
        x = np.linspace(-5, 5, 11)
        y = parabola(x, a=2, x0=1, y0=3)
        expected_at_vertex = 3
        vertex_idx = 6  # x=1 is at index 6
        assert y[vertex_idx] == pytest.approx(expected_at_vertex, abs=0.001)

    with subtests.test("negative a (downward)"):
        x = np.linspace(-5, 5, 11)
        y = parabola(x, a=-1, x0=0, y0=5)
        assert y[5] == pytest.approx(5.0, abs=0.001)  # Maximum at vertex
        assert y[6] < y[5]  # Values decrease away from vertex
        assert y[4] < y[5]


def test_hyperbola(subtests):
    """Test hyperbola function."""
    with subtests.test("basic"):
        z = np.linspace(-10, 10, 100)
        w = hyperbola(z, w0=1, z0=0, zr=1)
        # Minimum at z0
        min_idx = np.argmin(w)
        assert z[min_idx] == pytest.approx(0.0, abs=0.2)
        assert w[min_idx] == pytest.approx(1.0, abs=0.1)

    with subtests.test("value at z0"):
        z = np.array([0, 1, 2, 3, 4, 5])
        w = hyperbola(z, w0=1, z0=2, zr=1)
        assert w[2] == pytest.approx(1.0, abs=0.001)

    with subtests.test("value at z0 +/- zr"):
        w_at_zr = hyperbola(np.array([1, 3]), w0=1, z0=2, zr=1)
        expected = np.sqrt(2)
        np.testing.assert_array_almost_equal(w_at_zr, [expected, expected], decimal=5)

    with subtests.test("symmetry"):
        z_sym = np.array([0, 1, 3, 4])
        w_sym = hyperbola(z_sym, w0=1, z0=2, zr=1)
        assert w_sym[0] == pytest.approx(w_sym[3], abs=0.001)  # z=0 and z=4
        assert w_sym[1] == pytest.approx(w_sym[2], abs=0.001)  # z=1 and z=3


def test_cos(subtests):
    """Test cosine fit function."""
    with subtests.test("basic"):
        x = np.linspace(0, 2 * np.pi, 100)
        y = cos(x, b=0, a=2, c=1, k=1)
        # Check amplitude range
        assert np.max(y) == pytest.approx(3.0, abs=0.1)
        assert np.min(y) == pytest.approx(1.0, abs=0.1)

    with subtests.test("default k"):
        x = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        y = cos(x, b=0, a=2, c=1)
        expected = np.array([3, 2, 1, 2, 3])
        np.testing.assert_array_almost_equal(y, expected, decimal=5)

    with subtests.test("phase offset"):
        x = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        y = cos(x, b=np.pi / 2, a=2, c=1)
        expected = np.array([2, 3, 2, 1, 2])
        np.testing.assert_array_almost_equal(y, expected, decimal=5)

    with subtests.test("k parameter"):
        x = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        y = cos(x, b=0, a=2, c=1, k=2)
        expected = np.array([3, 1, 3, 1, 3])
        np.testing.assert_array_almost_equal(y, expected, decimal=5)


def test_lorentzian(subtests):
    """Test Lorentzian function."""
    with subtests.test("basic"):
        x = np.linspace(990, 1010, 1000)
        y = lorentzian(x, x0=1000, a=10, c=1, w=1)
        # Peak at x0
        max_idx = np.argmax(y)
        assert x[max_idx] == pytest.approx(1000.0, abs=0.1)
        # Check amplitude
        assert y[max_idx] == pytest.approx(11.0, abs=0.1)

    with subtests.test("width comparison"):
        x = np.linspace(990, 1010, 100)
        y_high_q = lorentzian(x, x0=1000, a=10, c=1, w=1)
        y_low_q = lorentzian(x, x0=1000, a=10, c=1, w=10)
        # High Q should be narrower (values away from center should be lower)
        assert y_high_q[10] < y_low_q[10]
        assert y_high_q[50] == pytest.approx(11.0, abs=0.5)
        assert y_low_q[50] == pytest.approx(11.0, abs=0.5)

    with subtests.test("at resonance center"):
        y_center = lorentzian(np.array([1000]), x0=1000, a=10, c=1, w=10)
        assert y_center[0] == pytest.approx(11.0, abs=0.001)  # c + a


def test_gaussian(subtests):
    """Test 1D Gaussian function."""
    with subtests.test("basic"):
        x = np.linspace(-10, 10, 200)
        y = gaussian(x, x0=0, a=10, c=1, w=2)
        # Peak at x0
        max_idx = np.argmax(y)
        assert x[max_idx] == pytest.approx(0.0, abs=0.1)
        # Check amplitude
        assert y[max_idx] == pytest.approx(11.0, abs=0.1)
        # Check baseline
        assert y[0] == pytest.approx(1.0, abs=0.2)

    with subtests.test("width comparison"):
        x = np.linspace(-10, 10, 100)
        gaussian(x, x0=0, a=10, c=1, w=0.1)
        gaussian(x, x0=0, a=10, c=1, w=10)

    with subtests.test("offset peak"):
        x = np.linspace(-10, 10, 100)
        y_offset = gaussian(x, x0=2, a=10, c=1, w=1)
        max_idx = np.argmax(y_offset)
        assert x[max_idx] == pytest.approx(2.0, abs=0.2)  # Peak at x0=2


# Test 2D fit functions


def test_gaussian2d(subtests):
    """Test 2D Gaussian function."""
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    xy = np.array([X, Y])

    with subtests.test("basic"):
        z = gaussian2d(xy, x0=0, y0=0, a=10, c=1, wx=2, wy=2)
        # Peak at center
        assert z[25, 25] == pytest.approx(11.0, abs=0.5)
        # Check baseline at edges
        assert z[0, 0] < 2.0

    with subtests.test("with shear"):
        z = gaussian2d(xy, x0=0, y0=0, a=10, c=1, wx=2, wy=2, wxy=0.5)
        # Should still have peak near center
        max_val = np.max(z)
        assert max_val == pytest.approx(11.0, abs=0.5)


def test_tophat2d(subtests):
    """Test 2D tophat function."""
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    xy = np.array([X, Y])

    z = tophat2d(xy, x0=0, y0=0, R=5, a=10, c=1)

    with subtests.test("center value"):
        assert z[50, 50] == pytest.approx(11.0)

    with subtests.test("edge value"):
        assert z[0, 0] == pytest.approx(1.0)

    with subtests.test("approximate area"):
        high_pixels = np.sum(z > 5)
        expected_pixels = np.pi * (5 / 0.2) ** 2  # R=5, step=0.2
        assert high_pixels == pytest.approx(expected_pixels, rel=0.2)


def test_sinc2d(subtests):
    """Test 2D sinc function."""
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    xy = np.array([X, Y])

    with subtests.test("basic"):
        z = sinc2d(xy, x0=0, y0=0, R=2, a=10, c=0, d=1)
        # Peak at center
        max_idx = np.unravel_index(np.argmax(z), z.shape)
        assert max_idx[0] == pytest.approx(50, abs=2)
        assert max_idx[1] == pytest.approx(50, abs=2)
        # Check that it's symmetric
        assert z[50, 60] == pytest.approx(z[50, 40], abs=0.1)
        assert z[60, 50] == pytest.approx(z[40, 50], abs=0.1)

    with subtests.test("alternate parameters"):
        z = sinc2d(xy, x0=0, y0=0, R=2, a=10, b=0, c=0, d=1)
        # Peak at center
        max_idx = np.unravel_index(np.argmax(z), z.shape)
        assert max_idx[0] == pytest.approx(50, abs=2)
        assert max_idx[1] == pytest.approx(50, abs=2)
