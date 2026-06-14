"""
Unit tests for slmsuite.holography.toolbox.phase module.
"""
import pytest
import numpy as np

from slmsuite.holography.toolbox import phase
from slmsuite.holography.toolbox.phase import (
    _parse_focal_length,
    _zernike_indices_parse,
    _cantor_pairing,
    _inverse_cantor_pairing,
    _parse_out,
    _determine_source_radius,
    _zernike_build_order,
    _zernike_build_indices,
    _zernike_coefficients,
    _zernike_populate_basis_map,
)


@pytest.fixture
def simple_grid():
    """Create a simple 2D grid for testing."""
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    return (X, Y)


@pytest.fixture
def normalized_grid():
    """Create a normalized grid (typical SLM coordinates in wavelengths)."""
    x = np.linspace(-500, 500, 256)  # -500 to 500 wavelengths
    y = np.linspace(-500, 500, 256)
    X, Y = np.meshgrid(x, y)
    return (X, Y)


def test_blaze(simple_grid, subtests, benchmark):
    """Test blaze() phase pattern generation."""
    with subtests.test("benchmark"):
        benchmark(phase.blaze, simple_grid, vector=(0.1, 0.05))

    with subtests.test("zero vector returns zeros"):
        result = phase.blaze(simple_grid, vector=(0, 0))
        assert result.shape == simple_grid[0].shape
        assert np.allclose(result, 0)

    with subtests.test("x-only vector varies only in x"):
        result = phase.blaze(simple_grid, vector=(1, 0))
        assert not np.allclose(result[0, :], result[0, 0])
        assert np.allclose(result[:, 50], result[0, 50])

    with subtests.test("y-only vector varies only in y"):
        result = phase.blaze(simple_grid, vector=(0, 1))
        assert not np.allclose(result[:, 0], result[0, 0])
        assert np.allclose(result[50, :], result[50, 0])

    with subtests.test("linear in k-vector"):
        result1 = phase.blaze(simple_grid, vector=(1, 1))
        result2 = phase.blaze(simple_grid, vector=(2, 2))
        assert np.allclose(2 * result1, result2)

    with subtests.test("3D vector includes focusing term"):
        result_2d = phase.blaze(simple_grid, vector=(1, 1))
        result_3d = phase.blaze(simple_grid, vector=(1, 1, 1))
        assert result_3d.shape == result_2d.shape
        assert not np.allclose(result_2d, result_3d)
        diff = result_3d - result_2d
        r_squared = simple_grid[0]**2 + simple_grid[1]**2
        assert np.allclose(diff, np.pi * r_squared)

        phase_3d = phase.blaze(simple_grid, vector=(0.1, 0.2, 0.5))
        phase_2d = phase.blaze(simple_grid, vector=(0.1, 0.2))
        assert not np.allclose(phase_3d, phase_2d)

    with subtests.test("larger vector produces steeper gradient"):
        blaze_small = phase.blaze(simple_grid, vector=(0.01, 0.01))
        blaze_large = phase.blaze(simple_grid, vector=(0.1, 0.1))
        grad_small = np.max(np.gradient(blaze_small)[0])
        grad_large = np.max(np.gradient(blaze_large)[0])
        assert grad_large > grad_small


def test_sinusoid(simple_grid, subtests):
    """Test sinusoid() phase pattern generation."""
    with subtests.test("zero vector gives constant"):
        result = phase.sinusoid(simple_grid, vector=(0, 0))
        assert np.allclose(result, result[0, 0])

    with subtests.test("output range with a=pi, b=0"):
        result = phase.sinusoid(simple_grid, vector=(1, 0), a=np.pi, b=0)
        assert np.min(result) >= -0.1
        assert np.max(result) <= np.pi + 0.1

    with subtests.test("b offset shifts range"):
        result = phase.sinusoid(simple_grid, vector=(1, 0), a=np.pi, b=np.pi/2)
        assert np.min(result) >= np.pi/2 - 0.1
        assert np.max(result) <= np.pi + 0.1

    with subtests.test("shift parameter changes pattern"):
        result1 = phase.sinusoid(simple_grid, vector=(1, 0), shift=0)
        result2 = phase.sinusoid(simple_grid, vector=(1, 0), shift=np.pi)
        assert not np.allclose(result1, result2)

    with subtests.test("custom amplitude and offset range"):
        phase_custom = phase.sinusoid(simple_grid, vector=(0.1, 0.2), a=2*np.pi, b=np.pi/2)
        assert np.max(phase_custom) <= 2*np.pi + np.pi/2 + 0.1
        assert np.min(phase_custom) >= np.pi/2 - 0.1

    with subtests.test("shift vs unshifted differ"):
        phase_shifted = phase.sinusoid(simple_grid, vector=(0.1, 0.2), shift=np.pi/4)
        phase_unshifted = phase.sinusoid(simple_grid, vector=(0.1, 0.2), shift=0)
        assert not np.allclose(phase_shifted, phase_unshifted)


def test_binary(simple_grid, normalized_grid, subtests):
    """Test binary() grating generation."""
    with subtests.test("produces exactly two levels"):
        result = phase.binary(simple_grid, vector=(1, 0), a=np.pi, b=0)
        unique_vals = np.unique(np.round(result, 6))
        assert len(unique_vals) == 2
        assert 0 in unique_vals or np.isclose(unique_vals.min(), 0)
        assert np.pi in unique_vals or np.isclose(unique_vals.max(), np.pi)

    with subtests.test("duty cycle 0.25"):
        result = phase.binary(simple_grid, vector=(.1, .1), duty_cycle=0.25, a=1, b=0)
        high_fraction = np.sum(result > 0.5) / result.size
        assert high_fraction == pytest.approx(0.25, abs=0.05)

    with subtests.test("integer period with duty cycle 0.75"):
        result = phase.binary(normalized_grid, vector=(4, 0), duty_cycle=0.75, a=np.pi, b=0)
        assert result.shape == normalized_grid[0].shape
        high_fraction = np.sum(result > 0.5) / result.size
        assert high_fraction == pytest.approx(0.75, abs=0.05)

    with subtests.test("extreme duty cycles differ and have structure"):
        phase_thin = phase.binary(simple_grid, vector=(0.1, 0.2), duty_cycle=0.1)
        phase_thick = phase.binary(simple_grid, vector=(0.1, 0.2), duty_cycle=0.9)
        assert not np.allclose(phase_thin, phase_thick)
        assert np.std(phase_thin) > 0
        assert np.std(phase_thick) > 0

    with subtests.test("zero vector, no shift returns b"):
        result = phase.binary(simple_grid, vector=(0, 0), a=np.pi, b=0)
        assert np.allclose(result, 0)

    with subtests.test("zero vector with shift beyond duty returns a"):
        result = phase.binary(simple_grid, vector=(0, 0), a=np.pi, b=0,
                              shift=np.pi, duty_cycle=0.25)
        assert np.allclose(result, np.pi)

    with subtests.test("zero vector with small shift returns b"):
        result = phase.binary(simple_grid, vector=(0, 0), a=np.pi, b=0,
                              shift=0.1, duty_cycle=0.5)
        assert np.allclose(result, 0)

    with subtests.test("x-only vector uses single-axis path"):
        result = phase.binary(simple_grid, vector=(0.1, 0), a=np.pi, b=0)
        unique = np.unique(np.round(result, 6))
        assert len(unique) == 2

    with subtests.test("y-only vector uses single-axis path"):
        result = phase.binary(simple_grid, vector=(0, 0.1), a=np.pi, b=0)
        unique = np.unique(np.round(result, 6))
        assert len(unique) == 2

    with subtests.test("pixel-period mode (vector > 1)"):
        result = phase.binary(simple_grid, vector=(10, 0), a=np.pi, b=0)
        assert result.shape == simple_grid[0].shape
        assert np.std(result) > 0


def test_lens(simple_grid, subtests, benchmark):
    """Test lens() phase pattern generation."""
    with subtests.test("benchmark"):
        benchmark(phase.lens, simple_grid, f=(1000, 1000))

    with subtests.test("infinite focal length gives zeros"):
        result = phase.lens(simple_grid, f=(np.inf, np.inf))
        assert np.allclose(result, 0)

    with subtests.test("symmetric about center"):
        result = phase.lens(simple_grid, f=(100, 100))
        centery = result.shape[0] // 2
        centerx = result.shape[1] // 2
        d = 20
        dx = result[centery, centerx-d] - result[centery, centerx+d]
        dy = result[centery-d, centerx] - result[centery+d, centerx]
        assert dx == pytest.approx(0, abs=.1)
        assert dy == pytest.approx(0, abs=.1)

    with subtests.test("negative focal length negates phase"):
        phase_pos = phase.lens(simple_grid, f=(10, 10))
        phase_neg = phase.lens(simple_grid, f=(-10, -10))
        assert np.allclose(phase_pos, -phase_neg, atol=1e-10)

    with subtests.test("mixed focal lengths bounded"):
        phase_pos = phase.lens(simple_grid, f=(10, 10))
        phase_neg = phase.lens(simple_grid, f=(-10, -10))
        phase_pos_neg = phase.lens(simple_grid, f=(10, -10))
        assert np.all(np.abs(phase_pos_neg) <= phase_pos - phase_neg + 1e-10)

    with subtests.test("y-only cylindrical lens"):
        result = phase.lens(simple_grid, f=(np.inf, 100))
        assert not np.allclose(result[:, 50], result[0, 50])
        expected = (np.pi / 100) * np.square(simple_grid[1])
        assert np.allclose(result, expected)

    with subtests.test("scalar focal length"):
        result = phase.lens(simple_grid, f=50)
        expected = (np.pi / 50) * (np.square(simple_grid[0]) + np.square(simple_grid[1]))
        assert np.allclose(result, expected)


def test_axicon(simple_grid, normalized_grid, subtests):
    """Test axicon() phase pattern generation."""
    with subtests.test("infinite focal length gives constant"):
        result = phase.axicon(simple_grid, f=(np.inf, np.inf))
        assert np.allclose(result, result[0, 0])
        assert np.std(result) < 0.1

    with subtests.test("linear in radius shape check"):
        result = phase.axicon(normalized_grid, f=(1000, 1000))
        r = np.sqrt(normalized_grid[0]**2 + normalized_grid[1]**2)
        center = result.shape[0] // 2
        r_center = r[center-50:center+50, center-50:center+50]
        result_center = result[center-50:center+50, center-50:center+50]
        assert result_center.shape == r_center.shape

    with subtests.test("y-only axicon (x-axis infinite)"):
        result = phase.axicon(simple_grid, f=(np.inf, 100), w=5.0)
        assert result.shape == simple_grid[0].shape
        assert np.all(np.isfinite(result))
        angle = 5.0 / 100 / 2
        expected = (2 * np.pi * angle) * np.abs(simple_grid[1])
        assert np.allclose(result, expected)

    with subtests.test("x-only axicon (y-axis infinite)"):
        result = phase.axicon(simple_grid, f=(100, np.inf), w=5.0)
        angle = 5.0 / 100 / 2
        expected = (2 * np.pi * angle) * np.abs(simple_grid[0])
        assert np.allclose(result, expected)

    with subtests.test("both finite gives sqrt form"):
        result = phase.axicon(simple_grid, f=(100, 200), w=5.0)
        assert result.shape == simple_grid[0].shape
        assert np.all(result >= 0)


def test_zernike(normalized_grid, subtests):
    """Test zernike() polynomials and related utilities."""
    with subtests.test("piston mode (j=0) is constant"):
        result = phase.zernike(normalized_grid, index=0)
        assert np.allclose(result, result[0, 0])

    with subtests.test("x-tilt mode (j=1) varies in x"):
        result = phase.zernike(normalized_grid, index=1)
        assert not np.allclose(result[128, :], result[0, :])

    with subtests.test("y-tilt mode (j=2) varies in y"):
        result = phase.zernike(normalized_grid, index=2)
        assert not np.allclose(result[:, 128], result[:, 0])

    with subtests.test("weight parameter scales linearly"):
        result1 = phase.zernike(normalized_grid, index=1, weight=1)
        result2 = phase.zernike(normalized_grid, index=1, weight=2)
        assert np.allclose(2 * result1, result2)

    with subtests.test("higher order (j=10) is nontrivial"):
        z_high = phase.zernike(normalized_grid, index=10)
        assert z_high.shape == normalized_grid[0].shape
        assert not np.allclose(z_high, 0)

    with subtests.test("weight scaling for j=5"):
        z_normal = phase.zernike(normalized_grid, index=5, weight=1.0)
        z_scaled = phase.zernike(normalized_grid, index=5, weight=2.0)
        np.testing.assert_array_almost_equal(z_scaled, 2 * z_normal)


def test_quadrants(simple_grid):
    """Test that quadrants creates four distinct regions."""
    result = phase.quadrants(simple_grid, radius=.001*np.sqrt(2), center=(0, 0))
    result -= phase.blaze(simple_grid, vector=(.001, .001))

    result = np.around(result * 1000)
    vals, counts = np.unique(result, return_counts=True)
    mode = vals[np.argmax(counts)]

    assert np.sum(mode == result) == pytest.approx(.25 * result.size, abs=0.05 * result.size)


def test_bahtinov(simple_grid):
    """Test Bahtinov mask generation."""
    result = phase.bahtinov(simple_grid, radius=0.005)
    assert result.shape == simple_grid[0].shape
    assert np.all(np.isfinite(result))


def test_phase_functions_general(simple_grid, subtests):
    """Test general properties across all phase functions."""
    functions_to_test = [
        (phase.blaze, {"vector": (1, 1)}),
        (phase.sinusoid, {"vector": (1, 1)}),
        (phase.binary, {"vector": (1, 1)}),
        (phase.lens, {"f": (100, 100)}),
    ]

    expected_shape = simple_grid[0].shape

    for func, kwargs in functions_to_test:
        with subtests.test(f"{func.__name__} returns real finite values"):
            result = func(simple_grid, **kwargs)
            assert np.isrealobj(result), f"{func.__name__} returned complex values"
            assert np.all(np.isfinite(result)), f"{func.__name__} returned non-finite values"

        with subtests.test(f"{func.__name__} preserves shape"):
            result = func(simple_grid, **kwargs)
            assert result.shape == expected_shape, f"{func.__name__} changed shape"


def test_zernike_aperture(normalized_grid, subtests):
    """Test zernike_aperture() scaling helper."""
    with subtests.test("circular aperture is isotropic"):
        x_scale, y_scale = phase.zernike_aperture(normalized_grid, aperture="circular")
        assert x_scale == pytest.approx(y_scale)
        # Scale times max coordinate should give 1
        assert x_scale * np.nanmax(normalized_grid[0]) == pytest.approx(1, rel=1e-6)

    with subtests.test("elliptical aperture may be anisotropic"):
        x = np.linspace(-200, 200, 128)
        y = np.linspace(-500, 500, 128)
        X, Y = np.meshgrid(x, y)
        rect_grid = (X, Y)
        x_scale, y_scale = phase.zernike_aperture(rect_grid, aperture="elliptical")
        # Each axis maps independently
        assert x_scale == pytest.approx(1 / 200, rel=1e-6)
        assert y_scale == pytest.approx(1 / 500, rel=1e-6)

    with subtests.test("cropped aperture circumscribes rectangle"):
        x_scale, y_scale = phase.zernike_aperture(normalized_grid, aperture="cropped")
        assert x_scale == pytest.approx(y_scale)
        # For a square grid the corner distance is sqrt(2)*max
        max_coord = np.nanmax(normalized_grid[0])
        expected = 1 / np.sqrt(2 * max_coord**2)
        assert x_scale == pytest.approx(expected, rel=1e-6)

    with subtests.test("scalar aperture"):
        x_scale, y_scale = phase.zernike_aperture(normalized_grid, aperture=0.005)
        assert x_scale == pytest.approx(0.005)
        assert y_scale == pytest.approx(0.005)

    with subtests.test("tuple aperture"):
        x_scale, y_scale = phase.zernike_aperture(normalized_grid, aperture=(0.01, 0.02))
        assert x_scale == pytest.approx(0.01)
        assert y_scale == pytest.approx(0.02)

    with subtests.test("invalid string raises ValueError"):
        with pytest.raises(ValueError):
            phase.zernike_aperture(normalized_grid, aperture="invalid")

    with subtests.test("None aperture defaults to cropped for grids"):
        x_scale, y_scale = phase.zernike_aperture(normalized_grid, aperture=None)
        x_s2, y_s2 = phase.zernike_aperture(normalized_grid, aperture="cropped")
        assert x_scale == pytest.approx(x_s2)
        assert y_scale == pytest.approx(y_s2)

    with subtests.test("SLM-like object with get_source_zernike_scaling"):
        class FakeSLM:
            def __init__(self, grid):
                self.x_grid, self.y_grid = grid
            def get_source_zernike_scaling(self):
                return (0.01, 0.02)
        fake = FakeSLM(normalized_grid)
        fake.x_grid = normalized_grid[0]
        fake.y_grid = normalized_grid[1]
        x_scale, y_scale = phase.zernike_aperture(fake, aperture=None)
        assert x_scale == 0.01
        assert y_scale == 0.02

    with subtests.test("CameraSLM-like object delegates to slm"):
        class FakeCameraSLM:
            def __init__(self, grid):
                self.x_grid, self.y_grid = grid
                self.slm = type('FakeSLM', (), {
                    'get_source_zernike_scaling': lambda self_: (0.03, 0.04),
                    'x_grid': grid[0],
                    'y_grid': grid[1],
                })()
                self.cam = True
        fake = FakeCameraSLM(normalized_grid)
        x_scale, y_scale = phase.zernike_aperture(fake, aperture=None)
        assert x_scale == 0.03
        assert y_scale == 0.04

    with subtests.test("unrecognized type raises ValueError"):
        with pytest.raises(ValueError, match="not recognized"):
            phase.zernike_aperture(normalized_grid, aperture=object())


def test_zernike_get_string(subtests):
    """Test zernike_get_string() LaTeX representations."""
    with subtests.test("piston (j=0) is constant '1'"):
        s = phase.zernike_get_string(0)
        assert s == "1"

    with subtests.test("tilt (j=1) contains y"):
        s = phase.zernike_get_string(1)
        assert "y" in s

    with subtests.test("tilt (j=2) contains x"):
        s = phase.zernike_get_string(2)
        assert "x" in s

    with subtests.test("defocus (j=4) contains x^2 and y^2"):
        s = phase.zernike_get_string(4)
        assert "x^2" in s
        assert "y^2" in s

    with subtests.test("derivative reduces order"):
        s_orig = phase.zernike_get_string(4, derivative=(0, 0))
        s_dx = phase.zernike_get_string(4, derivative=(1, 0))
        # Derivative should not contain x^2
        assert "x^2" not in s_dx
        assert "x" in s_dx  # Should still contain x^1 term

    with subtests.test("derivative zeroes out constant term"):
        s = phase.zernike_get_string(0, derivative=(1, 0))
        assert s == "0"

    with subtests.test("second derivative"):
        s = phase.zernike_get_string(4, derivative=(2, 0))
        assert "x" not in s

    with subtests.test("higher order index"):
        s = phase.zernike_get_string(10)
        assert len(s) > 0


def test_zernike_convert_index(subtests):
    """Test zernike_convert_index() roundtrip conversions."""
    with subtests.test("ansi -> radial -> ansi roundtrip"):
        indices = np.arange(15)
        radial = phase.zernike_convert_index(indices, from_index="ansi", to_index="radial")
        back = phase.zernike_convert_index(radial, from_index="radial", to_index="ansi")
        np.testing.assert_array_equal(back.ravel(), indices)

    with subtests.test("ansi -> noll"):
        noll = phase.zernike_convert_index([0, 1, 2, 3], from_index="ansi", to_index="noll")
        # Noll is 1-indexed; j=0 (piston) -> noll=1
        assert noll.ravel()[0] == 1

    with subtests.test("ansi -> fringe"):
        fringe = phase.zernike_convert_index([0, 1, 2], from_index="ansi", to_index="fringe")
        # Fringe is 1-indexed; piston j=0->1, x-tilt j=1->3 (odd), y-tilt j=2->2
        np.testing.assert_array_equal(fringe.ravel(), [1, 3, 2])

    with subtests.test("ansi -> wyant"):
        wyant = phase.zernike_convert_index([0, 1, 2], from_index="ansi", to_index="wyant")
        assert wyant.ravel()[0] == 0  # Wyant is 0-indexed

    with subtests.test("invalid index raises ValueError"):
        with pytest.raises(ValueError):
            phase.zernike_convert_index([0], from_index="bogus", to_index="ansi")

    with subtests.test("scalar index"):
        result = phase.zernike_convert_index(3, from_index="ansi", to_index="radial")
        assert result.shape == (1, 2)
        assert isinstance(result[0, 0], (int, np.integer))

    with subtests.test("list of indices"):
        result = phase.zernike_convert_index([3, 4, 5], from_index="ansi", to_index="radial")
        assert result.shape == (3, 2)
        assert isinstance(result[0, 0], (int, np.integer))

    with subtests.test("ansi -> noll roundtrip via radial"):
        indices_ansi = np.arange(10)
        noll = phase.zernike_convert_index(indices_ansi, "ansi", "noll")
        assert noll.ravel()[0] == 1
        assert len(noll.ravel()) == 10

    with subtests.test("ansi -> wyant conversion"):
        wyant = phase.zernike_convert_index([0, 1, 2, 3, 4], "ansi", "wyant")
        assert wyant.ravel()[0] == 0

    with subtests.test("radial -> fringe conversion"):
        radial = np.array([[0, 0], [1, -1], [1, 1], [2, 0], [2, -2]])
        fringe = phase.zernike_convert_index(radial, "radial", "fringe")
        assert fringe.ravel()[0] == 1

    with subtests.test("radial -> noll"):
        radial = np.array([[0, 0], [1, -1], [1, 1], [2, -2]])
        noll = phase.zernike_convert_index(radial, "radial", "noll")
        assert noll.ravel()[0] == 1

    with subtests.test("radial -> wyant"):
        radial = np.array([[0, 0], [1, -1], [1, 1]])
        wyant = phase.zernike_convert_index(radial, "radial", "wyant")
        assert len(wyant.ravel()) == 3

    with subtests.test("radial -> ansi"):
        radial = np.array([[0, 0], [1, -1], [1, 1], [2, 0]])
        ansi = phase.zernike_convert_index(radial, "radial", "ansi")
        np.testing.assert_array_equal(ansi.ravel(), [0, 1, 2, 4])

    with subtests.test("invalid to_index raises ValueError"):
        with pytest.raises(ValueError):
            phase.zernike_convert_index([0], "ansi", "bogus")

    with subtests.test("noll from_index raises NotImplementedError"):
        with pytest.raises(NotImplementedError):
            phase.zernike_convert_index([1], "noll", "ansi")

    with subtests.test("wyant from_index raises NotImplementedError"):
        with pytest.raises(NotImplementedError):
            phase.zernike_convert_index([0], "wyant", "ansi")

    with subtests.test("same index is identity"):
        indices = np.arange(5)
        result = phase.zernike_convert_index(indices, "ansi", "ansi")
        np.testing.assert_array_equal(result.ravel(), indices)


def test_zernike_sum(normalized_grid, subtests, benchmark):
    """Test zernike_sum() advanced features."""
    with subtests.test("benchmark"):
        rng = np.random.default_rng(42)
        coeffs = rng.normal(0, 0.1, 10)
        benchmark(phase.zernike_sum, normalized_grid, indices=list(range(len(coeffs))), weights=coeffs)

    with subtests.test("use_mask='return' gives boolean mask"):
        mask = phase.zernike_sum(
            normalized_grid, indices=[0], weights=[1], use_mask="return"
        )
        assert mask.dtype == bool
        assert mask.shape == normalized_grid[0].shape

    with subtests.test("use_mask=True zeros outside aperture"):
        result = phase.zernike_sum(
            normalized_grid, indices=[4], weights=[1], use_mask=True, aperture="circular"
        )
        mask = phase.zernike_sum(
            normalized_grid, indices=[0], weights=[1],
            use_mask="return", aperture="circular"
        )
        # Outside the mask should be zero
        assert np.allclose(result[~mask], 0)

    with subtests.test("use_mask=False does not crop"):
        result = phase.zernike_sum(
            normalized_grid, indices=[4], weights=[1], use_mask=False
        )
        # Should have non-zero values everywhere (defocus is nonzero at corners)
        assert not np.allclose(result, 0)

    with subtests.test("derivative (1,0) of tilt-x is constant"):
        # Z_2 = x, so d/dx = 1 (up to scaling)
        result = phase.zernike_sum(
            normalized_grid, indices=[2], weights=[1],
            use_mask=False, derivative=(1, 0)
        )
        # Should be approximately constant and nonzero
        assert np.std(result) < 1e-10
        assert not np.allclose(result, 0)

    with subtests.test("stacked weights (D, N) returns 3D"):
        weights_2d = np.array([[1, 0], [0, 1]])  # Two polynomials, two stacks
        result = phase.zernike_sum(
            normalized_grid, indices=[1, 2], weights=weights_2d
        )
        assert result.ndim == 3
        assert result.shape[0] == 2

    with subtests.test("out parameter reuses memory"):
        out = np.zeros((1, *normalized_grid[0].shape), dtype=normalized_grid[0].dtype)
        result = phase.zernike_sum(
            normalized_grid, indices=[1], weights=[1], out=out
        )
        # result should share memory with out
        assert np.shares_memory(result, out)

    with subtests.test("produces valid array"):
        indices = [0, 1, 2]
        weights = [1, 0.5, 0.3]
        result = phase.zernike_sum(normalized_grid, indices, weights)
        assert result.shape == normalized_grid[0].shape
        assert np.all(np.isfinite(result))

    with subtests.test("scalar index and scalar weight"):
        result = phase.zernike_sum(normalized_grid, indices=4, weights=1.0)
        assert result.shape == normalized_grid[0].shape

    with subtests.test("use_mask=nan gives nan outside"):
        result = phase.zernike_sum(
            normalized_grid, indices=[4], weights=[1],
            use_mask=np.nan, aperture="circular"
        )
        assert np.any(np.isnan(result))

    with subtests.test("derivative length != 2 raises"):
        with pytest.raises(ValueError, match="Expected derivative"):
            phase.zernike_sum(normalized_grid, [0], [1], derivative=(1,))

    with subtests.test("weights 3D raises"):
        with pytest.raises(ValueError, match="1D or 2D"):
            phase.zernike_sum(normalized_grid, [0, 1, 2], np.ones((3, 2, 2)))

    with subtests.test("mismatched weights raises"):
        with pytest.raises(ValueError, match="common dimension"):
            phase.zernike_sum(normalized_grid, [0, 1], [1.0, 2.0, 3.0])

    with subtests.test("indices=None defaults by D"):
        result = phase.zernike_sum(normalized_grid, indices=None, weights=[1, 1])
        assert result.shape == normalized_grid[0].shape

    with subtests.test("second derivative d^2/dx^2 of Z4"):
        result = phase.zernike_sum(
            normalized_grid, indices=[4], weights=[1],
            use_mask=False, derivative=(2, 0)
        )
        assert result.shape == normalized_grid[0].shape
        assert np.std(result) < 1e-8
        assert np.mean(result) != 0

    with subtests.test("mixed derivative d/dxdy of Z3"):
        result = phase.zernike_sum(
            normalized_grid, indices=[3], weights=[1],
            use_mask=False, derivative=(1, 1)
        )
        assert np.std(result) < 1e-8


def test_polynomial(simple_grid, subtests):
    """Test polynomial() monomial summation."""
    with subtests.test("constant term (x^0 * y^0)"):
        # term (0,0) with weight 5 should give constant 5
        result = phase.polynomial(simple_grid, weights=[5.0], terms=np.array([[0, 0]]))
        # Result is (1, H, W); squeeze it
        result = result.squeeze()
        assert result.shape == simple_grid[0].shape
        assert np.allclose(result, 5.0)

    with subtests.test("linear x term"):
        # term (1,0) = x with weight 1
        result = phase.polynomial(simple_grid, weights=[1.0], terms=np.array([[1, 0]]))
        result = result.squeeze()
        assert np.allclose(result, simple_grid[0])

    with subtests.test("linear y term"):
        # term (0,1) = y with weight 1
        result = phase.polynomial(simple_grid, weights=[1.0], terms=np.array([[0, 1]]))
        result = result.squeeze()
        assert np.allclose(result, simple_grid[1])

    with subtests.test("quadratic x^2 + y^2"):
        terms = np.array([[2, 0], [0, 2]])
        weights = [1.0, 1.0]
        result = phase.polynomial(simple_grid, weights=weights, terms=terms).squeeze()
        expected = simple_grid[0]**2 + simple_grid[1]**2
        assert np.allclose(result, expected)

    with subtests.test("stacked weights produce multiple outputs"):
        terms = np.array([[1, 0], [0, 1]])
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])  # (D=2, N=2)
        result = phase.polynomial(simple_grid, weights=weights, terms=terms)
        assert result.shape[0] == 2
        # First stack: 1*x + 3*y
        expected_0 = 1.0 * simple_grid[0] + 3.0 * simple_grid[1]
        assert np.allclose(result[0], expected_0)
        # Second stack: 2*x + 4*y
        expected_1 = 2.0 * simple_grid[0] + 4.0 * simple_grid[1]
        assert np.allclose(result[1], expected_1)

    with subtests.test("1D cantor terms"):
        result = phase.polynomial(simple_grid, weights=[1.0, 1.0], terms=np.array([1, 2]))
        result = result.squeeze()
        expected = simple_grid[0] + simple_grid[1]
        assert np.allclose(result, expected)

    with subtests.test("pathing=False disables optimization"):
        terms = np.array([[2, 0], [0, 2]])
        result = phase.polynomial(simple_grid, weights=[1.0, 1.0],
                                  terms=terms, pathing=False)
        result = result.squeeze()
        expected = simple_grid[0]**2 + simple_grid[1]**2
        assert np.allclose(result, expected)

    with subtests.test("bad terms shape raises"):
        with pytest.raises(ValueError, match="Terms must be"):
            phase.polynomial(simple_grid, weights=[1.0],
                            terms=np.array([[1, 0, 0]]))

    with subtests.test("mismatched weights 1D raises"):
        with pytest.raises(ValueError, match="common dimension"):
            phase.polynomial(simple_grid, weights=[1.0, 2.0, 3.0],
                            terms=np.array([[1, 0], [0, 1]]))

    with subtests.test("mismatched weights 2D raises"):
        with pytest.raises(ValueError, match="common dimension"):
            phase.polynomial(simple_grid, weights=np.ones((3, 1)),
                            terms=np.array([[1, 0], [0, 1]]))

    with subtests.test("3D weights raises"):
        with pytest.raises(ValueError, match="1D or 2D"):
            phase.polynomial(simple_grid, weights=np.ones((2, 1, 1)),
                            terms=np.array([[1, 0], [0, 1]]))

    with subtests.test("vortex waveplate term (-1, 0)"):
        terms = np.array([[-1, 0]])
        result = phase.polynomial(simple_grid, weights=[1.0], terms=terms)
        result = result.squeeze()
        expected = np.arctan2(simple_grid[1], simple_grid[0])
        assert np.allclose(result, expected)

    with subtests.test("unrecognized negative term raises"):
        terms = np.array([[-2, 0]])
        with pytest.raises(ValueError, match="Unrecognized terms"):
            phase.polynomial(simple_grid, weights=[1.0], terms=terms)

    with subtests.test("path reset on non-monotonic terms"):
        terms = np.array([[2, 0], [0, 2], [1, 0]])
        result = phase.polynomial(simple_grid, weights=[1.0, 1.0, 1.0], terms=terms)
        result = result.squeeze()
        expected = simple_grid[0]**2 + simple_grid[1]**2 + simple_grid[0]
        assert np.allclose(result, expected)

    with subtests.test("out parameter reuses memory"):
        terms = np.array([[1, 0]])
        out = np.zeros((1, *simple_grid[0].shape), dtype=simple_grid[0].dtype)
        result = phase.polynomial(simple_grid, weights=[1.0], terms=terms, out=out)
        assert np.shares_memory(result, out)


def test_laguerre_gaussian(simple_grid, subtests):
    """Test laguerre_gaussian() structured light generation."""
    with subtests.test("l=0, p=0 gives zero (scalar)"):
        result = phase.laguerre_gaussian(simple_grid, l=0, p=0)
        # With l=0, p=0 the function returns the scalar 0
        assert np.allclose(result, 0)

    with subtests.test("l=1 produces vortex (azimuthal variation)"):
        result = phase.laguerre_gaussian(simple_grid, l=1, p=0)
        assert result.shape == simple_grid[0].shape
        # Should wrap 2pi around the center
        assert np.ptp(result) > 0

    with subtests.test("l=-1 is negation of l=1"):
        result_pos = phase.laguerre_gaussian(simple_grid, l=1, p=0)
        result_neg = phase.laguerre_gaussian(simple_grid, l=-1, p=0)
        assert np.allclose(result_pos, -result_neg)

    with subtests.test("higher l produces steeper vortex"):
        result_l1 = phase.laguerre_gaussian(simple_grid, l=1, p=0)
        result_l3 = phase.laguerre_gaussian(simple_grid, l=3, p=0)
        # l=3 should have 3x the angular phase ramp
        assert np.allclose(result_l3, 3 * result_l1)

    with subtests.test("p>0 adds radial rings"):
        result_p0 = phase.laguerre_gaussian(simple_grid, l=1, p=0)
        result_p1 = phase.laguerre_gaussian(simple_grid, l=1, p=1)
        # p=1 should differ from p=0
        assert not np.allclose(result_p0, result_p1)
        # p=1 should contain pi-shifted regions (from Heaviside on Laguerre)
        unique_extra = np.unique(np.round(result_p1 - result_p0, 4))
        assert len(unique_extra) > 1

    with subtests.test("custom w parameter"):
        result_default = phase.laguerre_gaussian(simple_grid, l=1, p=1)
        result_custom = phase.laguerre_gaussian(simple_grid, l=1, p=1, w=2.0)
        # Different w should give different patterns
        assert not np.allclose(result_default, result_custom)

    with subtests.test("w=None uses default"):
        result = phase.laguerre_gaussian(simple_grid, l=1, p=1, w=None)
        assert result.shape == simple_grid[0].shape
        assert np.all(np.isfinite(result))

    with subtests.test("p=0 l=0 w=None gives scalar zero"):
        result = phase.laguerre_gaussian(simple_grid, l=0, p=0, w=None)
        assert np.allclose(result, 0)


def test_hermite_gaussian(simple_grid, subtests):
    """Test hermite_gaussian() structured light generation."""
    with subtests.test("n=0, m=0 gives flat phase"):
        result = phase.hermite_gaussian(simple_grid, n=0, m=0)
        assert result.shape == simple_grid[0].shape
        # HG_00 Hermite is constant positive, so phase should be pi everywhere
        assert np.allclose(result, np.pi) or np.allclose(result, 0)

    with subtests.test("n=1, m=0 has checkerboard in x"):
        result = phase.hermite_gaussian(simple_grid, n=1, m=0)
        assert result.shape == simple_grid[0].shape
        unique_vals = np.unique(result)
        # Should have exactly two phase levels: 0 and pi
        assert len(unique_vals) == 2
        assert np.isclose(unique_vals[0], 0)
        assert np.isclose(unique_vals[1], np.pi)

    with subtests.test("n=0, m=1 has checkerboard in y"):
        result = phase.hermite_gaussian(simple_grid, n=0, m=1)
        unique_vals = np.unique(result)
        assert len(unique_vals) == 2

    with subtests.test("higher orders produce more regions"):
        result_low = phase.hermite_gaussian(simple_grid, n=1, m=0)
        result_high = phase.hermite_gaussian(simple_grid, n=3, m=0)
        # Higher n should have more sign transitions
        transitions_low = np.sum(np.abs(np.diff(result_low, axis=1)) > 0.1)
        transitions_high = np.sum(np.abs(np.diff(result_high, axis=1)) > 0.1)
        assert transitions_high > transitions_low

    with subtests.test("custom w parameter changes pattern"):
        # Use a grid where default w and custom w give different binary patterns
        x = np.linspace(-50, 50, 200)
        y = np.linspace(-50, 50, 200)
        X, Y = np.meshgrid(x, y)
        wide_grid = (X, Y)
        # default w = min(50,50)/4 = 12.5
        result_default = phase.hermite_gaussian(wide_grid, n=2, m=0)
        result_custom = phase.hermite_gaussian(wide_grid, n=2, m=0, w=3.0)
        assert not np.allclose(result_default, result_custom)

    with subtests.test("w=None uses default"):
        result = phase.hermite_gaussian(simple_grid, n=1, m=1, w=None)
        assert result.shape == simple_grid[0].shape

    with subtests.test("n=2, m=2 produces multi-region pattern"):
        result = phase.hermite_gaussian(simple_grid, n=2, m=2, w=None)
        unique = np.unique(result)
        assert len(unique) == 2


def test_ince_gaussian(simple_grid, subtests):
    """Test ince_gaussian() structured light generation."""
    from slmsuite.holography.toolbox.phase import _ince_polynomial

    with subtests.test("even parity invalid m>p raises ValueError"):
        with pytest.raises(ValueError, match="invalid Ince"):
            phase.ince_gaussian(simple_grid, p=2, m=5, parity=1)

    with subtests.test("odd parity m=0 raises ValueError"):
        with pytest.raises(ValueError, match="invalid Ince"):
            phase.ince_gaussian(simple_grid, p=2, m=0, parity=-1)

    with subtests.test("parity mismatch p even m odd raises ValueError"):
        with pytest.raises(ValueError, match="same parity"):
            phase.ince_gaussian(simple_grid, p=2, m=1, parity=1)

    with subtests.test("parity mismatch p odd m even raises ValueError"):
        with pytest.raises(ValueError, match="same parity"):
            phase.ince_gaussian(simple_grid, p=3, m=2, parity=-1)

    with subtests.test("even mode shape matches grid"):
        result = phase.ince_gaussian(simple_grid, p=2, m=2, parity=1)
        assert result.shape == simple_grid[0].shape

    with subtests.test("odd mode shape matches grid"):
        result = phase.ince_gaussian(simple_grid, p=3, m=1, parity=-1)
        assert result.shape == simple_grid[0].shape

    with subtests.test("helical mode shape matches grid"):
        result = phase.ince_gaussian(simple_grid, p=4, m=2, parity=0)
        assert result.shape == simple_grid[0].shape

    with subtests.test("even mode all finite"):
        result = phase.ince_gaussian(simple_grid, p=2, m=2, parity=1)
        assert np.all(np.isfinite(result))

    with subtests.test("odd mode all finite"):
        result = phase.ince_gaussian(simple_grid, p=3, m=1, parity=-1)
        assert np.all(np.isfinite(result))

    with subtests.test("helical mode all finite"):
        result = phase.ince_gaussian(simple_grid, p=4, m=2, parity=0)
        assert np.all(np.isfinite(result))

    with subtests.test("even mode binary phase (0 or pi)"):
        result = phase.ince_gaussian(simple_grid, p=4, m=2, parity=1)
        unique = np.unique(np.round(result, 8))
        assert set(unique).issubset({0.0, np.round(np.pi, 8), np.round(-np.pi, 8)})

    with subtests.test("odd mode binary phase (0 or pi)"):
        result = phase.ince_gaussian(simple_grid, p=3, m=1, parity=-1)
        unique = np.unique(np.round(result, 8))
        assert set(unique).issubset({0.0, np.round(np.pi, 8), np.round(-np.pi, 8)})

    with subtests.test("helical mode continuous phase"):
        result = phase.ince_gaussian(simple_grid, p=4, m=2, parity=0)
        unique = np.unique(np.round(result, 4))
        # Helical modes should have many distinct phase values, not binary
        assert len(unique) > 10

    with subtests.test("helical phase in [-pi, pi]"):
        result = phase.ince_gaussian(simple_grid, p=4, m=2, parity=0)
        assert np.all(result >= -np.pi - 1e-10)
        assert np.all(result <= np.pi + 1e-10)

    with subtests.test("(p=2, m=0) even works"):
        result = phase.ince_gaussian(simple_grid, p=2, m=0, parity=1)
        assert result.shape == simple_grid[0].shape
        assert np.all(np.isfinite(result))

    with subtests.test("(p=4, m=4) even works"):
        result = phase.ince_gaussian(simple_grid, p=4, m=4, parity=1)
        assert result.shape == simple_grid[0].shape
        assert np.all(np.isfinite(result))

    with subtests.test("(p=5, m=3) odd works"):
        result = phase.ince_gaussian(simple_grid, p=5, m=3, parity=-1)
        assert result.shape == simple_grid[0].shape
        assert np.all(np.isfinite(result))

    with subtests.test("(p=6, m=2) helical works"):
        result = phase.ince_gaussian(simple_grid, p=6, m=2, parity=0)
        assert result.shape == simple_grid[0].shape
        assert np.all(np.isfinite(result))

    with subtests.test("different p changes pattern"):
        r1 = phase.ince_gaussian(simple_grid, p=2, m=2, parity=1)
        r2 = phase.ince_gaussian(simple_grid, p=4, m=2, parity=1)
        assert not np.allclose(r1, r2)

    with subtests.test("different m changes pattern"):
        r1 = phase.ince_gaussian(simple_grid, p=4, m=2, parity=1)
        r2 = phase.ince_gaussian(simple_grid, p=4, m=4, parity=1)
        assert not np.allclose(r1, r2)

    with subtests.test("different ellipticity changes pattern"):
        r1 = phase.ince_gaussian(simple_grid, p=4, m=2, parity=1, ellipticity=1)
        r2 = phase.ince_gaussian(simple_grid, p=4, m=2, parity=1, ellipticity=5)
        assert not np.allclose(r1, r2)

    with subtests.test("different w changes pattern"):
        r1 = phase.ince_gaussian(simple_grid, p=4, m=2, parity=1, w=2.0)
        r2 = phase.ince_gaussian(simple_grid, p=4, m=2, parity=1, w=5.0)
        assert not np.allclose(r1, r2)

    with subtests.test("Ince poly orthogonality (same p, different m, weighted)"):
        # C_p^m and C_p^{m'} with same p but different m are orthogonal
        # under weight exp(-eps*cos(2z)/2) (Sturm-Liouville for Whittaker-Hill eq)
        eps = 2.0
        z = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
        dz = z[1] - z[0]
        weight = np.exp(-eps * np.cos(2 * z) / 2)
        f1 = _ince_polynomial(4, 0, 1, eps, z)
        f2 = _ince_polynomial(4, 4, 1, eps, z)
        integral = np.sum(f1 * f2 * weight) * dz
        assert abs(integral) < 0.05, f"Orthogonality failed: integral={integral}"

    with subtests.test("Ince poly norm is approximately 1"):
        # Normalization: (1/pi) int_0^{2pi} (C_p^m)^2 dz = 1
        eps = 2.0
        z = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        f = _ince_polynomial(4, 2, 1, eps, z)
        dz = z[1] - z[0]
        norm = np.sum(f ** 2) * dz / np.pi
        assert abs(norm - 1.0) < 0.02, f"Norm failed: {norm} != 1"

    with subtests.test("Ince poly eps=0 even limit: cos(m*z)"):
        z = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        f = _ince_polynomial(4, 2, 1, 1e-10, z)
        expected = np.cos(2 * z)
        # Normalize both for comparison
        f_norm = f / np.sqrt(np.sum(f ** 2))
        e_norm = expected / np.sqrt(np.sum(expected ** 2))
        assert np.allclose(np.abs(f_norm), np.abs(e_norm), atol=0.01)

    with subtests.test("Ince poly eps=0 odd limit: sin(m*z)"):
        z = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        f = _ince_polynomial(3, 1, -1, 1e-10, z)
        expected = np.sin(1 * z)
        f_norm = f / np.sqrt(np.sum(f ** 2))
        e_norm = expected / np.sqrt(np.sum(expected ** 2))
        assert np.allclose(np.abs(f_norm), np.abs(e_norm), atol=0.01)

    with subtests.test("Ince poly handles complex argument"):
        z_complex = 1j * np.linspace(0, 3, 50)
        result = _ince_polynomial(2, 2, 1, 2.0, z_complex)
        assert np.all(np.isfinite(result))
        assert result.shape == z_complex.shape


def test_mathieu_gaussian(simple_grid, subtests):
    """Test mathieu_gaussian() structured light generation."""
    with subtests.test("even mode shape matches grid"):
        result = phase.mathieu_gaussian(simple_grid, r=1, q=5)
        assert result.shape == simple_grid[0].shape

    with subtests.test("odd mode shape matches grid"):
        result = phase.mathieu_gaussian(simple_grid, r=-1, q=5)
        assert result.shape == simple_grid[0].shape

    with subtests.test("even mode all finite"):
        result = phase.mathieu_gaussian(simple_grid, r=1, q=5)
        assert np.all(np.isfinite(result))

    with subtests.test("odd mode all finite"):
        result = phase.mathieu_gaussian(simple_grid, r=-1, q=5)
        assert np.all(np.isfinite(result))

    with subtests.test("r=0 even mode works"):
        result = phase.mathieu_gaussian(simple_grid, r=0, q=5)
        assert result.shape == simple_grid[0].shape
        assert np.all(np.isfinite(result))

    with subtests.test("phase within [-pi, pi]"):
        result = phase.mathieu_gaussian(simple_grid, r=2, q=10)
        assert np.all(result >= -np.pi - 1e-10)
        assert np.all(result <= np.pi + 1e-10)

    with subtests.test("different r changes pattern"):
        r1 = phase.mathieu_gaussian(simple_grid, r=1, q=5)
        r2 = phase.mathieu_gaussian(simple_grid, r=2, q=5)
        assert not np.allclose(r1, r2)

    with subtests.test("even vs odd changes pattern"):
        r_even = phase.mathieu_gaussian(simple_grid, r=1, q=5)
        r_odd = phase.mathieu_gaussian(simple_grid, r=-1, q=5)
        assert not np.allclose(r_even, r_odd)

    with subtests.test("different q changes pattern"):
        r1 = phase.mathieu_gaussian(simple_grid, r=1, q=2)
        r2 = phase.mathieu_gaussian(simple_grid, r=1, q=20)
        assert not np.allclose(r1, r2)

    with subtests.test("different w changes pattern"):
        r1 = phase.mathieu_gaussian(simple_grid, r=1, q=5, w=2.0)
        r2 = phase.mathieu_gaussian(simple_grid, r=1, q=5, w=5.0)
        assert not np.allclose(r1, r2)

    with subtests.test("even r=3 works"):
        result = phase.mathieu_gaussian(simple_grid, r=3, q=5)
        assert result.shape == simple_grid[0].shape
        assert np.all(np.isfinite(result))

    with subtests.test("odd r=-3 works"):
        result = phase.mathieu_gaussian(simple_grid, r=-3, q=5)
        assert result.shape == simple_grid[0].shape
        assert np.all(np.isfinite(result))

    with subtests.test("q=0 works (circular limit)"):
        result = phase.mathieu_gaussian(simple_grid, r=1, q=0)
        assert result.shape == simple_grid[0].shape
        assert np.all(np.isfinite(result))

    with subtests.test("w=None uses default"):
        result = phase.mathieu_gaussian(simple_grid, r=1, q=5, w=None)
        assert result.shape == simple_grid[0].shape
        assert np.all(np.isfinite(result))


def test_airy(simple_grid, subtests):
    """Test airy() cubic phase generation."""
    with subtests.test("shape matches grid"):
        result = phase.airy(simple_grid, f=(1.0, 1.0))
        assert result.shape == simple_grid[0].shape

    with subtests.test("all finite"):
        result = phase.airy(simple_grid, f=(1.0, 1.0))
        assert np.all(np.isfinite(result))

    with subtests.test("f=(inf,inf) gives zeros"):
        result = phase.airy(simple_grid)
        assert np.allclose(result, 0)

    with subtests.test("explicit f=(inf,inf) gives zeros"):
        result = phase.airy(simple_grid, f=(np.inf, np.inf))
        assert np.allclose(result, 0)

    with subtests.test("f=(1, inf) only varies in x"):
        result = phase.airy(simple_grid, f=(1.0, np.inf))
        # Rows should all be identical (no y dependence)
        for i in range(1, result.shape[0]):
            assert np.allclose(result[i, :], result[0, :])

    with subtests.test("f=(inf, 1) only varies in y"):
        result = phase.airy(simple_grid, f=(np.inf, 1.0))
        # Columns should all be identical (no x dependence)
        for j in range(1, result.shape[1]):
            assert np.allclose(result[:, j], result[:, 0])

    with subtests.test("cubic phase is antisymmetric in x"):
        result = phase.airy(simple_grid, f=(1.0, np.inf))
        # x^3 is odd: phase(-x) = -phase(x)
        flipped = result[:, ::-1]
        assert np.allclose(result, -flipped, atol=1e-10)

    with subtests.test("cubic phase is antisymmetric in y"):
        result = phase.airy(simple_grid, f=(np.inf, 1.0))
        flipped = result[::-1, :]
        assert np.allclose(result, -flipped, atol=1e-10)

    with subtests.test("smaller f gives larger phase magnitude"):
        r_small = phase.airy(simple_grid, f=(0.5, np.inf))
        r_large = phase.airy(simple_grid, f=(2.0, np.inf))
        assert np.max(np.abs(r_small)) > np.max(np.abs(r_large))

    with subtests.test("phase scales as 1/f^3"):
        f1, f2 = 1.0, 2.0
        r1 = phase.airy(simple_grid, f=(f1, np.inf))
        r2 = phase.airy(simple_grid, f=(f2, np.inf))
        # (x/(f1*w))^3 / (x/(f2*w))^3 = (f2/f1)^3
        ratio = (f2 / f1) ** 3
        assert np.allclose(r1, ratio * r2, atol=1e-10)

    with subtests.test("2D airy is sum of x and y contributions"):
        rx = phase.airy(simple_grid, f=(1.0, np.inf))
        ry = phase.airy(simple_grid, f=(np.inf, 1.0))
        rxy = phase.airy(simple_grid, f=(1.0, 1.0))
        assert np.allclose(rxy, rx + ry)


def test_parse_focal_length(subtests):
    """Test _parse_focal_length() input handling."""
    with subtests.test("scalar returns pair"):
        result = _parse_focal_length(10.0)
        assert len(result) == 2
        assert result[0] == result[1] == 10.0

    with subtests.test("pair passes through"):
        result = _parse_focal_length([5.0, 10.0])
        assert result[0] == 5.0
        assert result[1] == 10.0

    with subtests.test("wrong size raises ValueError"):
        with pytest.raises(ValueError, match="Expected two terms"):
            _parse_focal_length([1, 2, 3])

    with subtests.test("zero focal length raises ValueError"):
        with pytest.raises(ValueError, match="focal length of zero"):
            _parse_focal_length([0, 10])

    with subtests.test("both zero raises ValueError"):
        with pytest.raises(ValueError, match="focal length of zero"):
            _parse_focal_length(0.0)


def test_zernike_indices_parse(subtests):
    """Test _zernike_indices_parse() branch coverage (L910-940)."""
    with subtests.test("D=2 gives [2,1]"):
        result = _zernike_indices_parse(indices=None, D=2)
        np.testing.assert_array_equal(result, [2, 1])

    with subtests.test("D=3 gives [2,1,4]"):
        result = _zernike_indices_parse(indices=None, D=3)
        np.testing.assert_array_equal(result, [2, 1, 4])

    with subtests.test("D=4 gives [2,1,4,3]"):
        result = _zernike_indices_parse(indices=None, D=4)
        np.testing.assert_array_equal(result, [2, 1, 4, 3])

    with subtests.test("D=6 gives extended basis"):
        result = _zernike_indices_parse(indices=None, D=6)
        assert len(result) == 6
        np.testing.assert_array_equal(result[:4], [2, 1, 4, 3])

    with subtests.test("scalar indices, D=None"):
        result = _zernike_indices_parse(indices=3, D=None)
        assert len(result) == 3

    with subtests.test("scalar indices with matching D"):
        result = _zernike_indices_parse(indices=4, D=4)
        assert len(result) == 4

    with subtests.test("scalar indices with mismatched D raises"):
        with pytest.raises(ValueError):
            _zernike_indices_parse(indices=3, D=5)

    with subtests.test("None indices, None D raises"):
        with pytest.raises(ValueError, match="Either dimension"):
            _zernike_indices_parse(indices=None, D=None)

    with subtests.test("explicit indices pass through"):
        result = _zernike_indices_parse(indices=[5, 6, 7], D=3)
        np.testing.assert_array_equal(result, [5, 6, 7])

    with subtests.test("smaller_okay allows D < len(indices)"):
        result = _zernike_indices_parse(indices=5, D=3, smaller_okay=True)
        assert len(result) >= 3

    with subtests.test("smaller_okay=False with D mismatch raises"):
        with pytest.raises(ValueError):
            _zernike_indices_parse(indices=[1, 2, 3], D=5, smaller_okay=False)


def test_cantor_pairing(subtests):
    """Test _cantor_pairing and _inverse_cantor_pairing roundtrip."""
    with subtests.test("roundtrip"):
        xy = np.array([[0, 0], [1, 0], [0, 1], [2, 3], [5, 5]])
        z = _cantor_pairing(xy)
        recovered = _inverse_cantor_pairing(z)
        np.testing.assert_array_equal(recovered, xy)

    with subtests.test("known values"):
        assert _cantor_pairing([[0, 0]]) == 0
        assert _cantor_pairing([[1, 0]]) == 1
        assert _cantor_pairing([[0, 1]]) == 2

    with subtests.test("inverse with negative index"):
        result = _inverse_cantor_pairing(np.array([-1, 0, 1]))
        assert result[0, 0] == -1
        assert result[0, 1] == 0

    with subtests.test("inverse non-1D raises"):
        with pytest.raises(ValueError):
            _inverse_cantor_pairing(np.array([[1, 2]]))


def test_parse_out(subtests):
    """Test _parse_out() helper."""
    x = np.zeros((10, 10), dtype=np.float64)

    with subtests.test("None allocates new array"):
        out = _parse_out(x, None, stack=1)
        assert out.shape == (1, 10, 10)
        assert out.dtype == x.dtype

    with subtests.test("None with stack>1"):
        out = _parse_out(x, None, stack=3)
        assert out.shape == (3, 10, 10)

    with subtests.test("provided out reshapes"):
        buf = np.zeros(200, dtype=np.float64)
        out = _parse_out(x, buf, stack=2)
        assert out.shape == (2, 10, 10)

    with subtests.test("wrong size raises"):
        with pytest.raises(ValueError, match="same size"):
            _parse_out(x, np.zeros(50, dtype=np.float64), stack=1)

    with subtests.test("wrong dtype raises"):
        with pytest.raises(ValueError, match="same type"):
            _parse_out(x, np.zeros((1, 10, 10), dtype=np.float32), stack=1)


def test_determine_source_radius(simple_grid, subtests):
    """Test _determine_source_radius() branches (L1815, 1817)."""
    with subtests.test("w provided passes through"):
        assert _determine_source_radius(simple_grid, w=5.0) == 5.0

    with subtests.test("w=None from grid"):
        w = _determine_source_radius(simple_grid, w=None)
        assert w == pytest.approx(np.min([np.amax(simple_grid[0]),
                                           np.amax(simple_grid[1])]) / 4)

    with subtests.test("SLM-like with get_source_radius"):
        class FakeSLM:
            x_grid = simple_grid[0]
            y_grid = simple_grid[1]
            def get_source_radius(self):
                return 42.0
        assert _determine_source_radius(FakeSLM(), w=None) == 42.0

    with subtests.test("CameraSLM-like delegates to slm"):
        class FakeCameraSLM:
            x_grid = simple_grid[0]
            y_grid = simple_grid[1]
            slm = type('FakeSLM', (), {
                'get_source_radius': lambda self: 99.0,
                'x_grid': simple_grid[0],
                'y_grid': simple_grid[1],
            })()
            cam = True
        assert _determine_source_radius(FakeCameraSLM(), w=None) == 99.0


def test_zernike_order_number(subtests):
    """Test zernike_order_number() formula (n+1)(n+2)//2."""
    with subtests.test("order 0 gives 1 (only piston)"):
        assert phase.zernike_order_number(0) == 1

    with subtests.test("order 1 gives 3"):
        assert phase.zernike_order_number(1) == 3

    with subtests.test("order 2 gives 6"):
        assert phase.zernike_order_number(2) == 6

    with subtests.test("order 3 gives 10"):
        assert phase.zernike_order_number(3) == 10

    with subtests.test("order N matches formula"):
        for n in range(8):
            assert phase.zernike_order_number(n) == (n + 1) * (n + 2) // 2


def test_zernike_build_and_coefficients(subtests):
    """Test _zernike_build_order, _zernike_build_indices, _zernike_coefficients."""
    with subtests.test("build_order populates cache"):
        _zernike_build_order(3)
        # After build_order(3), indices up to (3+1)*(3+2)//2 = 10 should be cached
        for i in range(10):
            coeffs = _zernike_coefficients(i)
            assert isinstance(coeffs, dict)

    with subtests.test("build_indices for specific set"):
        _zernike_build_indices([0, 5, 10])
        for i in [0, 5, 10]:
            assert isinstance(_zernike_coefficients(i), dict)

    with subtests.test("coefficient for piston is {(0,0): 1}"):
        coeffs = _zernike_coefficients(0)
        assert (0, 0) in coeffs
        assert coeffs[(0, 0)] == 1


def test_zernike_populate_basis_map(subtests):
    """Test _zernike_populate_basis_map()."""
    with subtests.test("small indices produce valid maps"):
        indices = np.array([0, 1, 2, 4])
        c_md, i_md, pxy_m = _zernike_populate_basis_map(indices)
        assert c_md.dtype == np.float32
        assert i_md.dtype == np.int32
        assert pxy_m.dtype == np.int32

    with subtests.test("two indices"):
        c_md, i_md, pxy_m = _zernike_populate_basis_map(np.array([0, 1]))
        assert c_md.shape[1] == 2  # Two Zernikes


def test_zernike_pyramid_plot(normalized_grid, subtests):
    """Test zernike_pyramid_plot()."""
    with subtests.test("order=2 runs without error"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        phase.zernike_pyramid_plot(normalized_grid, order=2, use_mask=False)
        plt.close("all")


@pytest.mark.gpu
def test_zernike_sum_gpu(benchmark, has_cupy):
    """GPU variant of zernike_sum() using cupy arrays and CUDA kernels."""
    import cupy as cp

    x = cp.linspace(-1, 1, 256)
    grid = cp.meshgrid(x, x)
    rng = np.random.default_rng(42)
    coeffs = rng.normal(0, 0.1, 10)

    def run():
        phase.zernike_sum(grid, indices=list(range(len(coeffs))), weights=coeffs)

    result = benchmark(run)
    assert grid[0].shape == (256, 256)
