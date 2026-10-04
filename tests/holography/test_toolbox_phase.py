"""
Unit tests for slmsuite.holography.toolbox.phase module.
"""
import gc
import threading
import warnings
import weakref

import pytest
import numpy as np

from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.holography.toolbox import phase, Aperture
from slmsuite.misc.xp import as_numpy
from slmsuite.holography.toolbox.phase import _zernike as Z
from slmsuite.holography.toolbox.phase import (
    ZernikeBasis,
    clear_zernike_basis_cache,
    _parse_focal_length,
    _zernike_indices_parse,
    _cantor_pairing,
    _inverse_cantor_pairing,
    _parse_out,
    _determine_source_radius,
    _ince_polynomial,
    _zernike_build_order,
    _zernike_coefficients,
    _zernike_populate_basis_map,
)


@pytest.fixture
def simple_grid():
    """A small 2D grid for testing."""
    x = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, x)
    return (X, Y)


def _phase_close(a, b, atol=1e-6):
    """Whether two phase patterns agree modulo 2 pi."""
    return np.max(np.abs(np.angle(np.exp(1j * (np.asarray(a, float) - b))))) < atol


def _assert_float32(function, grid, fraction=1, **kwargs):
    """
    Assert that ``function`` on a float32 grid returns float32, matching the float64
    phase on the same rounded grid modulo 2 pi on at least ``fraction`` of the pixels.
    """
    grid32 = tuple(np.asarray(g, dtype=np.float32) for g in grid)
    result = function(grid32, **kwargs)
    reference = function(tuple(g.astype(float) for g in grid32), **kwargs)
    assert result.dtype == np.float32

    np.testing.assert_array_equal(np.isnan(result), np.isnan(reference))
    error = np.abs(np.angle(np.exp(1j * (result.astype(float) - reference))))
    # Single-precision arithmetic errs by a few ulps of the largest phase.
    tolerance = 8 * np.finfo(np.float32).eps * np.nanmax(np.abs(reference))
    assert np.mean(np.nan_to_num(error) <= tolerance) >= fraction


@pytest.fixture
def normalized_grid():
    """A grid of typical SLM coordinates, in wavelengths."""
    x = np.linspace(-500, 500, 256)
    X, Y = np.meshgrid(x, x)
    return (X, Y)


@pytest.fixture
def fine_grid():
    """A grid fine enough to locate the nodes of a structured mode."""
    x = np.linspace(-100, 100, 801)
    X, Y = np.meshgrid(x, x)
    return (X, Y)


def _nodes(profile, coordinate):
    """Coordinates of the sign changes in a binary phase profile."""
    return coordinate[:-1][np.abs(np.diff(profile)) > 0.1]


def test_blaze(simple_grid, subtests, benchmark):
    """Test blaze() phase pattern generation."""
    with subtests.test("benchmark"):
        benchmark(phase.blaze, simple_grid, vector=(0.1, 0.05))

    with subtests.test("the phase is 2 pi per unit of the vector along each axis"):
        (x_grid, y_grid) = simple_grid
        assert np.allclose(phase.blaze(simple_grid, (0.1, 0)), 2 * np.pi * 0.1 * x_grid)
        assert np.allclose(phase.blaze(simple_grid, (0, 0.1)), 2 * np.pi * 0.1 * y_grid)

    with subtests.test("a zero vector is a flat phase"):
        assert np.allclose(phase.blaze(simple_grid, vector=(0, 0)), 0)

    with subtests.test("the phase is linear in the vector"):
        assert np.allclose(
            phase.blaze(simple_grid, vector=(1, 2)),
            phase.blaze(simple_grid, vector=(1, 0)) + phase.blaze(simple_grid, vector=(0, 2)),
        )
        assert np.allclose(
            phase.blaze(simple_grid, vector=(2, 2)), 2 * phase.blaze(simple_grid, vector=(1, 1))
        )

    with subtests.test("a single-axis vector varies only along that axis"):
        x_only = phase.blaze(simple_grid, vector=(1, 0))
        y_only = phase.blaze(simple_grid, vector=(0, 1))
        assert np.allclose(x_only, x_only[[0], :])
        assert np.allclose(y_only, y_only[:, [0]])

    with subtests.test("the third vector component adds pi|x|^2 of focus"):
        focus = (
            phase.blaze(simple_grid, vector=(1, 1, 1)) - phase.blaze(simple_grid, vector=(1, 1))
        )
        assert np.allclose(focus, np.pi * (simple_grid[0] ** 2 + simple_grid[1] ** 2))

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        _assert_float32(phase.blaze, simple_grid, vector=(0.1, 0.05, 0.001))


def test_triangle(simple_grid, subtests):
    """Test triangle() phase pattern generation."""
    vector = (0.1, 0.05)

    with subtests.test("a bias of one is a blaze, of minus one the opposite blaze"):
        blazed = np.mod(phase.blaze(simple_grid, vector=vector), 2*np.pi)
        assert np.allclose(phase.triangle(simple_grid, vector=vector, bias=1), blazed)
        assert np.allclose(
            np.exp(1j * phase.triangle(simple_grid, vector=vector, bias=-1)),
            np.exp(-1j * blazed),
        )
        # Bias is clipped, so nothing lies beyond those two gratings.
        assert np.allclose(phase.triangle(simple_grid, vector=vector, bias=5), blazed)

    with subtests.test("bias sets the rising fraction of each period"):
        for bias in (-0.5, 0, 0.5):
            result = phase.triangle(simple_grid, vector=(0.13, 0), bias=bias)
            rising = np.mean(np.diff(result[0, :]) > 0)
            assert rising == pytest.approx((bias + 1) / 2, abs=0.1)

    with subtests.test("zero bias is symmetric under reflection"):
        assert np.allclose(
            phase.triangle(simple_grid, vector=vector, bias=0),
            phase.triangle(simple_grid, vector=(-vector[0], -vector[1]), bias=0),
        )

    with subtests.test("a and b bound the ramp"):
        (a, b) = (np.pi, np.pi/2)
        result = phase.triangle(simple_grid, vector=(1, 0), a=a, b=b)
        assert np.min(result) >= b - 1e-9
        assert np.max(result) <= a + 1e-9
        # The peak falls between samples, so the ramp only nearly fills the range.
        assert np.ptp(result) > 0.75 * (a - b)

    with subtests.test("shift translates the pattern"):
        shift = np.pi / 2
        shifted = phase.triangle(simple_grid, vector=(1, 0), shift=shift)
        translated = phase.triangle(
            (simple_grid[0] + shift / (2*np.pi), simple_grid[1]), vector=(1, 0)
        )
        assert np.allclose(shifted, translated)

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        _assert_float32(phase.triangle, simple_grid, vector=vector)


def test_sinusoid(simple_grid, subtests):
    """Test sinusoid() phase pattern generation."""
    with subtests.test("a zero vector is flat at a"):
        assert np.allclose(phase.sinusoid(simple_grid, vector=(0, 0), a=2.0, b=0), 2.0)

    with subtests.test("a and b bound the sinusoid and are nearly attained"):
        for (a, b) in ((np.pi, 0), (np.pi, np.pi/2), (2*np.pi, np.pi/2)):
            result = phase.sinusoid(simple_grid, vector=(1, 0), a=a, b=b)
            assert np.min(result) >= b - 1e-9
            assert np.max(result) <= a + 1e-9
            assert np.ptp(result) > 0.99 * (a - b)

    with subtests.test("shifting by pi reflects about the midpoint"):
        kwargs = dict(vector=(1, 0), a=np.pi, b=0.5)
        assert np.allclose(
            phase.sinusoid(simple_grid, shift=0, **kwargs)
            + phase.sinusoid(simple_grid, shift=np.pi, **kwargs),
            np.pi + 0.5,
        )

    with subtests.test("the default amplitude extinguishes the 0th order"):
        # J_0(|a-b|/2) = 0 at the default; the duplicated grid endpoint sets the 1e-2 floor.
        field = np.mean(np.exp(1j * phase.sinusoid(simple_grid, vector=(0.5, 0))))
        assert np.abs(field) < 0.02

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        _assert_float32(phase.sinusoid, simple_grid, vector=(0.1, 0.05))


def test_binary(simple_grid, subtests):
    """Test binary() grating generation."""
    with subtests.test("the grating takes exactly the two values a and b"):
        result = phase.binary(simple_grid, vector=(0.1, 0.1), a=np.pi, b=0)
        np.testing.assert_array_equal(np.unique(result), [0, np.pi])

    with subtests.test("the duty cycle is the fraction of the period held at a"):
        for duty_cycle in (0.1, 0.25, 0.5, 0.75, 0.9):
            result = phase.binary(simple_grid, vector=(.1, .1), duty_cycle=duty_cycle, a=1, b=0)
            assert np.mean(result > 0.5) == pytest.approx(duty_cycle, abs=0.02)

    with subtests.test("a vector component above one is a period in pixels"):
        result = phase.binary(simple_grid, vector=(10, 0), a=np.pi, b=0)
        np.testing.assert_array_equal(result[0, :10], [np.pi] * 5 + [0] * 5)
        np.testing.assert_array_equal(result[:, :-10], result[:, 10:])
        result = phase.binary(simple_grid, vector=(4, 0), duty_cycle=0.75, a=np.pi, b=0)
        np.testing.assert_array_equal(result[0, :4], [np.pi] * 3 + [0])

    with subtests.test("a single-axis vector varies only along that axis"):
        x_only = phase.binary(simple_grid, vector=(0.1, 0), a=np.pi, b=0)
        y_only = phase.binary(simple_grid, vector=(0, 0.1), a=np.pi, b=0)
        np.testing.assert_array_equal(x_only, np.broadcast_to(x_only[[0], :], x_only.shape))
        np.testing.assert_array_equal(y_only, np.broadcast_to(y_only[:, [0]], y_only.shape))

    with subtests.test("a zero vector selects a or b by whether the shift lies in the duty"):
        for (shift, duty, expected) in ((0, 0.5, np.pi), (0.1, 0.5, np.pi), (np.pi, 0.25, 0)):
            result = phase.binary(
                simple_grid, vector=(0, 0), a=np.pi, b=0, shift=shift, duty_cycle=duty
            )
            assert np.allclose(result, expected)

    with subtests.test("a float32 grid gives the float64 grating in float32"):
        # Rounding the grid may move a duty-cycle edge across a pixel.
        for vector in ((0.1, 0.05), (4, 0), (0, 0)):
            _assert_float32(phase.binary, simple_grid, fraction=.99, vector=vector)


def test_lens(simple_grid, subtests, benchmark):
    """Test lens() phase pattern generation."""
    with subtests.test("benchmark"):
        benchmark(phase.lens, simple_grid, f=(1000, 1000))

    with subtests.test("infinite focal length gives zeros"):
        assert np.allclose(phase.lens(simple_grid, f=(np.inf, np.inf)), 0)

    with subtests.test("the lens is even in both axes"):
        result = phase.lens(simple_grid, f=(100, 200))
        assert np.allclose(result, result[:, ::-1])
        assert np.allclose(result, result[::-1, :])

    with subtests.test("negative focal length negates phase"):
        assert np.allclose(
            phase.lens(simple_grid, f=(10, 10)), -phase.lens(simple_grid, f=(-10, -10))
        )

    with subtests.test("the axes separate into two cylindrical lenses"):
        assert np.allclose(
            phase.lens(simple_grid, f=(100, 200)),
            phase.lens(simple_grid, f=(100, np.inf)) + phase.lens(simple_grid, f=(np.inf, 200)),
        )

    with subtests.test("a scalar focal length is the isotropic pi|x|^2/f"):
        result = phase.lens(simple_grid, f=50)
        assert np.allclose(result, phase.lens(simple_grid, f=(50, 50)))
        assert np.allclose(
            result,
            phase.polynomial(
                simple_grid, weights=[np.pi/50, np.pi/50], terms=np.array([[2, 0], [0, 2]])
            ).squeeze(),
        )

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        _assert_float32(phase.lens, simple_grid, f=(100, 200))


def test_axicon(simple_grid, subtests):
    """Test axicon() phase pattern generation."""
    with subtests.test("infinite focal length gives zeros"):
        assert np.allclose(phase.axicon(simple_grid, f=(np.inf, np.inf)), 0)

    with subtests.test("a cylindrical axicon blazes by w/f/2 along its axis"):
        angle = 5.0 / 100 / 2
        assert np.allclose(
            phase.axicon(simple_grid, f=(100, np.inf), w=5.0),
            (2 * np.pi * angle) * np.abs(simple_grid[0]),
        )
        assert np.allclose(
            phase.axicon(simple_grid, f=(np.inf, 100), w=5.0),
            (2 * np.pi * angle) * np.abs(simple_grid[1]),
        )

    with subtests.test("an elliptical axicon is the hypotenuse of its two cylinders"):
        assert np.allclose(
            phase.axicon(simple_grid, f=(100, 200), w=5.0),
            np.hypot(
                phase.axicon(simple_grid, f=(100, np.inf), w=5.0),
                phase.axicon(simple_grid, f=(np.inf, 200), w=5.0),
            ),
        )

    with subtests.test("a diverging axicon negates the converging phase"):
        assert np.allclose(
            phase.axicon(simple_grid, f=(-100, -100), w=5.0),
            -phase.axicon(simple_grid, f=(100, 100), w=5.0),
        )

    with subtests.test("mixed focal signs raise"):
        with pytest.raises(ValueError, match="cannot converge"):
            phase.axicon(simple_grid, f=(100, -200), w=5.0)

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        _assert_float32(phase.axicon, simple_grid, f=(100, 200), w=5.0)


def test_zernike(normalized_grid, subtests):
    """Test zernike() single-polynomial evaluation."""
    (x_scale, y_scale) = Aperture.resolve(normalized_grid, None).scale

    with subtests.test("piston (j=0) is unity over the pupil"):
        assert np.allclose(phase.zernike(normalized_grid, index=0), 1)

    with subtests.test("the tilts (j=1, j=2) are the normalized y and x coordinates"):
        assert np.allclose(phase.zernike(normalized_grid, index=1), normalized_grid[1] * y_scale)
        assert np.allclose(phase.zernike(normalized_grid, index=2), normalized_grid[0] * x_scale)

    with subtests.test("defocus (j=4) is 2 rho^2 - 1"):
        rho_squared = (normalized_grid[0] * x_scale)**2 + (normalized_grid[1] * y_scale)**2
        assert np.allclose(phase.zernike(normalized_grid, index=4), 2 * rho_squared - 1)

    with subtests.test("weight scales the polynomial linearly"):
        for index in (1, 5, 10):
            assert np.allclose(
                phase.zernike(normalized_grid, index=index, weight=2.0),
                2 * phase.zernike(normalized_grid, index=index),
            )

    with subtests.test("zernike(j) is zernike_sum() over the single index j"):
        assert np.allclose(
            phase.zernike(normalized_grid, index=10, weight=2.0),
            phase.zernike_sum(normalized_grid, [10], [2.0]),
        )

    with subtests.test("the third- and fourth-order terms match the OSA closed forms"):
        axis = np.linspace(-0.7, 0.7, 29)
        (x, y) = np.meshgrid(axis, axis)
        r2 = x**2 + y**2
        for (index, expected) in (
            (6, 3 * x**2 * y - y**3),
            (7, (3 * r2 - 2) * y),
            (8, (3 * r2 - 2) * x),
            (9, x**3 - 3 * x * y**2),
            (10, 4 * x**3 * y - 4 * x * y**3),
            (11, (4 * r2 - 3) * 2 * x * y),
            (12, 6 * r2**2 - 6 * r2 + 1),
            (13, (4 * r2 - 3) * (x**2 - y**2)),
            (14, x**4 - 6 * x**2 * y**2 + y**4),
        ):
            np.testing.assert_allclose(
                phase.zernike((x, y), index, aperture=1.0, use_mask=False), expected,
                atol=1e-12, err_msg=f"Z{index}",
            )


def test_quadrants(simple_grid, subtests):
    """Each quadrant of quadrants() is a blaze toward that quadrant, offset by center."""
    (radius, center) = (0.001, (0.0005, -0.0005))
    v = radius / np.sqrt(2)
    odd_grid = (simple_grid[0][:-1, :-1], simple_grid[1][:-1, :-1])

    for (name, grid) in (("even", simple_grid), ("odd", odd_grid)):
        with subtests.test(f"every pixel of an {name} grid lies in a quadrant"):
            result = phase.quadrants(grid, radius=radius, center=center)
            (rows, cols) = result.shape
            for (row, col, vector) in (
                (slice(None, rows//2), slice(cols//2, None), (v, -v)),
                (slice(rows//2, None), slice(cols//2, None), (v, v)),
                (slice(None, rows//2), slice(None, cols//2), (-v, -v)),
                (slice(rows//2, None), slice(None, cols//2), (-v, v)),
            ):
                expected = phase.blaze(grid, vector=np.add(vector, center))
                assert np.allclose(result[row, col], expected[row, col])

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        _assert_float32(phase.quadrants, simple_grid, radius=radius, center=center)


def test_bahtinov(simple_grid, subtests):
    """Test bahtinov() focusing mask generation."""
    (radius, angle) = (0.005, 10 * np.pi / 180)

    with subtests.test("each quadrant is a binary grating tilted by the mask angle"):
        result = phase.bahtinov(simple_grid, radius=radius, angle=angle)
        (rows, cols) = result.shape
        (s, c) = (radius * np.sin(angle), radius * np.cos(angle))

        for (row, col, vector) in (
            (slice(None, rows//2), slice(cols//2, None), (s, c)),
            (slice(rows//2, None), slice(cols//2, None), (s, -c)),
            (slice(None, None), slice(None, cols//2), (0, radius)),
        ):
            expected = phase.binary(simple_grid, vector=vector)
            assert np.allclose(result[row, col], expected[row, col])

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        _assert_float32(phase.bahtinov, simple_grid, radius=radius, angle=angle)


def test_zernike_get_string(subtests):
    """Test zernike_get_string() LaTeX representations."""
    with subtests.test("the low orders are the cartesian Zernike monomials"):
        for (index, string) in enumerate(
            ["1", "1y", "1x", "2xy", "2y^2+2x^2-1", "-1y^2+1x^2"]
        ):
            assert phase.zernike_get_string(index) == string

    with subtests.test("a derivative differentiates the monomials"):
        assert phase.zernike_get_string(4, derivative=(1, 0)) == "4x"
        assert phase.zernike_get_string(4, derivative=(2, 0)) == "4"
        assert phase.zernike_get_string(0, derivative=(1, 0)) == "0"


def test_zernike_convert_index(subtests):
    """Test zernike_convert_index() conversions between indexing conventions."""
    with subtests.test("the output is (N, 1) for linear conventions and (N, 2) for radial"):
        assert phase.zernike_convert_index(3, "ansi", "radial").shape == (1, 2)
        assert phase.zernike_convert_index([3, 4, 5], "ansi", "radial").shape == (3, 2)
        assert np.issubdtype(
            phase.zernike_convert_index([3, 4, 5], "ansi", "noll").dtype, np.integer
        )

    with subtests.test("an unknown convention raises"):
        with pytest.raises(ValueError):
            phase.zernike_convert_index([0], from_index="bogus", to_index="ansi")
        with pytest.raises(ValueError):
            phase.zernike_convert_index([0], "ansi", "bogus")

    # The full matrix of conventions, over a range every convention can express.
    schemes = ["ansi", "noll", "fringe", "wyant", "radial"]
    source = {s: phase.zernike_convert_index(np.arange(15), "ansi", s) for s in schemes}

    with subtests.test("every from_index x to_index pair converts"):
        for a in schemes:
            for b in schemes:
                result = phase.zernike_convert_index(source[a], a, b)
                np.testing.assert_array_equal(
                    result, np.reshape(source[b], result.shape), f"{a} -> {b}"
                )

    with subtests.test("noll indices beyond radial order 12 round-trip through ansi"):
        Z._zernike_inverse_cache.clear()
        noll = np.arange(1, 400)
        ansi = phase.zernike_convert_index(noll, "noll", "ansi")
        assert np.all(ansi >= 0)
        np.testing.assert_array_equal(
            np.ravel(phase.zernike_convert_index(ansi, "ansi", "noll")), noll
        )

    with subtests.test("round trips A -> B -> A are the identity"):
        for a in schemes:
            for b in schemes:
                there = phase.zernike_convert_index(source[a], a, b)
                back = phase.zernike_convert_index(there, b, a)
                np.testing.assert_array_equal(
                    back, np.reshape(source[a], back.shape), f"{a} -> {b} -> {a}"
                )

    with subtests.test("conversion is transitive: A -> B -> C == A -> C"):
        for a in schemes:
            for b in schemes:
                for c in schemes:
                    via = phase.zernike_convert_index(
                        phase.zernike_convert_index(source[a], a, b), b, c
                    )
                    direct = phase.zernike_convert_index(source[a], a, c)
                    np.testing.assert_array_equal(
                        via.ravel(), direct.ravel(), f"{a} -> {b} -> {c}"
                    )

    with subtests.test("Fringe 37 (Wyant 36) is R_12^0, ANSI 84"):
        np.testing.assert_array_equal(
            phase.zernike_convert_index([37], "fringe", "radial"), [[12, 0]]
        )
        assert phase.zernike_convert_index([37], "fringe", "ansi").ravel()[0] == 84
        assert phase.zernike_convert_index([36], "wyant", "ansi").ravel()[0] == 84
        assert phase.zernike_convert_index([84], "ansi", "fringe").ravel()[0] == 37
        assert phase.zernike_convert_index([84], "ansi", "wyant").ravel()[0] == 36

    undefined = phase.ZERNIKE_INDEX_UNDEFINED

    with subtests.test("indices outside the 37-term set are undefined, not a dead batch"):
        # ANSI 21 = (n=6, l=-6) is an ordinary term with no Fringe equivalent.
        batch = phase.zernike_convert_index(np.arange(22), "ansi", "fringe").ravel()
        assert np.issubdtype(batch.dtype, np.integer)
        assert batch[21] == undefined
        np.testing.assert_array_equal(
            batch[:21],
            phase.zernike_convert_index(np.arange(21), "ansi", "fringe").ravel(),
        )
        # And the same in the inverse direction.
        np.testing.assert_array_equal(
            phase.zernike_convert_index([1, 100, 2], "fringe", "ansi").ravel(), [0, undefined, 2]
        )

    with subtests.test("an empty batch converts to an empty batch"):
        for a in schemes:
            empty = np.empty((0, 2), int) if a == "radial" else np.array([], int)
            for b in schemes:
                assert np.size(phase.zernike_convert_index(empty, a, b)) == 0, f"{a} -> {b}"

    with subtests.test("the undefined sentinel survives every conversion"):
        # ANSI -1 is the vortex waveplate; Wyant is 0-indexed; Fringe and Noll are 1-indexed.
        assert undefined < -1
        for a in schemes:
            source_a = [[undefined, undefined]] if a == "radial" else [undefined]
            for b in schemes:
                result = phase.zernike_convert_index(source_a, a, b)
                np.testing.assert_array_equal(
                    result, np.full_like(result, undefined), f"{a} -> {b}"
                )

    with subtests.test("a round trip through Fringe/Wyant cannot fabricate an ANSI index"):
        # ANSI 21 = (n=6, l=-6) has no Fringe/Wyant index and must not come back as one.
        for scheme in ["fringe", "wyant"]:
            there = phase.zernike_convert_index(np.arange(22), "ansi", scheme).ravel()
            back = phase.zernike_convert_index(there, scheme, "ansi").ravel()
            np.testing.assert_array_equal(back[:21], np.arange(21), scheme)
            assert back[21] == undefined, scheme
            assert back[21] != -1, scheme       # Not the vortex.
            assert back[21] < 0, scheme         # Not any ANSI polynomial.

    with subtests.test("ANSI -> Noll matches Noll (1976)"):
        # J. Opt. Soc. Am. 66, 207, doi:10.1364/JOSA.66.000207, through radial order 6.
        np.testing.assert_array_equal(
            phase.zernike_convert_index(np.arange(28), "ansi", "noll").ravel(),
            [1, 3, 2, 5, 4, 6, 9, 7, 8, 10, 15, 13, 11, 12, 14,
             21, 19, 17, 16, 18, 20, 27, 25, 23, 22, 24, 26, 28],
        )

    with subtests.test("a malformed radial index raises"):
        for shape_error in ([[1, 2, 3]], [1, 2, 3, 4]):
            with pytest.raises(ValueError, match="Expected dimension"):
                phase.zernike_convert_index(shape_error, "radial", "ansi")
        for (n, l) in ((2, 3), (2, -4), (-2, 0)):
            with pytest.raises(ValueError, match="Invalid Zernike index"):
                phase.zernike_convert_index([[n, l]], "radial", "ansi")

    with subtests.test("ANSI -> Fringe matches the published table; Wyant is Fringe - 1"):
        ansi = np.arange(phase.zernike_order_number(12))
        fringe = phase.zernike_convert_index(ansi, "ansi", "fringe").ravel()
        wyant = phase.zernike_convert_index(ansi, "ansi", "wyant").ravel()
        np.testing.assert_array_equal(
            fringe[:15], [1, 3, 2, 6, 4, 5, 11, 8, 7, 10, 18, 13, 9, 12, 17]
        )
        mapped = fringe != undefined
        assert np.count_nonzero(mapped) == 37
        np.testing.assert_array_equal(wyant[mapped], fringe[mapped] - 1)
        np.testing.assert_array_equal(wyant[~mapped], undefined)


def test_zernike_sum(normalized_grid, subtests, benchmark):
    """Test zernike_sum() weighted summation of Zernike polynomials."""
    with subtests.test("benchmark"):
        rng = np.random.default_rng(42)
        coeffs = rng.normal(0, 0.1, 10)
        benchmark(
            phase.zernike_sum, normalized_grid, indices=list(range(len(coeffs))), weights=coeffs
        )

    with subtests.test("use_mask=True zeros outside the aperture, use_mask=nan voids it"):
        mask = Aperture.resolve(normalized_grid, "circular").mask
        masked = phase.zernike_sum(
            normalized_grid, indices=[4], weights=[1], use_mask=True, aperture="circular"
        )
        assert np.allclose(masked[~mask], 0)
        voided = phase.zernike_sum(
            normalized_grid, indices=[4], weights=[1], use_mask=np.nan, aperture="circular"
        )
        assert np.all(np.isnan(voided[~mask]))

    with subtests.test("use_mask=False leaves the corners of the grid populated"):
        result = phase.zernike_sum(normalized_grid, indices=[4], weights=[1], use_mask=False)
        assert np.all(np.isfinite(result))
        assert result[0, 0] != 0

    with subtests.test("derivatives are the analytic monomial coefficients"):
        # Z2 = x, Z4 = 2y^2 + 2x^2 - 1, Z3 = 2xy, in normalized pupil coordinates.
        for (index, derivative, expected) in ((2, (1, 0), 1), (4, (2, 0), 4), (3, (1, 1), 2)):
            result = phase.zernike_sum(
                normalized_grid, indices=[index], weights=[1],
                use_mask=False, derivative=derivative
            )
            assert np.allclose(result, expected)

    with subtests.test("weights of shape (D, N) return a stack of N patterns"):
        result = phase.zernike_sum(
            normalized_grid, indices=[1, 2], weights=np.array([[1, 0], [0, 1]])
        )
        assert result.shape == (2, *normalized_grid[0].shape)
        assert np.allclose(result[0], phase.zernike(normalized_grid, 1))
        assert np.allclose(result[1], phase.zernike(normalized_grid, 2))

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        for use_mask in (True, np.nan):
            _assert_float32(
                phase.zernike_sum, normalized_grid, indices=[1, 4, 7], weights=[1.0, 2.0, 3.0],
                use_mask=use_mask,
            )

    with subtests.test("every radial order a float32 grid computes silently is within 8 bits"):
        # m = 0 is the least accurate polynomial of each radial order.
        x = np.linspace(-1, 1, 201)
        grid32 = tuple(g.astype(np.float32) for g in np.meshgrid(x, x))
        grid64 = tuple(g.astype(float) for g in grid32)
        for n in range(2, 25, 2):
            index = np.ravel(phase.zernike_convert_index([[n, 0]], "radial", "ansi"))
            kwargs = dict(indices=index, weights=[1.0], aperture="circular", use_mask=np.nan)
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                single = phase.zernike_sum(grid32, **kwargs)
            if not any("float32" in str(w.message) for w in record):
                double = phase.zernike_sum(grid64, **kwargs)
                assert np.nanmax(np.abs(single - double)) < 2 * np.pi / 256, f"order {n}"

    with subtests.test("a float64 grid, or a float32 grid at low order, stays silent"):
        high = np.ravel(phase.zernike_convert_index([[24, 0]], "radial", "ansi"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            phase.zernike_sum(normalized_grid, indices=high, weights=[1], use_mask=np.nan)
            phase.zernike_sum(
                tuple(g.astype(np.float32) for g in normalized_grid),
                indices=[4], weights=[1], use_mask=np.nan,
            )

    with subtests.test("scalar index and weight give a single pattern"):
        result = phase.zernike_sum(normalized_grid, indices=4, weights=1.0)
        assert np.allclose(result, phase.zernike(normalized_grid, 4))

    with subtests.test("indices=None defaults to the first D of the default basis"):
        result = phase.zernike_sum(normalized_grid, indices=None, weights=[1, 1])
        assert np.allclose(result, phase.zernike_sum(normalized_grid, [2, 1], [1, 1]))

    with subtests.test("out parameter reuses memory"):
        out = np.zeros((1, *normalized_grid[0].shape), dtype=normalized_grid[0].dtype)
        result = phase.zernike_sum(normalized_grid, indices=[1], weights=[1], out=out)
        assert np.shares_memory(result, out)

    indices = [2, 1, 4, 3, 5, 6, 7, 8]
    basis = ZernikeBasis(normalized_grid, indices)

    with subtests.test("the cached and ZernikeBasis paths match the direct evaluation"):
        stack = np.random.default_rng(0).normal(0, 0.3, (len(indices), 3))
        for weights in (stack, stack[:, 0]):
            direct = Z._zernike_sum_direct(
                normalized_grid, indices, weights, None, True, (0, 0), None
            )
            np.testing.assert_allclose(
                phase.zernike_sum(normalized_grid, indices, weights), direct, atol=1e-9
            )
            np.testing.assert_allclose(phase.zernike_sum(basis, None, weights), direct, atol=1e-9)

    with subtests.test("a derivative of a ZernikeBasis raises"):
        with pytest.raises(ValueError, match="derivative"):
            phase.zernike_sum(basis, None, np.ones(len(indices)), derivative=(1, 0))

    for (kwargs, match) in (
        ({"indices": [0], "weights": [1], "derivative": (1,)}, "Expected derivative"),
        ({"indices": [0, 1, 2], "weights": np.ones((3, 2, 2))}, "1D or 2D"),
        ({"indices": [0, 1], "weights": [1.0, 2.0, 3.0]}, "common dimension"),
    ):
        with subtests.test(f"malformed input raises '{match}'"):
            with pytest.raises(ValueError, match=match):
                phase.zernike_sum(normalized_grid, **kwargs)


class TestZernikeBasis:
    """Tests for the ZernikeBasis cache of basis images."""

    indices = [2, 1, 4, 3, 5, 6, 7, 8]

    def test_init(self, normalized_grid, subtests):
        D = len(self.indices)
        basis = ZernikeBasis(normalized_grid, self.indices)

        with subtests.test("each index gets one basis image, and the gram is D x D"):
            assert basis.basis.shape == (D, *normalized_grid[0].shape)
            assert basis.basis_flat.shape == (D, normalized_grid[0].size)
            assert basis.mask.shape == normalized_grid[0].shape
            assert len(basis) == D
            assert basis.gram.shape == (D, D)
            assert basis.norm.shape == (D,)

        with subtests.test("each basis image is its polynomial"):
            for (image, index) in zip(basis.basis, self.indices):
                np.testing.assert_allclose(
                    image,
                    Z._zernike_sum_direct(normalized_grid, [index], [1], None, True, (0, 0), None),
                    atol=1e-12,
                )

    def test_getitem(self, normalized_grid, subtests):
        weights = np.random.default_rng(0).normal(0, 0.3, len(self.indices))
        sub = ZernikeBasis(normalized_grid, self.indices)[2:]

        with subtests.test("a slice selects modes positionally"):
            assert len(sub) == len(self.indices) - 2
            np.testing.assert_array_equal(sub.indices, self.indices[2:])
            direct = Z._zernike_sum_direct(
                normalized_grid, self.indices[2:], weights[2:], None, True, (0, 0), None
            )
            np.testing.assert_allclose(
                phase.zernike_sum(sub, None, weights[2:]), direct, atol=1e-9
            )


def test_zernike_get_basis(normalized_grid, subtests):
    """Test _zernike_get_basis(), the cache behind zernike_sum and image_zernike_fit."""
    indices = [2, 1, 4, 3, 5, 6]
    weights = np.random.default_rng(1).normal(0, 0.3, len(indices))

    with subtests.test("repeated calls build the basis once and reuse it"):
        clear_zernike_basis_cache()
        basis = Z._zernike_get_basis(normalized_grid, indices)
        assert Z._zernike_get_basis(normalized_grid, indices) is basis
        assert len(Z._ZERNIKE_BASIS_CACHE) == 1

    with subtests.test("distinct grids and apertures key to distinct entries"):
        clear_zernike_basis_cache()
        grid_b = (normalized_grid[0].copy(), normalized_grid[1].copy())
        phase.zernike_sum(normalized_grid, indices, weights)
        phase.zernike_sum(grid_b, indices, weights)
        phase.zernike_sum(normalized_grid, indices, weights, aperture="circular")
        assert len(Z._ZERNIKE_BASIS_CACHE) == 3

    with subtests.test("grids sharing only their x array key to distinct entries"):
        clear_zernike_basis_cache()
        y_interior = normalized_grid[1].copy()
        y_interior[5:-5, :] *= 0.5
        a = phase.zernike_sum(normalized_grid, [1], [1.0])
        b = phase.zernike_sum((normalized_grid[0], y_interior), [1], [1.0])
        assert not np.allclose(a, b)

    with subtests.test("a derivative or a nan mask is evaluated directly, uncached"):
        clear_zernike_basis_cache()
        phase.zernike_sum(normalized_grid, indices, weights, derivative=(1, 0))
        phase.zernike_sum(normalized_grid, indices, weights, use_mask=np.nan)
        assert len(Z._ZERNIKE_BASIS_CACHE) == 0

    with subtests.test("a discarded grid frees its cached basis"):
        clear_zernike_basis_cache()
        x = np.linspace(-500, 500, 64)
        grid = np.meshgrid(x, x)
        refs = [weakref.ref(Z._zernike_get_basis(grid, indices)), weakref.ref(grid[0])]
        del grid
        gc.collect()
        assert [ref() for ref in refs] == [None, None]

    with subtests.test("the cache holds at most its cap, evicting the least recently used"):
        clear_zernike_basis_cache()
        x = np.linspace(-1, 1, 8)
        grids = [np.meshgrid(x, x) for _ in range(Z._ZERNIKE_BASIS_CACHE_MAX + 1)]
        bases = [Z._zernike_get_basis(grid, [1]) for grid in grids[:-1]]
        Z._zernike_get_basis(grids[0], [1])
        newest = Z._zernike_get_basis(grids[-1], [1])

        cached = list(Z._ZERNIKE_BASIS_CACHE.values())
        assert len(cached) == Z._ZERNIKE_BASIS_CACHE_MAX
        assert any(b is newest for b in cached) and any(b is bases[0] for b in cached)
        assert not any(b is bases[1] for b in cached)
        clear_zernike_basis_cache()


def test_clear_zernike_basis_cache(normalized_grid, subtests):
    """Test clear_zernike_basis_cache(), which releases every cached basis."""
    indices = [2, 1, 4, 3, 5, 6]

    with subtests.test("clearing empties the cache"):
        phase.zernike_sum(normalized_grid, indices, np.ones(len(indices)))
        clear_zernike_basis_cache()
        assert len(Z._ZERNIKE_BASIS_CACHE) == 0

    with subtests.test("clearing frees the default fit grids"):
        from slmsuite.holography.analysis import image_zernike_fit

        clear_zernike_basis_cache()
        image_zernike_fit(np.zeros((24, 32)), order=[1, 2])
        grid = Z._zernike_fit_grid(24, 32, np)
        (basis,) = Z._ZERNIKE_BASIS_CACHE.values()
        refs = [weakref.ref(basis), weakref.ref(grid[0]), weakref.ref(grid[1])]
        del basis, grid
        clear_zernike_basis_cache()
        gc.collect()
        assert [ref() for ref in refs] == [None, None, None]

    with subtests.test("clearing waits on the lock that serializes the cache"):
        cleared = threading.Event()
        phase.zernike_sum(normalized_grid, indices, np.ones(len(indices)))
        with Z._ZERNIKE_BASIS_CACHE_LOCK:
            worker = threading.Thread(target=lambda: (clear_zernike_basis_cache(), cleared.set()))
            worker.start()
            assert not cleared.wait(0.2)
        worker.join()
        assert cleared.is_set() and len(Z._ZERNIKE_BASIS_CACHE) == 0


def test_polynomial(simple_grid, subtests):
    """Test polynomial() monomial summation."""
    (x, y) = simple_grid

    with subtests.test("each term is the monomial x^i y^j"):
        for (terms, weights, expected) in (
            (np.array([[0, 0]]), [5.0], 5.0 * np.ones_like(x)),
            (np.array([[1, 0]]), [1.0], x),
            (np.array([[0, 1]]), [1.0], y),
            (np.array([[2, 0], [0, 2]]), [1.0, 1.0], x**2 + y**2),
            (np.array([[-1, 0]]), [1.0], np.arctan2(y, x)),      # Vortex waveplate.
        ):
            result = phase.polynomial(simple_grid, weights=weights, terms=terms)
            assert np.allclose(result.squeeze(), expected)

    with subtests.test("1D terms are Cantor-paired monomials"):
        result = phase.polynomial(simple_grid, weights=[1.0, 1.0], terms=np.array([1, 2]))
        assert np.allclose(result.squeeze(), x + y)

    with subtests.test("weights of shape (D, N) return a stack of N patterns"):
        result = phase.polynomial(
            simple_grid, weights=np.array([[1.0, 2.0], [3.0, 4.0]]),
            terms=np.array([[1, 0], [0, 1]]),
        )
        assert np.allclose(result[0], 1.0 * x + 3.0 * y)
        assert np.allclose(result[1], 2.0 * x + 4.0 * y)

    with subtests.test("disabling pathing does not change the result"):
        terms = np.array([[2, 0], [0, 2]])
        assert np.allclose(
            phase.polynomial(simple_grid, weights=[1.0, 1.0], terms=terms, pathing=False),
            phase.polynomial(simple_grid, weights=[1.0, 1.0], terms=terms),
        )

    with subtests.test("pathing resets on non-monotonic terms"):
        result = phase.polynomial(
            simple_grid, weights=[1.0, 1.0, 1.0], terms=np.array([[2, 0], [0, 2], [1, 0]])
        )
        assert np.allclose(result.squeeze(), x**2 + y**2 + x)

    with subtests.test("out parameter reuses memory"):
        out = np.zeros((1, *x.shape), dtype=x.dtype)
        result = phase.polynomial(simple_grid, weights=[1.0], terms=np.array([[1, 0]]), out=out)
        assert np.shares_memory(result, out)

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        _assert_float32(
            phase.polynomial, simple_grid, weights=[0.1, 0.2], terms=np.array([[2, 0], [0, 1]])
        )

    for (weights, terms, match) in (
        ([1.0], np.array([[1, 0, 0]]), "Terms must be"),
        ([1.0, 2.0, 3.0], np.array([[1, 0], [0, 1]]), "common dimension"),
        (np.ones((3, 1)), np.array([[1, 0], [0, 1]]), "common dimension"),
        (np.ones((2, 1, 1)), np.array([[1, 0], [0, 1]]), "1D or 2D"),
        ([1.0], np.array([[-2, 0]]), "Unrecognized terms"),
    ):
        with subtests.test(f"malformed input raises '{match}'"):
            with pytest.raises(ValueError, match=match):
                phase.polynomial(simple_grid, weights=weights, terms=terms)


def test_laguerre_gaussian(simple_grid, fine_grid, subtests):
    """Test laguerre_gaussian() structured light generation."""
    with subtests.test("l = p = 0 is a flat phase"):
        assert np.allclose(phase.laguerre_gaussian(simple_grid, l=0, p=0), 0)

    with subtests.test("l is the counterclockwise azimuthal winding"):
        azimuth = np.arctan2(simple_grid[1], simple_grid[0])
        for l in (-3, -1, 1, 2, 3):
            assert np.allclose(phase.laguerre_gaussian(simple_grid, l=l, p=0), l * azimuth)

    with subtests.test("the p=1 radial node sits at r = w / sqrt(2)"):
        x = fine_grid[0][0, :]
        row = phase.laguerre_gaussian(fine_grid, l=0, p=1, w=40.0)[len(x) // 2, :]
        nodes = np.abs(_nodes(row, x))
        assert len(nodes) == 2
        assert np.allclose(nodes, 40.0 / np.sqrt(2), atol=2 * (x[1] - x[0]))

    with subtests.test("an omitted w is the source radius the grid implies"):
        assert np.array_equal(
            phase.laguerre_gaussian(simple_grid, l=1, p=1),
            phase.laguerre_gaussian(
                simple_grid, l=1, p=1, w=_determine_source_radius(simple_grid, None)
            ),
        )

    with subtests.test("a pure vortex needs no source radius, even on an elliptical SLM"):
        slm = SimulatedSLM((96, 64), pitch_um=(8, 8), wav_um=1.0)
        slm.set_aperture("elliptical")
        (x_grid, y_grid) = (as_numpy(g) for g in slm.grid)
        np.testing.assert_allclose(
            as_numpy(phase.laguerre_gaussian(slm, l=2)), 2 * np.arctan2(y_grid, x_grid),
            atol=1e-5,
        )

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        _assert_float32(phase.laguerre_gaussian, simple_grid, l=2, p=1)


def test_hermite_gaussian(simple_grid, fine_grid, subtests):
    """Test hermite_gaussian() structured light generation."""
    with subtests.test("the phase is binary, and flat for the n = m = 0 Gaussian"):
        assert np.allclose(phase.hermite_gaussian(simple_grid, n=0, m=0), np.pi)
        for (n, m) in ((1, 0), (0, 1), (2, 2)):
            np.testing.assert_array_equal(
                np.unique(phase.hermite_gaussian(simple_grid, n=n, m=m)), [0, np.pi]
            )

    with subtests.test("n and m count the nodal lines"):
        # A node is a sign change, encoded here as a jump of pi along a central cut.
        (rows, cols) = simple_grid[0].shape
        for order in (1, 2, 3):
            row = phase.hermite_gaussian(simple_grid, n=order, m=0)[rows // 2, :]
            col = phase.hermite_gaussian(simple_grid, n=0, m=order)[:, cols // 2]
            assert np.count_nonzero(np.abs(np.diff(row)) > np.pi / 2) == order
            assert np.count_nonzero(np.abs(np.diff(col)) > np.pi / 2) == order

    with subtests.test("the n=2 nodes sit at |x| = w / 2"):
        x = fine_grid[0][0, :]
        row = phase.hermite_gaussian(fine_grid, n=2, m=0, w=40.0)[len(x) // 2, :]
        nodes = np.abs(_nodes(row, x))
        assert len(nodes) == 2
        assert np.allclose(nodes, 40.0 / 2, atol=2 * (x[1] - x[0]))

    with subtests.test("an omitted w is the source radius the grid implies"):
        assert np.array_equal(
            phase.hermite_gaussian(simple_grid, n=2, m=0),
            phase.hermite_gaussian(
                simple_grid, n=2, m=0, w=_determine_source_radius(simple_grid, None)
            ),
        )

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        _assert_float32(phase.hermite_gaussian, simple_grid, n=1, m=1)


def test_ince_polynomial(subtests):
    """Test _ince_polynomial() against Ince's equation, DLMF 28.31.1."""
    ellipticity = 2.0
    cases = (
        (4, 0, 1), (4, 4, 1), (3, 1, 1), (5, 3, 1),
        (4, 2, -1), (6, 2, -1), (3, 1, -1), (5, 5, -1),
    )
    # Spectral derivatives are exact for a trigonometric polynomial this well sampled.
    z = 2 * np.pi * np.arange(256) / 256
    k = np.fft.fftfreq(len(z), 1 / len(z))

    def derivative(f, order):
        return np.real(np.fft.ifft((1j * k) ** order * np.fft.fft(f)))

    with subtests.test("each polynomial solves w'' + e sin(2z) w' + (eta - p e cos(2z)) w = 0"):
        for (p, m, parity) in cases:
            f = _ince_polynomial(p, m, parity, ellipticity, z)
            operator = (
                derivative(f, 2) + ellipticity * np.sin(2 * z) * derivative(f, 1)
                - p * ellipticity * np.cos(2 * z) * f
            )
            eta = -np.dot(operator, f) / np.dot(f, f)
            assert np.max(np.abs(operator + eta * f)) < 1e-9 * np.max(np.abs(f)), (p, m, parity)

    with subtests.test("C_p^m is even and S_p^m is odd in z"):
        for (p, m, parity) in cases:
            np.testing.assert_allclose(
                _ince_polynomial(p, m, parity, ellipticity, -z),
                parity * _ince_polynomial(p, m, parity, ellipticity, z),
                atol=1e-12, err_msg=str((p, m, parity)),
            )

    with subtests.test("the polynomial has m zeros in [0, pi)"):
        # Sampled off the symmetry points, where zeros sit exactly; S_p^m also vanishes at 0.
        interior = (np.arange(512) + 0.5) * np.pi / 512
        for (p, m, parity) in cases:
            f = _ince_polynomial(p, m, parity, ellipticity, interior)
            assert np.count_nonzero(np.diff(np.sign(f))) + (parity == -1) == m, (p, m, parity)

    with subtests.test("the polynomials are normalized to (1/pi) int (C_p^m)^2 dz = 1"):
        for (p, m, parity) in cases:
            f = _ince_polynomial(p, m, parity, ellipticity, z)
            assert np.sum(f ** 2) * (z[1] - z[0]) / np.pi == pytest.approx(1, abs=1e-12)

    with subtests.test("ellipticity -> 0 recovers cos(mz) and sin(mz)"):
        for (p, m, parity, expected) in ((4, 2, 1, np.cos(2 * z)), (3, 1, -1, np.sin(z))):
            f = _ince_polynomial(p, m, parity, 1e-10, z)
            assert np.allclose(
                np.abs(f / np.linalg.norm(f)), np.abs(expected / np.linalg.norm(expected)),
                atol=0.01,
            )

    with subtests.test("an imaginary argument continues the Fourier series analytically"):
        xi = np.linspace(0, 1, 11)
        coarse = 2 * np.pi * np.arange(16) / 16
        continuation = np.exp(-np.outer(xi, np.fft.fftfreq(16, 1 / 16)))
        for (p, m, parity) in cases:
            coefficients = np.fft.fft(_ince_polynomial(p, m, parity, ellipticity, coarse)) / 16
            np.testing.assert_allclose(
                _ince_polynomial(p, m, parity, ellipticity, 1j * xi), continuation @ coefficients,
                rtol=1e-9, atol=1e-12, err_msg=str((p, m, parity)),
            )


def test_ince_gaussian(simple_grid, subtests):
    """Test ince_gaussian() structured light generation."""
    with subtests.test("an invalid (p, m, parity) raises"):
        for (p, m, parity, match) in (
            (2, 5, 1, "invalid Ince"),
            (2, 0, -1, "invalid Ince"),
            (2, 1, 1, "same parity"),
            (3, 2, -1, "same parity"),
        ):
            with pytest.raises(ValueError, match=match):
                phase.ince_gaussian(simple_grid, p=p, m=m, parity=parity)

    with subtests.test("even and odd modes have binary phase, distinct in p and m"):
        cases = ((2, 0, 1), (2, 2, 1), (4, 2, 1), (4, 4, 1), (3, 1, -1), (5, 3, -1))
        results = [
            phase.ince_gaussian(simple_grid, p=p, m=m, parity=parity) for (p, m, parity) in cases
        ]
        for result in results:
            assert np.all(np.isin(np.round(np.abs(result), 8), [0.0, np.round(np.pi, 8)]))
        assert all(
            not np.allclose(results[i], results[j])
            for i in range(len(results)) for j in range(i)
        )

    with subtests.test("a helical mode winds continuously and negates under y -> -y"):
        for (p, m) in ((4, 2), (6, 2)):
            result = phase.ince_gaussian(simple_grid, p=p, m=m, parity=0)
            assert len(np.unique(np.round(result, 4))) > 10
            assert np.all(np.abs(result) <= np.pi + 1e-10)
            assert np.allclose(result, -result[::-1, :])

    with subtests.test("ellipticity -> 0 recovers laguerre_gaussian"):
        for (p, radial) in ((2, 1), (4, 2)):
            assert np.allclose(
                phase.ince_gaussian(simple_grid, p, 0, ellipticity=1e-3),
                phase.laguerre_gaussian(simple_grid, l=0, p=radial),
            )

    with subtests.test("the mode depends on the grid only through x / w"):
        stretched = (2.5 * simple_grid[0], 2.5 * simple_grid[1])
        for parity in (1, -1, 0):
            assert _phase_close(
                phase.ince_gaussian(simple_grid, p=4, m=2, parity=parity, w=2.0),
                phase.ince_gaussian(stretched, p=4, m=2, parity=parity, w=5.0),
                atol=1e-12,
            )

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        for parity in (1, -1, 0):
            _assert_float32(phase.ince_gaussian, simple_grid, p=4, m=2, parity=parity)


def test_mathieu_gaussian(simple_grid, fine_grid, subtests):
    """Test mathieu_gaussian() structured light generation."""
    with subtests.test("even and odd modes have binary phase, distinct in r"):
        results = [phase.mathieu_gaussian(simple_grid, r=r, q=5) for r in (0, 1, 2, 3, -1, -2, -3)]
        for result in results:
            assert np.all(np.isin(np.round(np.abs(result), 8), [0.0, np.round(np.pi, 8)]))
        assert all(
            not np.allclose(results[i], results[j])
            for i in range(len(results)) for j in range(i)
        )

    with subtests.test("the q=0 rings sit at the zeros of J_r"):
        # The circular limit is J_r(2 sqrt(2) rho / w), so each ring is one of its zeros.
        x = fine_grid[0][0, :]
        half = x[len(x) // 2:]
        for (r, bessel_zero) in ((0, 2.404826), (1, 3.831706), (2, 5.135622)):
            row = phase.mathieu_gaussian(fine_grid, r=r, q=0, w=40.0)[len(x) // 2, len(x) // 2:]
            ring = _nodes(row, half)[0]
            assert ring == pytest.approx(
                bessel_zero * 40.0 / (2 * np.sqrt(2)), abs=2 * (x[1] - x[0])
            )

    with subtests.test("the q=0 branch is the limit of its own q -> 0+"):
        for r in (0, 1, 2, -1, -2):
            circular = phase.mathieu_gaussian(simple_grid, r=r, q=0)
            nearly = phase.mathieu_gaussian(simple_grid, r=r, q=1e-6)
            agree = np.abs(np.angle(np.exp(1j * (circular - nearly)))) < 1e-6
            assert np.mean(agree) > 0.98, f"r={r} is discontinuous at q=0"

    with subtests.test("an omitted w is the source radius the grid implies"):
        assert np.array_equal(
            phase.mathieu_gaussian(simple_grid, r=1, q=5),
            phase.mathieu_gaussian(
                simple_grid, r=1, q=5, w=_determine_source_radius(simple_grid, None)
            ),
        )

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        for q in (0, 5):
            _assert_float32(phase.mathieu_gaussian, simple_grid, r=1, q=q)


def test_airy(simple_grid, subtests):
    """Test airy() cubic phase generation."""
    with subtests.test("infinite focal length gives zeros"):
        assert np.allclose(phase.airy(simple_grid), 0)
        assert np.allclose(phase.airy(simple_grid, f=(np.inf, np.inf)), 0)

    with subtests.test("a single-axis ramp varies only along that axis"):
        x_only = phase.airy(simple_grid, f=(1.0, np.inf))
        y_only = phase.airy(simple_grid, f=(np.inf, 1.0))
        assert np.allclose(x_only, x_only[[0], :])
        assert np.allclose(y_only, y_only[:, [0]])

    with subtests.test("the cubic phase is odd about the origin"):
        assert np.allclose(
            phase.airy(simple_grid, f=(1.0, np.inf)),
            -phase.airy(simple_grid, f=(1.0, np.inf))[:, ::-1],
        )
        assert np.allclose(
            phase.airy(simple_grid, f=(np.inf, 1.0)),
            -phase.airy(simple_grid, f=(np.inf, 1.0))[::-1, :],
        )

    with subtests.test("the phase scales as 1/f^3"):
        assert np.allclose(
            phase.airy(simple_grid, f=(1.0, np.inf)),
            (2.0 / 1.0) ** 3 * phase.airy(simple_grid, f=(2.0, np.inf)),
        )

    with subtests.test("the 2D ramp is the sum of its two axes"):
        assert np.allclose(
            phase.airy(simple_grid, f=(1.0, 1.0)),
            phase.airy(simple_grid, f=(1.0, np.inf)) + phase.airy(simple_grid, f=(np.inf, 1.0)),
        )

    with subtests.test("a float32 grid gives the float64 phase in float32"):
        _assert_float32(phase.airy, simple_grid, f=(10.0, 20.0))


def test_parse_focal_length(subtests):
    """Test _parse_focal_length() input handling."""
    with subtests.test("a scalar becomes an isotropic pair"):
        assert list(_parse_focal_length(10.0)) == [10.0, 10.0]

    with subtests.test("a pair passes through"):
        assert list(_parse_focal_length([5.0, 10.0])) == [5.0, 10.0]

    with subtests.test("the wrong number of terms raises"):
        with pytest.raises(ValueError, match="Expected two terms"):
            _parse_focal_length([1, 2, 3])

    with subtests.test("a zero focal length raises"):
        for f in (0.0, [0, 10]):
            with pytest.raises(ValueError, match="focal length of zero"):
                _parse_focal_length(f)


def test_zernike_indices_parse(subtests):
    """Test _zernike_indices_parse() defaults and consistency checks."""
    with subtests.test("D alone gives the default basis of that dimension"):
        for (D, expected) in (
            (2, [2, 1]), (3, [2, 1, 4]), (4, [2, 1, 4, 3]), (6, [2, 1, 4, 3, 5, 6])
        ):
            np.testing.assert_array_equal(_zernike_indices_parse(indices=None, D=D), expected)

    with subtests.test("a scalar requests that many indices"):
        np.testing.assert_array_equal(
            _zernike_indices_parse(indices=3, D=None), _zernike_indices_parse(indices=None, D=3)
        )
        assert len(_zernike_indices_parse(indices=4, D=4)) == 4

    with subtests.test("explicit indices pass through"):
        np.testing.assert_array_equal(_zernike_indices_parse(indices=[5, 6, 7], D=3), [5, 6, 7])

    with subtests.test("smaller_okay allows D < len(indices)"):
        np.testing.assert_array_equal(
            _zernike_indices_parse(indices=5, D=3, smaller_okay=True), [2, 1, 4, 3, 5]
        )
        np.testing.assert_array_equal(
            _zernike_indices_parse(indices=[5, 6, 7, 8], D=3, smaller_okay=True), [5, 6, 7, 8]
        )

    with subtests.test("a dimension inconsistent with the indices raises"):
        for kwargs in (
            {"indices": 3, "D": 5},
            {"indices": [1, 2, 3], "D": 5, "smaller_okay": False},
            {"indices": None, "D": None},
        ):
            with pytest.raises(ValueError):
                _zernike_indices_parse(**kwargs)


def test_cantor_pairing(subtests):
    """Test _cantor_pairing() enumeration of monomials."""
    with subtests.test("the first pairs enumerate in Cantor order"):
        np.testing.assert_array_equal(_cantor_pairing([[0, 0], [1, 0], [0, 1]]), [0, 1, 2])

    with subtests.test("the pairing is injective over a block of monomials"):
        xy = np.stack(np.meshgrid(np.arange(8), np.arange(8)), -1).reshape(-1, 2)
        assert len(np.unique(_cantor_pairing(xy))) == len(xy)


def test_inverse_cantor_pairing(subtests):
    """Test _inverse_cantor_pairing() recovery of monomials."""
    with subtests.test("it inverts _cantor_pairing"):
        xy = np.array([[0, 0], [1, 0], [0, 1], [2, 3], [5, 5]])
        np.testing.assert_array_equal(_inverse_cantor_pairing(_cantor_pairing(xy)), xy)

    with subtests.test("a negative index is the vortex waveplate (-1, 0)"):
        np.testing.assert_array_equal(_inverse_cantor_pairing(np.array([-1]))[0], [-1, 0])

    with subtests.test("a non-1D input raises"):
        with pytest.raises(ValueError):
            _inverse_cantor_pairing(np.array([[1, 2]]))


def test_parse_out(subtests):
    """Test _parse_out() buffer allocation and validation."""
    x = np.zeros((10, 10), dtype=np.float64)

    with subtests.test("None allocates a (stack, *shape) array of the grid dtype"):
        for stack in (1, 3):
            out = _parse_out(x, None, stack=stack)
            assert out.shape == (stack, 10, 10)
            assert out.dtype == x.dtype

    with subtests.test("a provided buffer is reshaped in place"):
        buf = np.zeros(200, dtype=np.float64)
        out = _parse_out(x, buf, stack=2)
        assert out.shape == (2, 10, 10)
        assert np.shares_memory(out, buf)

    with subtests.test("a buffer of the wrong size or dtype raises"):
        with pytest.raises(ValueError, match="same size"):
            _parse_out(x, np.zeros(50, dtype=np.float64), stack=1)
        with pytest.raises(ValueError, match="same type"):
            _parse_out(x, np.zeros((1, 10, 10), dtype=np.float32), stack=1)


def test_determine_source_radius(simple_grid, subtests):
    """Test _determine_source_radius() sources of the beam radius."""
    with subtests.test("w provided passes through"):
        assert _determine_source_radius(simple_grid, w=5.0) == 5.0

    with subtests.test("w=None is a quarter of the smallest grid half-width"):
        assert _determine_source_radius(simple_grid, w=None) == pytest.approx(
            min(np.amax(simple_grid[0]), np.amax(simple_grid[1])) / 4
        )

    with subtests.test("an anisotropic SLM aperture asks for w explicitly"):
        slm = SimulatedSLM((96, 64), pitch_um=(8, 8), wav_um=1.0)
        slm.set_aperture("elliptical")
        with pytest.raises(ValueError, match="w="):
            _determine_source_radius(slm, w=None)

    with subtests.test("an SLM-like object supplies its own source_radius"):
        class FakeSLM:
            x_grid = simple_grid[0]
            y_grid = simple_grid[1]
            source_radius = 42.0
        assert _determine_source_radius(FakeSLM(), w=None) == 42.0

    with subtests.test("a CameraSLM-like object delegates to its slm"):
        class FakeCameraSLM:
            x_grid = simple_grid[0]
            y_grid = simple_grid[1]
            slm = type('FakeSLM', (), {
                'source_radius': 99.0,
                'x_grid': simple_grid[0],
                'y_grid': simple_grid[1],
            })()
            cam = True
        assert _determine_source_radius(FakeCameraSLM(), w=None) == 99.0


def test_zernike_order_number():
    """The number of ANSI indices through each radial order."""
    assert [phase.zernike_order_number(n) for n in range(6)] == [1, 3, 6, 10, 15, 21]


def test_zernike_coefficients(subtests):
    """Test the monomial coefficients of the Zernike polynomials and the caches feeding them."""
    with subtests.test("the polynomials are orthogonal, with norm pi (1 + d_m0) / (2n + 2)"):
        order = 12
        _zernike_build_order(order)
        D = phase.zernike_order_number(order)
        # Gauss-Legendre in rho^2 and a uniform azimuth integrate these products exactly.
        (u, u_weights) = np.polynomial.legendre.leggauss(order + 2)
        theta = np.linspace(0, 2 * np.pi, 2 * order + 4, endpoint=False)
        (rho, t) = np.meshgrid(np.sqrt((u + 1) / 2), theta, indexing="ij")
        area = np.outer(u_weights / 4, np.full(len(theta), 2 * np.pi / len(theta)))

        images = phase.zernike_sum(
            (rho * np.cos(t), rho * np.sin(t)), np.arange(D), np.eye(D),
            aperture=1.0, use_mask=False,
        )
        gram = np.einsum("ixy,jxy,xy->ij", images, images, area)
        (n, m) = phase.zernike_convert_index(np.arange(D), to_index="radial").T
        norm = np.pi * (1 + (m == 0)) / (2 * n + 2)
        np.testing.assert_allclose(gram, np.diag(norm), atol=1e-12)

    with subtests.test("coefficient for piston is {(0,0): 1}"):
        coeffs = _zernike_coefficients(0)
        assert (0, 0) in coeffs
        assert coeffs[(0, 0)] == 1

    with subtests.test("coefficients are exact: R_n^m(1) = 1 through order 44"):
        # Along theta=0 the y-free monomials sum to R_n^m(1), which is 1 for the
        # cosine terms (l >= 0) and 0 for the sine terms (l < 0).
        for index in [0, 4, 12, 65, 100, 300, 500, 703, 840, 900, 1012, 1034]:
            (n, l) = phase.zernike_convert_index(index, to_index="radial")[0]
            total = sum(w for (a, b), w in _zernike_coefficients(index).items() if b == 0)
            assert total == (1 if l >= 0 else 0), f"ANSI {index} (n={n}, l={l}): {total}"

    with subtests.test("order 40 coefficients are exact integers"):
        assert _zernike_coefficients(840)[(26, 0)] == -44431862428800

    with subtests.test("order 44 generates and stays within the unit pupil"):
        x = np.linspace(-1, 1, 65)
        grid = np.meshgrid(x, x)
        assert len(_zernike_coefficients(1012)) > 0
        out = phase.zernike_sum(grid, [1012], [1.0], aperture="circular")
        assert np.nanmax(np.abs(out)) < 1.01

    with subtests.test("high radial order warns once about float64 precision"):
        warned = Z._zernike_precision_warned
        cached = Z._zernike_cache.pop(1012, None)
        try:
            Z._zernike_precision_warned = False
            with pytest.warns(UserWarning, match="float64 precision"):
                _zernike_coefficients(1012)
            Z._zernike_cache.pop(1012, None)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                _zernike_coefficients(1012)
        finally:
            Z._zernike_precision_warned = warned
            if cached is None:
                Z._zernike_cache.pop(1012, None)
            else:
                Z._zernike_cache[1012] = cached


def test_zernike_get_cantor():
    """High-order derivatives keep the exact power-rule factor."""
    # d^2/dx^2 x^25 = 600 x^23, the only source of x^23 in the differentiated Z_25^1.
    index = int(np.ravel(phase.zernike_convert_index([[25, 1]], "radial", "ansi"))[0])
    (pairs, weights) = Z._zernike_get_cantor([index], np.ones((1, 1)), derivative=(2, 0))
    term = [k for (k, (a, b)) in enumerate(pairs) if (a, b) == (23, 0)]
    assert weights[term, 0] == 25 * 24 * Z._zernike_coefficients(index)[(25, 0)]


def test_zernike_populate_basis_map():
    """The CUDA basis map is typed for the kernel and one column wide per Zernike."""
    (c_md, i_md, pxy_m) = _zernike_populate_basis_map(np.array([0, 1, 2, 4]))
    assert c_md.dtype == np.float32
    assert i_md.dtype == np.int32
    assert pxy_m.dtype == np.int32
    assert c_md.shape[1] == 4


def test_zernike_pyramid_plot(normalized_grid, mpl_test, subtests):
    """Test zernike_pyramid_plot(), one panel per polynomial through the order."""
    figure = mpl_test.figure(figsize=(6, 6))
    phase.zernike_pyramid_plot(normalized_grid, order=2, use_mask=False)

    with subtests.test("there is one panel per polynomial through the order"):
        assert len(figure.axes) == phase.zernike_order_number(2)

    with subtests.test("each panel shows its polynomial, titled by its ANSI index"):
        for (index, ax) in enumerate(figure.axes):
            assert ax.get_title().split("\n")[0] == str(index)
            np.testing.assert_allclose(
                ax.images[0].get_array(), phase.zernike(normalized_grid, index, use_mask=False)
            )


@pytest.mark.gpu
def test_ince_gaussian_gpu(simple_grid, has_cupy):
    """GPU variant of ince_gaussian(): a cupy grid gives the host result on the device."""
    import cupy as cp

    grid32 = tuple(g.astype(np.float32) for g in simple_grid)
    result = phase.ince_gaussian(tuple(cp.asarray(g) for g in grid32), p=4, m=2, parity=0)
    assert isinstance(result, cp.ndarray) and result.dtype == np.float32
    reference = phase.ince_gaussian(tuple(g.astype(float) for g in grid32), p=4, m=2, parity=0)
    assert _phase_close(cp.asnumpy(result), reference)


@pytest.mark.gpu
def test_mathieu_gaussian_gpu(simple_grid, has_cupy):
    """GPU variant of mathieu_gaussian(): a cupy grid gives the host result on the device."""
    import cupy as cp

    grid_cp = tuple(cp.asarray(g) for g in simple_grid)
    for q in (0, 5):
        result = phase.mathieu_gaussian(grid_cp, r=1, q=q)
        assert isinstance(result, cp.ndarray)
        np.testing.assert_array_equal(
            cp.asnumpy(result), phase.mathieu_gaussian(simple_grid, r=1, q=q)
        )


@pytest.mark.gpu
def test_zernike_sum_gpu(benchmark, has_cupy, subtests):
    """GPU variant of zernike_sum() using cupy arrays and CUDA kernels."""
    import cupy as cp

    x = cp.linspace(-1, 1, 256)
    grid = cp.meshgrid(x, x)
    rng = np.random.default_rng(42)
    coeffs = rng.normal(0, 0.1, 10)

    def run():
        phase.zernike_sum(grid, indices=list(range(len(coeffs))), weights=coeffs)

    benchmark(run)

    with subtests.test("every zernike_sum path matches numpy on a device grid"):
        host = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
        device = tuple(cp.asarray(g) for g in host)
        for kwargs in ({}, {"use_mask": np.nan}, {"derivative": (1, 1)}, {"use_mask": False}):
            args = ([1, 4, 7, 12], [1.0, -2.0, 0.5, 0.3], "circular")
            result = phase.zernike_sum(device, *args, **kwargs)
            assert isinstance(result, cp.ndarray)
            np.testing.assert_allclose(
                cp.asnumpy(result), phase.zernike_sum(host, *args, **kwargs),
                atol=1e-12, equal_nan=True, err_msg=str(kwargs),
            )


@pytest.mark.gpu
def test_zernike_get_basis_gpu(has_cupy, subtests):
    """GPU variant of _zernike_get_basis(): a cached basis lives no longer than its grid."""
    import cupy as cp

    pool = cp.get_default_memory_pool()

    def synthesize_on_a_fresh_grid():
        x = cp.linspace(-1, 1, 128)
        phase.zernike_sum(tuple(cp.meshgrid(x, x)), [1, 2, 4], [1.0, 1.0, 1.0])

    with subtests.test("a discarded device grid returns the memory pool to its start"):
        clear_zernike_basis_cache()
        synthesize_on_a_fresh_grid()    # Any lazy device state is allocated here, not below.
        gc.collect()
        before = pool.used_bytes()
        synthesize_on_a_fresh_grid()
        gc.collect()
        assert pool.used_bytes() == before


@pytest.mark.gpu
def test_zernike_basis_gpu(has_cupy):
    """ZernikeBasis on the GPU: cupy-resident basis, and numpy/cupy parity."""
    import cupy as cp
    from slmsuite.holography.toolbox.phase import ZernikeBasis
    from slmsuite.holography.analysis import image_zernike_fit

    indices = [2, 1, 4, 3, 5, 6]
    D = len(indices)
    x = np.linspace(-1, 1, 128)
    grid_np = np.meshgrid(x, x)
    grid_cp = (cp.asarray(grid_np[0]), cp.asarray(grid_np[1]))

    basis_np = ZernikeBasis(grid_np, indices)
    basis_cp = ZernikeBasis(grid_cp, indices)
    assert isinstance(basis_cp.basis_flat, cp.ndarray)
    assert isinstance(basis_cp.mask, cp.ndarray)

    rng = np.random.default_rng(1)
    weights = rng.normal(0, 0.3, D)

    # Synthesis parity between numpy and cupy.
    synth_np = phase.zernike_sum(basis_np, None, weights)
    synth_cp = phase.zernike_sum(basis_cp, None, weights)
    assert isinstance(synth_cp, cp.ndarray)
    assert np.allclose(synth_np, cp.asnumpy(synth_cp), atol=1e-6)

    # Exact least-squares fit recovers the weights on the GPU.
    coeffs_cp = image_zernike_fit(synth_cp, basis_cp, leastsquares=True)
    assert isinstance(coeffs_cp, cp.ndarray)
    assert np.allclose(cp.asnumpy(coeffs_cp)[:, 0], weights, atol=1e-5)

    # Gradient-mode fit: numpy/cupy parity, and recovery through phase wraps.
    wrapped_np = np.mod(synth_np, 2 * np.pi)
    wrapped_cp = cp.asarray(wrapped_np)
    grad_np = image_zernike_fit(wrapped_np, basis_np, gradient=True)
    grad_cp = image_zernike_fit(wrapped_cp, basis_cp, gradient=True)
    assert isinstance(grad_cp, cp.ndarray)
    assert np.allclose(cp.asnumpy(grad_cp), grad_np, atol=1e-5)
    assert np.allclose(cp.asnumpy(grad_cp)[:, 0], weights, atol=1e-2)
