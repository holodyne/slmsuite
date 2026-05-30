"""
Unit tests for slmsuite.holography.toolbox module.
"""

import numpy as np
import pytest
from scipy.spatial import distance

from slmsuite.holography import toolbox
from slmsuite.holography.toolbox import *
from slmsuite.holography.toolbox import phase


def test_convert_vector(slm, subtests):
    """Comprehensive tests for convert_vector unit conversions."""
    vec = np.array([[0.1], [-0.2]])
    hw = {"hardware": slm}
    knm_shape = (512, 512)
    knm_kw = {"hardware": slm, "shape": knm_shape}

    # Units that need no hardware, SLM-only, and knm (needs shape too).
    no_hw_units = ["norm", "kxy", "rad", "mrad", "deg"]
    slm_units = ["freq", "lpmm", "zernike"]
    all_roundtrip_units = no_hw_units + slm_units + ["knm"]

    for unit in no_hw_units:
        with subtests.test(f"identity {unit}"):
            result = convert_vector(vec, from_units=unit, to_units=unit)
            np.testing.assert_allclose(result, vec)

    with subtests.test("bad from_units"):
        with pytest.raises(ValueError, match="not recognized"):
            convert_vector((0, 0), from_units="bogus", to_units="norm")

    with subtests.test("bad to_units"):
        with pytest.raises(ValueError, match="not recognized"):
            convert_vector((0, 0), from_units="norm", to_units="bogus")

    expected = np.array([[1.0], [2.0]])
    for label, inp in [
        ("tuple", (1, 2)),
        ("list", [1, 2]),
        ("1-D array", np.array([1.0, 2.0])),
        ("(2,N) array", np.array([[1.0, 3.0], [2.0, 4.0]])),
    ]:
        with subtests.test(f"accepts {label}"):
            result = convert_vector(inp)
            if result.shape[1] == 1:
                np.testing.assert_allclose(result, expected)
            else:
                np.testing.assert_allclose(result, inp)

    with subtests.test("norm/kxy/rad are aliases"):
        v = np.array([[0.05], [-0.03]])
        for a, b in [("norm", "kxy"), ("kxy", "rad"), ("rad", "norm")]:
            np.testing.assert_allclose(convert_vector(v, a, b), v)

    with subtests.test("norm <-> mrad"):
        np.testing.assert_allclose(convert_vector(vec, "norm", "mrad"), vec * 1000)
        np.testing.assert_allclose(convert_vector(vec * 1000, "mrad", "norm"), vec)

    with subtests.test("norm <-> deg"):
        np.testing.assert_allclose(convert_vector(vec, "norm", "deg"), vec * 180 / np.pi)
        np.testing.assert_allclose(convert_vector(vec * 180 / np.pi, "deg", "norm"), vec)

    pitch_um = toolbox.format_2vectors(slm.pitch_um)
    wav_um = slm.wav_um

    with subtests.test("norm <-> freq"):
        np.testing.assert_allclose(
            convert_vector(vec, "norm", "freq", **hw), vec * pitch_um / wav_um
        )
        np.testing.assert_allclose(
            convert_vector(vec * pitch_um / wav_um, "freq", "norm", **hw), vec
        )

    with subtests.test("norm <-> lpmm"):
        np.testing.assert_allclose(convert_vector(vec, "norm", "lpmm", **hw), vec * 1000 / wav_um)
        np.testing.assert_allclose(convert_vector(vec * 1000 / wav_um, "lpmm", "norm", **hw), vec)

    shape_vec = toolbox.format_2vectors(np.flip(np.squeeze(np.array(knm_shape, dtype=float))))
    knm_conv = toolbox.format_2vectors(slm.pitch) * shape_vec

    with subtests.test("norm <-> knm"):
        np.testing.assert_allclose(
            convert_vector(vec, "norm", "knm", **knm_kw),
            vec * knm_conv + shape_vec / 2.0,
        )
        np.testing.assert_allclose(
            convert_vector(vec * knm_conv + shape_vec / 2.0, "knm", "norm", **knm_kw),
            vec,
        )

    with subtests.test("zero norm maps to knm shape/2"):
        np.testing.assert_allclose(convert_vector((0, 0), "norm", "knm", **knm_kw), shape_vec / 2.0)

    with subtests.test("knm defaults to slm.shape"):
        shape_default = toolbox.format_2vectors(
            np.flip(np.squeeze(np.array(slm.shape, dtype=float)))
        )
        np.testing.assert_allclose(convert_vector((0, 0), "norm", "knm", **hw), shape_default / 2.0)

    zernike_scale = 2 * np.pi * np.reciprocal(slm.get_source_zernike_scaling())

    with subtests.test("norm <-> zernike"):
        np.testing.assert_allclose(
            convert_vector(vec, "norm", "zernike", **hw), vec * zernike_scale
        )
        np.testing.assert_allclose(
            convert_vector(vec * zernike_scale, "zernike", "norm", **hw), vec
        )

    for unit in all_roundtrip_units:
        with subtests.test(f"roundtrip {unit}"):
            kw = knm_kw if unit == "knm" else (hw if unit in slm_units else {})
            rt = convert_vector(convert_vector(vec, "norm", unit, **kw), unit, "norm", **kw)
            np.testing.assert_allclose(rt, vec)

    for unit in ["freq", "lpmm", "knm"]:
        with subtests.test(f"{unit} without hardware warns"):
            with pytest.warns(UserWarning):
                result = convert_vector(vec, from_units=unit, to_units="norm")
            assert np.all(np.isnan(result))

    for unit in ["ij", "um"]:
        with subtests.test(f"{unit} without cameraslm warns"):
            with pytest.warns(UserWarning, match="CameraSLM"):
                result = convert_vector(vec, from_units=unit, to_units="norm")
            assert np.all(np.isnan(result))

    with subtests.test("cross-unit mrad -> deg"):
        vec_mrad = np.array([[100.0], [-200.0]])
        np.testing.assert_allclose(
            convert_vector(vec_mrad, "mrad", "deg"),
            (vec_mrad / 1000) * (180 / np.pi),
        )

    with subtests.test("cross-unit freq -> lpmm"):
        vec_freq = np.array([[0.01], [-0.02]])
        direct = convert_vector(vec_freq, "freq", "lpmm", **hw)
        via_norm = convert_vector(
            convert_vector(vec_freq, "freq", "norm", **hw), "norm", "lpmm", **hw
        )
        np.testing.assert_allclose(direct, via_norm)

    vecs_batch = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])

    with subtests.test("batch shape preserved"):
        result = convert_vector(vecs_batch, "norm", "mrad")
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result, vecs_batch * 1000)

    zero = np.array([[0.0], [0.0]])
    for unit in no_hw_units:
        with subtests.test(f"zero norm -> {unit}"):
            np.testing.assert_allclose(convert_vector(zero, "norm", unit), zero, atol=1e-15)

    vec_3d = np.array([[0.1], [-0.2], [0.5]])

    with subtests.test("3D norm identity"):
        result = convert_vector(vec_3d, "norm", "norm")
        np.testing.assert_allclose(result, vec_3d)
        assert result.shape == (3, 1)

    with subtests.test("3D norm -> mrad: xy scaled, z unchanged"):
        result = convert_vector(vec_3d, "norm", "mrad")
        np.testing.assert_allclose(result[0, 0], 100.0)
        np.testing.assert_allclose(result[1, 0], -200.0)
        np.testing.assert_allclose(result[2, 0], 0.5)

    with subtests.test("3D batch shape (3, N)"):
        vecs_3d = np.array([[0.1, 0.2], [-0.1, -0.2], [0.3, 0.4]])
        assert convert_vector(vecs_3d, "norm", "mrad").shape == (3, 2)

    for unit in ["mrad", "deg"] + slm_units:
        with subtests.test(f"symmetry {unit}"):
            pos = convert_vector(vec, "norm", unit, **hw)
            neg = convert_vector(-vec, "norm", unit, **hw)
            np.testing.assert_allclose(pos, -neg, atol=1e-12)

    with subtests.test("convert_blaze_vector alias"):
        with pytest.warns(UserWarning, match="convert_blaze_vector"):
            result = toolbox.convert_blaze_vector((0.1, -0.2))
        np.testing.assert_allclose(result, np.array([[0.1], [-0.2]]))


def test_imprint(slm, subtests, benchmark):
    """Comprehensive tests for imprint."""
    H, W = 40, 60
    x = np.arange(W, dtype=float)
    y = np.arange(H, dtype=float)
    grid = np.meshgrid(x, y)

    # (x, w, y, h) — upper-left (10,5), size 20×15
    win = [10, 20, 5, 15]
    sl = (slice(5, 20), slice(10, 30))

    with subtests.test("benchmark"):
        bench_H, bench_W = 512, 512
        bench_x = np.arange(bench_W, dtype=float)
        bench_y = np.arange(bench_H, dtype=float)
        bench_grid = np.meshgrid(bench_x, bench_y)
        bench_mat = np.zeros((bench_H, bench_W))
        bench_win = [50, 400, 50, 400]
        benchmark(imprint, bench_mat, bench_win, phase.blaze, grid=bench_grid, vector=(0.1, 0.05))

    with subtests.test("float replace"):
        mat = np.zeros((H, W))
        result = imprint(mat, win, 7.0)
        assert result is mat
        np.testing.assert_array_equal(mat[sl], 7.0)
        mat[sl] = 0
        np.testing.assert_array_equal(mat, 0)

    with subtests.test("float add"):
        mat = np.ones((H, W))
        imprint(mat, win, 3.0, imprint_operation="add")
        np.testing.assert_array_equal(mat[sl], 4.0)
        mat[sl] = 1.0
        np.testing.assert_array_equal(mat, 1.0)

    with subtests.test("callable replace"):
        mat = np.full((H, W), 99.0)
        imprint(mat, win, phase.blaze, grid=grid, vector=(0, 0))
        # blaze with zero vector returns zeros in the window
        np.testing.assert_allclose(mat[sl], 0.0)
        # outside window is untouched
        assert mat[0, 0] == 99.0

    with subtests.test("callable add"):
        mat = np.ones((H, W))
        imprint(mat, win, phase.blaze, grid=grid, vector=(0, 0), imprint_operation="add")
        np.testing.assert_allclose(mat[sl], 1.0)  # 1 + 0
        assert mat[0, 0] == 1.0

    with subtests.test("callable produces nonzero phase"):
        mat = np.zeros((H, W))
        imprint(mat, win, phase.blaze, grid=grid, vector=(0.1, 0))
        assert not np.allclose(mat[sl], 0)
        mat[sl] = 0
        np.testing.assert_array_equal(mat, 0)

    with subtests.test("bad imprint_operation"):
        with pytest.raises(ValueError, match="Unrecognized"):
            imprint(np.zeros((H, W)), win, 1.0, imprint_operation="multiply")

    with subtests.test("grid=None with callable raises"):
        with pytest.raises(ValueError, match="grid cannot be None"):
            imprint(np.zeros((H, W)), win, phase.blaze, grid=None)

    with subtests.test("grid=None with float is fine"):
        mat = np.zeros((H, W))
        imprint(mat, win, 5.0, grid=None)
        np.testing.assert_array_equal(mat[sl], 5.0)

    with subtests.test("boolean mask window"):
        mat = np.zeros((H, W))
        mask = np.zeros((H, W), dtype=bool)
        mask[0, 0] = True
        mask[H - 1, W - 1] = True
        imprint(mat, mask, 42.0)
        assert mat[0, 0] == 42.0
        assert mat[H - 1, W - 1] == 42.0
        assert mat[0, 1] == 0.0

    with subtests.test("index-pair window"):
        mat = np.zeros((H, W))
        y_idx = np.array([0, 1, 2])
        x_idx = np.array([5, 5, 5])
        imprint(mat, (y_idx, x_idx), 10.0)
        for yi, xi in zip(y_idx, x_idx):
            assert mat[yi, xi] == 10.0
        assert mat[3, 5] == 0.0

    with subtests.test("clip=True clips out-of-bounds"):
        mat = np.zeros((H, W))
        big_win = [W - 5, 20, H - 5, 20]
        imprint(mat, big_win, 1.0, clip=True)
        # Clipped region should be filled (clip caps slice ends to shape-1)
        assert mat[H - 5, W - 5] == 1.0
        assert mat[0, 0] == 0.0
        # Total filled pixels should be less than full 20x20 window
        assert 0 < np.sum(mat) < 20 * 20

    with subtests.test("centered window"):
        mat = np.zeros((H, W))
        # centered: (cx, w, cy, h) — center at (20, 10) size 6×4
        cwin = [20, 6, 10, 4]
        imprint(mat, cwin, 1.0, centered=True)
        csl = window_slice(cwin, centered=True)
        np.testing.assert_array_equal(mat[csl], 1.0)
        # Verify the region differs from non-centered interpretation
        ncsl = window_slice(cwin, centered=False)
        assert csl != ncsl

    with subtests.test("SLM as grid"):
        mat = np.zeros(slm.shape)
        small_win = [0, 10, 0, 10]
        imprint(mat, small_win, phase.blaze, grid=slm, vector=(0, 0))
        np.testing.assert_allclose(mat[:10, :10], 0.0)

    with subtests.test("shift=True centers sub-grid"):
        mat1 = np.zeros((H, W))
        mat2 = np.zeros((H, W))
        # With shift=True the sub-grid is recentered so blaze has different absolute values
        imprint(mat1, win, phase.blaze, grid=grid, vector=(0.1, 0), shift=(0, 0))
        imprint(mat2, win, phase.blaze, grid=grid, vector=(0.1, 0), shift=True)
        assert not np.allclose(mat1[sl], mat2[sl])

    with subtests.test("transform rotates sub-grid"):
        mat1 = np.zeros((H, W))
        mat2 = np.zeros((H, W))
        imprint(mat1, win, phase.blaze, grid=grid, vector=(0.1, 0), transform=0)
        imprint(mat2, win, phase.blaze, grid=grid, vector=(0.1, 0), transform=np.pi / 4)
        assert not np.allclose(mat1[sl], mat2[sl])


def test_format_vectors(subtests):
    """Comprehensive tests for format_vectors and format_2vectors."""

    for label, inp in [
        ("tuple", (1, 2)),
        ("list", [1, 2]),
        ("1D array", np.array([1, 2])),
    ]:
        with subtests.test(f"2vec from {label}"):
            result = format_vectors(inp)
            assert result.shape == (2, 1)
            np.testing.assert_array_equal(result, [[1], [2]])

    with subtests.test("(2,N) passthrough"):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = format_vectors(arr)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, arr)

    with subtests.test("row vector transposed"):
        arr = np.array([[1, 2]])
        result = format_vectors(arr)
        assert result.shape == (2, 1)

    with subtests.test("3-vectors"):
        result = format_vectors(np.array([[1, 2], [3, 4], [5, 6]]), expected_dimension=3)
        assert result.shape == (3, 2)

    with subtests.test("too few dims raises"):
        with pytest.raises(ValueError):
            format_vectors(np.array([[1, 2]]), expected_dimension=3)

    with subtests.test("handle_dimension error"):
        with pytest.raises(ValueError, match="Expected 2-vectors"):
            format_vectors(
                np.array([[1], [2], [3]]),
                expected_dimension=2,
                handle_dimension="error",
            )

    with subtests.test("handle_dimension crop"):
        result = format_vectors(
            np.array([[1], [2], [3]]),
            expected_dimension=2,
            handle_dimension="crop",
        )
        assert result.shape == (2, 1)
        np.testing.assert_array_equal(result, [[1], [2]])

    with subtests.test("handle_dimension pass"):
        result = format_vectors(
            np.array([[1], [2], [3]]),
            expected_dimension=2,
            handle_dimension="pass",
        )
        assert result.shape == (3, 1)

    with subtests.test("bad handle_dimension"):
        with pytest.raises(ValueError, match="not recognized"):
            format_vectors(np.array([1, 2]), handle_dimension="bad")

    with subtests.test("format_2vectors crops 3D"):
        result = format_2vectors(np.array([[1], [2], [3]]))
        assert result.shape == (2, 1)
        np.testing.assert_array_equal(result, [[1], [2]])

    with subtests.test("format_2vectors basic"):
        result = format_2vectors((5, 10))
        assert result.shape == (2, 1)
        np.testing.assert_array_equal(result, [[5], [10]])

    with subtests.test("scalar raises"):
        with pytest.raises((ValueError, TypeError)):
            format_vectors(5)

    with subtests.test("float preserved"):
        result = format_vectors(np.array([1.5, 2.5]))
        assert result.dtype == np.float64

    for n in [1, 5, 100]:
        with subtests.test(f"batch N={n}"):
            arr = np.random.rand(2, n)
            result = format_vectors(arr)
            assert result.shape == (2, n)
            np.testing.assert_array_equal(result, arr)


def test_fit_3pt(subtests):
    """Comprehensive tests for fit_3pt."""

    with subtests.test("identity affine dict"):
        d = fit_3pt((0, 0), (1, 0), (0, 1), N=None)
        np.testing.assert_allclose(d["M"], np.eye(2), atol=1e-14)
        np.testing.assert_allclose(d["b"], np.zeros((2, 1)), atol=1e-14)

    with subtests.test("identity grid 3x3"):
        result = fit_3pt((0, 0), (1, 0), (0, 1), N=(3, 3))
        assert result.shape == (2, 9)
        # First point should be (0,0)
        np.testing.assert_allclose(result[:, 0], [0, 0], atol=1e-14)

    with subtests.test("translation"):
        d = fit_3pt((10, 20), (11, 20), (10, 21), N=None)
        np.testing.assert_allclose(d["M"], np.eye(2), atol=1e-14)
        np.testing.assert_allclose(d["b"], [[10], [20]], atol=1e-14)

    with subtests.test("2x scaling"):
        d = fit_3pt((0, 0), (2, 0), (0, 2), N=None)
        np.testing.assert_allclose(d["M"], 2 * np.eye(2), atol=1e-14)
        np.testing.assert_allclose(d["b"], np.zeros((2, 1)), atol=1e-14)

    with subtests.test("90 degree rotation"):
        d = fit_3pt((0, 0), (0, 1), (-1, 0), N=None)
        expected_M = np.array([[0, -1], [1, 0]], dtype=float)
        np.testing.assert_allclose(d["M"], expected_M, atol=1e-14)

    with subtests.test("N scalar"):
        result = fit_3pt((0, 0), (1, 0), (0, 1), N=4)
        assert result.shape == (2, 16)

    for label, n_val in [("N=0", 0), ("N=-1", -1), ("N=None", None)]:
        with subtests.test(f"affine return {label}"):
            d = fit_3pt((0, 0), (1, 0), (0, 1), N=n_val)
            assert isinstance(d, dict)
            assert "M" in d and "b" in d

    with subtests.test("custom x indices"):
        # Points at x=(2,0) and x=(0,3) instead of (1,0) and (0,1)
        d = fit_3pt((0, 0), (4, 0), (0, 6), N=None, x0=(0, 0), x1=(2, 0), x2=(0, 3))
        np.testing.assert_allclose(d["M"], 2 * np.eye(2), atol=1e-14)

    with subtests.test("difference mode x1=None"):
        origin = np.array([10, 20])
        dv1 = np.array([1, 0])
        dv2 = np.array([0, 1])
        d1 = fit_3pt(origin, origin + dv1, origin + dv2, N=None)
        d2 = fit_3pt(origin, dv1, dv2, N=None, x1=None, x2=None)
        np.testing.assert_allclose(d1["M"], d2["M"], atol=1e-14)
        np.testing.assert_allclose(d1["b"], d2["b"], atol=1e-14)

    with subtests.test("colinear raises"):
        with pytest.raises(ValueError, match="colinear"):
            fit_3pt((0, 0), (1, 0), (2, 0), x0=(0, 0), x1=(1, 0), x2=(2, 0))

    with subtests.test("orientation_check"):
        full = fit_3pt((0, 0), (1, 0), (0, 1), N=(3, 3))
        trimmed = fit_3pt((0, 0), (1, 0), (0, 1), N=(3, 3), orientation_check=True)
        assert trimmed.shape[1] == full.shape[1] - 2
        np.testing.assert_allclose(trimmed, full[:, :-2])

    with subtests.test("N as ndarray"):
        pts = np.array([[0, 1, 2], [0, 0, 0]])
        result = fit_3pt((5, 10), (6, 10), (5, 11), N=pts)
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result[:, 0], [5, 10], atol=1e-14)
        np.testing.assert_allclose(result[:, 1], [6, 10], atol=1e-14)

    with subtests.test("roundtrip"):
        d = fit_3pt((3, 7), (5, 8), (4, 10), N=None)
        M, b = d["M"], d["b"]
        # Evaluate at x=(0,0), (1,0), (0,1)
        np.testing.assert_allclose(M @ [[0], [0]] + b, [[3], [7]], atol=1e-14)
        np.testing.assert_allclose(M @ [[1], [0]] + b, [[5], [8]], atol=1e-14)
        np.testing.assert_allclose(M @ [[0], [1]] + b, [[4], [10]], atol=1e-14)


def test_smallest_distance(subtests):
    """Comprehensive tests for smallest_distance."""

    for label, vecs in [
        ("single point", np.array([[5], [3]])),
        ("empty (0 cols)", np.empty((2, 0))),
    ]:
        with subtests.test(label):
            assert smallest_distance(vecs) == np.inf

    with subtests.test("two points chebyshev"):
        vecs = np.array([[0, 3], [0, 4]])
        # chebyshev = max(|3-0|, |4-0|) = 4
        assert smallest_distance(vecs) == pytest.approx(4.0)

    with subtests.test("minimum among many"):
        vecs = np.array([[0, 10, 11, 50], [0, 10, 11, 50]])
        # closest pair is (10,10)-(11,11), chebyshev = 1
        assert smallest_distance(vecs) == pytest.approx(1.0)

    with subtests.test("duplicate points"):
        vecs = np.array([[1, 2, 1], [3, 4, 3]])
        assert smallest_distance(vecs) == pytest.approx(0.0)

    with subtests.test("negative coordinates"):
        vecs = np.array([[-5, -3], [10, 10]])
        # chebyshev = max(|-3-(-5)|, |10-10|) = 2
        assert smallest_distance(vecs) == pytest.approx(2.0)

    for label, inp in [
        ("list of tuples", [(0, 0), (3, 4)]),
        ("tuple pair", ((0, 3), (0, 4))),
    ]:
        with subtests.test(f"input: {label}"):
            result = smallest_distance(inp)
            assert np.isfinite(result)

    pts = np.array([[0, 3], [0, 4]])
    metric_expected = {
        "chebyshev": 4.0,
        "euclidean": 5.0,
        "cityblock": 7.0,
    }
    for metric, expected in metric_expected.items():
        with subtests.test(f"metric {metric}"):
            assert smallest_distance(pts, metric=metric) == pytest.approx(expected)

    with subtests.test("custom callable metric"):
        vecs = np.array([[0, 3, 10], [0, 4, 10]])
        euclidean_fn = lambda a, b: np.sqrt(np.sum((a - b) ** 2))
        result = smallest_distance(vecs, metric=euclidean_fn)
        assert result == pytest.approx(5.0)

    with subtests.test("collinear equally spaced"):
        vecs = np.array([[0, 2, 4, 6, 8], [0, 0, 0, 0, 0]])
        assert smallest_distance(vecs) == pytest.approx(2.0)

    with subtests.test("collinear unequally spaced"):
        vecs = np.array([[0, 1, 5, 20], [0, 0, 0, 0]])
        assert smallest_distance(vecs) == pytest.approx(1.0)

    with subtests.test("large N divide-and-conquer"):
        rng = np.random.default_rng(42)
        n = 500
        vecs = rng.uniform(0, 1000, size=(2, n))
        result_str = smallest_distance(vecs, metric="chebyshev")
        # Compare against brute-force via pdist
        expected = distance.pdist(vecs.T, metric="chebyshev").min()
        assert result_str == pytest.approx(expected, rel=1e-10)

    with subtests.test("large N euclidean"):
        rng = np.random.default_rng(123)
        n = 500
        vecs = rng.uniform(0, 1000, size=(2, n))
        result = smallest_distance(vecs, metric="euclidean")
        expected = distance.pdist(vecs.T, metric="euclidean").min()
        assert result == pytest.approx(expected, rel=1e-10)

    with subtests.test("str vs callable agree"):
        rng = np.random.default_rng(7)
        vecs = rng.uniform(0, 100, size=(2, 50))
        str_result = smallest_distance(vecs, metric="euclidean")
        fn_result = smallest_distance(vecs, metric=lambda a, b: np.sqrt(np.sum((a - b) ** 2)))
        assert str_result == pytest.approx(fn_result, rel=1e-10)


def test_lloyds_algorithm(subtests):
    """Comprehensive tests for lloyds_algorithm."""
    shape = (100, 100)
    grid = np.meshgrid(range(shape[1]), range(shape[0]))

    for n in [3, 5, 10]:
        with subtests.test(f"output shape {n} points"):
            seeds = np.array(
                [
                    np.linspace(10, 90, n),
                    np.linspace(10, 90, n),
                ]
            )
            result = lloyds_algorithm(grid, seeds, iterations=5)
            assert result.shape == (2, n)

    with subtests.test("zero iterations"):
        seeds = np.array([[20, 50, 80], [20, 50, 80]])
        result = lloyds_algorithm(grid, seeds, iterations=0)
        np.testing.assert_allclose(result, seeds.astype(float))

    with subtests.test("points stay in bounds"):
        rng = np.random.default_rng(42)
        seeds = rng.uniform(5, 95, size=(2, 15))
        result = lloyds_algorithm(grid, seeds, iterations=20)
        assert np.all(result[0] >= 0) and np.all(result[0] <= shape[1])
        assert np.all(result[1] >= 0) and np.all(result[1] <= shape[0])

    with subtests.test("spacing improves"):
        rng = np.random.default_rng(99)
        seeds = rng.uniform(5, 95, size=(2, 12))
        before = smallest_distance(seeds, metric="euclidean")
        result = lloyds_algorithm(grid, seeds, iterations=30)
        after = smallest_distance(result, metric="euclidean")
        assert after >= before * 0.95  # allow small tolerance

    with subtests.test("shape tuple grid"):
        seeds = np.array([[20, 50, 80], [20, 50, 80]])
        result = lloyds_algorithm(shape, seeds, iterations=5)
        assert result.shape == (2, 3)
        assert np.all(result[0] >= 0) and np.all(result[0] <= shape[1])
        assert np.all(result[1] >= 0) and np.all(result[1] <= shape[0])

    with subtests.test("deterministic"):
        seeds = np.array([[10, 30, 70, 90], [50, 50, 50, 50]])
        r1 = lloyds_algorithm(grid, seeds, iterations=10)
        r2 = lloyds_algorithm(grid, seeds, iterations=10)
        np.testing.assert_allclose(r1, r2)

    with subtests.test("convergence"):
        seeds = np.array([[10, 11, 12, 88, 89, 90], [50, 50, 50, 50, 50, 50]])
        r5 = lloyds_algorithm(grid, seeds, iterations=5)
        r50 = lloyds_algorithm(grid, seeds, iterations=50)
        spread_5 = smallest_distance(r5, metric="euclidean")
        spread_50 = smallest_distance(r50, metric="euclidean")
        assert spread_50 >= spread_5 - 1e-6

    with subtests.test("two points converge"):
        seeds = np.array([[10, 90], [50, 50]])
        result = lloyds_algorithm(grid, seeds, iterations=50)
        # Should be near x=25, x=75 (or y-centers at 50)
        xs = np.sort(result[0])
        assert xs[0] == pytest.approx(25, abs=5)
        assert xs[1] == pytest.approx(75, abs=5)

    with subtests.test("rectangular grid"):
        rect_shape = (50, 200)
        rect_grid = np.meshgrid(range(rect_shape[1]), range(rect_shape[0]))
        seeds = np.array([[50, 100, 150], [10, 25, 40]])
        result = lloyds_algorithm(rect_grid, seeds, iterations=10)
        assert result.shape == (2, 3)
        assert np.all(result[0] >= 0) and np.all(result[0] <= rect_shape[1])
        assert np.all(result[1] >= 0) and np.all(result[1] <= rect_shape[0])


def test_lloyds_points(subtests):
    """Tests for lloyds_points (wrapper around lloyds_algorithm)."""
    shape = (100, 100)

    with subtests.test("output shape"):
        np.random.seed(42)
        result = lloyds_points(shape, 7, iterations=5)
        assert result.shape == (2, 7)

    with subtests.test("meshgrid input"):
        np.random.seed(11)
        grid = np.meshgrid(range(shape[1]), range(shape[0]))
        result = lloyds_points(grid, 6, iterations=10)
        assert result.shape == (2, 6)

    with subtests.test("no duplicates"):
        np.random.seed(0)
        result = lloyds_points(shape, 10, iterations=10)
        assert smallest_distance(result) > 0

    with subtests.test("single point near center"):
        np.random.seed(22)
        result = lloyds_points(shape, 1, iterations=20)
        assert result.shape == (2, 1)
        assert result[0, 0] == pytest.approx(50, abs=10)
        assert result[1, 0] == pytest.approx(50, abs=10)


def test_assign_vectors(subtests):
    """Comprehensive tests for assign_vectors."""

    with subtests.test("exact matches"):
        options = np.array([[1, 2, 3], [1, 2, 3]])
        vectors = np.array([[1, 2, 3], [1, 2, 3]])
        np.testing.assert_array_equal(assign_vectors(vectors, options), [0, 1, 2])

    with subtests.test("nearest neighbor"):
        options = np.array([[0, 10, 20], [0, 10, 20]])
        vectors = np.array([[1, 11], [1, 11]])
        np.testing.assert_array_equal(assign_vectors(vectors, options), [0, 1])

    with subtests.test("single vector single option"):
        result = assign_vectors(np.array([[5], [5]]), np.array([[0], [0]]))
        np.testing.assert_array_equal(result, [0])

    with subtests.test("all map to same"):
        options = np.array([[0, 100], [0, 100]])
        vectors = np.array([[1, 2, 3], [1, 2, 3]])
        np.testing.assert_array_equal(assign_vectors(vectors, options), [0, 0, 0])

    with subtests.test("equidistant picks lower index"):
        options = np.array([[-1, 1], [0, 0]])
        vectors = np.array([[0], [0]])
        result = assign_vectors(vectors, options)
        assert result[0] == 0  # argmin returns first

    with subtests.test("output shape"):
        options = np.array([[0, 10, 20], [0, 10, 20]])
        vectors = np.array([[5, 15, 25, 35], [5, 15, 25, 35]])
        result = assign_vectors(vectors, options)
        assert result.shape == (4,)

    with subtests.test("tuple inputs"):
        result = assign_vectors([(5, 5)], [(0, 10), (0, 10)])
        assert np.isfinite(result).all()


def test_format_shape(subtests):
    """Comprehensive tests for format_shape."""

    for label, inp, expected in [
        ("tuple", (10, 20), (10, 20)),
        ("list", [10, 20], (10, 20)),
        ("numpy array", np.array([10, 20]), (10, 20)),
    ]:
        with subtests.test(f"valid: {label}"):
            assert format_shape(inp) == expected

    with subtests.test("any dim: 3D"):
        assert format_shape((2, 3, 4), expected_dimension=None) == (2, 3, 4)

    with subtests.test("wrong dimension raises"):
        with pytest.raises(ValueError, match="dimensions"):
            format_shape((1, 2, 3), expected_dimension=2)

    for label, inp in [("zero", (0, 5)), ("negative", (5, -1))]:
        with subtests.test(f"bad dim: {label}"):
            with pytest.raises(ValueError, match="positive integer"):
                format_shape(inp)

    with subtests.test("float raises"):
        with pytest.raises(ValueError, match="positive integer"):
            format_shape((1.5, 2.5))


def test_pad_unpad(subtests):
    """Comprehensive tests for pad and unpad."""
    mat = np.arange(12).reshape(3, 4)

    with subtests.test("pad shape"):
        result = pad(mat, (7, 10))
        assert result.shape == (7, 10)

    with subtests.test("pad preserves center"):
        result = pad(mat, (7, 10))
        # Original 3x4 data centered in 7x10
        b = (7 - 3) // 2  # =2
        l = (10 - 4) // 2  # =3
        np.testing.assert_array_equal(result[b : b + 3, l : l + 4], mat)

    with subtests.test("pad zeros in border"):
        result = pad(mat, (7, 10))
        total_nonzero = np.count_nonzero(result)
        mat_nonzero = np.count_nonzero(mat)
        assert total_nonzero == mat_nonzero

    with subtests.test("pad None returns original"):
        result = pad(mat, None)
        np.testing.assert_array_equal(result, mat)

    with subtests.test("pad same shape"):
        result = pad(mat, mat.shape)
        np.testing.assert_array_equal(result, mat)

    with subtests.test("pad too small raises"):
        with pytest.raises(ValueError, match="too large"):
            pad(mat, (2, 2))

    with subtests.test("unpad shape"):
        big = pad(mat, (7, 10))
        result = unpad(big, (3, 4))
        assert result.shape == (3, 4)

    with subtests.test("unpad None returns original"):
        result = unpad(mat, None)
        np.testing.assert_array_equal(result, mat)

    with subtests.test("unpad same shape"):
        result = unpad(mat, mat.shape)
        np.testing.assert_array_equal(result, mat)

    with subtests.test("unpad too large raises"):
        with pytest.raises(ValueError, match="too small"):
            unpad(mat, (10, 10))

    with subtests.test("unpad shape returns slicing args"):
        args = unpad((7, 10), (3, 4))
        assert len(args) == 4
        b, t, l, r = args
        big = pad(mat, (7, 10))
        np.testing.assert_array_equal(big[b:t, l:r], mat)

    with subtests.test("unpad shape None returns full range"):
        args = unpad((7, 10), None)
        assert args == (0, 7, 0, 10)

    for label, target in [
        ("even", (8, 12)),
        ("odd", (9, 11)),
        ("asymmetric", (5, 10)),
    ]:
        with subtests.test(f"roundtrip {label}"):
            np.testing.assert_array_equal(unpad(pad(mat, target), mat.shape), mat)

    with subtests.test("odd delta padding"):
        small = np.ones((2, 3))
        result = pad(small, (3, 4))
        assert result.shape == (3, 4)
        recovered = unpad(result, (2, 3))
        np.testing.assert_array_equal(recovered, small)
