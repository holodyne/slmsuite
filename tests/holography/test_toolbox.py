"""
Unit tests for slmsuite.holography.toolbox module.
"""
import logging

import pytest
import numpy as np

from scipy.spatial import distance

from slmsuite.holography import toolbox
from slmsuite.holography.toolbox import *
from slmsuite.holography.toolbox import phase


def test_convert_vector(slm, camera, fourierslm_calibrated, subtests, caplog):
    """Test convert_vector's unit conversions."""
    vec = np.array([[0.1], [-0.2]])
    hw = {"hardware": slm}
    knm_shape = (256, 512)
    knm_kw = {"hardware": slm, "shape": knm_shape}
    (height, width) = slm.shape

    with subtests.test("an unrecognized unit raises"):
        with pytest.raises(ValueError, match="not recognized"):
            convert_vector((0, 0), from_units="bogus", to_units="norm")
        with pytest.raises(ValueError, match="not recognized"):
            convert_vector((0, 0), from_units="norm", to_units="bogus")

    with subtests.test("every unit is its own identity"):
        for unit in toolbox.BLAZE_UNITS:
            np.testing.assert_allclose(convert_vector(vec, unit, unit), vec, err_msg=unit)

    with subtests.test("input is cleaned into (2, N) columns"):
        for inp in [(1, 2), [1, 2], np.array([1.0, 2.0]), np.array([[1.0, 2.0]])]:
            np.testing.assert_allclose(convert_vector(inp), [[1.0], [2.0]])
        batch = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
        np.testing.assert_allclose(convert_vector(batch, "norm", "mrad"), batch * 1000)

    with subtests.test("norm, kxy and rad are one unit under three names"):
        for a, b in [("norm", "kxy"), ("kxy", "rad"), ("rad", "norm")]:
            np.testing.assert_allclose(convert_vector(vec, a, b), vec)

    with subtests.test("angle units are exact multiples of the paraxial radian"):
        np.testing.assert_allclose(convert_vector(vec, "norm", "mrad"), vec * 1000)
        np.testing.assert_allclose(convert_vector(vec, "norm", "deg"), vec * 180 / np.pi)
        np.testing.assert_allclose(convert_vector(vec * 1000, "mrad", "deg"), vec * 180 / np.pi)

    # The blaze this vector describes, against which the grating units are measured.
    ramp = phase.blaze(slm, (vec[0, 0], vec[1, 0]))
    cycles = np.array([
        [(ramp[0, -1] - ramp[0, 0]) / (width - 1)],
        [(ramp[-1, 0] - ramp[0, 0]) / (height - 1)],
    ]) / (2 * np.pi)

    with subtests.test("freq is the blaze's phase cycles per pixel"):
        # An SLM stores its grid in float32, hence the tolerance here and below.
        np.testing.assert_allclose(convert_vector(vec, "norm", "freq", **hw), cycles, rtol=1e-6)

    with subtests.test("lpmm is the blaze's phase cycles per millimeter"):
        per_mm = cycles * 1000 / toolbox.format_2vectors(slm.pitch_um)
        np.testing.assert_allclose(convert_vector(vec, "norm", "lpmm", **hw), per_mm, rtol=1e-6)

    shape_xy = np.array([[knm_shape[1]], [knm_shape[0]]], dtype=float)

    with subtests.test("knm is the freq grating's DFT bin, offset to the shape's center"):
        freq = convert_vector(vec, "norm", "freq", **hw)
        np.testing.assert_allclose(
            convert_vector(vec, "norm", "knm", **knm_kw), freq * shape_xy + shape_xy / 2
        )

    with subtests.test("an unblazed beam sits at the knm origin, shape/2"):
        np.testing.assert_allclose(convert_vector((0, 0), "norm", "knm", **knm_kw), shape_xy / 2)
        np.testing.assert_allclose(
            convert_vector((0, 0), "norm", "knm", **hw), [[width / 2], [height / 2]]
        )

    with subtests.test("zernike gives the tilt weights that rebuild the blaze"):
        coeff = convert_vector(vec, "norm", "zernike", **hw)
        tilt = phase.zernike_sum(
            slm, indices=(2, 1), weights=(coeff[0, 0], coeff[1, 0]), use_mask=False
        )
        np.testing.assert_allclose(tilt, ramp, atol=1e-3)

    with subtests.test("every unit inverts back to norm"):
        for unit in ["kxy", "rad", "mrad", "deg", "freq", "lpmm", "zernike", "knm"]:
            kw = knm_kw if unit == "knm" else hw
            roundtrip = convert_vector(convert_vector(vec, "norm", unit, **kw), unit, "norm", **kw)
            np.testing.assert_allclose(roundtrip, vec, err_msg=unit)

    with subtests.test("the z component carries focal power, untouched by the xy scaling"):
        vec_3d = np.array([[0.1], [-0.2], [0.5]])
        np.testing.assert_allclose(
            convert_vector(vec_3d, "norm", "mrad"), [[100.0], [-200.0], [0.5]]
        )
        assert convert_vector(np.hstack((vec_3d, vec_3d)), "norm", "mrad").shape == (3, 2)

    with subtests.test("a unit needing an SLM warns and returns nan without one"):
        for unit in ["freq", "lpmm", "knm", "zernike"]:
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                caplog.clear()
                result = convert_vector(vec, from_units=unit, to_units="norm")
            assert any(r.levelno == logging.WARNING for r in caplog.records), unit
            assert np.all(np.isnan(result)), unit

    with subtests.test("a camera unit warns and returns nan without a CameraSLM"):
        for unit in ["ij", "um"]:
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                caplog.clear()
                result = convert_vector(vec, from_units=unit, to_units="norm")
            assert any("CameraSLM" in r.getMessage() for r in caplog.records), unit
            assert np.all(np.isnan(result)), unit

    camera.set_binning(2)
    camera.set_woi((10, 100, 20, 80))
    ij = np.array([[5.0], [7.0]])

    with subtests.test("ij and ijraw differ by the camera's own sensor affine"):
        raw = convert_vector(ij, "ij", "ijraw", hardware=camera)
        np.testing.assert_allclose(raw, camera._get_ijcam_to_ijraw() @ ij)
        np.testing.assert_allclose(convert_vector(raw, "ijraw", "ij", hardware=camera), ij)

    with subtests.test("ijraw depth scales by the isotropic binning"):
        ij_3d = np.array([[5.0], [7.0], [0.5]])
        raw_3d = convert_vector(ij_3d, "ij", "ijraw", hardware=camera)
        scale = np.sqrt(np.abs(camera._get_ijcam_to_ijraw().det()))
        np.testing.assert_allclose(raw_3d[2, 0], ij_3d[2, 0] * scale)
        np.testing.assert_allclose(convert_vector(raw_3d, "ijraw", "ij", hardware=camera), ij_3d)

    with subtests.test("ijraw reaches the blaze units through a FourierSLM"):
        v = np.array([[0.01], [-0.02]])
        ij_cal = convert_vector(v, "norm", "ij", hardware=fourierslm_calibrated)
        np.testing.assert_allclose(
            convert_vector(v, "norm", "ijraw", hardware=fourierslm_calibrated),
            fourierslm_calibrated.cam._get_ijcam_to_ijraw() @ ij_cal,
        )

    with subtests.test("a bare Camera cannot reach a blaze unit"):
        with caplog.at_level(logging.WARNING, logger="slmsuite"):
            caplog.clear()
            result = convert_vector(ij, "ijraw", "norm", hardware=camera)
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert np.all(np.isnan(result))

    with subtests.test("convert_blaze_vector is a deprecated alias"):
        with pytest.warns(UserWarning, match="convert_blaze_vector"):
            result = toolbox.convert_blaze_vector((0.1, -0.2))
        np.testing.assert_allclose(result, [[0.1], [-0.2]])


def test_convert_radius(slm, subtests):
    """Test convert_radius' scalar radius conversions."""
    units = ["mrad", "deg", "freq", "lpmm", "zernike"]

    with subtests.test("angle units scale the radius by their definition"):
        assert convert_radius(0.05, "norm", "norm") == pytest.approx(0.05)
        assert convert_radius(0.0, "norm", "mrad") == pytest.approx(0.0)
        assert convert_radius(0.1, "norm", "mrad") == pytest.approx(100.0)
        assert convert_radius(0.1, "norm", "deg") == pytest.approx(0.1 * 180 / np.pi)

    with subtests.test("the radius is the length convert_vector gives that displacement"):
        for unit in units:
            origin = convert_vector((0, 0), "norm", unit, hardware=slm)
            offset = convert_vector((0.1, 0), "norm", unit, hardware=slm)
            assert convert_radius(0.1, "norm", unit, hardware=slm) == pytest.approx(
                float(np.linalg.norm(offset - origin))
            ), unit

    with subtests.test("every unit inverts back to norm"):
        for unit in units:
            radius = convert_radius(0.05, "norm", unit, hardware=slm)
            assert convert_radius(radius, unit, "norm", hardware=slm) == pytest.approx(0.05), unit


def test_imprint(slm, subtests, benchmark):
    """Test imprint's in-place write of a function into a windowed region."""
    (H, W) = (40, 60)
    grid = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    win = [10, 20, 5, 15]                       # (x, w, y, h): upper-left (10, 5), size 20x15
    sl = (slice(5, 20), slice(10, 30))
    sub = (grid[0][sl], grid[1][sl])
    vector = (0.1, -0.05)

    with subtests.test("benchmark"):
        bench_grid = np.meshgrid(np.arange(512, dtype=float), np.arange(512, dtype=float))
        benchmark(
            imprint, np.zeros((512, 512)), [50, 400, 50, 400],
            phase.blaze, grid=bench_grid, vector=vector,
        )

    with subtests.test("replace fills exactly the window and nothing else"):
        mat = np.zeros((H, W))
        assert imprint(mat, win, 7.0) is mat
        np.testing.assert_array_equal(mat[sl], 7.0)
        mat[sl] = 0
        np.testing.assert_array_equal(mat, 0)

    with subtests.test("a callable is evaluated on the windowed sub-grid"):
        mat = np.full((H, W), 99.0)
        imprint(mat, win, phase.blaze, grid=grid, vector=vector)
        np.testing.assert_allclose(mat[sl], phase.blaze(sub, vector))
        mat[sl] = 99.0
        np.testing.assert_array_equal(mat, 99.0)

    with subtests.test("add accumulates onto the existing values"):
        mat = np.ones((H, W))
        imprint(mat, win, 3.0, imprint_operation="add")
        np.testing.assert_array_equal(mat[sl], 4.0)
        mat = np.ones((H, W))
        imprint(mat, win, phase.blaze, grid=grid, vector=vector, imprint_operation="add")
        np.testing.assert_allclose(mat[sl], 1.0 + phase.blaze(sub, vector))

    with subtests.test("transform and shift act on the cropped sub-grid alone"):
        mat = np.zeros((H, W))
        imprint(mat, win, phase.blaze, grid=grid, vector=vector, transform=np.pi / 4, shift=True)
        np.testing.assert_allclose(
            mat[sl], phase.blaze(transform_grid(sub, np.pi / 4, True), vector)
        )

    with subtests.test("an SLM stands in for its own grid"):
        mat = np.zeros(slm.shape)
        imprint(mat, win, phase.blaze, grid=slm, vector=vector)
        np.testing.assert_allclose(
            mat[sl], phase.blaze((slm.grid[0][sl], slm.grid[1][sl]), vector)
        )

    with subtests.test("centered puts (x, y) at the middle of the window"):
        mat = np.zeros((H, W))
        imprint(mat, [20, 6, 10, 4], 1.0, centered=True)
        np.testing.assert_array_equal(mat[8:12, 17:23], 1.0)
        assert np.sum(mat) == 6 * 4

    with subtests.test("a boolean mask imprints exactly its True pixels"):
        mat = np.zeros((H, W))
        mask = np.zeros((H, W), dtype=bool)
        mask[0, 0] = mask[H - 1, W - 1] = True
        imprint(mat, mask, 42.0)
        np.testing.assert_array_equal(mat[mask], 42.0)
        assert np.sum(mat) == 2 * 42.0

    with subtests.test("a (y_ind, x_ind) pair imprints exactly those pixels"):
        mat = np.zeros((H, W))
        imprint(mat, (np.array([0, 1, 2]), np.array([5, 5, 5])), 10.0)
        np.testing.assert_array_equal(mat[0:3, 5], 10.0)
        assert np.sum(mat) == 3 * 10.0

    with subtests.test("clip keeps only the in-bounds corner of the window"):
        mat = np.zeros((H, W))
        imprint(mat, [W - 5, 20, H - 5, 20], 1.0, clip=True)
        np.testing.assert_array_equal(mat[H - 5:, W - 5:], 1.0)
        assert np.sum(mat) == 5 * 5

    with subtests.test("clip=False rejects a window that leaves the matrix"):
        for kwargs in ({}, {"circular": True}):
            mat = np.zeros((H, W))
            with pytest.raises(ValueError, match="extends past"):
                imprint(mat, [1, 10, 1, 10], 1.0, centered=True, clip=False, **kwargs)
            np.testing.assert_array_equal(mat, 0)

    with subtests.test("a matrix beyond 2D is rejected, not silently skipped"):
        # The window slices the leading axes, so a stack would otherwise imprint nothing.
        for clip in (True, False):
            mat = np.zeros((3, H, W))
            with pytest.raises(ValueError):
                imprint(mat, [2, 5, 3, 4], 1.0, clip=clip)
            np.testing.assert_array_equal(mat, 0)

    with subtests.test("an unusable operation or missing grid raises"):
        with pytest.raises(ValueError, match="Unrecognized"):
            imprint(np.zeros((H, W)), win, 1.0, imprint_operation="multiply")
        with pytest.raises(ValueError, match="grid cannot be None"):
            imprint(np.zeros((H, W)), win, phase.blaze, grid=None)


def test_format_vectors(subtests):
    """Test format_vectors' cleaning of vectors into (M, N) columns."""
    with subtests.test("a lone 2-vector becomes a (2, 1) column"):
        for inp in [(1, 2), [1, 2], np.array([1, 2]), np.array([[1, 2]])]:
            np.testing.assert_array_equal(format_vectors(inp), [[1], [2]])

    with subtests.test("an (M, N) array passes through untouched"):
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(format_vectors(arr), arr)
        arr3 = np.array([[1, 2], [3, 4], [5, 6]])
        np.testing.assert_array_equal(format_vectors(arr3, expected_dimension=3), arr3)

    with subtests.test("handle_dimension decides the fate of a surplus dimension"):
        vec3 = np.array([[1], [2], [3]])
        np.testing.assert_array_equal(format_vectors(vec3, 2, "crop"), [[1], [2]])
        np.testing.assert_array_equal(format_vectors(vec3, 2, "pass"), vec3)
        with pytest.raises(ValueError, match="Expected 2-vectors"):
            format_vectors(vec3, 2, "error")

    with subtests.test("malformed input raises"):
        with pytest.raises(ValueError):
            format_vectors(np.array([[1, 2]]), expected_dimension=3)
        with pytest.raises(ValueError, match="not recognized"):
            format_vectors(np.array([1, 2]), handle_dimension="bad")
        with pytest.raises((ValueError, TypeError)):
            format_vectors(5)


def test_format_2vectors(subtests):
    """Test format_2vectors, the two-dimensional wrapper of format_vectors."""
    with subtests.test("a 2-vector becomes a (2, 1) column"):
        np.testing.assert_array_equal(format_2vectors((5, 10)), [[5], [10]])

    with subtests.test("a surplus third dimension is cropped away"):
        np.testing.assert_array_equal(format_2vectors(np.array([[1], [2], [3]])), [[1], [2]])


def test_fit_3pt(subtests):
    """Test fit_3pt's affine fit through three points."""
    cases = {
        "the identity": ((0, 0), (1, 0), (0, 1), np.eye(2), [[0], [0]]),
        "a translation": ((10, 20), (11, 20), (10, 21), np.eye(2), [[10], [20]]),
        "a doubling": ((0, 0), (2, 0), (0, 2), 2 * np.eye(2), [[0], [0]]),
        "a quarter turn": ((0, 0), (0, 1), (-1, 0), [[0, -1], [1, 0]], [[0], [0]]),
    }
    for name, (y0, y1, y2, M, b) in cases.items():
        with subtests.test(f"{name} is recovered exactly"):
            affine = fit_3pt(y0, y1, y2, N=None)
            np.testing.assert_allclose(affine["M"], M, atol=1e-14)
            np.testing.assert_allclose(affine["b"], b, atol=1e-14)

    with subtests.test("the fit maps each index back onto its point"):
        points = [((0, 0), (3, 7)), ((1, 0), (5, 8)), ((0, 1), (4, 10))]
        affine = fit_3pt(*[y for (_, y) in points], N=None)
        for x, y in points:
            np.testing.assert_allclose(
                affine["M"] @ format_2vectors(x) + affine["b"], format_2vectors(y), atol=1e-14
            )

    with subtests.test("non-unit indices rescale the basis vectors"):
        affine = fit_3pt((0, 0), (4, 0), (0, 6), N=None, x0=(0, 0), x1=(2, 0), x2=(0, 3))
        np.testing.assert_allclose(affine["M"], 2 * np.eye(2), atol=1e-14)

    with subtests.test("x1=None reads y1 and y2 as basis vectors, not positions"):
        (origin, dv1, dv2) = (np.array([10, 20]), np.array([1, 0]), np.array([0, 1]))
        positions = fit_3pt(origin, origin + dv1, origin + dv2, N=None)
        differences = fit_3pt(origin, dv1, dv2, N=None, x1=None, x2=None)
        np.testing.assert_allclose(positions["M"], differences["M"], atol=1e-14)
        np.testing.assert_allclose(positions["b"], differences["b"], atol=1e-14)

    with subtests.test("a positive N evaluates the fit on that lattice of indices"):
        grid = fit_3pt((0, 0), (1, 0), (0, 1), N=(3, 3))
        assert grid.shape == (2, 9)
        assert set(map(tuple, grid.T)) == {(i, j) for i in range(3) for j in range(3)}
        assert fit_3pt((0, 0), (1, 0), (0, 1), N=4).shape == (2, 16)

    with subtests.test("an ndarray N supplies the indices directly"):
        indices = np.array([[0, 1, 2], [0, 0, 0]])
        np.testing.assert_allclose(
            fit_3pt((5, 10), (6, 10), (5, 11), N=indices), indices + [[5], [10]], atol=1e-14
        )

    with subtests.test("a non-positive N returns the affine instead of a lattice"):
        for n in (0, -1, None):
            assert set(fit_3pt((0, 0), (1, 0), (0, 1), N=n)) == {"M", "b"}

    with subtests.test("orientation_check drops the last two lattice points"):
        full = fit_3pt((0, 0), (1, 0), (0, 1), N=(3, 3))
        trimmed = fit_3pt((0, 0), (1, 0), (0, 1), N=(3, 3), orientation_check=True)
        np.testing.assert_allclose(trimmed, full[:, :-2])

    with subtests.test("colinear indices raise"):
        with pytest.raises(ValueError, match="colinear"):
            fit_3pt((0, 0), (1, 0), (2, 0), x0=(0, 0), x1=(1, 0), x2=(2, 0))


def test_smallest_distance(subtests):
    """Test smallest_distance's closest-pair search."""
    pair = np.array([[0, 3], [0, 4]])
    cases = {
        "a lone point has no pair": (np.array([[5], [3]]), "chebyshev", np.inf),
        "an empty set has no pair": (np.empty((2, 0)), "chebyshev", np.inf),
        "chebyshev is the largest coordinate difference": (pair, "chebyshev", 4.0),
        "euclidean is the straight-line distance": (pair, "euclidean", 5.0),
        "cityblock sums the coordinate differences": (pair, "cityblock", 7.0),
        "the closest pair wins, not the first": (
            np.array([[0, 10, 11, 50], [0, 10, 11, 50]]), "chebyshev", 1.0,
        ),
        "duplicated points are zero apart": (np.array([[1, 2, 1], [3, 4, 3]]), "chebyshev", 0.0),
        "negative coordinates are signed": (np.array([[-5, -3], [10, 10]]), "chebyshev", 2.0),
        "evenly spaced collinear points give the spacing": (
            np.array([[0, 2, 4, 6, 8], [0, 0, 0, 0, 0]]), "chebyshev", 2.0,
        ),
        "unevenly spaced collinear points give the tightest gap": (
            np.array([[0, 1, 5, 20], [0, 0, 0, 0]]), "chebyshev", 1.0,
        ),
    }
    for name, (vectors, metric, expected) in cases.items():
        with subtests.test(name):
            assert smallest_distance(vectors, metric=metric) == pytest.approx(expected)

    with subtests.test("the brute-force callable path agrees with the string path"):
        vectors = np.random.default_rng(7).uniform(0, 100, size=(2, 50))
        assert smallest_distance(
            vectors, metric=lambda a, b: np.sqrt(np.sum((a - b) ** 2))
        ) == pytest.approx(smallest_distance(vectors, metric="euclidean"), rel=1e-10)

    with subtests.test("divide and conquer matches brute force across random layouts"):
        # A merge fault only shows for layouts whose closest pair straddles the split, so
        # the trials are randomized rather than seeded once.
        metrics = ["euclidean", "chebyshev", "cityblock"]
        for trial in range(200):
            rng = np.random.default_rng(trial)
            n = int(rng.integers(400, 800))     # >= 2*min_div, so divide and conquer runs
            metric = metrics[trial % len(metrics)]
            layout = trial % 4
            if layout == 0:
                vectors = rng.uniform(0, 5000, size=(2, n))
            elif layout == 1:
                vectors = rng.uniform(0, 30, size=(2, n))
            elif layout == 2:
                centers = rng.uniform(0, 1000, size=(int(rng.integers(2, 6)), 2))
                vectors = (centers[rng.integers(0, len(centers), n)]
                           + rng.normal(0, 0.5, size=(n, 2))).T
            else:
                vectors = np.vstack((np.sort(rng.uniform(0, 1000, n)), rng.uniform(0, 5, n)))
            expected = distance.pdist(vectors.T, metric=metric).min()
            assert smallest_distance(vectors, metric=metric) == pytest.approx(
                expected, rel=1e-9, abs=1e-9
            ), f"trial {trial}: n={n}, metric={metric}, layout={layout}"

    with subtests.test("a closest pair straddling the split line is still found"):
        for trial in range(80):
            rng = np.random.default_rng(5000 + trial)
            n = 500
            xs = np.linspace(0, 1000, n) + rng.uniform(-0.1, 0.1, n)
            ys = np.linspace(0, 1000, n)[rng.permutation(n)]
            mid = n // 2
            xmid = 0.5 * (xs[mid] + xs[mid + 1])
            (xs[mid], xs[mid + 1]) = (xmid - 5e-5, xmid + 5e-5)
            ys[mid] = ys[mid + 1] = 500.0
            vectors = np.vstack((xs, ys))
            expected = distance.pdist(vectors.T, metric="euclidean").min()
            assert smallest_distance(vectors, metric="euclidean") == pytest.approx(
                expected, rel=1e-9, abs=1e-12
            ), f"trial {trial}"


def test_lloyds_algorithm(subtests):
    """Test lloyds_algorithm's relaxation of seeds toward even spacing."""
    shape = (100, 100)
    grid = np.meshgrid(range(shape[1]), range(shape[0]))

    with subtests.test("zero iterations returns the seeds untouched"):
        seeds = np.array([[20, 50, 80], [20, 50, 80]])
        np.testing.assert_allclose(lloyds_algorithm(grid, seeds, iterations=0), seeds)

    with subtests.test("two seeds relax onto the centroids of the two halves"):
        result = lloyds_algorithm(grid, np.array([[10, 90], [50, 50]]), iterations=50)
        np.testing.assert_allclose(np.sort(result[0]), [25, 75], atol=1e-6)
        np.testing.assert_allclose(result[1], [50, 50], atol=1e-6)

    with subtests.test("four seeds relax onto the centroids of the quadrants"):
        seeds = np.array([[20, 60, 30, 70], [30, 20, 70, 80]], dtype=float)
        result = lloyds_algorithm(grid, seeds, iterations=50)
        np.testing.assert_allclose(np.sort(result[0]), [25, 25, 75, 75], atol=1e-6)
        np.testing.assert_allclose(np.sort(result[1]), [25, 25, 75, 75], atol=1e-6)

    with subtests.test("points stay inside the grid, however the grid is given"):
        rng = np.random.default_rng(42)
        rect = (50, 200)
        for space, (h, w) in [
            (grid, shape),
            (shape, shape),
            (np.meshgrid(range(rect[1]), range(rect[0])), rect),
        ]:
            seeds = np.vstack((rng.uniform(5, w - 5, 12), rng.uniform(5, h - 5, 12)))
            result = lloyds_algorithm(space, seeds, iterations=20)
            assert result.shape == (2, 12)
            assert np.all(result[0] >= 0) and np.all(result[0] <= w)
            assert np.all(result[1] >= 0) and np.all(result[1] <= h)

    with subtests.test("the relaxation is deterministic"):
        seeds = np.array([[10, 30, 70, 90], [50, 50, 50, 50]])
        np.testing.assert_allclose(
            lloyds_algorithm(grid, seeds, iterations=10),
            lloyds_algorithm(grid, seeds, iterations=10),
        )


def test_lloyds_points(subtests):
    """Test lloyds_points, which seeds lloyds_algorithm at random."""
    shape = (100, 100)

    with subtests.test("n_points distinct points come back, from either form of grid"):
        np.random.seed(42)
        for space in (shape, np.meshgrid(range(shape[1]), range(shape[0]))):
            result = lloyds_points(space, 7, iterations=5)
            assert result.shape == (2, 7)
            assert smallest_distance(result) > 0

    with subtests.test("a lone point relaxes onto the center of the grid"):
        np.random.seed(22)
        result = lloyds_points(shape, 1, iterations=50)
        np.testing.assert_allclose(result.ravel(), [50, 50], atol=1)


def test_assign_vectors(subtests):
    """Test assign_vectors' nearest-option assignment."""
    diagonal = np.array([[0, 10, 20], [0, 10, 20]])
    cases = {
        "exact matches map onto themselves": (diagonal, diagonal, [0, 1, 2]),
        "each vector takes its nearest option": (np.array([[1, 11], [1, 11]]), diagonal, [0, 1]),
        "a distant cluster still takes the nearest": (
            np.array([[1, 2, 3], [1, 2, 3]]), np.array([[0, 100], [0, 100]]), [0, 0, 0],
        ),
        "a tie goes to the lower index": (np.array([[0], [0]]), np.array([[-1, 1], [0, 0]]), [0]),
        "options may be reused and outnumbered": (
            np.array([[5, 15, 25, 35], [5, 15, 25, 35]]), diagonal, [0, 1, 2, 2],
        ),
    }
    for name, (vectors, options, expected) in cases.items():
        with subtests.test(name):
            np.testing.assert_array_equal(assign_vectors(vectors, options), expected)


def test_format_shape(subtests):
    """Test format_shape's validation of a shape tuple."""
    with subtests.test("array-likes become an (h, w) tuple"):
        for inp in [(10, 20), [10, 20], np.array([10, 20])]:
            assert format_shape(inp) == (10, 20)

    with subtests.test("expected_dimension=None accepts any rank"):
        assert format_shape((2, 3, 4), expected_dimension=None) == (2, 3, 4)

    with subtests.test("the wrong number of dimensions raises"):
        with pytest.raises(ValueError, match="dimensions"):
            format_shape((1, 2, 3), expected_dimension=2)

    with subtests.test("a dimension that is not a positive integer raises"):
        for inp in [(0, 5), (5, -1), (1.5, 2.5)]:
            with pytest.raises(ValueError, match="positive integer"):
                format_shape(inp)


def test_pad(subtests):
    """Test pad's centered zero-padding."""
    mat = np.arange(12).reshape(3, 4)

    with subtests.test("the data lands centered in a field of zeros"):
        expected = np.zeros((7, 10))
        expected[2:5, 3:7] = mat
        np.testing.assert_array_equal(pad(mat, (7, 10)), expected)

    with subtests.test("an odd padding puts the extra row and column last"):
        expected = np.zeros((3, 4))
        expected[0:2, 0:3] = 1
        np.testing.assert_array_equal(pad(np.ones((2, 3)), (3, 4)), expected)

    with subtests.test("None or the same shape returns the matrix"):
        np.testing.assert_array_equal(pad(mat, None), mat)
        np.testing.assert_array_equal(pad(mat, mat.shape), mat)

    with subtests.test("padding to a smaller shape raises"):
        with pytest.raises(ValueError, match="too large"):
            pad(mat, (2, 2))


def test_unpad(subtests):
    """Test unpad's centered crop, the inverse of pad."""
    mat = np.arange(12).reshape(3, 4)

    with subtests.test("unpadding recovers exactly what pad wrapped"):
        for target in [(8, 12), (9, 11), (5, 10), (3, 4)]:
            np.testing.assert_array_equal(unpad(pad(mat, target), mat.shape), mat)

    with subtests.test("a shape argument returns the slicing indices instead"):
        assert unpad((7, 10), (3, 4)) == (2, 5, 3, 7)
        assert unpad((7, 10), None) == (0, 7, 0, 10)

    with subtests.test("None returns the matrix"):
        np.testing.assert_array_equal(unpad(mat, None), mat)

    with subtests.test("unpadding to a larger shape raises"):
        with pytest.raises(ValueError, match="too small"):
            unpad(mat, (10, 10))


def test_window_slice(subtests):
    """Test window_slice's parsing of the several window formats."""
    cases = {
        "None is the whole array": (
            (None, {}), (slice(None), slice(None)),
        ),
        "(x, w, y, h) is an upper-left corner plus an extent": (
            ([10, 20, 5, 15], {}), (slice(5, 20), slice(10, 30)),
        ),
        "a unit window is a single pixel": (
            ([7, 1, 4, 1], {}), (slice(4, 5), slice(7, 8)),
        ),
        "centered puts (x, y) at the middle of the window": (
            ([10, 20, 5, 15], {"centered": True}), (slice(-2, 13), slice(0, 20)),
        ),
        "shape clips the far edge": (
            ([0, 20, 0, 20], {"shape": (10, 10)}), (slice(0, 10), slice(0, 10)),
        ),
        "shape clips a negative start up to zero": (
            ([-5, 10, -5, 10], {"shape": (20, 20)}), (slice(0, 5), slice(0, 5)),
        ),
    }
    for name, ((window, kwargs), expected) in cases.items():
        with subtests.test(name):
            assert window_slice(window, **kwargs) == expected

    with subtests.test("a (y_ind, x_ind) pair indexes exactly those pixels"):
        (y_ind, x_ind) = (np.array([1, 2, 3]), np.array([5, 5, 5]))
        mat = np.zeros((10, 10))
        mat[window_slice((y_ind, x_ind))] = 1
        np.testing.assert_array_equal(np.nonzero(mat), (y_ind, x_ind))

    with subtests.test("a boolean mask is its own window"):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 3] = True
        assert window_slice(mask) is mask

    with subtests.test("circular inscribes an ellipse, dropping the window's corners"):
        mat = np.zeros((10, 10))
        mat[window_slice([0, 5, 0, 5], shape=(10, 10), circular=True)] = 1
        expected = np.zeros((10, 10))
        expected[0:5, 0:5] = 1
        expected[[0, 0, 4, 4], [0, 4, 0, 4]] = 0
        np.testing.assert_array_equal(mat, expected)


def test_window_extent(subtests):
    """Test window_extent's bounding rectangle around a window."""
    rect = np.zeros((10, 10), dtype=bool)
    rect[2:5, 3:7] = True
    pixel = np.zeros((10, 10), dtype=bool)
    pixel[5, 7] = True
    ell = np.zeros((10, 10), dtype=bool)
    ell[1:5, 2:4] = True
    ell[3:6, 2:7] = True
    square = np.zeros((20, 20), dtype=bool)
    square[5:10, 5:10] = True

    cases = {
        "a rectangle is its own extent": ((rect, {}), (3, 4, 2, 3)),
        "a single pixel has unit extent": ((pixel, {}), (7, 1, 5, 1)),
        "a full mask spans the whole array": ((np.ones((8, 6), dtype=bool), {}), (0, 6, 0, 8)),
        "an L-shape gives its bounding box": ((ell, {}), (2, 5, 1, 5)),
        "an (x, w, y, h) window comes back unchanged": (((3, 4, 2, 5), {}), (3, 4, 2, 5)),
        "padding_frac grows the extent proportionally": (
            (square, {"padding_frac": 0.5}), (4, 7, 4, 7),
        ),
        "padding_pix grows the extent by whole pixels": (
            (square, {"padding_pix": 3}), (2, 11, 2, 11),
        ),
    }
    for name, ((window, kwargs), expected) in cases.items():
        with subtests.test(name):
            assert window_extent(window, **kwargs) == expected

    with subtests.test("the extent of a rectangular mask slices back to that mask"):
        mask = np.zeros((12, 15), dtype=bool)
        mask[1:4, 2:8] = True
        recovered = np.zeros_like(mask)
        recovered[window_slice(window_extent(mask))] = True
        np.testing.assert_array_equal(recovered, mask)


def test_transform_grid(subtests):
    """Test transform_grid's affine transformation of a coordinate basis."""
    axis = np.linspace(0.0, 1.0, 5)
    (x_grid, y_grid) = np.meshgrid(axis, axis)
    grid = (x_grid, y_grid)

    with subtests.test("no transform and no shift returns a copy of the grid"):
        (x_out, y_out) = transform_grid(grid)
        np.testing.assert_allclose(x_out, x_grid)
        np.testing.assert_allclose(y_out, y_grid)
        assert x_out is not x_grid and y_out is not y_grid

    with subtests.test("shift adds its offset"):
        (x_out, y_out) = transform_grid(grid, shift=(0.5, -0.3))
        np.testing.assert_allclose(x_out, x_grid + 0.5)
        np.testing.assert_allclose(y_out, y_grid - 0.3)

    with subtests.test("shift=True centers the grid on zero"):
        (x_out, y_out) = transform_grid(grid, shift=True)
        np.testing.assert_allclose(np.mean(x_out), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.mean(y_out), 0.0, atol=1e-14)

    with subtests.test("a scalar transform rotates counterclockwise"):
        for angle, (x_expected, y_expected) in [
            (np.pi / 2, (-y_grid, x_grid)),
            (np.pi, (-x_grid, -y_grid)),
        ]:
            (x_out, y_out) = transform_grid(grid, transform=angle)
            np.testing.assert_allclose(x_out, x_expected, atol=1e-13)
            np.testing.assert_allclose(y_out, y_expected, atol=1e-13)

    with subtests.test("a matrix transform is applied row by row"):
        (x_out, y_out) = transform_grid(grid, transform=np.array([[2.0, 0.0], [0.0, 3.0]]))
        np.testing.assert_allclose(x_out, 2.0 * x_grid)
        np.testing.assert_allclose(y_out, 3.0 * y_grid)

    with subtests.test("fwd rotates before it shifts"):
        (x_out, y_out) = transform_grid(grid, transform=np.pi / 2, shift=(1.0, 0.0))
        np.testing.assert_allclose(x_out, -y_grid + 1.0, atol=1e-14)
        np.testing.assert_allclose(y_out, x_grid, atol=1e-14)

    with subtests.test("rev undoes fwd"):
        (angle, shift) = (np.pi / 3, (0.2, -0.3))
        forward = transform_grid(grid, transform=angle, shift=shift)
        (x_out, y_out) = transform_grid(forward, transform=angle, shift=shift, direction="rev")
        np.testing.assert_allclose(x_out, x_grid, atol=1e-13)
        np.testing.assert_allclose(y_out, y_grid, atol=1e-13)


def test_voronoi_windows(subtests):
    """Test voronoi_windows' partition of a grid into cells around each vector."""
    shape = (40, 40)
    vectors = np.array([[10, 30, 10, 30], [10, 10, 30, 30]])
    windows = voronoi_windows(shape, vectors)

    with subtests.test("the windows tile the grid, one boolean cell per vector"):
        assert len(windows) == len(vectors.T)
        counts = np.zeros(shape, dtype=int)
        for window in windows:
            assert window.shape == shape and window.dtype == bool
            counts += window
        np.testing.assert_array_equal(counts, 1)

    with subtests.test("a cell holds the pixels nearest its own vector"):
        (x_grid, y_grid) = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        radii = np.stack([np.hypot(x_grid - x, y_grid - y) for (x, y) in vectors.T])
        nearest = np.argmin(radii, axis=0)
        radii.sort(axis=0)
        # cv2's polygon fill dilates a cell slightly, so the shared borders are excluded.
        interior = (radii[1] - radii[0]) > 2
        for i, window in enumerate(windows):
            assert np.all(window[interior & (nearest == i)]), i

    with subtests.test("radius crops each cell without evicting its vector"):
        cropped = voronoi_windows(shape, vectors, radius=5)
        for i, (window, crop) in enumerate(zip(windows, cropped)):
            assert np.sum(crop) <= np.sum(window)
            assert crop[vectors[1, i], vectors[0, i]]

    with subtests.test("a lone vector owns every pixel"):
        lone = voronoi_windows(shape, np.array([[20], [20]]))
        assert len(lone) == 1 and np.all(lone[0])


class TestAperture:
    """Tests for the Aperture class."""

    def test_crops(self, subtests):
        """Test Aperture.crops, the cheap precursor of Aperture.mask."""
        axis = np.linspace(-1.0, 1.0, 32)
        grid = np.meshgrid(axis, axis)

        with subtests.test("a centered 'cropped' aperture masks nothing"):
            aperture = Aperture(grid, "cropped")
            assert not aperture.crops
            assert np.all(aperture.mask)

        with subtests.test("an off-center 'cropped' aperture does crop"):
            aperture = Aperture(grid, "cropped", center=(0.3, 0.2))
            assert aperture.crops
            assert not np.all(aperture.mask)

        with subtests.test("crops is never False while the mask excludes pixels"):
            for spec in ("cropped", "circular", 2.0):
                for center in (None, (0.0, 0.0), (0.3, 0.0)):
                    aperture = Aperture(grid, spec, center=center)
                    if not np.all(aperture.mask):
                        assert aperture.crops, (spec, center)
