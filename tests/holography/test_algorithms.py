"""
Unit tests for slmsuite.holography.algorithms module.
"""
import pytest
import numpy as np
from types import SimpleNamespace
from scipy import ndimage

import slmsuite._plotting
from slmsuite.holography import toolbox
from slmsuite.holography.algorithms import (
    Hologram, SpotHologram, CompressedSpotHologram, MultiplaneHologram, FeedbackHologram
)
from slmsuite.holography.algorithms._feedback import (
    _IJCAM_TO_KNMSLM_CACHE, clear_ijcam_to_knmslm_cache
)
from slmsuite.holography.algorithms._header import cp_affine_transform, cp as xp
from slmsuite.holography.analysis import Affine
from slmsuite.holography.analysis.files import load_h5

from conftest import seed_for, install_ground_truth_calibration, spot_size_ij, view_kxy_grid

try:
    import cupy as cp
except ImportError:
    cp = None


def _np(array):
    """Return numpy array regardless of whether input is numpy or cupy."""
    return array.get() if hasattr(array, "get") else array


def _spot_target(shape, *spots):
    """A ``shape`` target with unit amplitude at each ``(row, col)`` in ``spots``."""
    target = np.zeros(shape, dtype=np.float32)
    for spot in spots:
        target[spot] = 1.0
    return target


@pytest.fixture
def keep_figures(mpl_test, monkeypatch):
    """``plt``, with slmsuite's show handler deferred to ``flush()`` so a plot can be read."""
    handler = slmsuite._plotting._current_handler
    monkeypatch.setattr("slmsuite._plotting._current_handler", lambda *args, **kwargs: None)

    def flush(name=None):
        if handler is not None:
            handler(name=name)
        mpl_test.close("all")

    monkeypatch.setattr(mpl_test, "flush", flush, raising=False)
    return mpl_test


def _multiplane(shape=(64, 64), slm_shape=None, n=2, amp=None):
    """A :class:`MultiplaneHologram` whose ``n`` children each target one distinct spot."""
    slm_shape = shape if slm_shape is None else slm_shape
    if amp is None:
        amp = np.ones(slm_shape, dtype=np.float32) / np.sqrt(np.prod(slm_shape))
    return MultiplaneHologram([
        Hologram(target=_spot_target(shape, spot), amp=amp, slm_shape=slm_shape)
        for spot in [(12, 12), (20, 44), (44, 20)][:n]
    ])


class TestHologram:

    def test_init(self, subtests):
        with subtests.test("dtype fixes the matching complex precision"):
            h = Hologram(target=np.zeros((64, 64), dtype=np.float32))
            assert h.dtype == np.float32 and h.dtype_complex == np.complex64
            h = Hologram(target=np.zeros((64, 64)), dtype=np.float64)
            assert h.dtype == np.float64 and h.dtype_complex == np.complex128

        with subtests.test("accepts every spelling of a float dtype"):
            for spelling in (np.float32, "float32", np.dtype("float32")):
                assert Hologram((16, 16), dtype=spelling).dtype == np.float32
            assert Hologram((16, 16), dtype=float).dtype == np.float64

        with subtests.test("rejects non-float dtypes"):
            for spelling in (np.complex64, np.int32):
                with pytest.raises(ValueError):
                    Hologram((16, 16), dtype=spelling)

        with subtests.test("slm_shape defaults to the computational shape"):
            h = Hologram(target=np.zeros((64, 128)))
            assert h.slm_shape == (64, 128) and h.shape == (64, 128)

        with subtests.test("slm_shape can differ from the computational shape"):
            h = Hologram(target=np.zeros((64, 64)), slm_shape=(32, 32))
            assert h.slm_shape == (32, 32) and h.shape == (64, 64)

        with subtests.test("a uniform nearfield costs a scalar, not an slm_shape array"):
            for amp in (None, 1.0):
                h = Hologram((64, 48), amp=amp)
                assert np.ndim(h.amp) == 0
                assert h.amp.dtype == h.dtype
                assert h.amp == pytest.approx(1 / np.sqrt(64 * 48))

        with subtests.test("a hologram's own amp is a valid amp"):
            h = Hologram((64, 64))
            assert Hologram((64, 64), amp=h.amp).amp == h.amp

        with subtests.test("an array amp is kept and normalized"):
            h = Hologram((64, 48), amp=np.ones((64, 48)))
            assert h.amp.shape == (64, 48)
            assert h.amp.dtype == h.dtype

        with subtests.test("an amp that does not match slm_shape raises"):
            with pytest.raises(ValueError):
                Hologram(
                    target=np.zeros((64, 64)), phase=np.zeros((64, 64)), amp=np.ones((32, 32))
                )

    def test_reset(self, subtests):
        with subtests.test("clears iter and stats"):
            h = Hologram(target=np.zeros((64, 64)))
            h.optimize(method="GS", maxiter=5, verbose=False)
            h.reset()
            assert h.iter == 0
            assert h.stats == {"method": [], "flags": {}, "stats": {}}

        h = Hologram(target=np.zeros((64, 64), dtype=np.float32))
        (nearfield, farfield) = (h.nearfield, h.farfield)

        with subtests.test("buffers of the right shape are reused, not reallocated"):
            h.reset()
            assert h.nearfield is nearfield
            assert h.farfield is farfield

        with subtests.test("reused buffers are zeroed"):
            h.nearfield.fill(1)
            h.reset()
            assert not np.any(_np(h.nearfield))

        with subtests.test("a shape change reallocates"):
            h.target = cp.zeros((32, 32), dtype=h.dtype) if cp is not None \
                else np.zeros((32, 32), dtype=h.dtype)
            h.reset()
            assert h.farfield is not farfield
            assert h.farfield.shape == (32, 32)

    def test_reset_phase(self, subtests):
        h = Hologram(target=np.zeros((64, 64)))

        with subtests.test("phase lies in [0, 2 pi]"):
            phase = h.get_phase()
            assert phase.min() >= 0.0
            assert phase.max() <= 2 * np.pi + 1e-5

        with subtests.test("redrawing gives a different phase"):
            before = h.get_phase().copy()
            h.reset_phase()
            assert not np.allclose(before, h.get_phase())

    def test_reset_weights(self):
        """Weights copy the target, with MRAF's NaN noise region zeroed."""
        target = np.ones((32, 32), dtype=np.float32)
        target[4:8, 4:8] = np.nan
        h = Hologram(target=np.nan_to_num(target, nan=0.0))
        h.target = cp.asarray(target) if cp is not None else target

        h.reset_weights()
        assert np.allclose(_np(h.weights), np.nan_to_num(target, nan=0.0))

    def test_get_padded_shape(self, subtests):
        with subtests.test("padding_order raises each dimension to that power of two"):
            # The worked example from the docstring.
            for (order, expected) in ((0, (720, 1280)), (1, (1024, 2048)), (2, (2048, 4096))):
                assert Hologram.get_padded_shape(
                    (720, 1280), padding_order=order, square_padding=False
                ) == expected

        with subtests.test("a shape already a power of two is left alone"):
            assert Hologram.get_padded_shape(
                (128, 256), padding_order=1, square_padding=False
            ) == (128, 256)

        with subtests.test("square padding takes the larger dimension"):
            assert Hologram.get_padded_shape((128, 256)) == (256, 256)

    def test_set_target(self, subtests):
        with subtests.test("the constructor L2 normalizes"):
            h = Hologram(target=np.random.rand(64, 64).astype(np.float32) + 0.1)
            assert float(np.sum(_np(h.target) ** 2)) == pytest.approx(1.0, rel=1e-4)

        with subtests.test("set_target L2 normalizes"):
            h = Hologram(target=np.zeros((64, 64)))
            h.set_target(np.ones((64, 64)) * 5.0)
            assert float(np.sum(_np(h.target) ** 2)) == pytest.approx(1.0, rel=1e-4)

    def test_get_phase(self):
        """Phase comes back at ``slm_shape``, not the padded computational shape."""
        h = Hologram(target=np.zeros((64, 64)), slm_shape=(32, 32))
        assert h.get_phase().shape == (32, 32)

    def test_get_farfield(self):
        """The farfield carries the hologram's complex precision."""
        assert Hologram((64, 64)).get_farfield().dtype == np.complex64

    def test_unpad_slice(self, subtests):
        """Memoized on the shapes, because subclasses adjust them after ``__init__``."""
        h = Hologram(target=np.zeros((64, 64)), slm_shape=(32, 32))

        with subtests.test("matches toolbox.unpad"):
            assert h._unpad_slice == toolbox.unpad(h.shape, h.slm_shape)

        with subtests.test("repeated access is cached"):
            assert h._unpad_slice is h._unpad_slice

        with subtests.test("a shape change invalidates the cache"):
            h.slm_shape = (64, 64)
            assert h._unpad_slice == (0, 64, 0, 64)

    def test_optimize(self, subtests):
        with subtests.test("iter accumulates across calls"):
            h = Hologram(target=np.zeros((64, 64)))
            h.optimize(method="GS", maxiter=5, verbose=False)
            h.optimize(method="GS", maxiter=5, verbose=False)
            assert h.iter == 10

        with subtests.test("an unrecognized method raises"):
            with pytest.raises(ValueError, match="Unrecognized method"):
                Hologram(target=np.zeros((64, 64))).optimize(
                    method="INVALID", maxiter=1, verbose=False
                )

        with subtests.test("an unrecognized stat group raises"):
            with pytest.raises(ValueError):
                Hologram(target=np.zeros((64, 64))).optimize(
                    method="GS", maxiter=1, verbose=False, stat_groups=["INVALID_GROUP"]
                )

        with subtests.test("an MRAF target leaves no NaN in the phase"):
            target = np.full((64, 64), np.nan, dtype=np.float32)
            target[20, 20] = target[40, 40] = 1.0
            h = Hologram(target=target)
            h.optimize(method="GS", maxiter=10, verbose=False)
            assert not np.any(np.isnan(h.get_phase()))

    def test_optimize_gs(self, subtests):
        with subtests.test("a single spot reaches better than 90% efficiency"):
            h = Hologram(target=_spot_target((64, 64), (16, 48)))
            h.optimize(method="GS", maxiter=40, verbose=False, stat_groups=["computational"])
            eff = h.stats["stats"]["computational"]["efficiency"][-1]
            assert eff > 0.9, f"single-spot GS efficiency {eff:.4f}"

        with subtests.test("the farfield peak lands on the target pixel"):
            spot = (20, 44)
            h = Hologram(target=_spot_target((64, 64), spot))
            h.optimize(method="GS", maxiter=40, verbose=False)
            ff = np.abs(h.get_farfield())
            assert np.unravel_index(np.argmax(ff), ff.shape) == spot

        with subtests.test("efficiency improves over iterations"):
            h = Hologram(
                target=_spot_target((64, 64), (13, 17), (30, 44), (50, 10), (10, 50))
            )
            h.optimize(method="GS", maxiter=20, verbose=False, stat_groups=["computational"])
            effs = h.stats["stats"]["computational"]["efficiency"]
            assert effs[-1] > effs[0]

    @pytest.mark.parametrize("method", ["WGS-Leonardo", "WGS-Kim", "WGS-Nogrette"])
    def test_optimize_wgs(self, method, subtests):
        """Weighting drives the spots toward equal power as the iterations proceed."""
        seed_for(f"wgs_uniformity-{method}")    # The initial phase decides the outcome.
        h = Hologram(
            target=_spot_target((64, 64), (13, 17), (30, 44), (50, 10), (10, 50), (32, 32))
        )
        h.optimize(method=method, maxiter=30, verbose=False, stat_groups=["computational"])
        comp = h.stats["stats"]["computational"]

        with subtests.test("weighting beats plain GS from the same starting phase"):
            seed_for(f"wgs_uniformity-{method}")
            gs = Hologram(
                target=_spot_target((64, 64), (13, 17), (30, 44), (50, 10), (10, 50), (32, 32))
            )
            gs.optimize(method="GS", maxiter=30, verbose=False, stat_groups=["computational"])
            assert (
                comp["uniformity"][-1]
                > gs.stats["stats"]["computational"]["uniformity"][-1]
            )

        with subtests.test("uniformity does not decrease"):
            assert comp["uniformity"][-1] >= comp["uniformity"][1]

        with subtests.test("std_err does not increase"):
            assert comp["std_err"][-1] <= comp["std_err"][1]

    def test_calculate_stats(self, subtests):
        with subtests.test("a feedback equal to its target is perfect"):
            amp = np.array([1.0, 2.0, 3.0, 4.0])
            stats = Hologram._calculate_stats(amp, amp, xp=np)
            assert stats["efficiency"] == pytest.approx(1.0)
            assert stats["uniformity"] == pytest.approx(1.0)
            assert stats["pkpk_err"] == pytest.approx(0.0, abs=1e-12)
            assert stats["std_err"] == pytest.approx(0.0, abs=1e-12)

        with subtests.test("both arguments are normalized, so a gain moves nothing"):
            feedback = np.array([1.0, 0.5, 2.0, 1.5])
            base = Hologram._calculate_stats(feedback, np.ones(4), xp=np)
            for gain in (1e-3, 1e3):
                scaled = Hologram._calculate_stats(feedback * gain, np.ones(4), xp=np)
                for key in ("efficiency", "uniformity"):
                    assert scaled[key] == pytest.approx(base[key], rel=1e-9)

        with subtests.test("NaN feedback is ignored, not propagated"):
            # Any partial-coverage camera: `ijcam_to_knmslm` NaN-fills the knm it misses.
            amp = np.ones(4)
            clean = Hologram._calculate_stats(amp, amp, xp=np)
            dirty = Hologram._calculate_stats(
                np.append(amp, np.nan), np.append(amp, np.nan), xp=np
            )
            assert dirty["efficiency"] == pytest.approx(clean["efficiency"])

    def test_update_stats(self, subtests):
        N = 10
        h = Hologram(target=_spot_target((64, 64), (20, 30), (40, 50)))
        h.optimize(method="GS", maxiter=N, verbose=False, stat_groups=["computational"])
        comp = h.stats["stats"]["computational"]

        with subtests.test("one entry per iteration"):
            for key in ("efficiency", "uniformity", "std_err"):
                assert len(comp[key]) == N

        with subtests.test("efficiency and uniformity stay within [0, 1]"):
            for key in ("efficiency", "uniformity"):
                values = np.array(comp[key])
                assert np.all(values >= 0) and np.all(values <= 1 + 1e-9), key

        with subtests.test("std_err is finite"):
            assert np.all(np.isfinite(comp["std_err"]))

    def test_update_flags(self, subtests):
        h = Hologram(target=np.zeros((32, 32)))

        with subtests.test("experimental bases are valid stat groups"):
            h._update_flags("GS", None, ["experimental_knm", "experimental_ij"])

        with subtests.test("but are not valid feedback"):
            with pytest.raises(ValueError):
                h._update_flags("GS", "experimental_knm", [])

    def test_save_stats(self, tmp_path):
        """A WGS method leaves ``fix_phase_efficiency`` None, which h5 must still carry."""
        h = Hologram(target=_spot_target((32, 32), (8, 8)))
        h.optimize(method="WGS-Kim", maxiter=3, verbose=False, stat_groups=["computational"])

        path = str(tmp_path / "stats.h5")
        h.save_stats(path)
        assert load_h5(path)["flags"]["fix_phase_efficiency"] is None

    def test_load_stats(self, tmp_path, subtests):
        """The h5 round trip, which must give back the lists that the update path appends to."""
        h = Hologram(target=_spot_target((32, 32), (8, 8), (20, 24)))
        h.reset_phase(random_phase=0)
        h.optimize(method="WGS-Kim", maxiter=4, verbose=False, stat_groups=["computational"])

        path = str(tmp_path / "stats.h5")
        h.save_stats(path)
        loaded = Hologram(target=np.zeros((16, 16), dtype=np.float32))
        loaded.load_stats(path)

        with subtests.test("the stats hierarchy survives the round trip"):
            np.testing.assert_equal(loaded.stats, h.stats)

            # h5 gives back arrays; the update path appends, so they must be lists again.
            assert isinstance(loaded.stats["method"], list)
            for group in loaded.stats["stats"].values():
                assert all(isinstance(series, list) for series in group.values())

        with subtests.test("include_state restores the geometry as tuples of int"):
            assert loaded.shape == tuple(h.shape)
            assert loaded.slm_shape == tuple(h.slm_shape)
            assert all(isinstance(n, int) for n in loaded.shape + loaded.slm_shape)
            assert loaded.iter == h.iter
            np.testing.assert_allclose(_np(loaded.phase), _np(h.phase))

        with subtests.test("a stats-only file cannot restore state"):
            bare_path = str(tmp_path / "stats_only.h5")
            h.save_stats(bare_path, include_state=False)

            with pytest.raises(ValueError, match="State was not stored"):
                Hologram(target=np.zeros((16, 16), dtype=np.float32)).load_stats(bare_path)

            bare = Hologram(target=np.zeros((16, 16), dtype=np.float32))
            bare.load_stats(bare_path, include_state=False)
            np.testing.assert_equal(bare.stats, h.stats)

    def test_compute_limits(self, subtests):
        """The zoom box: it contains every lit pixel and never leaves the frame."""
        source = np.zeros((40, 60))
        source[10:21, 5:16] = 1.0

        with subtests.test("the box wraps the lit region by a pixel"):
            assert Hologram._compute_limits(source, limit_padding=0) == [(4, 17), (9, 22)]

        with subtests.test("padding widens it on every side"):
            assert Hologram._compute_limits(source, limit_padding=0.5) == [(0, 22), (4, 27)]

        with subtests.test("nothing lit falls back to the whole frame"):
            for empty in (np.zeros((40, 60)), np.full((40, 60), np.nan)):
                assert Hologram._compute_limits(empty) == [(0, 59), (0, 39)]

        with subtests.test("the box is clipped to the frame"):
            corners = np.zeros((40, 60))
            corners[0, 0] = corners[-1, -1] = 1.0
            assert Hologram._compute_limits(corners, limit_padding=1.0) == [(0, 59), (0, 39)]

    def test_plot_nearfield(self, keep_figures, subtests):
        """Two panels over the SLM plane, and the padded choice of how much of it to show."""
        h = Hologram(target=_spot_target((64, 64), (16, 16)), slm_shape=(32, 32))

        def panels():
            return [im.get_array() for ax in keep_figures.gcf().axes for im in ax.images]

        with subtests.test("the default view is the unpadded SLM"):
            h.plot_nearfield()
            assert [np.shape(panel) for panel in panels()] == [tuple(h.slm_shape)] * 2
            keep_figures.flush()

        with subtests.test("padded shows the whole computational nearfield"):
            h.plot_nearfield(padded=True, title="Hi", cbar=True)
            fig = keep_figures.gcf()
            assert [np.shape(panel) for panel in panels()] == [tuple(h.shape)] * 2
            assert [ax.get_title() for ax in fig.axes[:2]] == ["Hi: Amplitude", "Hi: Phase"]
            assert len(fig.axes) == 4, "a colorbar axis beside each panel"
            keep_figures.flush()

        with subtests.test("a complex source supplies both panels"):
            # Off the branch cut, so np.angle's (-pi, pi] range maps back onto it exactly.
            phase = np.linspace(0.1, 2 * np.pi - 0.1, 32 * 32).reshape(32, 32)
            h.plot_nearfield(source=0.5 * np.exp(1j * phase))
            (amp_panel, phase_panel) = panels()

            assert np.amax(amp_panel) == pytest.approx(0.5)
            np.testing.assert_allclose(
                phase_panel, np.mod(phase, 2 * np.pi) / np.pi, atol=1e-6
            )

    def test_plot_farfield(self, simulated_system_factory, keep_figures, subtests):
        """The overview and zoom pair: where it crops, what it names, and the units it labels."""
        h = Hologram(target=_spot_target((64, 64), (20, 20), (44, 44)))
        h.reset_phase(random_phase=0)
        h.optimize(method="GS", maxiter=3, verbose=False)

        target_limits = Hologram._compute_limits(_np(h.target))

        with subtests.test("the zoom crops to the target by default"):
            np.testing.assert_array_equal(h.plot_farfield(), target_limits)
            keep_figures.flush()

        with subtests.test("each named source titles its own plot"):
            for (source, title) in [
                ("amp_ff", "Farfield Amplitude"),
                ("phase_ff", "Farfield Phase"),
                ("target", "Target Amplitude"),
            ]:
                h.plot_farfield(source=source)
                assert keep_figures.gcf().axes[0].get_title() == title + ": Full"
                keep_figures.flush()

        with subtests.test("an unknown source, unit, or empty zoom raises"):
            with pytest.raises(ValueError, match="Did not recognize source"):
                h.plot_farfield(source="nonsense")
            with pytest.raises(ValueError, match="not a valid unit"):
                h.plot_farfield(units="ij")
            with pytest.raises(ValueError, match="valid blaze unit"):
                h.plot_farfield(units="bogus")
            with pytest.raises(ValueError, match="zero length"):
                h.plot_farfield(limits=[(10, 10), (10, 10)])
            keep_figures.flush()

        with subtests.test("supplied axes are drawn into rather than a new figure"):
            (_, axs) = keep_figures.subplots(1, 2)
            open_figures = len(keep_figures.get_fignums())

            np.testing.assert_array_equal(
                h.plot_farfield(axs=axs, cbar=True), target_limits
            )
            assert len(keep_figures.get_fignums()) == open_figures
            keep_figures.flush()

        def camera_hologram(case):
            fs = simulated_system_factory(case)
            install_ground_truth_calibration(fs)
            canvas = np.zeros(fs.cam.shape, dtype=np.float32)
            canvas[60:70, 60:70] = 1.0
            return FeedbackHologram(shape=(128, 128), target_ij=canvas, cameraslm=fs)

        with subtests.test("a camera reaching past the farfield outlines both"):
            camera_hologram("fov_larger").plot_farfield(source="target")
            overview = keep_figures.gcf().axes[0]

            assert {text.get_text() for text in overview.texts} == {
                "SLM FoV", "Camera FoV", "Zoom"
            }
            knm_extent = overview.images[0].get_extent()
            keep_figures.flush()

        with subtests.test("a camera inside the farfield leaves the SLM outline off"):
            camera_hologram("fov_smaller").plot_farfield(source="target")

            assert {text.get_text() for text in keep_figures.gcf().axes[0].texts} == {
                "Camera FoV", "Zoom"
            }
            keep_figures.flush()

        with subtests.test("kxy rebases the extent off the pixel grid"):
            camera_hologram("fov_larger").plot_farfield(source="target", units="kxy")
            kxy_extent = keep_figures.gcf().axes[0].images[0].get_extent()
            keep_figures.flush()

            assert np.amax(np.abs(knm_extent)) > 1, "knm counts pixels"
            assert np.amax(np.abs(kxy_extent)) < 1, "kxy is a normalized angle"

    def test_plot_stats(self, keep_figures, subtests):
        """The convergence plot: one color per stat group, shaded where the phase was fixed."""
        h = Hologram(target=_spot_target((32, 32), (8, 8), (20, 24)))
        h.reset_phase(random_phase=0)
        h.optimize(
            method="WGS-Kim", maxiter=4, verbose=False,
            stat_groups=["computational"], fix_phase_iteration=2,
        )

        with subtests.test("the axes are named for the class and scaled logarithmically"):
            ax = h.plot_stats()
            assert ax.get_title() == "Hologram Statistics"
            assert ax.get_yscale() == "log"
            assert ax.get_xlim() == pytest.approx((-0.75, h.iter - 0.25))
            keep_figures.flush()

        with subtests.test("the legend names the group, the metrics, and the shading"):
            ax = h.plot_stats()
            assert [text.get_text() for text in ax.get_legend().get_texts()] == [
                "computational", "inefficiency", "nonuniformity",
                "pkpk_err", "std_err", "fixed_phase",
            ]
            keep_figures.flush()

        with subtests.test("an explicit stats_dict and ylim are honored"):
            ax = h.plot_stats(
                stats_dict=h.stats, stat_groups=["computational"],
                ylim=(1e-3, 1.0), show=True,
            )
            assert ax.get_ylim() == pytest.approx((1e-3, 1.0))

    def test_populate_results(self, subtests):
        h = Hologram(target=_spot_target((64, 64), (16, 16)))
        h.reset_phase(random_phase=0)
        h.optimize(method="GS", maxiter=3, verbose=False)

        with subtests.test("amp_ff is the farfield magnitude"):
            assert np.allclose(_np(h.amp_ff), np.abs(_np(h.farfield)), atol=1e-6)

        with subtests.test("phase_ff is the farfield argument"):
            assert np.allclose(_np(h.phase_ff), np.angle(_np(h.farfield)), atol=1e-5)

    @pytest.mark.parametrize("method", ["GS", "WGS-Leonardo", "WGS-Kim", "WGS-Nogrette"])
    def test_optimize_speed(self, random_seed, method, benchmark):
        rng = np.random.default_rng(random_seed)
        target = np.zeros((1024, 1024))
        target[rng.integers(0, 1024, 20), rng.integers(0, 1024, 20)] = 1

        hologram = Hologram(target=target)
        benchmark(hologram.optimize, method=method, maxiter=20, verbose=False, stat_groups=[])

    @pytest.mark.gpu
    @pytest.mark.parametrize("method", ["GS", "WGS-Leonardo", "WGS-Kim", "WGS-Nogrette"])
    def test_optimize_speed_gpu(self, random_seed, method, benchmark, has_cupy):
        import cupy as cp
        rng = np.random.default_rng(random_seed)
        target = cp.zeros((1024, 1024))
        target[rng.integers(0, 1024, 20), rng.integers(0, 1024, 20)] = 1

        hologram = Hologram(target=target)
        benchmark(hologram.optimize, method=method, maxiter=20, verbose=False, stat_groups=[])


class TestSpotHologram:

    def test_init(self, simulated_system_factory, subtests):
        with subtests.test("a spot puts all of the target power on its pixel"):
            h = SpotHologram(
                shape=(64, 64), spot_vectors=np.array([[32.0], [32.0]]), basis="knm"
            )
            target = _np(h.target)
            assert target[32, 32] > 0
            rest = target.copy()
            rest[32, 32] = 0.0
            assert np.all(np.nan_to_num(rest) == 0)

        with subtests.test("len is the number of spots"):
            spots = np.array([[10.0 + 5 * i for i in range(7)]] * 2)
            assert len(SpotHologram(shape=(64, 64), spot_vectors=spots, basis="knm")) == 7

        with subtests.test("uniform spot amplitudes give equal target pixel powers"):
            spots = np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
            h = SpotHologram(shape=(64, 64), spot_vectors=spots, basis="knm")
            target = _np(h.target)
            powers = [float(target[int(spots[1, i]), int(spots[0, i])] ** 2) for i in range(3)]
            assert np.allclose(powers, powers[0], rtol=1e-4)

        with subtests.test("external_spot_amp defaults to a copy of spot_amp"):
            spots = np.array([[10.0, 20.0, 30.0, 40.0], [10.0, 20.0, 30.0, 40.0]])
            h = SpotHologram(
                shape=(64, 64), spot_vectors=spots, basis="knm",
                spot_amp=np.array([1.0, 2.0, 3.0, 4.0]),
            )
            assert np.allclose(h.external_spot_amp, h.spot_amp)
            assert h.external_spot_amp is not h.spot_amp

        with subtests.test("a spot outside the computational space raises"):
            with pytest.raises(ValueError, match="[Bb]ounds"):
                SpotHologram(
                    shape=(64, 64), spot_vectors=np.array([[100.0], [100.0]]), basis="knm"
                )

        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)

        spots_kxy = view_kxy_grid(fs, count=2, frac=0.3)
        spots_ij = fs.kxyslm_to_ijcam(spots_kxy)
        spots_knm = toolbox.convert_vector(
            spots_kxy, "kxy", "knm", hardware=fs.slm, shape=fs.slm.shape
        )
        null_kxy = view_kxy_grid(fs, count=2, frac=0.6)

        with subtests.test("the kxy, ij and knm bases describe the same spots"):
            for (vectors, basis) in [(spots_kxy, "kxy"), (spots_ij, "ij"), (spots_knm, "knm")]:
                h = SpotHologram(fs.slm.shape, vectors, basis=basis, cameraslm=fs)
                np.testing.assert_allclose(h.spot_knm, spots_knm)
                np.testing.assert_allclose(h.spot_kxy, spots_kxy)
                np.testing.assert_allclose(h.spot_ij, spots_ij)

        with subtests.test("kxy and ij need a cameraslm, and ij needs a fourier calibration"):
            for basis in ("kxy", "ij"):
                with pytest.raises(ValueError, match="cameraslm"):
                    SpotHologram(fs.slm.shape, spots_knm, basis=basis)

            uncalibrated = simulated_system_factory("matched")
            with pytest.raises(ValueError, match="fourier"):
                SpotHologram(
                    uncalibrated.slm.shape, spots_ij, basis="ij", cameraslm=uncalibrated
                )

            with pytest.raises(Exception, match="[Uu]nrecognized basis"):
                SpotHologram(fs.slm.shape, spots_knm, basis="nonsense", cameraslm=fs)

        with subtests.test("null_vectors reach the same knm points from either basis"):
            null_radius_kxy = 0.01

            from_kxy = SpotHologram(
                fs.slm.shape, spots_kxy, basis="kxy", cameraslm=fs,
                null_vectors=null_kxy, null_radius=null_radius_kxy,
            )
            from_ij = SpotHologram(
                fs.slm.shape, spots_ij, basis="ij", cameraslm=fs,
                null_vectors=fs.kxyslm_to_ijcam(null_kxy),
                null_radius=toolbox.convert_radius(
                    null_radius_kxy, "kxy", "ij", hardware=fs, shape=fs.slm.shape
                ),
            )

            np.testing.assert_allclose(from_kxy.null_knm, from_ij.null_knm)
            assert from_kxy.null_radius_knm == from_ij.null_radius_knm

        with subtests.test("null_vectors without a null_radius fall back on the spot spacing"):
            h = SpotHologram(
                fs.slm.shape, spots_kxy, basis="kxy", cameraslm=fs, null_vectors=null_kxy
            )
            spacing = toolbox.smallest_distance(np.hstack((h.null_knm, h.spot_knm)))
            assert h.null_radius_knm == int(h.null_radius_knm)
            assert 0 < h.null_radius_knm < spacing / 2, "the nulled discs must not overlap"

        with subtests.test("spot_amp needs one amplitude per spot"):
            with pytest.raises(ValueError, match="spot_amp"):
                SpotHologram(
                    shape=(64, 64), spot_vectors=np.array([[10.0, 20.0], [10.0, 20.0]]),
                    basis="knm", spot_amp=np.ones(3),
                )

    def test_optimize(self):
        """GS concentrates most of the power onto two well-separated spots."""
        h = SpotHologram(
            shape=(64, 64), spot_vectors=np.array([[16.0, 48.0], [16.0, 48.0]]), basis="knm"
        )
        h.optimize(method="GS", maxiter=30, verbose=False, stat_groups=["computational"])
        eff = h.stats["stats"]["computational"]["efficiency"][-1]
        assert eff > 0.5, f"SpotHologram GS efficiency {eff:.3f}"

    def test_update_weights(self, simulated_system_factory, subtests):
        """The ``"external_spot"`` feedback branch, which the farfield never enters."""
        spots = np.array([[16.0, 32.0, 48.0], [16.0, 32.0, 48.0]])

        def weighted(external, seed=None):
            h = SpotHologram(shape=(64, 64), spot_vectors=spots, basis="knm")
            if seed is not None:
                h.reset_phase(np.random.default_rng(seed).uniform(0, 2 * np.pi, h.slm_shape))
            h.external_spot_amp = h.spot_amp * np.asarray(external)

            at_spots = (h.spot_knm_rounded[1, :], h.spot_knm_rounded[0, :])
            before = _np(h.weights[at_spots]).copy()
            h.optimize(
                method="WGS-Kim", feedback="external_spot", maxiter=3,
                verbose=False, stat_groups=[],
            )
            return (before, _np(h.weights[at_spots]))

        with subtests.test("feedback proportional to spot_amp leaves the weights alone"):
            (before, after) = weighted([3.0, 3.0, 3.0])
            np.testing.assert_allclose(after, before)

        with subtests.test("a spot reading above its share is weighted down"):
            (_, after) = weighted([2.0, 1.0, 0.5])
            assert after[0] < after[1] < after[2]

        with subtests.test("external feedback does not consult the farfield"):
            (_, from_one_phase) = weighted([2.0, 1.0, 0.5], seed=0)
            (_, from_another) = weighted([2.0, 1.0, 0.5], seed=1)
            np.testing.assert_array_equal(from_one_phase, from_another)

        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)

        with subtests.test("'experimental' is an alias for 'experimental_spot'"):
            h = SpotHologram(
                fs.slm.shape, view_kxy_grid(fs, count=2, frac=0.3), basis="kxy", cameraslm=fs
            )
            h.optimize(
                method="WGS-Kim", feedback="experimental", maxiter=3,
                verbose=False, stat_groups=[],
            )
            assert h.flags["feedback"] == "experimental_spot"

        with subtests.test("'computational_spot' drives the spots to a uniform farfield"):
            h = SpotHologram(
                fs.slm.shape, view_kxy_grid(fs, count=2, frac=0.3), basis="kxy", cameraslm=fs
            )
            h.optimize(
                method="WGS-Kim", feedback="computational_spot", maxiter=10,
                verbose=False, stat_groups=["computational_spot"],
            )
            assert h.stats["stats"]["computational_spot"]["uniformity"][-1] > 0.999

    def test_update_stats(self, simulated_system_factory, subtests):
        """The two spot stat groups that the computational tests do not reach."""
        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)
        spots = view_kxy_grid(fs, count=2, frac=0.3)

        with subtests.test("'external_spot' reports the supplied amplitudes"):
            h = SpotHologram(fs.slm.shape, spots, basis="kxy", cameraslm=fs)
            h.external_spot_amp = h.spot_amp * np.array([2.0, 1.0, 1.0, 1.0])
            h.optimize(
                method="WGS-Kim", feedback="external_spot", maxiter=3,
                verbose=False, stat_groups=["external_spot"],
            )

            # Powers of 4:1:1:1 against a flat target: 1 - (4 - 1) / (4 + 1), every iteration.
            assert h.stats["stats"]["external_spot"]["uniformity"] == pytest.approx([0.4] * 3)

        with subtests.test("'computational_spot' reports the spots at either farfield shape"):
            # One spot per pixel at the SLM shape; a window to integrate once padded.
            for shape in (fs.slm.shape, (256, 256)):
                h = SpotHologram(shape, spots, basis="kxy", cameraslm=fs)
                h.optimize(
                    method="WGS-Kim", feedback="computational_spot", maxiter=10,
                    verbose=False, stat_groups=["computational_spot"],
                )
                assert h.stats["stats"]["computational_spot"]["uniformity"][-1] > 0.99

    def test_refine_offset(self, simulated_system_factory, subtests):
        """The ``"kxy"`` basis, which steers the k-vectors instead of the camera targets."""
        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)

        h = SpotHologram(
            fs.slm.shape, view_kxy_grid(fs, count=2, frac=0.3), basis="kxy", cameraslm=fs
        )
        h.optimize(method="WGS-Kim", maxiter=5, verbose=False, stat_groups=[])
        img = fs.cam.get_image()

        with subtests.test("basis='kxy' steers by the measured shift"):
            spot_ij = h.spot_ij.copy()
            spot_knm = h.spot_knm.copy()
            shift = h.refine_offset(img=img, basis="kxy")

            np.testing.assert_allclose(fs.kxyslm_to_ijcam(h.spot_kxy), spot_ij - shift)
            np.testing.assert_array_equal(h.spot_ij, spot_ij)
            assert np.any(h.spot_knm != spot_knm)

        with subtests.test("an unrecognized basis raises"):
            with pytest.raises(Exception, match="[Uu]nrecognized basis"):
                h.refine_offset(img=img, basis="nonsense")


class TestCompressedSpotHologram:

    def test_init(self, simulated_system_factory):
        """``external_spot_amp`` defaults as in `SpotHologram`: a copy of ``spot_amp``."""
        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)

        h = CompressedSpotHologram(
            view_kxy_grid(fs, count=2, frac=0.4), basis="kxy",
            spot_amp=np.array([1.0, 2.0, 3.0, 4.0]), cameraslm=fs, cuda=False,
        )
        assert np.allclose(h.external_spot_amp, h.spot_amp)
        assert h.external_spot_amp is not h.spot_amp
        with pytest.raises(ValueError, match="spot_amp"):
            CompressedSpotHologram(
                view_kxy_grid(fs, count=2, frac=0.4), basis="kxy",
                spot_amp=np.ones(3), cameraslm=fs, cuda=False,
            )

    def test_optimize(self, simulated_system_factory):
        """The ``cuda=False`` path, which builds its kernel through ``zernike_sum``."""
        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)

        h = CompressedSpotHologram(
            view_kxy_grid(fs, count=3, frac=0.5), basis="kxy", cameraslm=fs, cuda=False
        )
        assert len(h) == 9

        h.optimize(
            method="WGS-Kim", maxiter=10, verbose=False, stat_groups=["computational_spot"]
        )
        assert h.stats["stats"]["computational_spot"]["uniformity"][-1] > 0.5

    def test_refine_offset(self, simulated_system_factory, subtests):
        """Camera frames are integers, and spots can sit against the sensor edge."""
        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)

        h = CompressedSpotHologram(
            view_kxy_grid(fs, count=2, frac=0.4), basis="kxy", cameraslm=fs, cuda=False
        )
        h.optimize(method="WGS-Kim", maxiter=3, verbose=False)
        img = fs.cam.get_image()
        assert not np.issubdtype(img.dtype, np.floating)

        bound = spot_size_ij(fs)[:, np.newaxis]

        with subtests.test("integer camera frame"):
            shift = h.refine_offset(img=img, basis=None)
            assert np.all(np.abs(shift) < bound)

        with subtests.test("spot against the sensor edge"):
            h.spot_ij[:, 0] = [1.0, 1.0]    # As a previous refine_offset(basis="ij") could.
            shift = h.refine_offset(img=img, basis=None)
            assert np.all(np.isfinite(shift[:, 0]))
            assert np.all(np.abs(shift[:, 1:]) < bound)

        with subtests.test("basis='ij' neither writes through the caller nor truncates"):
            caller = np.rint(h.spot_ij).astype(int)
            h.spot_ij = caller
            before = caller.copy()
            h.refine_offset(img=img, basis="ij")

            assert h.spot_ij is not caller
            np.testing.assert_array_equal(caller, before)
            assert np.issubdtype(h.spot_ij.dtype, np.floating)
            assert np.any(h.spot_ij[[0, 1]] != np.rint(h.spot_ij[[0, 1]]))

    def test_update_weights(self, simulated_system_factory, subtests):
        """The ``"external_spot"`` feedback branch, as in `SpotHologram`."""
        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)
        spots = view_kxy_grid(fs, count=2, frac=0.3)

        def weighted(external, feedback="external_spot"):
            h = CompressedSpotHologram(spots, basis="kxy", cameraslm=fs, cuda=False)
            h.external_spot_amp = h.spot_amp * np.asarray(external)

            before = _np(h.weights).copy()
            h.optimize(
                method="WGS-Kim", feedback=feedback, maxiter=3, verbose=False, stat_groups=[]
            )
            return (h, before, _np(h.weights))

        with subtests.test("feedback proportional to spot_amp leaves the weights alone"):
            (_, before, after) = weighted([3.0, 3.0, 3.0, 3.0])
            np.testing.assert_allclose(after, before)

        with subtests.test("a spot reading above its share is weighted down"):
            (_, _, after) = weighted([2.0, 1.0, 1.0, 1.0])
            after = after.ravel()
            assert after[0] < after[1]
            np.testing.assert_allclose(after[1:], after[1])

        with subtests.test("'computational' and 'experimental' mean their spot forms"):
            for (feedback, spot_feedback) in [
                ("computational", "computational_spot"),
                ("experimental", "experimental_spot"),
            ]:
                (h, _, _) = weighted([1.0, 1.0, 1.0, 1.0], feedback=feedback)
                assert h.flags["feedback"] == spot_feedback


class TestMultiplaneHologram:

    def test_init(self, subtests):
        mph = _multiplane(n=2)

        with subtests.test("len is the number of children"):
            assert len(mph) == 2

        with subtests.test("the children share one phase array"):
            assert mph.holograms[0].phase is mph.holograms[1].phase

        with subtests.test("a non-hologram child is rejected"):
            with pytest.raises(ValueError):
                MultiplaneHologram([mph.holograms[0], "not a hologram"])

        with subtests.test("a nested MultiplaneHologram is rejected"):
            with pytest.raises(ValueError):
                MultiplaneHologram([mph, mph.holograms[0]])

    def test_nearfield_extract(self, simulated_system_factory, subtests):
        """Extracting the meta phase drops every child's frame, so each iteration re-measures."""
        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)
        children = [
            SpotHologram.make_rectangular_array(
                fs.slm.shape, (2, 2), (20, 20), array_center=(64 + offset, 64),
                basis="knm", cameraslm=fs,
            )
            for offset in (0, 25)
        ]
        mph = MultiplaneHologram(children)

        grabs = []
        for child in children:
            inner = child.measure
            def measure(basis="ij", child=child, inner=inner):
                fresh = child.img_ij is None
                out = inner(basis)
                if fresh:
                    grabs.append(float(np.nansum(_np(child.img_ij))))
                return out
            child.measure = measure

        mph.optimize(
            method="WGS-Kim", maxiter=3, verbose=False,
            feedback="experimental_spot", stat_groups=["experimental_spot"],
        )

        with subtests.test("every child grabs a frame every iteration"):
            assert len(grabs) == 3 * len(children)

        with subtests.test("one distinct frame per iteration, shared by both children"):
            assert len({round(total, 3) for total in grabs}) == 3

    def test_set_target(self):
        """The meta hologram holds no target; the children's targets are the interface."""
        with pytest.raises(RuntimeError):
            _multiplane().set_target(np.zeros((64, 64)))

    def test_reset(self, subtests):
        """``super().reset()`` calls this class's ``reset_weights()``, itself a loop over
        the children, so the children must not reset their weights a second time."""
        mph = _multiplane(n=3)
        counts = {id(h): 0 for h in mph.holograms}
        for h in mph.holograms:
            def counted(_h=h, _orig=h.reset_weights):
                counts[id(_h)] += 1
                return _orig()
            h.reset_weights = counted

        with subtests.test("each child resets its weights exactly once"):
            mph.reset()
            assert set(counts.values()) == {1}, counts

        with subtests.test("reset_weights=False skips them entirely"):
            counts.update(dict.fromkeys(counts, 0))
            mph.reset(reset_weights=False)
            assert set(counts.values()) == {0}, counts

    def test_optimize(self, subtests):
        mph = _multiplane(n=2)
        mph.optimize(method="GS", maxiter=40, verbose=False)

        with subtests.test("every child's target draws farfield power"):
            # The children share phase, so one composite farfield serves them all.
            ff = np.abs(_np(mph.holograms[0].get_farfield()))
            for child in mph.holograms:
                spot = np.unravel_index(np.argmax(_np(child.target)), np.shape(child.target))
                assert ff[spot] > 0.3 * ff.max(), f"spot {spot}"

        with subtests.test("the meta hologram's amp_ff is populated"):
            assert mph.amp_ff is not None

        with subtests.test("a child built without an amp keeps the scalar nearfield"):
            target = _spot_target((64, 64), (12, 12))
            plain = MultiplaneHologram([Hologram(target.copy()), Hologram(target.copy())])
            plain.optimize(maxiter=3, verbose=False)
            uniform = 1 / np.sqrt(np.prod(plain.slm_shape))
            for child in plain.holograms:
                assert np.ndim(child.amp) == 0
                assert child.amp == pytest.approx(uniform)

    @pytest.mark.gpu
    def test_can_batch_routines(self, subtests):
        """The batched amplitude replacement collapses every child onto one branch, so it
        must decline whenever the children would not agree on that branch."""
        mph = _multiplane(n=2)
        mph.reset_phase(random_phase=0)
        mph.optimize(method="GS", maxiter=1, verbose=False)
        plain = [{"mraf_enabled": False}] * len(mph)

        with subtests.test("plain GS batches"):
            mph.flags["method"] = "GS"
            assert mph._can_batch_routines(plain)

        with subtests.test("WGS declines"):
            mph.flags["method"] = "WGS-Leonardo"
            assert not mph._can_batch_routines(plain)

        with subtests.test("a fixed phase declines"):
            mph.flags["method"] = "GS"
            mph.flags["fixed_phase"] = True
            assert not mph._can_batch_routines(plain)
            mph.flags["fixed_phase"] = False

        with subtests.test("MRAF declines"):
            assert not mph._can_batch_routines([{"mraf_enabled": True}] * len(mph))

    @pytest.mark.gpu
    @pytest.mark.parametrize("slm_shape", [(64, 64), (32, 32)], ids=["unpadded", "padded"])
    def test_set_propagation_kernels(self, slm_shape, subtests):
        """Handing over the stack must be indistinguishable from assigning each child
        individually -- that equivalence is the whole basis for using it."""
        rng = np.random.default_rng(3)
        stack = (rng.random((3,) + slm_shape) * 6.28).astype(np.float32)

        legacy = _multiplane(slm_shape=slm_shape, n=3)
        for (i, h) in enumerate(legacy.holograms):
            h.propagation_kernel = cp.asarray(stack[i]) if cp is not None else stack[i]
        legacy._refresh_batched_kernels()

        batched = _multiplane(slm_shape=slm_shape, n=3)
        batched.set_propagation_kernels(stack)

        with subtests.test("the batched phasor matches"):
            assert np.allclose(
                _np(batched._batched_kernel_phasor),
                _np(legacy._batched_kernel_phasor),
                atol=1e-6,
            )

        with subtests.test("each child sees its own kernel"):
            for (i, h) in enumerate(batched.holograms):
                assert np.allclose(_np(h.propagation_kernel), stack[i], atol=1e-6)

        with subtests.test("no restack is triggered afterwards"):
            phasor = batched._batched_kernel_phasor
            batched._refresh_batched_kernels()
            assert batched._batched_kernel_phasor is phasor

        with subtests.test("GS agrees with per-child assignment"):
            for mph in (legacy, batched):
                mph.reset_phase(random_phase=0)
                mph.optimize(method="GS", maxiter=5, verbose=False)
            assert np.allclose(_np(legacy.phase), _np(batched.phase), atol=1e-4)

        with subtests.test("a single kernel broadcasts to every plane"):
            one = (rng.random(slm_shape) * 6.28).astype(np.float32)
            mph = _multiplane(slm_shape=slm_shape, n=3)
            mph.set_propagation_kernels(one)
            for h in mph.holograms:
                assert np.allclose(_np(h.propagation_kernel), one, atol=1e-6)

    @pytest.mark.gpu
    def test_refresh_batched_kernels(self, subtests):
        """The next transform multiplies by the phasor unconditionally, so a refresh must
        always leave one built and agreeing with the children's kernels."""
        mph = _multiplane(n=3)

        with subtests.test("all-None kernels still build a unit phasor"):
            assert all(h.propagation_kernel is None for h in mph.holograms)
            mph._refresh_batched_kernels()
            assert np.allclose(_np(mph._batched_kernel_phasor), 1.0, atol=1e-6)

        with subtests.test("a missing phasor is rebuilt even when no child changed"):
            mph._batched_kernel_phasor = None
            mph._refresh_batched_kernels()
            assert mph._batched_kernel_phasor is not None

        with subtests.test("rebinding one child's kernel is picked up"):
            # In-place mutation is explicitly outside the contract; see `_restack_children`.
            kernel = (np.random.default_rng(9).random((64, 64)) * 6.28).astype(np.float32)
            mph.holograms[1].propagation_kernel = (
                cp.asarray(kernel) if cp is not None else kernel
            )
            mph._refresh_batched_kernels()
            phasor = _np(mph._batched_kernel_phasor)
            assert np.allclose(phasor[1], np.exp(1j * kernel), atol=1e-5)
            assert np.allclose(phasor[0], 1.0, atol=1e-6), "untouched planes stay put"

    @pytest.mark.parametrize(
        "slm_shape", [(64, 64), (32, 32)], ids=["unpadded", "padded"]
    )
    @pytest.mark.gpu
    def test_gs_farfield_routines_batched(self, slm_shape, monkeypatch, subtests):
        """One pass over the batched tensors must land where the per-child loop does."""
        def run(iterations=8):
            mph = _multiplane(slm_shape=slm_shape, n=2)
            mph.reset_phase(random_phase=0)
            mph.optimize(method="GS", maxiter=iterations, verbose=False)
            return mph

        reference = run()
        monkeypatch.setattr(
            MultiplaneHologram, "_can_batch_routines", lambda self, mraf: False
        )
        legacy = run()

        with subtests.test("phase"):
            assert np.allclose(_np(reference.phase), _np(legacy.phase), atol=1e-4)

        with subtests.test("child amp_ff"):
            for (i, (a, b)) in enumerate(zip(reference.holograms, legacy.holograms)):
                assert np.allclose(_np(a.amp_ff), _np(b.amp_ff), atol=1e-5), f"child {i}"

    # "odd-pad": an odd shape difference makes `unpad` return a 0-based slice that under-covers.
    @pytest.mark.parametrize(
        "shape, slm_shape",
        [((64, 64), (64, 64)), ((64, 64), (32, 32)), ((65, 65), (64, 64))],
        ids=["unpadded", "padded", "odd-pad"],
    )
    @pytest.mark.gpu
    def test_farfield2nearfield_batched(self, shape, slm_shape, monkeypatch):
        """The batched path fuses the propagation kernel into the nearfield build, so
        only non-trivial kernels against `Hologram._build_nearfield` can pin it."""
        rng = np.random.default_rng(5)
        stack = (rng.random((3,) + slm_shape) * 6.28).astype(np.float32)

        # A non-uniform source: GS renormalizes, so a uniform amp hides a dropped factor.
        (yy, xx) = np.meshgrid(
            np.linspace(-1, 1, slm_shape[0]), np.linspace(-1, 1, slm_shape[1]), indexing="ij"
        )
        amp = np.exp(-(xx**2 + yy**2) / 0.5).astype(np.float32)
        amp /= np.sqrt(np.sum(amp**2))

        def run(force_unbatched):
            if force_unbatched:
                monkeypatch.setattr(MultiplaneHologram, "_can_batch", lambda self: False)
            mph = _multiplane(shape, slm_shape, n=3, amp=amp)
            mph.set_propagation_kernels(stack)
            mph.reset_phase(random_phase=0)
            mph.optimize(method="GS", maxiter=6, verbose=False)
            return mph

        batched = run(False)
        assert batched._batched, "expected the batched fast path for this configuration"
        unbatched = run(True)
        assert not unbatched._batched

        assert np.allclose(_np(batched.phase), _np(unbatched.phase), atol=1e-4)


class _StubSLM:
    """Minimal duck-typed SLM: just the geometry `ijcam_to_knmslm` reads."""

    def __init__(self, shape=(32, 32), pitch=(1e-5, 1e-5)):
        self.shape = shape
        self.pitch = np.array(pitch, dtype=float)

    def set_phase(self, phase=None, **kwargs):
        """Never called; `toolbox.convert_vector` duck-types an SLM by this method."""

    def _get_source_amplitude(self):
        amp = np.ones(self.shape, dtype=np.float32)
        return amp / np.sqrt(np.sum(amp**2))


class _StubCameraSLM:
    """
    Minimal duck-typed FourierSLM carrying a synthetic Fourier calibration. Enough for
    `FeedbackHologram` to build targets; nothing here talks to hardware.
    """

    def __init__(self, cam_shape=(60, 80), slm=None, calibrated=True):
        self.slm = slm if slm is not None else _StubSLM()
        self.cam = SimpleNamespace(shape=cam_shape)
        self._calibrated = calibrated
        self.calibrations = {"fourier": {}} if calibrated else {}
        # Anisotropic kxy -> ij scales, off-center so the composite affine is not trivial.
        self.fourier_affine = Affine(
            M=[[1.92e-4, 0.0], [0.0, 1.60e-4]],
            b=[[cam_shape[1] / 2 + 3], [cam_shape[0] / 2 - 4]],
        )

    def ijcam_to_kxyslm(self, ij):
        M_inv = np.linalg.inv(self.fourier_affine.M)
        return np.matmul(M_inv, np.array(ij, dtype=float) - self.fourier_affine.b)


class TestFeedbackHologram:
    """
    A sub-image target must be indistinguishable from the same sub-image pasted into a
    full camera-sized NaN canvas -- that equivalence is the entire justification for the
    ROI path, which exists only to avoid uploading the canvas.
    """

    SHAPE = (32, 32)
    CAM_SHAPE = (60, 80)
    # The knm grid's footprint is ~19 x 16 camera pixels about (26, 43); this roi clears it.
    ROI = (12, 29)
    PATCH_SHAPE = (28, 28)

    def _canvas_and_patch(self, roi=None, patch_shape=None, seed=7):
        roi = self.ROI if roi is None else roi
        patch_shape = self.PATCH_SHAPE if patch_shape is None else patch_shape
        rng = np.random.default_rng(seed)
        patch = rng.random(patch_shape).astype(np.float32)
        canvas = np.full(self.CAM_SHAPE, np.nan, dtype=np.float32)
        (y0, x0) = roi
        canvas[y0:y0 + patch_shape[0], x0:x0 + patch_shape[1]] = patch
        return canvas, patch

    def _pair(self, roi=None, patch_shape=None, seed=7):
        """The same target as a full NaN canvas and as an ROI sub-image."""
        (canvas, patch) = self._canvas_and_patch(roi, patch_shape, seed)
        full = FeedbackHologram(
            self.SHAPE, target_ij=canvas, cameraslm=_StubCameraSLM(self.CAM_SHAPE)
        )
        sub = FeedbackHologram(
            self.SHAPE, target_ij=patch, target_ij_roi=self.ROI if roi is None else roi,
            cameraslm=_StubCameraSLM(self.CAM_SHAPE),
        )
        return (full, sub, canvas, patch)

    def test_init(self, subtests):
        (full, sub, _, patch) = self._pair()

        with subtests.test("the target is actually populated"):
            assert np.count_nonzero(np.nan_to_num(_np(full.target))) > 100

        with subtests.test("a sub-image target reaches the same knm grid as the canvas"):
            assert np.allclose(_np(full.target), _np(sub.target), equal_nan=True)

        with subtests.test("the roi is recorded"):
            assert sub.target_ij_roi == self.ROI
            assert full.target_ij_roi is None

        with subtests.test("a roi without a Fourier calibration is rejected"):
            # Uncalibrated, set_target() never runs, so the roi would drop silently.
            with pytest.raises(ValueError, match="Fourier calibration"):
                FeedbackHologram(
                    self.SHAPE, target_ij=patch, target_ij_roi=self.ROI,
                    cameraslm=_StubCameraSLM(self.CAM_SHAPE, calibrated=False),
                )

    def test_set_target(self, subtests):
        (full, sub, _, _) = self._pair()
        (canvas2, patch2) = self._canvas_and_patch(seed=11)

        full.set_target(canvas2, reset_weights=True)
        sub.set_target(patch2, reset_weights=True, roi=self.ROI)

        with subtests.test("target lands where the constructor put it"):
            assert np.allclose(_np(full.target), _np(sub.target), equal_nan=True)

        with subtests.test("weights follow the target"):
            assert np.allclose(_np(full.weights), _np(sub.weights), equal_nan=True)

    def test_validate_roi(self, subtests):
        """A sub-image hanging off the frame has samples no camera could supply. The check
        lives at the `_ijcam_to_knmslm_resampler` choke point, so every entry point sees it."""
        (canvas, patch) = self._canvas_and_patch()
        h = FeedbackHologram(
            self.SHAPE, target_ij=canvas, cameraslm=_StubCameraSLM(self.CAM_SHAPE)
        )

        for roi in [(-1, 0), (0, -1), (52, 34), (24, 72)]:
            with subtests.test(f"the constructor rejects {roi}"):
                with pytest.raises(ValueError, match="camera frame"):
                    FeedbackHologram(
                        self.SHAPE, target_ij=patch, target_ij_roi=roi,
                        cameraslm=_StubCameraSLM(self.CAM_SHAPE),
                    )

            with subtests.test(f"set_target rejects {roi}"):
                with pytest.raises(ValueError, match="camera frame"):
                    h.set_target(patch, roi=roi)

            with subtests.test(f"ijcam_to_knmslm rejects {roi}"):
                with pytest.raises(ValueError, match="camera frame"):
                    h.ijcam_to_knmslm(patch, order=0, roi=roi)

    def test_roi_window(self, subtests):
        """The window is the roi sub-image of a full frame, or the frame itself."""
        (full, sub, canvas, patch) = self._pair()
        frame = np.nan_to_num(canvas, nan=0.0).astype(np.float32)

        with subtests.test("no roi passes the frame through"):
            assert np.array_equal(_np(full._roi_window(frame)), frame)

        with subtests.test("a roi cuts out exactly the sub-image"):
            assert np.array_equal(_np(sub._roi_window(frame)), patch)

    def test_ijcam_to_knmslm(self, subtests):
        """The roi may only ever drop samples relative to the canvas, never invent them,
        and it must carry every interior sample at identical relative weight."""
        roi = (24, 34)
        (full, sub, _, _) = self._pair(roi=roi, patch_shape=(12, 12))
        (a, b) = (_np(full.target), _np(sub.target))

        defined_in_both = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
        assert np.count_nonzero(defined_in_both) > 50, "nothing left to compare"

        with subtests.test("interior samples agree up to one global normalization"):
            ratio = b[defined_in_both] / a[defined_in_both]
            assert np.allclose(ratio, np.median(ratio), rtol=1e-5)

        with subtests.test("no sample is invented"):
            assert np.all((np.isfinite(b) & (b != 0)) <= (np.isfinite(a) & (a != 0)))

        with subtests.test("only the outer rim is left undefined"):
            rim = ndimage.binary_dilation(np.isnan(a), np.ones((3, 3), bool))
            assert np.all(np.isnan(b) <= rim)

        with subtests.test("a dark frame has no power to normalize by, warm cache or cold"):
            dark = np.zeros_like(_np(full.target_ij))
            for _ in range(2):      # The second call runs against a warmed resampler cache.
                with pytest.raises(ValueError, match="No power in hologram"):
                    full.ijcam_to_knmslm(dark)

    @pytest.mark.parametrize("order", [0, 3])
    def test_ijcam_to_knmslm_resampler(self, order, subtests):
        """The augmented (2, 3) matrix `affine_transform` is handed must be equivalent to
        a separate `offset=`, NaN fill included: MRAF's undefined region depends on it."""
        (canvas, _) = self._canvas_and_patch((24, 34), patch_shape=(20, 20))
        canvas = np.nan_to_num(canvas, nan=0.0)     # affine_transform cannot blur NaNs.
        h = FeedbackHologram(
            self.SHAPE, target_ij=canvas, cameraslm=_StubCameraSLM(self.CAM_SHAPE)
        )

        matrix = h._ijcam_to_knmslm_resampler(np.shape(canvas), None)
        assert matrix.shape == (2, 3)

        reference = cp_affine_transform(
            input=xp.array(canvas, dtype=h.dtype),
            matrix=matrix[:, :-1].copy(),
            offset=matrix[:, -1].copy(),
            output_shape=h.shape,
            order=order,
            mode="constant",
            cval=np.nan,
        )
        # Mirror the abs + normalize that `ijcam_to_knmslm` applies after the transform.
        reference = xp.abs(reference)
        reference /= Hologram._norm(reference)

        with subtests.test("the augmented matrix matches a separate offset"):
            assert np.allclose(
                _np(h.ijcam_to_knmslm(canvas, order=order)), _np(reference), equal_nan=True
            )

    def test_ijcam_to_knmslm_cache(self, subtests):
        """The cache holds per-geometry device arrays sized like a hologram, so it needs a
        bound and a way to release what it holds."""
        clear_ijcam_to_knmslm_cache()
        canvas = np.zeros(self.CAM_SHAPE, dtype=np.float32)
        canvas[24:44, 34:54] = 1.0

        with subtests.test("repeated calls reuse one entry"):
            h = FeedbackHologram(
                (32, 32), target_ij=canvas, cameraslm=_StubCameraSLM(self.CAM_SHAPE)
            )
            n = len(_IJCAM_TO_KNMSLM_CACHE)
            for _ in range(3):
                h.set_target(canvas)
            assert len(_IJCAM_TO_KNMSLM_CACHE) == n

        with subtests.test("evicts past capacity"):
            for size in range(8, 8 + 2 * (_IJCAM_TO_KNMSLM_CACHE.maxsize + 4), 2):
                FeedbackHologram(
                    (size, size), target_ij=canvas, cameraslm=_StubCameraSLM(self.CAM_SHAPE)
                )
            assert len(_IJCAM_TO_KNMSLM_CACHE) <= _IJCAM_TO_KNMSLM_CACHE.maxsize

        with subtests.test("clear releases everything"):
            clear_ijcam_to_knmslm_cache()
            assert len(_IJCAM_TO_KNMSLM_CACHE) == 0

    def test_measure(self, subtests):
        """`measure("knm")` must land on the same knm grid whether the hologram was given a
        full-frame target or a sub-image, else `img_knm` and `target` disagree pixel for
        pixel and every feedback statistic is computed against a shifted image."""
        (full, sub, canvas, _) = self._pair()
        frame = np.nan_to_num(canvas, nan=0.0).astype(np.float32)

        # Computational feedback keeps `measure` on its "img_ij already present" branch.
        for h in (full, sub):
            h.flags.update({"feedback": "computational", "stat_groups": []})
            h.img_ij, h.img_knm = frame, None
            h.measure("knm")

        with subtests.test("the measurement is actually populated"):
            assert np.count_nonzero(np.nan_to_num(_np(sub.img_knm))) > 100

        with subtests.test("img_knm agrees exactly at order=0, the order set_target uses"):
            windows = [
                h.ijcam_to_knmslm(
                    np.square(h._roi_window(frame)), roi=h.target_ij_roi, order=0
                )
                for h in (full, sub)
            ]
            assert np.array_equal(_np(windows[0]), _np(windows[1]), equal_nan=True)

        with subtests.test("img_knm agrees to prefilter tolerance at the cubic default"):
            # Cropping the input perturbs cubic prefiltering everywhere; a shift would be O(1).
            assert np.allclose(
                _np(full.img_knm), _np(sub.img_knm), atol=2e-3, rtol=0, equal_nan=True
            )

    def test_measure_freshness(self, simulated_system_factory, subtests):
        """A changed phase drops the cached frame, and the projection policy still rules."""
        fs = simulated_system_factory("matched")
        install_ground_truth_calibration(fs)
        h = SpotHologram.make_rectangular_array(
            fs.slm.shape, (2, 2), (20, 20), basis="knm", cameraslm=fs
        )
        h.optimize(method="GS", maxiter=3, verbose=False, feedback="computational")

        counts = {"write": 0, "grab": 0}
        (set_phase, get_image) = (fs.slm.set_phase, fs.cam.get_image)
        fs.slm.set_phase = lambda p, *a, **k: (
            counts.update(write=counts["write"] + 1), set_phase(p, *a, **k))[1]
        fs.cam.get_image = lambda *a, **k: (
            counts.update(grab=counts["grab"] + 1), get_image(*a, **k))[1]
        try:
            with subtests.test("computational feedback leaves the SLM to its owner"):
                h.measure(basis="ij")
                assert counts["write"] == 0
                assert counts["grab"] == 1

            with subtests.test("a further optimize drops the frame, so the next grab is fresh"):
                h.optimize(method="GS", maxiter=1, verbose=False, feedback="computational")
                assert h.img_ij is None
                h.measure(basis="ij")
                assert counts["grab"] == 2

            with subtests.test("an unchanged phase reuses the cached frame"):
                h.measure(basis="ij")
                assert counts["grab"] == 2

            with subtests.test("an unrecognized basis raises before touching the hardware"):
                h.img_ij = None
                with pytest.raises(ValueError, match="Unrecognized measurement basis"):
                    h.measure(basis="bogus")
                assert counts["grab"] == 2
        finally:
            (fs.slm.set_phase, fs.cam.get_image) = (set_phase, get_image)
