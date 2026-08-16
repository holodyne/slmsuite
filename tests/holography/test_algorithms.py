"""
Unit tests for slmsuite.holography.algorithms module.
"""
import pytest
import numpy as np
import logging
import copy

from slmsuite.holography.algorithms import *
from slmsuite.holography import analysis
from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.holography.toolbox import convert_vector, format_vectors
from slmsuite.holography.toolbox.phase import blaze
import matplotlib.pyplot as plt

# Module-level logger for test output
logger = logging.getLogger(__name__)

class TestHologram:
    """Tests for Hologram class."""

    def test_hologram_construction(self):
        """Test the primitives for hologram formation."""
        slm_shape = (256, 256)
        shape = (512, 512)

        random_phase = np.random.uniform(0, 2 * np.pi, slm_shape).astype(np.float32)
        random_amplitude = np.random.uniform(0, 1, slm_shape).astype(np.float32) + 1e-2

        target = np.zeros(shape, dtype=np.float32)
        hologram = Hologram(
            target=target,
            slm_shape=slm_shape,
            phase=random_phase,
            amp=random_amplitude
        )

        # Check shape conventions
        assert hologram.slm_shape == slm_shape
        assert hologram.shape == shape

        # Check dtype conversions
        assert hologram.dtype == np.float32
        assert hologram.dtype_complex == np.complex64

        # Check initial conditions
        phase_diff = hologram.get_phase() - random_phase
        assert np.allclose(phase_diff, phase_diff.flat[0])
        amp_ratio = hologram.get_amp() / random_amplitude
        assert np.allclose(amp_ratio, amp_ratio.flat[0])

    def test_remove_vortices(self):
        # Flat-top target, which GS fills with vortices.
        target = np.zeros((128, 128), dtype=np.float32)
        target[48:80, 48:80] = 1
        hologram = Hologram(target=target, slm_shape=(64, 64))
        hologram.optimize(method="GS", maxiter=5, verbose=False)

        mask = target > 0
        vortices_before = np.count_nonzero(analysis.image_vortices(hologram.phase_ff)[mask])
        assert vortices_before > 0

        figures_before = plt.get_fignums()
        hologram.remove_vortices()
        vortices_after = np.count_nonzero(analysis.image_vortices(hologram.phase_ff)[mask])

        # Removal happens without plotting, and actually removes vortices.
        assert plt.get_fignums() == figures_before
        assert vortices_after < vortices_before

    def test_remove_vortices_callback(self):
        # Used as the docstring documents it, the removal has to change the
        # optimization, not just phase_ff. The loop recomputes phase_ff from the
        # farfield after the callback returns, so a removal that does not reach
        # the farfield is discarded in the same iteration.
        target = np.zeros((128, 128), dtype=np.float32)
        target[40:88, 40:88] = 1
        mask = target > 0

        def remove_vortices_callback(holo):
            if holo.iter % 5 == 4:
                holo.remove_vortices()

        # Same start for both arms, so the only difference is the callback.
        initial_phase = np.random.default_rng(0).uniform(0, 2 * np.pi, (64, 64))

        def run(callback):
            hologram = Hologram(target=target, slm_shape=(64, 64))
            hologram.reset_phase(custom_phase=initial_phase)
            hologram.optimize(method="GS", maxiter=20, verbose=False, callback=callback)
            return np.count_nonzero(analysis.image_vortices(hologram.phase_ff)[mask])

        without = run(None)
        with_callback = run(remove_vortices_callback)

        assert without > 0
        assert with_callback < without

    @pytest.mark.parametrize("method", ["GS", "WGS-Leonardo", "WGS-Kim", "WGS-Nogrette"])
    def test_gs_validity(self, random_seed, method):

        # Create a single far-field spot
        target = np.zeros((64, 64))

        rng = np.random.default_rng(random_seed)
        test_point = (rng.integers(0, 64), rng.integers(0, 64))
        logger.info(f'GS Convergence Test Point: {test_point}')
        target[test_point] = 1
        hologram = Hologram(target=target)

        hologram.optimize(method=method, maxiter=20, verbose=False, stat_groups=["computational"])

        # Check that output matches the expected grating
        slm = SimulatedSLM(hologram.target.shape)
        kxy = convert_vector(format_vectors(test_point[::-1]), "knm","norm", hardware=slm)
        blaze_phase = copy.deepcopy(slm.set_phase(blaze(slm,kxy)))
        holo_phase = copy.deepcopy(slm.set_phase(hologram.get_phase()))
        phase_err = holo_phase - blaze_phase
        rel_err = np.amax(np.abs(phase_err - phase_err.flat[0])) / (2*np.pi)

        # Comparison plot
        fig, axs = plt.subplots(2, 2, constrained_layout=True)
        hologram.plot_farfield(axs=axs[0])
        axs[0,1].cla()
        axs[0,1].imshow(phase_err)
        axs[0,1].set_title('Phase Error')
        slm.plot(phase=blaze_phase, title='Blaze Phase', ax=axs[1,0], cbar=False)
        slm.plot(phase=holo_phase, title='Hologram Phase', ax=axs[1,1], cbar=False)
        fig.suptitle(f'{method} | Relative Error: {rel_err:.2e}', fontsize=12)
        plt.show()

        assert np.allclose(phase_err, phase_err.flat[0], rtol=0.1, atol=0.1)

    @pytest.mark.parametrize("method", ["GS", "WGS-Leonardo", "WGS-Kim", "WGS-Nogrette"])
    def test_gs_convergence(self, random_seed, method):

        # Create a single far-field spot
        target = np.zeros((64, 64))

        rng = np.random.default_rng(random_seed)
        for i in range(20):
            test_point = (rng.integers(0, 64), rng.integers(0, 64))
            logger.info(f'Adding GS test point at: {test_point}')
            target[test_point] = 1
        hologram = Hologram(target=target)

        hologram.optimize(method=method, maxiter=20, verbose=False, stat_groups=["computational"])
        stats = hologram.stats["stats"]["computational"]
        hologram.plot_stats()

        # Comparison plot - show target, result...
        fig, axs = plt.subplots(2, 2, constrained_layout=True)
        hologram.plot_farfield(source=hologram.target,axs=axs[0])
        hologram.plot_farfield(axs=axs[1])
        fig.suptitle(f'{method} | Relative Error: {stats["std_err"][-1]:.2e}', fontsize=12)
        plt.show()

        # Check that efficiency improves
        assert stats["efficiency"][-1] >= stats["efficiency"][0]

        # Check that efficiency converges
        recent_efficiencies = stats["efficiency"][-5:]
        assert np.std(recent_efficiencies) < 0.05

        # Check that error decreases
        if method != "GS": # Basic GS may have non-monotonic error
            assert stats["std_err"][-1] <= stats["std_err"][1]

    @pytest.mark.parametrize("method", ["GS", "WGS-Leonardo", "WGS-Kim", "WGS-Nogrette"])
    def test_gs_speed(self, random_seed, method, benchmark):
        """CPU speed benchmark for GS algorithms (no stats overhead)."""
        target = np.zeros((1024, 1024))

        rng = np.random.default_rng(random_seed)
        for i in range(20):
            test_point = (rng.integers(0, 1024), rng.integers(0, 1024))
            target[test_point] = 1
        hologram = Hologram(target=target)
        benchmark(hologram.optimize, method=method, maxiter=20, verbose=False, stat_groups=[])

    @pytest.mark.gpu
    @pytest.mark.parametrize("method", ["GS", "WGS-Leonardo", "WGS-Kim", "WGS-Nogrette"])
    def test_gs_speed_gpu(self, random_seed, method, benchmark, has_cupy):
        """GPU speed benchmark for GS algorithms (no stats overhead)."""
        import cupy as cp
        target = cp.zeros((1024, 1024))

        rng = np.random.default_rng(random_seed)
        for i in range(20):
            test_point = (rng.integers(0, 1024), rng.integers(0, 1024))
            target[test_point] = 1
        hologram = Hologram(target=target)
        benchmark(hologram.optimize, method=method, maxiter=20, verbose=False, stat_groups=[])
