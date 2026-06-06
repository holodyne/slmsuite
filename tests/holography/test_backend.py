"""
Tests for the array-backend abstraction (:mod:`slmsuite.misc.backend`) and the
PyTorch-differentiable paths it enables on simulated hardware.

These tests run on CPU torch in float64 (for ``gradcheck``) and do not require a GPU.
The default numpy/cupy paths are also exercised to guard against regressions.
"""
import subprocess
import sys
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from slmsuite.misc import backend
from slmsuite.holography.algorithms import Hologram
from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.hardware.cameras.simulated import SimulatedCamera


# ---------------------------------------------------------------------------
# Regression: cupy <-> torch cuBLAS load-order (Windows)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_cublas_loadorder_regression():
    """Importing cupy before slmsuite (which imports torch) must not break cupy's cuBLAS.

    The historical failure: with cupy's cuBLAS loaded first and torch's loaded afterwards, cupy
    ``gemmEx`` succeeded exactly once then returned CUBLAS_STATUS_INVALID_VALUE on every call.
    backend.py warms cupy's cuBLAS before importing torch to avoid this. This runs in a fresh
    subprocess to reproduce the exact import order (the conflict is process-global and one-shot).
    """
    cp = pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for the cuBLAS load-order regression.")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No cupy-visible GPU.")
    except Exception:
        pytest.skip("cupy GPU unavailable.")

    code = (
        "import cupy as cp, numpy as np\n"   # cupy module loaded FIRST (the historically-bad order)
        "from slmsuite.misc import backend\n"# warms cupy cuBLAS, then imports torch
        "import torch\n"
        "ok = 0\n"
        "for _ in range(8):\n"
        "    a = cp.asarray(np.random.rand(6, 4096))\n"
        "    try:\n"
        "        c = a @ a.T; cp.cuda.runtime.deviceSynchronize(); ok += 1\n"
        "    except Exception:\n"
        "        pass\n"
        "print('GEMMS_OK', ok)\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, f"subprocess failed:\n{res.stderr}"
    assert "GEMMS_OK 8" in res.stdout, (
        "cupy cuBLAS broke after torch import (load-order regression):\n"
        f"stdout={res.stdout!r}\nstderr={res.stderr!r}"
    )


# ---------------------------------------------------------------------------
# Backend dispatch primitives
# ---------------------------------------------------------------------------
class TestBackendModule:
    def test_get_module_dispatch(self):
        assert backend.get_module(np.zeros(3)) is np
        assert backend.get_module(torch.zeros(3)) is torch
        assert backend.is_torch(torch.zeros(3))
        assert not backend.is_torch(np.zeros(3))

    def test_pad_matches_numpy(self):
        a = np.arange(12, dtype=np.float64).reshape(3, 4)
        pad_width = [(1, 2), (2, 1)]
        ref = np.pad(a, pad_width, mode="constant", constant_values=0)
        # numpy backend
        np.testing.assert_array_equal(backend.pad(a, pad_width), ref)
        # torch backend matches numpy semantics
        t = backend.pad(torch.from_numpy(a), pad_width)
        assert backend.is_torch(t)
        np.testing.assert_array_equal(t.numpy(), ref)

    def test_to_numpy(self):
        t = torch.arange(4, dtype=torch.float64, requires_grad=True)
        out = backend.to_numpy(t)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.arange(4))

    def test_to_backend_fast_path_identity(self):
        # A tensor already on the target device/dtype must be returned unchanged (no .to() copy).
        t = torch.arange(4, dtype=torch.float64)
        assert backend.to_backend(t, t) is t
        # A dtype mismatch must trigger a real (non-identity) conversion.
        other = torch.arange(4, dtype=torch.float32)
        converted = backend.to_backend(t, other)
        assert converted is not t and converted.dtype == torch.float32

    def test_to_backend_scalar_bypass(self):
        # Python native scalars must be returned directly as-is
        ref = torch.zeros(3)
        assert backend.to_backend(2.0, ref) == 2.0
        assert isinstance(backend.to_backend(2.0, ref), float)
        assert backend.to_backend(3 + 4j, ref) == 3 + 4j
        assert isinstance(backend.to_backend(3 + 4j, ref), complex)

        # NumPy scalars must be converted to native Python scalars
        assert backend.to_backend(np.float32(1.5), ref) == 1.5
        assert isinstance(backend.to_backend(np.float32(1.5), ref), float)
        assert backend.to_backend(np.complex64(1 + 2j), ref) == 1 + 2j
        assert isinstance(backend.to_backend(np.complex64(1 + 2j), ref), complex)


# ---------------------------------------------------------------------------
# Elementwise wrapper parity (numpy vs torch) + autograd safety
# ---------------------------------------------------------------------------
class TestWrapperParity:
    """numpy and torch must agree, and the torch path must stay autograd-safe (no out= leakage)."""

    # Positive, finite inputs so every op (incl. reciprocal/power) is well-defined.
    _x = np.array([0.2, 0.7, 1.3, 2.1], dtype=np.float64)
    _y = np.array([0.5, 1.1, 0.3, 1.9], dtype=np.float64)

    @pytest.mark.parametrize("name", ["exp", "tanh", "reciprocal", "abs", "conj", "angle", "sinc"])
    def test_unary_parity(self, name):
        f = getattr(backend, name)
        ref = f(self._x)
        got = backend.to_numpy(f(torch.from_numpy(self._x.copy())))
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("name", ["add", "subtract", "multiply", "divide", "power"])
    def test_binary_parity(self, name):
        f = getattr(backend, name)
        ref = f(self._x, self._y)
        got = backend.to_numpy(f(torch.from_numpy(self._x.copy()), torch.from_numpy(self._y.copy())))
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)

    def test_power_regression_torch_pow(self):
        # Ensure that power maps correctly to torch.pow (avoiding the nonexistent torch.power) and
        # preserves the autograd graph for fractional/negative exponents (e.g. the WGS-Leonardo case).
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, requires_grad=True)
        out = backend.power(a, -0.8)
        assert backend.is_torch(out) and out.requires_grad
        out.sum().backward()
        assert a.grad is not None and torch.isfinite(a.grad).all()

    @pytest.mark.parametrize("name", ["clip", "mod"])
    def test_clip_mod_drop_out_on_autograd(self, name):
        # Ensure that out= arguments are safely dropped for torch, avoiding graph-severing
        # or runtime errors on graph-carrying tensors.
        vals = np.array([0.2, 1.5, 3.0])
        a = torch.tensor(vals, dtype=torch.float64, requires_grad=True)
        if name == "clip":
            out = backend.clip(a, 0.5, 2.0)
            np.testing.assert_allclose(backend.to_numpy(out), np.clip(vals, 0.5, 2.0))
        else:
            out = backend.mod(a, 2.0)
            np.testing.assert_allclose(backend.to_numpy(out), np.mod(vals, 2.0))
        assert out.requires_grad  # graph preserved (no out= leakage)

    def test_unary_gradcheck(self):
        for name in ["exp", "tanh", "reciprocal", "abs"]:
            f = getattr(backend, name)
            x0 = torch.rand(5, dtype=torch.float64, requires_grad=True) + 0.5
            assert torch.autograd.gradcheck(lambda z: f(z), (x0,), eps=1e-6, atol=1e-5)

    def test_binary_gradcheck(self):
        for name in ["add", "subtract", "multiply", "divide", "power"]:
            f = getattr(backend, name)
            a0 = torch.rand(5, dtype=torch.float64, requires_grad=True) + 0.5
            b0 = torch.rand(5, dtype=torch.float64, requires_grad=True) + 0.5
            assert torch.autograd.gradcheck(lambda u, v: f(u, v), (a0, b0), eps=1e-6, atol=1e-5)

    def test_embed_and_scatter_gradcheck(self):
        # embed: centered pad must be differentiable wrt the source.
        src0 = torch.rand(4, 4, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda s: backend.embed(s, (8, 8), (4, 4)), (src0,), eps=1e-6, atol=1e-5
        )
        # scatter_update: out-of-place clone must preserve grad to the unselected entries.
        base = torch.arange(6, dtype=torch.float64, requires_grad=True)
        def f(z):
            return backend.scatter_update(z, (slice(0, 2),), z[2:4] * 2.0)
        assert torch.autograd.gradcheck(f, (base,), eps=1e-6, atol=1e-5)

    def test_where_replace_gradcheck(self):
        x0 = torch.linspace(-1, 1, 6, dtype=torch.float64, requires_grad=True)
        def f(z):
            return backend.where_replace(z, z > 0, 1.0)
        assert torch.autograd.gradcheck(f, (x0,), eps=1e-6, atol=1e-5)


# ---------------------------------------------------------------------------
# Explicit backend= / device= constructor selection
# ---------------------------------------------------------------------------
class TestExplicitBackend:
    """Hologram(backend=..., device=...) places the stored arrays on the chosen backend/device.
    The default selection is unchanged (cupy if available, else numpy)."""

    @staticmethod
    def _target():
        tgt = np.zeros((32, 32), np.float32)
        tgt[10:14, 18:22] = 1.0
        return tgt

    def test_resolve_backend(self):
        assert backend.resolve_backend(None)[0] is backend.cp     # default: cupy if avail, else numpy
        assert backend.resolve_backend("auto")[0] is backend.cp
        assert backend.resolve_backend("numpy")[0] is np
        mod, dev = backend.resolve_backend("torch", "cpu")
        assert mod is torch and dev == torch.device("cpu")
        with pytest.raises(ValueError):
            backend.resolve_backend("bogus")

    def test_default_backend_not_torch(self):
        h = Hologram(target=(16, 16), dtype=np.float32)
        assert not backend.is_torch(h.phase)  # default is numpy/cupy, never torch

    def test_numpy_backend_forces_numpy_through_optimize(self):
        # Forces CPU numpy even when cupy is installed; full WGS + host getters must work.
        h = Hologram(target=self._target(), dtype=np.float32, backend="numpy")
        assert backend.get_module(h.phase) is np
        assert backend.get_module(h.nearfield) is np and backend.get_module(h.weights) is np
        h.optimize("WGS-Leonardo", maxiter=4, verbose=False)
        assert backend.get_module(h.phase) is np          # stayed numpy across optimize
        assert isinstance(h.get_phase(), np.ndarray)
        assert isinstance(h.get_farfield(get=True), np.ndarray)
        assert isinstance(h.get_weights(), np.ndarray)

    def test_torch_backend_places_and_survives_reset(self):
        h = Hologram(target=(32, 32), dtype=np.float64, backend="torch", device="cpu")
        for name in ("phase", "nearfield", "weights", "target"):
            assert backend.is_torch(getattr(h, name)), name
        assert h.phase.device.type == "cpu"
        ff = h.get_farfield(get=False)
        assert backend.is_torch(ff)
        # Host getters still bridge to numpy.
        assert isinstance(h.get_phase(), np.ndarray)
        # reset() rebuilds in the default backend then re-places onto torch.
        h.reset()
        assert backend.is_torch(h.phase) and backend.is_torch(h.weights)

    def test_torch_backend_farfield_differentiable(self):
        h = Hologram(target=(16, 16), dtype=np.float64, backend="torch", device="cpu")
        h.phase.requires_grad_(True)
        ff = h.get_farfield(get=False)
        (ff.real ** 2 + ff.imag ** 2).sum().backward()
        assert h.phase.grad is not None and torch.isfinite(h.phase.grad).all()

    def test_bad_backend_raises(self):
        with pytest.raises(ValueError):
            Hologram(target=(8, 8), backend="bogus")


# ---------------------------------------------------------------------------
# Differentiable propagation (Hologram.get_farfield / _nearfield2farfield)
# ---------------------------------------------------------------------------
class TestDifferentiablePropagation:
    @staticmethod
    def _torch_holo(N):
        holo = Hologram(target=(N, N), slm_shape=(N, N), dtype=np.float64)
        amp = torch.rand(N, N, dtype=torch.float64)
        amp = amp / torch.sqrt(torch.sum(amp ** 2))
        holo.amp = amp
        holo.amp_ff = None
        holo.phase_ff = None
        holo.propagation_kernel = None
        return holo

    def test_get_farfield_returns_torch(self):
        holo = self._torch_holo(8)
        holo.phase = torch.randn(8, 8, dtype=torch.float64)
        ff = holo.get_farfield(get=False)
        assert backend.is_torch(ff)
        assert ff.shape == (8, 8)

    def test_gradcheck_farfield_intensity(self):
        holo = self._torch_holo(8)

        def f(phase):
            holo.phase = phase
            ff = holo.get_farfield(get=False)
            return ff.real ** 2 + ff.imag ** 2

        phase0 = torch.randn(8, 8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(f, (phase0,), eps=1e-6, atol=1e-5, rtol=1e-3)

    def test_numpy_torch_farfield_equivalence(self):
        N = 8
        amp = np.random.rand(N, N).astype(np.float64)
        amp /= np.sqrt(np.sum(amp ** 2))
        phase = np.random.uniform(-np.pi, np.pi, (N, N)).astype(np.float64)

        holo_np = Hologram(target=(N, N), slm_shape=(N, N), dtype=np.float64)
        holo_np.amp = amp
        holo_np.phase = phase
        holo_np.amp_ff = None
        holo_np.phase_ff = None
        holo_np.propagation_kernel = None
        I_np = np.abs(holo_np.get_farfield(get=False)) ** 2

        holo_t = self._torch_holo(N)
        holo_t.amp = torch.from_numpy(amp.copy())
        holo_t.phase = torch.from_numpy(phase.copy())
        ff_t = holo_t.get_farfield(get=False)
        I_t = (ff_t.real ** 2 + ff_t.imag ** 2).numpy()

        assert np.max(np.abs(I_np - I_t)) < 1e-10


# ---------------------------------------------------------------------------
# Default (numpy/cupy) path regressions + the unified optimize_cg
# ---------------------------------------------------------------------------
class TestDefaultPathAndCG:
    def test_gs_runs_and_returns_array(self):
        holo = Hologram(target=(64, 64), dtype=np.float32)
        holo.optimize("GS", maxiter=5, verbose=False)
        assert holo.iter == 5
        ff = holo.get_farfield(get=True)
        assert isinstance(ff, np.ndarray)  # host array (numpy or cupy.get())
        assert not backend.is_torch(holo.phase)

    def test_optimize_cg_reduces_loss(self):
        # The unified, autograd-correct CG path: a real target's loss must decrease.
        tgt = np.zeros((64, 64), np.float32)
        tgt[20:26, 38:44] = 1.0
        holo = Hologram(target=tgt, dtype=np.float32)
        losses = []
        holo.optimize(
            "CG", maxiter=40, verbose=False,
            callback=lambda h: losses.append(h.flags.get("loss_result")) or False,
        )
        assert losses[-1] < 0.5 * losses[0]
        # Post-CG state is restored to the default backend (not torch).
        assert not backend.is_torch(holo.phase)
        assert not backend.is_torch(holo.nearfield)


# ---------------------------------------------------------------------------
# Differentiable simulated hardware (backend-transparent set_phase / get_image)
# ---------------------------------------------------------------------------
class TestDifferentiableSimulatedHardware:
    @staticmethod
    def _slm_cam(N, return_complex=False):
        slm = SimulatedSLM((N, N))
        cam = SimulatedCamera(slm, return_complex=return_complex)  # identity affine, non-interp
        return slm, cam

    def test_get_image_returns_torch_and_matches_reference(self):
        N = 16
        slm, cam = self._slm_cam(N)
        amp_sim = np.asarray(slm.source["amplitude_sim"])
        phase_sim = np.asarray(slm.source["phase_sim"])

        phase = torch.randn(N, N, dtype=torch.float64)
        slm.set_phase(phase)
        img = cam.get_image()
        assert backend.is_torch(img)

        nf = amp_sim * np.exp(1j * (phase.numpy() + phase_sim))
        ff = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(nf), norm="ortho"))
        ref = (np.abs(ff) ** 2) * (cam.exposure_s * cam.gain)
        assert np.max(np.abs(img.detach().numpy() - ref)) < 1e-10

    def test_gradcheck_camera(self):
        N = 12
        slm, cam = self._slm_cam(N)

        def f(phase):
            slm.set_phase(phase)
            return cam.get_image()

        phase0 = torch.randn(N, N, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(f, (phase0,), eps=1e-6, atol=1e-5, rtol=1e-3)

    def test_return_complex_fields(self):
        N = 16
        slm, cam = self._slm_cam(N, return_complex=True)
        phase = torch.randn(N, N, dtype=torch.float64, requires_grad=True)
        slm.set_phase(phase)
        img = cam.get_image()

        assert cam.field_far is not None and cam.field_pupil is not None
        # |U|^2 * exposure * gain reproduces the intensity image.
        recon = (cam.field_far.real ** 2 + cam.field_far.imag ** 2) * (cam.exposure_s * cam.gain)
        assert torch.allclose(recon, img)
        # Gradients flow through the exposed complex field too.
        cam.field_far.abs().sum().backward()
        assert phase.grad is not None and torch.isfinite(phase.grad).all()

    def test_interpolated_camera_is_differentiable(self):
        # With an affine (M, b) the camera takes the interpolated path
        # (backend.map_coordinates); it must keep the image on torch and differentiable wrt
        # the SLM phase. A near-identity affine keeps the far-field within camera k-space.
        N = 16
        slm = SimulatedSLM((N, N))
        M = np.eye(2)
        b = np.zeros(2)
        cam = SimulatedCamera(slm, resolution=(N, N), M=M, b=b)
        assert cam._interpolate
        phase = torch.randn(N, N, dtype=torch.float64, requires_grad=True)
        slm.set_phase(phase)
        img = cam.get_image()
        assert backend.is_torch(img)
        img.sum().backward()
        assert phase.grad is not None and torch.isfinite(phase.grad).all()


# ---------------------------------------------------------------------------
# Capability gate: only compute-only SLMs accept a torch phase
# ---------------------------------------------------------------------------
class TestTorchPhaseCapability:
    def test_simulated_slm_supports_torch_phase(self):
        slm = SimulatedSLM((16, 16))
        assert slm._supports_torch_phase is True
        slm.set_phase(torch.randn(16, 16, dtype=torch.float64))
        assert backend.is_torch(slm.phase)

    def test_unsupported_slm_rejects_torch_phase(self):
        # Emulate a physical SLM, which cannot display an analog torch phase.
        slm = SimulatedSLM((16, 16))
        slm._supports_torch_phase = False
        with pytest.raises(TypeError, match="cannot accept a torch.Tensor phase"):
            slm.set_phase(torch.randn(16, 16, dtype=torch.float64))


# ---------------------------------------------------------------------------
# Differentiable spot holograms (CompressedSpotHologram & SpotHologram)
# ---------------------------------------------------------------------------
class TestDifferentiableSpots:
    def test_compressed_spot_hologram_autograd(self):
        from slmsuite.holography.algorithms import CompressedSpotHologram
        from slmsuite.hardware.cameraslms import FourierSLM
        
        N = 16
        slm = SimulatedSLM((N, N))
        cam = SimulatedCamera(slm)
        fs = FourierSLM(cam, slm)
        
        # Calibrate fs to enable cameraslm coordinates
        fs.fourier_calibrate_analytic(np.eye(2) * 20.0, np.array([[8.0], [8.0]]))
        
        # 4 spots in kxy basis
        spot_vectors = np.array([[-0.03125, 0.03125, -0.03125, 0.03125], [-0.03125, -0.03125, 0.03125, 0.03125]], dtype=np.float64)
        
        holo = CompressedSpotHologram(
            spot_vectors=spot_vectors,
            basis="kxy",
            cameraslm=fs,
            dtype=np.float64
        )
        
        # Ensure we use cupy path fallback since cuda RawKernel is not compiled for CPU torch
        assert holo.cuda is False
        
        try:
            import cupy as cp
        except ImportError:
            cp = np
        device = 'cuda' if (cp is not np and torch.cuda.is_available()) else 'cpu'
        
        # Test out-of-place autograd-preserving path
        phase = torch.randn(N, N, dtype=torch.float64, device=device, requires_grad=True)
        holo.phase = phase
        
        ff = holo._nearfield2farfield()
        assert backend.is_torch(ff)
        assert ff.requires_grad
        assert ff.shape == (4,)
        
        # Compute some simple loss and backpropagate
        loss = torch.sum(ff.real ** 2 + ff.imag ** 2)
        loss.backward()
        
        assert phase.grad is not None
        assert torch.any(phase.grad != 0)
        
        # Test _farfield2nearfield under torch
        holo.farfield = torch.randn(4, dtype=torch.complex128, device=device, requires_grad=True)
        holo._farfield2nearfield()
        assert backend.is_torch(holo.nearfield)
        assert holo.nearfield.requires_grad
        assert holo.nearfield.shape == (N, N)
        
        # Test _update_weights
        holo.amp_ff = torch.rand(4, dtype=torch.float64, device=device, requires_grad=True)
        holo.target = np.ones(4, dtype=np.float64)
        holo.weights = np.ones(4, dtype=np.float64)
        if cp is not np:
            holo.target = cp.array(holo.target)
            holo.weights = cp.array(holo.weights)
        
        holo.flags["feedback"] = "computational_spot"
        holo.flags["method"] = "wgs-multiplicative"
        holo._update_weights()
        assert backend.is_torch(holo.weights)

    def test_compressed_spot_hologram_large_batch(self):
        # Verify batched processing by overriding N_BATCH_MAX to 2
        from slmsuite.holography.algorithms import CompressedSpotHologram
        from slmsuite.hardware.cameraslms import FourierSLM
        from slmsuite.holography.algorithms import _spots
        
        old_batch_max = _spots.N_BATCH_MAX
        _spots.N_BATCH_MAX = 2
        try:
            N = 16
            slm = SimulatedSLM((N, N))
            cam = SimulatedCamera(slm)
            fs = FourierSLM(cam, slm)
            fs.fourier_calibrate_analytic(np.eye(2) * 20.0, np.array([[8.0], [8.0]]))
            
            spot_vectors = np.array([[-0.03125, 0.03125, -0.03125, 0.03125], [-0.03125, -0.03125, 0.03125, 0.03125]], dtype=np.float64)
            
            holo = CompressedSpotHologram(
                spot_vectors=spot_vectors,
                basis="kxy",
                cameraslm=fs,
                dtype=np.float64
            )
            
            try:
                import cupy as cp
            except ImportError:
                cp = np
            device = 'cuda' if (cp is not np and torch.cuda.is_available()) else 'cpu'
            
            phase = torch.randn(N, N, dtype=torch.float64, device=device, requires_grad=True)
            holo.phase = phase
            
            ff = holo._nearfield2farfield()
            assert ff.shape == (4,)
            assert ff.requires_grad
            
            loss = torch.sum(ff.real ** 2 + ff.imag ** 2)
            loss.backward()
            assert phase.grad is not None
            
            # Test batched _farfield2nearfield
            holo.farfield = torch.randn(4, dtype=torch.complex128, device=device, requires_grad=True)
            holo._farfield2nearfield()
            assert backend.is_torch(holo.nearfield)
            assert holo.nearfield.requires_grad
            
        finally:
            _spots.N_BATCH_MAX = old_batch_max

    def test_spot_hologram_autograd(self):
        from slmsuite.holography.algorithms import SpotHologram
        from slmsuite.hardware.cameraslms import FourierSLM
        
        N = 16
        slm = SimulatedSLM((N, N))
        cam = SimulatedCamera(slm)
        fs = FourierSLM(cam, slm)
        fs.fourier_calibrate_analytic(np.eye(2) * 20.0, np.array([[8.0], [8.0]]))
        
        spot_vectors = np.array([[-0.03125, 0.03125, -0.03125, 0.03125], [-0.03125, -0.03125, 0.03125, 0.03125]], dtype=np.float64)
        
        holo = SpotHologram(
            shape=(N, N),
            spot_vectors=spot_vectors,
            basis="kxy",
            cameraslm=fs,
            dtype=np.float64
        )
        
        try:
            import cupy as cp
        except ImportError:
            cp = np
        device = 'cuda' if (cp is not np and torch.cuda.is_available()) else 'cpu'
        
        # Test computational spot weighting path
        holo.amp_ff = torch.randn(N, N, dtype=torch.float64, device=device, requires_grad=True)
        holo.target = np.ones((N, N), dtype=np.float64)
        holo.weights = np.ones((N, N), dtype=np.float64)
        if cp is not np:
            holo.target = cp.array(holo.target)
            holo.weights = cp.array(holo.weights)
        
        holo.spot_integration_width_knm = 3
        holo.flags["feedback"] = "computational_spot"
        holo.flags["method"] = "wgs-multiplicative"
        
        holo._update_weights()
        assert backend.is_torch(holo.weights)
        assert holo.weights.requires_grad
        
        # Check that backward passes through the updated weights successfully back to amp_ff
        loss = torch.sum(holo.weights)
        loss.backward()
        assert holo.amp_ff.grad is not None
        assert torch.any(holo.amp_ff.grad != 0)


# ---------------------------------------------------------------------------
# WGS weighting under torch (exercises backend.power -> torch.pow)
# ---------------------------------------------------------------------------
class TestWGSWeightingTorch:
    """WGS-Leonardo / WGS-Kim weighting routes through ``backend.power``; under torch this must
    dispatch to ``torch.pow`` (not the nonexistent ``torch.power``) and stay differentiable."""

    @pytest.mark.parametrize("method", ["WGS-Leonardo", "WGS-Kim"])
    def test_weighting_power_path_backprops(self, method):
        holo = Hologram(target=(16, 16), dtype=np.float64)
        holo.flags["method"] = method
        holo.flags["feedback_exponent"] = 0.8

        feedback = (torch.rand(8, dtype=torch.float64) + 0.5).requires_grad_(True)  # leaf
        weight = torch.ones(8, dtype=torch.float64)
        target = torch.ones(8, dtype=torch.float64)

        updated = holo._update_weights_generic(weight, feedback, target)
        assert backend.is_torch(updated) and updated.requires_grad
        updated.sum().backward()
        assert feedback.grad is not None and torch.isfinite(feedback.grad).all()


# ---------------------------------------------------------------------------
# Phase 2/3 additions: new elementwise/logical/constructor wrappers
# ---------------------------------------------------------------------------
class TestNewWrapperParity:
    """numpy<->torch parity for the wrappers added during the backend unification."""

    _x = np.array([0.2, -0.7, 1.3, np.nan], dtype=np.float64)

    @pytest.mark.parametrize("name", ["square", "sqrt"])
    def test_new_unary_parity(self, name):
        f = getattr(backend, name)
        vals = np.array([0.2, 0.7, 1.3, 2.1], dtype=np.float64)
        ref = f(vals)
        got = backend.to_numpy(f(torch.from_numpy(vals.copy())))
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)

    def test_isnan_logical_sum_any(self):
        x = self._x
        xt = torch.from_numpy(x.copy())
        np.testing.assert_array_equal(backend.to_numpy(backend.isnan(xt)), np.isnan(x))
        np.testing.assert_array_equal(
            backend.to_numpy(backend.logical_not(backend.isnan(xt))), ~np.isnan(x)
        )
        a = np.array([True, False, True]); b = np.array([False, False, True])
        np.testing.assert_array_equal(
            backend.to_numpy(backend.logical_or(torch.from_numpy(a), torch.from_numpy(b))),
            np.logical_or(a, b),
        )
        ints = np.array([1, 2, 3, 4])
        assert int(backend.sum(torch.from_numpy(ints))) == int(ints.sum())
        assert bool(backend.any(torch.from_numpy(np.array([False, True])))) is True

    def test_linspace_meshgrid_constructors(self):
        # Witness-array form selects backend/device; meshgrid defaults to "xy" like numpy/cupy.
        ref_x = np.linspace(-1, 1, 7)
        xt = backend.linspace(-1, 1, 7, like=torch.zeros(1, dtype=torch.float64))
        assert backend.is_torch(xt)
        np.testing.assert_allclose(backend.to_numpy(xt), ref_x, atol=1e-12)
        gx, gy = backend.meshgrid(xt, xt)
        gxn, gyn = np.meshgrid(ref_x, ref_x)
        np.testing.assert_allclose(backend.to_numpy(gx), gxn, atol=1e-12)
        np.testing.assert_allclose(backend.to_numpy(gy), gyn, atol=1e-12)

    def test_linspace_spec_form_and_torch_dtype_bool_int(self):
        # (module, device) spec form, plus _torch_dtype coverage of bool/int via backend.zeros.
        xt = backend.linspace(0, 1, 4, like=(backend.torch, torch.device("cpu")), dtype=np.float32)
        assert backend.is_torch(xt) and xt.dtype == torch.float32
        zb = backend.zeros((2, 2), backend.torch, bool, device=torch.device("cpu"))
        assert zb.dtype == torch.bool
        zi = backend.zeros((3,), backend.torch, np.int64, device=torch.device("cpu"))
        assert zi.dtype == torch.int64


# ---------------------------------------------------------------------------
# Affine / coordinate-map resampling (shared backend ops)
# ---------------------------------------------------------------------------
class TestAffineAndMapCoordinates:
    """backend.affine_transform / map_coordinates: numpy<->torch parity + autograd."""

    def _img(self):
        rng = np.random.default_rng(0)
        return rng.random((16, 20)).astype(np.float64)

    @pytest.mark.parametrize("cval", [0.0, float("nan")])
    def test_affine_transform_parity(self, cval):
        img = self._img()
        M = np.array([[1.1, 0.05], [0.0, 0.9]])
        b = np.array([1.5, -2.0])
        ref = backend.affine_transform(img, M, b, (16, 20), order=1, cval=cval)
        got = backend.to_numpy(
            backend.affine_transform(torch.from_numpy(img.copy()), M, b, (16, 20), order=1, cval=cval)
        )
        np.testing.assert_array_equal(np.isnan(ref), np.isnan(got))
        fin = ~np.isnan(ref)
        np.testing.assert_allclose(got[fin], ref[fin], rtol=1e-6, atol=1e-6)

    def test_affine_transform_complex_and_negative_strides(self):
        rng = np.random.default_rng(1)
        img = (rng.random((12, 14)) + 1j * rng.random((12, 14)))
        # np.flip yields negative-strided M/b, mirroring the feedback transform.
        M = np.flip(np.flip(np.array([[0.8, 0.0], [0.0, 1.2]]), axis=0), axis=1)
        b = np.flip(np.array([1.0, -1.0]))
        ref = backend.affine_transform(img, M, b, (12, 14), order=1, cval=0)
        got = backend.to_numpy(
            backend.affine_transform(torch.from_numpy(img.copy()), M, b, (12, 14), order=1, cval=0)
        )
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)

    def test_affine_transform_gradcheck(self):
        x0 = torch.rand(10, 12, dtype=torch.float64, requires_grad=True)
        M = np.array([[1.0, 0.1], [0.0, 1.0]]); b = np.array([0.5, -0.5])
        assert torch.autograd.gradcheck(
            lambda z: backend.affine_transform(z, M, b, (10, 12), order=1, cval=0),
            (x0,), eps=1e-6, atol=1e-5,
        )

    @pytest.mark.parametrize("order", [0, 1])
    def test_map_coordinates_parity(self, order):
        img = self._img()
        H, W = img.shape
        rng = np.random.default_rng(2)
        coords = np.stack([
            rng.uniform(-2, H + 2, size=(10, 11)),
            rng.uniform(-2, W + 2, size=(10, 11)),
        ])
        ref = backend.map_coordinates(img, coords, order=order, cval=0)
        got = backend.to_numpy(
            backend.map_coordinates(torch.from_numpy(img.copy()), torch.from_numpy(coords.copy()),
                                    order=order, cval=0)
        )
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)

    def test_map_coordinates_gradcheck(self):
        x0 = torch.rand(8, 9, dtype=torch.float64, requires_grad=True)
        rng = np.random.default_rng(3)
        coords = torch.from_numpy(np.stack([
            rng.uniform(0, 7, size=(6, 6)), rng.uniform(0, 8, size=(6, 6))
        ]))
        assert torch.autograd.gradcheck(
            lambda z: backend.map_coordinates(z, coords, order=1, cval=0),
            (x0,), eps=1e-6, atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Affine far-field transform on torch (previously NotImplementedError)
# ---------------------------------------------------------------------------
class TestAffineFarfieldTorch:
    def _holo(self, backend_name):
        target = np.zeros((32, 32)); target[16, 16] = 1.0; target[8, 20] = 1.0
        kwargs = {"backend": backend_name}
        if backend_name == "torch":
            kwargs["device"] = "cpu"
        return Hologram(target=target, **kwargs)

    def test_affine_farfield_runs_on_torch(self):
        affine = {"M": np.array([[1.0, 0.0], [0.0, 1.0]]), "b": np.array([2.0, -1.0])}
        ff = self._holo("torch").get_farfield(affine=affine, get=False)
        assert backend.is_torch(ff) and torch.is_complex(ff)
        assert tuple(ff.shape) == (32, 32)

    def test_affine_farfield_differentiable(self):
        holo = self._holo("torch")
        holo.phase = holo.phase.clone().requires_grad_(True)
        affine = {"M": np.array([[1.0, 0.0], [0.0, 1.0]]), "b": np.array([1.0, 0.0])}
        ff = holo.get_farfield(affine=affine, get=False)
        (ff.real ** 2 + ff.imag ** 2).sum().backward()
        assert holo.phase.grad is not None and bool((holo.phase.grad != 0).any())


# ---------------------------------------------------------------------------
# Backend speed benchmark (CuPy vs PyTorch CUDA performance capture)
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_backend_performance_recording(capsys):
    """Runs a non-failing speed test comparing CuPy and PyTorch CUDA and records results."""
    import time
    cp = pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for performance speed test.")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No cupy-visible GPU.")
    except Exception:
        pytest.skip("cupy GPU unavailable.")

    from slmsuite.holography.algorithms import SpotHologram

    slm_size = (1024, 1024)

    # 1. Warm up & Benchmark CuPy
    array_holo_cupy = SpotHologram.make_rectangular_array(
        slm_size,
        array_shape=(10, 10),
        array_pitch=(10, 10),
        basis='knm',
        backend='cupy'
    )
    # Warm up
    array_holo_cupy.optimize(method="GS", maxiter=2, stat_groups=[], verbose=False)
    cp.cuda.runtime.deviceSynchronize()

    # Benchmark
    t0 = time.time()
    array_holo_cupy.optimize(method="GS", maxiter=20, stat_groups=[], verbose=False)
    cp.cuda.runtime.deviceSynchronize()
    t_cupy = time.time() - t0
    cupy_ms = t_cupy * 1000 / 20

    del array_holo_cupy
    import gc
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()

    # 2. Warm up & Benchmark PyTorch
    array_holo_torch = SpotHologram.make_rectangular_array(
        slm_size,
        array_shape=(10, 10),
        array_pitch=(10, 10),
        basis='knm',
        backend='torch',
        device='cuda'
    )
    # Warm up
    array_holo_torch.optimize(method="GS", maxiter=2, stat_groups=[], verbose=False)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    array_holo_torch.optimize(method="GS", maxiter=20, stat_groups=[], verbose=False)
    torch.cuda.synchronize()
    t_torch = time.time() - t0
    torch_ms = t_torch * 1000 / 20

    del array_holo_torch
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()

    # Report results via pytest stdout
    print("\n================== BACKEND SPEED BENCHMARK ==================")
    print(f"CuPy Speed: {t_cupy:.4f} seconds ({cupy_ms:.2f} ms/iter)")
    print(f"PyTorch CUDA Speed: {t_torch:.4f} seconds ({torch_ms:.2f} ms/iter)")
    print(f"Overhead Ratio (Torch/CuPy): {torch_ms / cupy_ms:.2f}x")
    print("=============================================================")


