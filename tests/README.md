# slmsuite Testing Framework

This directory contains the unified testing framework for slmsuite, designed to work both with and without connected hardware.

## Quick Start

```bash
# Run all tests
pytest

# Run tests for specific module
pytest tests/holography/
pytest tests/hardware/
pytest tests/misc/

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/holography/test_algorithms.py

# Run specific test class or function
pytest tests/holography/test_algorithms.py::TestHologram::test_gs_converges
```

## Test Structure

The test directory structure mirrors s `slmsuite/`:

- `tests/hardware/` → `slmsuite/hardware/`
- `tests/holography/` → `slmsuite/holography/`
- `tests/misc/` → `slmsuite/misc/`
- etc

### No Hardware Required by Default
All tests use simulated hardware (SimulatedCamera, SimulatedSLM) and synthetic data. This allows:
- Fast test execution in CI/CD
- Development without physical hardware
- Consistent test results across environments

### Characteristics
- **Fast**: Most tests complete in <1 second (mark long tests with `@pytest.mark.slow`)
- **Isolated**: Each test is independent and doesn't require external state
- **No hardware?**: Defaults to SimulatedCamera/SimulatedSLM
- **Hardware-ready**: Some fixtures support real hardware via environment variables

## Fixtures

### Available Fixtures (from `conftest.py`)

The test fixtures are designed to work with **any** SLM or Camera subclass, not just simulated hardware.

#### Session Fixtures

```python
@pytest.fixture(scope="session")
def random_seed():
    """Generate and set a random seed for the session (logged for reproducibility)."""

@pytest.fixture(scope="session")
def has_cupy():
    """Check if CuPy is available for GPU tests."""
```

#### Hardware Fixtures

```python
@pytest.fixture
def slm():
    """
    Provides an SLM instance for testing.
    By default: SimulatedSLM (1920x1080, 8-bit, 780nm)
    Override via SLMSUITE_TEST_SLM_CLASS / SLMSUITE_TEST_SLM_ARGS env vars.
    """

@pytest.fixture
def slm_small():
    """Small-resolution SLM (128x128) for faster tests."""

@pytest.fixture
def camera():
    """
    Provides a Camera instance for testing.
    By default: SimulatedCamera (512x512, 8-bit)
    Override via SLMSUITE_TEST_CAMERA_CLASS / SLMSUITE_TEST_CAMERA_ARGS env vars.
    """

@pytest.fixture
def camera_small():
    """Small-resolution Camera (128x128) for faster tests."""

@pytest.fixture
def fourierslm():
    """Provides a FourierSLM (camera + SLM) instance for testing."""

@pytest.fixture
def fourierslm_small():
    """Small-resolution FourierSLM (128x128) for faster tests."""
```

#### Utility Fixtures

```python
@pytest.fixture
def temp_dir():
    """Provides a temporary directory for test file I/O."""

@pytest.fixture
def mpl_test():
    """Per-test matplotlib fixture with automatic figure cleanup."""

@pytest.fixture(autouse=True)
def test_logger():
    """Per-test logger (auto-applied). Logger name: {module}.{class}.{function}."""
```

### Using Fixtures

#### Basic Usage (Simulated Hardware - Default)

```python
def test_example(camera, temp_dir):
    """Test using camera and temp directory."""
    image = camera.get_image()
    assert image.shape == (512, 512)

    # Save to temp directory
    np.save(f"{temp_dir}/image.npy", image)
```

#### Testing with Real Hardware

To run tests with real hardware instead of simulated:

```bash
# Linux/Mac
export SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.screenmirrored.ScreenMirrored
export SLMSUITE_TEST_SLM_ARGS='{"display_number": 1}'
export SLMSUITE_TEST_CAMERA_CLASS=slmsuite.hardware.cameras.thorlabs.ThorCam
export SLMSUITE_TEST_CAMERA_ARGS='{"serial": "12345"}'
pytest

# Windows (PowerShell)
$env:SLMSUITE_TEST_SLM_CLASS="slmsuite.hardware.slms.screenmirrored.ScreenMirrored"
$env:SLMSUITE_TEST_SLM_ARGS='{"display_number": 1}'
pytest

# Windows (CMD)
set SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.screenmirrored.ScreenMirrored
set SLMSUITE_TEST_SLM_ARGS={"display_number": 1}
pytest
```

The test fixtures automatically handle:
- Instantiation with appropriate arguments
- Cleanup and resource management
- Compatibility with any SLM/Camera subclass

## Test Markers

Markers are defined in `pytest.ini` and can be used to categorize tests:

```python
@pytest.mark.gpu
def test_cupy_algorithm():
    """Test requiring CuPy/GPU."""
    if not HAS_CUPY:
        pytest.skip("CuPy not available")
    # ... GPU test code

@pytest.mark.slow
def test_long_optimization():
    """Test that takes >5 seconds."""
    # ... long-running test code
```

Run tests by marker:
```bash
# Skip GPU tests
pytest -m "not gpu"

# Run only slow tests
pytest -m slow

# Skip both GPU and slow tests
pytest -m "not gpu and not slow"
```

## Test Results

All tests are passing across all modules (hardware, holography, misc, examples).

## Writing New Tests

### Test Class Structure
```python
class TestFeatureName:
    """Tests for specific feature."""

    def test_basic_functionality(self):
        """Test the happy path."""
        result = my_function(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """Test boundary conditions."""
        assert my_function(edge_input) handles correctly

    def test_error_handling(self):
        """Test that errors are raised appropriately."""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### Testing Best Practices

1. **Usually one testing function per package function**: `blaze` is mirrored
   `test_blaze`.
2. **Use subtests**: Each test function can have multiple `subtests` which each test one aspect of the function. This allows grouping related assertions while keeping test functions focused.
3. **Use fixtures**: Avoid code duplication with shared fixtures
4. **Test edge cases**: Empty arrays, zero values, NaN, extreme sizes
5. **Use pytest.approx()**: For floating-point comparisons
6. **Parametrize**: Test multiple inputs with `@pytest.mark.parametrize`

### Example: Parametrized Test
```python
@pytest.mark.parametrize("bitdepth,expected", [
    (8, 256),
    (10, 1024),
    (12, 4096),
])
def test_bitresolution(bitdepth, expected):
    """Test bitresolution calculation for various bitdepths."""
    slm = SimulatedSLM(bitdepth=bitdepth)
    assert slm.bitresolution == expected
```

## GPU Testing

Tests can run on both CPU (NumPy) and GPU (CuPy) when available:

```python
@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
def test_gpu_acceleration():
    """Test that runs only if CuPy is installed."""
    import cupy as cp
    target = cp.zeros((128, 128))
    # ... GPU test code
```

## Debugging Failed Tests

```bash
# Show full traceback
pytest -v --tb=long

# Stop at first failure
pytest -x

# Show local variables in traceback
pytest -l

# Run last failed tests
pytest --lf

# Enter debugger on failure
pytest --pdb
```

## Logging and Plot Capture

### Output Directory Structure

Each pytest run creates a **timestamped output directory**:

```
tests/output/
├── 20250126_143015/         # First test run (YYYYMMDD_HHMMSS)
│   ├── pytest.log
│   └── test_*.png
├── 20250126_154522/         # Second test run
│   ├── pytest.log
│   └── test_*.png
└── latest -> 20250126_154522/  # Symlink to latest (if supported)
```

This allows you to:
- Compare outputs from different test runs
- Keep history of test results
- Isolate each run's artifacts

### Test Logging

All tests **automatically log** to `tests/output/{timestamp}/pytest.log` with start/end markers:

```bash
# View logs in terminal during test run
pytest --log-cli --log-cli-level=INFO

# View logs after test run (latest)
cat tests/output/latest/pytest.log
tail -f tests/output/latest/pytest.log

# View logs from specific run
cat tests/output/20250126_143015/pytest.log
```

**Automatic Logging**: Every test automatically gets a logger (via `autouse=True` fixture). No need to add `test_logger` parameter unless you want to use it explicitly:

```python
# Option 1: Automatic logging (most tests)
def test_example():
    # Test automatically logs start/end
    result = my_function()
    assert result == expected

# Option 2: Explicit logging (when needed)
def test_complex_operation(test_logger):
    test_logger.info("Starting complex operation...")
    test_logger.debug(f"Parameters: {params}")
    result = complex_function(params)
    test_logger.info(f"Result: {result}")
    assert result > 0
```

**Third-party Logging**: All external packages are automatically set to WARNING level. Only `slmsuite` package logs at INFO level.

**Log Format:**
```
2025-01-26 14:30:15 [    INFO] [slmsuite.tests.test_algorithms.TestHologram.test_gs_converges] === START: holography/test_algorithms.py::TestHologram::test_gs_converges ===
2025-01-26 14:30:16 [    INFO] [slmsuite.holography.algorithms] Optimizing with method=GS, maxiter=20
2025-01-26 14:30:17 [    INFO] [slmsuite.tests.test_algorithms.TestHologram.test_gs_converges] === PASSED: holography/test_algorithms.py::TestHologram::test_gs_converges ===
```

### Matplotlib Plot Capture

Plots are **automatically saved** to the current run's directory:

```bash
# Disable plot saving (default)
pytest

# Enable plot saving
pytest --save-plots

# Example output:
# 📁 Test output directory: tests/output/20250126_143015
# 📊 Saved plot: tests/output/20250126_143015/test_algorithms_TestHologram_test_hologram_construction_fig1.png
```

**Filename Format:** `{module}_{class}_{function}_fig{N}.png`

Examples:
- `test_algorithms_TestHologram_test_gs_converges_fig1.png`
- `test_toolbox_test_blaze_basic_fig1.png` (no class)

Existing tests using `plt.show()` work without modification - plots are automatically saved instead of displayed.

**View saved plots:**
```bash
# Latest run
ls tests/output/latest/*.png
open tests/output/latest/*.png

# Specific run
ls tests/output/20250126_143015/*.png

# Compare two runs
diff tests/output/20250126_143015/pytest.log tests/output/20250126_154522/pytest.log
```

### Cleanup

```bash
# Remove old test runs (keep last 5)
cd tests/output && ls -t | tail -n +6 | xargs rm -rf

# Remove all output
rm -rf tests/output/
```

## Benchmarking

Performance benchmarks use [pytest-benchmark](https://pytest-benchmark.readthedocs.io/).
Tests that accept the `benchmark` fixture automatically get timed with warmup, multiple rounds,
and statistical reporting. Results are printed as a table after the test run.

**Benchmarked functions:**

| Function | Test file | GPU variant? |
|---|---|---|
| `Hologram.optimize()` (GS/WGS) | `test_algorithms.py` | Yes |
| `SLM._phase2gray()` | `test_slms.py` | No |
| `SLM.set_phase()` | `test_slms.py` | No |
| `analysis.take()` | `test_analysis.py` | Yes |
| `analysis.image_moment()` | `test_analysis.py` | No |
| `analysis.image_fit()` | `test_analysis.py` | No |
| `phase.blaze()` | `test_toolbox_phase.py` | No |
| `phase.lens()` | `test_toolbox_phase.py` | No |
| `phase.zernike_sum()` | `test_toolbox_phase.py` | Yes |
| `toolbox.imprint()` | `test_toolbox.py` | No |

```bash
# Run everything including benchmarks
pytest

# Disable benchmark timing (tests still run, just not timed)
pytest --benchmark-disable

# Run only benchmarked tests
pytest --benchmark-only

# Compare against a saved baseline
pytest --benchmark-compare
```

## TODOs

- [ ] Integration tests for SpotHologram optimization
- [ ] CPU/GPU parametrization for all algorithm tests
- [x] Performance benchmarking tests
- [ ] Coverage reporting (pytest-cov)
- [ ] Hardware detection and optional hardware tests
- [ ] Test data fixtures for common calibration scenarios