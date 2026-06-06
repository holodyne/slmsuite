"""
Pytest configuration and fixtures for slmsuite tests.

The fixtures in this file support testing with any SLM or Camera subclass.
By default, SimulatedSLM and SimulatedCamera are used for fast, hardware-free testing.

To test with real hardware, set environment variables:
    SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.thorlabs.ThorlabsSLM
    SLMSUITE_TEST_SLM_ARGS='{"monitor_id": 1}'
    SLMSUITE_TEST_CAMERA_CLASS=slmsuite.hardware.cameras.thorlabs.ThorlabsCamera
    SLMSUITE_TEST_CAMERA_ARGS='{"serial": "12345"}'

Automatic Features:
- All tests automatically log to tests/output/YYYYMMDD_HHMMSS/pytest.log
- Matplotlib plots automatically saved to tests/output/YYYYMMDD_HHMMSS/
- Random seed generated per session (logged for reproducibility)
- slmsuite package logging: INFO level
- External packages logging: WARNING level and above only
"""
import pytest
import numpy as np
import tempfile
import sys, os
import json
import importlib
import logging
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Globals

_TEST_RUN_OUTPUT_DIR = None


def get_test_run_output_dir():
    """Helper function to get current test run output directory."""
    return _TEST_RUN_OUTPUT_DIR


try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False


@pytest.fixture(scope="session")
def has_cupy():
    """Check if CuPy is available for GPU tests."""
    return HAS_CUPY


@pytest.fixture(scope="session")
def random_seed():
    """
    Generate and return a random seed for the test session.

    The seed is logged and can be used to reproduce test runs.
    Also sets numpy's global random seed for reproducibility.

    Returns
    -------
    int
        Random seed value for this test session
    """
    import random
    seed = random.randint(0, 2**32 - 1)

    # Set numpy's random seed
    np.random.seed(seed)

    # Set CuPy's random seed if available
    if HAS_CUPY:
        cp.random.seed(seed)

    # Log the seed for reproducibility
    logger = logging.getLogger("conftest")
    logger.info(f"Random seed for this session: {seed}")
    print(f"\nRandom seed for this session: {seed}")

    return seed


# Fixtures for SLM and Camera instances, with dynamic configuration via environment variables.

from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.hardware.cameraslms import FourierSLM

_TEST_SMALL_RESOLUTION = (128, 128)


def _get_class_from_string(class_path):
    """
    Import and return a class from a module path string.

    Parameters
    ----------
    class_path : str
        Full path to class, e.g., 'slmsuite.hardware.slms.simulated.SimulatedSLM'

    Returns
    -------
    class
        The imported class
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@pytest.fixture
def slm_class():
    """
    Return the SLM class to use for testing.

    By default returns SimulatedSLM. Can be overridden via environment variable:
    SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.santec.Santec

    Returns
    -------
    class
        SLM subclass to instantiate
    """
    class_path = os.environ.get('SLMSUITE_TEST_SLM_CLASS', None)
    if class_path:
        return _get_class_from_string(class_path)
    return SimulatedSLM


@pytest.fixture
def slm_kwargs():
    """
    Return keyword arguments for SLM instantiation.

    By default returns arguments for SimulatedSLM. Can be overridden via:
    SLMSUITE_TEST_SLM_ARGS='{"bitdepth": 12}'

    Returns
    -------
    dict
        Keyword arguments for SLM constructor
    """
    args_json = os.environ.get('SLMSUITE_TEST_SLM_ARGS', None)
    if args_json:
        return json.loads(args_json)

    # Default args for SimulatedSLM
    return {
        'resolution': (1920, 1080),
        'pitch_um': (8.0, 8.0),
        'bitdepth': 8,
        'wav_um': 0.78
    }


@pytest.fixture
def slm(slm_class, slm_kwargs):
    """
    Fixture providing an SLM instance for testing.
    """
    slm_instance = slm_class(**slm_kwargs)
    yield slm_instance

    # Cleanup
    try:
        slm_instance.close()
    except Exception:
        pass


@pytest.fixture
def slm_small(slm_kwargs):
    """
    Fixture providing an SLM instance for testing.
    """
    kwargs = slm_kwargs.copy()
    kwargs['resolution'] = _TEST_SMALL_RESOLUTION
    slm_instance = SimulatedSLM(**kwargs)
    yield slm_instance

    # Cleanup
    try:
        slm_instance.close()
    except Exception:
        pass


@pytest.fixture
def camera_class():
    """
    Return the Camera class to use for testing.

    By default returns SimulatedCamera. Can be overridden via:
    SLMSUITE_TEST_CAMERA_CLASS=slmsuite.hardware.cameras.alliedvision.AlliedVision

    Returns
    -------
    class
        Camera subclass to instantiate
    """
    class_path = os.environ.get('SLMSUITE_TEST_CAMERA_CLASS', None)
    if class_path:
        return _get_class_from_string(class_path)
    return SimulatedCamera


@pytest.fixture
def camera_kwargs(slm):
    """
    Return keyword arguments for Camera instantiation.

    By default returns arguments for SimulatedCamera. Can be overridden via:
    SLMSUITE_TEST_CAMERA_ARGS='{"serial": "12345", "bitdepth": 8}'

    Parameters
    ----------
    slm : SLM
        SLM instance (required for SimulatedCamera)

    Returns
    -------
    dict
        Keyword arguments for Camera constructor
    """
    args_json = os.environ.get('SLMSUITE_TEST_CAMERA_ARGS', None)
    if args_json:
        return json.loads(args_json)

    # Default args for SimulatedCamera
    return {
        'slm': slm,
        'resolution': (512, 512),
        'pitch_um': (5.5, 5.5),
        'bitdepth': 8
    }


@pytest.fixture
def camera(camera_class, camera_kwargs):
    """
    Fixture providing a Camera instance for testing.

    By default returns SimulatedCamera, but can be configured to return any Camera subclass
    via environment variables SLMSUITE_TEST_CAMERA_CLASS and SLMSUITE_TEST_CAMERA_ARGS.
    """
    cam = camera_class(**camera_kwargs)
    yield cam

    # Cleanup
    try:
        cam.close()
    except Exception:
        pass


@pytest.fixture
def camera_small(slm_small, camera_kwargs):
    """
    Fixture providing a Camera instance for testing.
    """
    kwargs = camera_kwargs.copy()
    kwargs["resolution"] = _TEST_SMALL_RESOLUTION
    kwargs["slm"] = slm_small
    cam = SimulatedCamera(**kwargs)
    yield cam

    # Cleanup
    try:
        cam.close()
    except Exception:
        pass

@pytest.fixture
def fourierslm(camera, slm):
    """
    Fixture providing a FourierSLM instance for testing.
    """
    camera.set_exposure(0.1)  # Don't overexpose the fourier calibration.
    fs = FourierSLM(camera, slm, mag=1.0)
    yield fs

    # Cleanup
    try:
        fs.close()
    except Exception:
        pass

@pytest.fixture
def fourierslm_calibrated(fourierslm):
    """FourierSLM with a completed Fourier calibration."""
    fourierslm.fourier_calibrate(array_pitch=30, array_shape=10, plot=True)
    return fourierslm


# Matplotlib configuration (saving of plots)

@pytest.fixture(scope="session", autouse=True)
def configure_matplotlib_for_testing(request):
    """
    Configure matplotlib for testing environment.

    - Use Agg backend (non-interactive) to prevent blocking
    - Replace plt.show() to save figures with descriptive names
    - Save to current test run's timestamped directory

    TODO: determine when plt.show() is called outside of
    slmsuite's `slmsuite_plt_show()`.
    """
    # Check if we should save plots (default: True)
    save_plots = request.config.getoption("--save-plots", default=True)

    # Use non-interactive backend for testing
    matplotlib.use("Agg")
    plt.ioff()  # Disable interactive mode

    # Store original plt.show
    original_show = plt.show

    if save_plots:
        # Track figure count per test
        test_fig_counts = {}

        def custom_show(name=None):
            """
            Replacement for plt.show() that saves figures with descriptive names.

            Format: {module}_{class}_{function}[_{name}]_fig{N}.png
            Saved to: tests/output/{timestamp}/
            """
            # Get output directory for this test run
            output_dir = get_test_run_output_dir()
            if output_dir is None:
                print("Warning: Test run output directory not initialized")
                plt.close('all')
                return

            # Get current test info from pytest environment variable
            test_name = os.environ.get('PYTEST_CURRENT_TEST', '')

            if not test_name:
                # Fallback if called outside test context
                key = f"unknown_{name}" if name else "unknown"
                filename = output_dir / f"{key}_fig{len(test_fig_counts)}.png"
                figs = [plt.figure(n) for n in plt.get_fignums()]
                for fig in figs:
                    fig.savefig(filename, dpi=150, bbox_inches='tight')
            else:
                # Parse test path: "tests/holography/test_algorithms.py::TestHologram::test_gs_converges (call)"
                test_path = test_name.split(' ')[0]  # Remove "(call)" part

                # Extract components
                parts = []
                if '::' in test_path:
                    file_and_rest = test_path.split('::')
                    # Get module name from file path
                    module = file_and_rest[0].split('/')[-1].replace('.py', '')
                    parts.append(module)
                    # Add class and function if present
                    parts.extend(file_and_rest[1:])
                else:
                    parts.append('unknown')

                # Append plot-site name from _slmsuite_plt_show if provided
                if name:
                    parts.append(name)

                # Build filename key; each unique key has its own counter
                key = '_'.join(parts)

                # Save all open figures
                figs = [plt.figure(n) for n in plt.get_fignums()]
                for fig in figs:
                    test_fig_counts[key] = test_fig_counts.get(key, 0) + 1
                    filename = output_dir / f"{key}_fig{test_fig_counts[key]}.png"
                    fig.savefig(filename, dpi=150, bbox_inches='tight')
                    # Print relative path
                    rel_path = filename.relative_to(Path("tests/output"))
                    print(f"Saved plot: tests/output/{rel_path}")

            # Close figures to free memory
            plt.close('all')

        # Replace plt.show and configure slmsuite's internal handler
        plt.show = custom_show
        import slmsuite
        slmsuite.configure_plotting(custom_show)
    else:
        # If plots disabled, just close figures silently
        def no_show(_name=None, *_args, **_kwargs):
            plt.close('all')
        plt.show = no_show
        import slmsuite
        slmsuite.configure_plotting(no_show)

    yield

    # Restore original plt.show and slmsuite handler
    plt.show = original_show
    slmsuite.configure_plotting("show")


@pytest.fixture
def mpl_test(request):
    """
    Per-test fixture for matplotlib tests.

    Provides automatic figure cleanup and easy access to plt.
    """
    # Clear any existing figures before test
    plt.close('all')

    yield plt

    # Cleanup after test
    plt.close('all')


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--save-plots",
        action="store_true",
        default=True,
        help="Save matplotlib plots to tests/output/{timestamp}/ (default: True)"
    )
    parser.addoption(
        "--no-save-plots",
        action="store_false",
        dest="save_plots",
        help="Disable saving matplotlib plots"
    )

# Logging and final configuration

@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(autouse=True)
def test_logger(request):
    """
    Provides a test-specific logger with proper naming.

    This fixture is automatically used for every test (autouse=True).
    Tests can access the logger via request.node.test_logger if needed.

    Logger name format: {module}.{class}.{function}
    Example: test_algorithms.TestHologram.test_gs_converges
    """
    # Build logger name from test node
    parts = []
    if request.module:
        module_name = request.module.__name__.split('.')[-1]
        parts.append(module_name)
    if request.cls:
        parts.append(request.cls.__name__)
    if request.function:
        parts.append(request.function.__name__)

    logger_name = ".".join(parts)
    logger = logging.getLogger(logger_name)

    # Store logger in request for access by tests if needed
    request.node.test_logger = logger

    # Log test start
    logger.info("=== START ===")

    yield logger

    # Log test result
    if hasattr(request.node, 'rep_call'):
        outcome = request.node.rep_call.outcome
        logger.info(f"=== {outcome.upper()} ===")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for logging."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU-marked tests when CuPy is not available."""
    if not HAS_CUPY:
        skip_gpu = pytest.mark.skip(reason="CuPy not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


def pytest_configure(config):
    """Configure pytest with dynamic log file path and custom settings."""
    # Create output directory with timestamp for this test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path("tests/output")
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store for later use by fixtures
    global _TEST_RUN_OUTPUT_DIR
    _TEST_RUN_OUTPUT_DIR = output_dir

    # Create/update 'latest' symlink for convenience
    latest_link = output_base / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(timestamp, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Symlinks may fail on Windows without developer mode
        pass

    # Set log file path dynamically
    log_file = output_dir / "pytest.log"
    config.option.log_file = str(log_file)

    # Save benchmark results to the output directory
    config.option.benchmark_storage = str(output_dir)
    config.option.benchmark_autosave = True

    # Configure logging: suppress all external packages to WARNING level
    # Only allow INFO and above from slmsuite package
    logging.captureWarnings(True)

    # Set all loggers to WARNING by default (external packages)
    logging.getLogger().setLevel(logging.WARNING)

    # Explicitly set common external packages to WARNING
    for package in ['matplotlib', 'PIL', 'numpy', 'cupy', 'h5py']:
        logging.getLogger(package).setLevel(logging.WARNING)

    # Enable INFO for slmsuite package only
    logging.getLogger('slmsuite').setLevel(logging.INFO)

    print(f"\nTest output directory: {output_dir}")


def pytest_sessionfinish(session, exitstatus):
    """Print summary message at end of test session."""
    output_dir = get_test_run_output_dir()
    if output_dir:
        print(f"\nTest output saved to: {output_dir}")