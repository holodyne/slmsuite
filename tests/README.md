# Tests

Everything here runs against simulated hardware: `pytest -m "not slow"` passes on a machine
with no SLM or camera attached. The full `pytest` additionally downloads and executes the
example notebooks, which needs network access and tens of minutes.

```bash
pytest                                      # everything
pytest tests/holography/                    # one subpackage
pytest tests/holography/test_toolbox_phase.py::test_blaze
pytest -m "not gpu and not slow"            # the fast subset
```

`tests/` mirrors `slmsuite/`: `tests/holography/test_toolbox_phase.py` covers
`slmsuite/holography/toolbox/phase/`, `tests/hardware/test_slms.py` covers
`slmsuite/hardware/slms/`, and so on. A new module gets a new file in the matching
directory.

## Style

**One test per package function.** `blaze()` is covered by `test_blaze()`, `zernike_sum()`
by `test_zernike_sum()`. Private helpers and private methods get the same treatment:
`_parse_focal_length()` by `test_parse_focal_length()`, `_validate_roi()` by
`test_validate_roi()`. Keep the leading underscore when a public sibling would collide, so
`Camera.get_dtype()` and `_Common._get_dtype()` become `test_get_dtype` and
`test__get_dtype`. Classes get a `TestClassName` holding a method per public method it has
a test for. Finding the tests for a function should never require a search.

Three exceptions, all of which the suite already follows. `test()` is named `test_selftest`,
since `test_test()` is absurd. Behavior inherited from a shared base such as
`slmsuite/hardware/_common.py` is tested through the concrete subclasses that use it. And an
alias or thin wrapper is tested by asserting its identity with what it wraps, which is the
whole of its contract: `test_image_centroids` checks only that it forwards to
`image_positions`.

**Subtests, not many small functions.** One `test_*` covers a function's whole contract,
split into `subtests`:

```python
def test_lens(simple_grid, subtests):
    """Test lens() phase pattern generation."""
    with subtests.test("infinite focal length gives zeros"):
        ...

    with subtests.test("negative focal length negates phase"):
        ...
```

Each subtest reports separately and the rest still run after one fails, which beats a
dozen near-identical functions sharing a setup. Parametrize instead when the same
assertions apply to a list of inputs, and reach for a separate function only when the
setup genuinely differs.

The subtest string is what a reader sees in the failure report, so make it the property
being asserted, `"l is the counterclockwise azimuthal winding"` rather than `"test l=2"`.

**Say it in absolutes where you can.** An analytic value, a closed-form limit, or an exact
identity pins far more than an inequality: `lorentzian(x0 + w) == a / 2` beats `a narrower w
gives a sharper peak`, and both cost the same to write. Reserve one-sided bounds for
convergence, timing, and other genuinely inexact claims. A test taking the `slm`, `camera` or
`fourierslm` fixture is the exception, since the environment may point those at real
hardware: assert only what any device must satisfy there, and construct a `SimulatedSLM` or
`SimulatedCamera` explicitly when you want to pin a specific resolution or dtype.

**Do not reimplement the function to check it.** A transcription of the implementation
shares all of its assumptions and agrees with it for free, including where both are wrong.
Check against an independent source instead: a textbook value, a limit of another function, a
symmetry, or a second code path in the package. `np.testing.assert_equal` is the tool for a
save/load or pickle round trip, since it recurses through nested dicts and handles `None`.

**Comments are for the non-obvious.** Docstrings and comments follow the same rules as the
package (numpydoc and PEP 257; see `../CONTRIBUTING.md`) and a one-line summary is usually
the whole docstring. An assertion does not need a comment restating it. A tolerance, a
magic constant, a grid size, or a stub does: one line on where the number came from, or
what breaks without it. Comments describe what the test claims, never the history of a
bug it was written for.

## Fixtures

All from `conftest.py`, except `subtests`, which comes from the pytest-subtests plugin.

| Fixture | What you get |
|---|---|
| `slm`, `camera`, `fourierslm` | `SimulatedSLM` / `SimulatedCamera` / a `FourierSLM` pairing them |
| `slm_small`, `camera_small` | 128x128 versions, for tests that do not need resolution |
| `fourierslm_calibrated` | a `FourierSLM` with `fourier_calibrate()` already run |
| `simulated_system` | parameterized over every geometry case and both illuminations |
| `simulated_system_name`, `simulated_system_source` | which case and illumination the current parameterization is on |
| `simulated_system_factory` | one named case on demand: `factory("rotated", noise=...)` |
| `random_seed`, `has_cupy` | session-scoped seed (logged) and a CuPy probe |
| `temp_dir`, `mpl_test`, `test_output_dir` | scratch directory, figure cleanup at the end of the test, this run's output directory |

`conftest.py` also exports helpers to import directly rather than request: `seed_for`,
`driver_classes`, and the simulated-system ground truth (`ground_truth_affine` and friends).
Coordinate grids are file-local fixtures, since what counts as a useful grid differs per
module. Note `mpl_test` only cleans up when the test ends; a test comparing several figures
must close them itself, because `_Common._plot()` draws into `plt.gcf()` whenever one is open.

`SIMULATED_SYSTEM_CASES` in `conftest.py` defines the geometry cases: camera field of view
larger and smaller than the SLM farfield, rotation, shear, an off-center 0th order, a parity
flip, noise, anisotropy, and wavefront aberration. Their ground truth
lives in the simulated hardware and comes back through `ground_truth_affine(fs)` and
friends, so a calibration routine can be checked without depending on the calibration
under test.

Anything whose outcome depends on the starting phase should call `seed_for("some-name")`
first. Holograms draw from CuPy's generator when CuPy is installed, so seeding numpy alone
leaves them free-running.

## Markers

`gpu` for tests needing CUDA/CuPy, `slow` for anything over a few seconds. Both are
declared in `pytest.ini`, and `--strict-markers` rejects anything else.

## Output

Each run writes to a timestamped `tests/output/{YYYYMMDD_HHMMSS}/`. `pytest.log` holds
every test's start and end plus whatever `slmsuite` logged in between; third-party
loggers are pinned to WARNING. `--save-plots` writes figures there as
`{module}_{class}_{function}[_{name}]_fig{N}.png`, where `name` is the label an internal plot
site passes. Without the flag every figure is closed unshown, so existing `plt.show()` calls
need no change either way.

## Benchmarks

A test that takes the `benchmark` fixture is timed by
[pytest-benchmark](https://pytest-benchmark.readthedocs.io/) and reported in a table
afterwards; the covered hot paths are `Hologram.optimize()`, `SLM._phase2gray()`,
`SLM.set_phase()`, `analysis.take()`, `image_moment()`, `image_fit()`, `phase.blaze()`,
`lens()`, `zernike_sum()`, and `toolbox.imprint()`. Use `--benchmark-disable` to run them
untimed, `--benchmark-only` to run nothing else.

## Real hardware

The `slm`, `camera` and `fourierslm` fixtures instantiate whatever class the environment
names, so the suite runs against a real setup unchanged. The `_small` and `simulated_system`
fixtures stay simulated.

```bash
export SLMSUITE_TEST_SLM_CLASS=slmsuite.hardware.slms.screenmirrored.ScreenMirrored
export SLMSUITE_TEST_SLM_ARGS='{"display_number": 1}'
export SLMSUITE_TEST_CAMERA_CLASS=slmsuite.hardware.cameras.thorlabs.ThorCam
export SLMSUITE_TEST_CAMERA_ARGS='{"serial": "12345"}'
pytest
```
