"""Tests for :mod:`slmsuite._plotting` save-mode behavior."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import slmsuite
from slmsuite import _plotting


@pytest.fixture
def restore_handler():
    """Snapshot and restore the active plot handler around a test.

    The session-scoped ``configure_matplotlib_for_testing`` fixture installs a
    global handler; resetting to ``"show"`` would clobber it for later tests, so
    we save and restore the exact handler instead.
    """
    prev = _plotting._current_handler
    try:
        yield
    finally:
        _plotting._current_handler = prev
        plt.close("all")


def _draw():
    fig = plt.figure()
    plt.plot([0, 1], [0, 1])
    return fig


def test_save_extension_and_kwargs(tmp_path, restore_handler):
    slmsuite.configure_plotting(
        mode="save", save_dir=tmp_path, extension="pdf",
        savefig_kwargs={"dpi": 72},
    )
    _draw()
    _plotting._slmsuite_plt_show(name="demo")

    assert (tmp_path / "demo_00000.pdf").exists()
    # Figures are closed after saving.
    assert plt.get_fignums() == []


def test_save_default_is_png(tmp_path, restore_handler):
    slmsuite.configure_plotting(mode="save", save_dir=tmp_path)
    _draw()
    _plotting._slmsuite_plt_show(name="demo")

    assert (tmp_path / "demo_00000.png").exists()


def test_save_numbering_is_conflict_free(tmp_path, restore_handler):
    """A second save does not overwrite the first; the id increments on disk."""
    slmsuite.configure_plotting(mode="save", save_dir=tmp_path)

    _draw()
    _plotting._slmsuite_plt_show(name="demo")
    _draw()
    _plotting._slmsuite_plt_show(name="demo")

    assert (tmp_path / "demo_00000.png").exists()
    assert (tmp_path / "demo_00001.png").exists()


def test_save_single_figure_has_no_index_suffix(tmp_path, restore_handler):
    """A lone figure is named with the bare stem, not ``stem_0``."""
    slmsuite.configure_plotting(mode="save", save_dir=tmp_path)
    _draw()
    _plotting._slmsuite_plt_show(name="demo")

    assert (tmp_path / "demo_00000.png").exists()
    assert not (tmp_path / "demo_0_00000.png").exists()


def test_save_multiple_figures_get_index_suffix(tmp_path, restore_handler):
    """Several figures from one show are disambiguated with a ``_n`` suffix."""
    slmsuite.configure_plotting(mode="save", save_dir=tmp_path)
    _draw()
    _draw()
    _plotting._slmsuite_plt_show(name="demo")

    assert (tmp_path / "demo_0_00000.png").exists()
    assert (tmp_path / "demo_1_00000.png").exists()


def test_save_requires_save_dir():
    with pytest.raises(ValueError):
        slmsuite.configure_plotting(mode="save")
