"""Tests for :mod:`slmsuite._plotting` save-mode behavior."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from PIL import Image

import slmsuite
from slmsuite import _plotting


@pytest.fixture
def restore_handler():
    """Restore the session's plot handler so a test does not leak it to later tests."""
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


def test_configure_plotting(tmp_path, restore_handler, subtests):
    """Test configure_plotting() mode dispatch and save-mode file output."""
    with subtests.test("mode='show' clears the handler"):
        slmsuite.configure_plotting(mode="show")
        assert _plotting._current_handler is None

    with subtests.test("mode='suppress' closes open figures"):
        slmsuite.configure_plotting(mode="suppress")
        _draw()
        _plotting._slmsuite_plt_show(name="demo")
        assert plt.get_fignums() == []

    with subtests.test("mode='save' without save_dir raises ValueError"):
        with pytest.raises(ValueError):
            slmsuite.configure_plotting(mode="save")

    with subtests.test("mode='save' defaults to a .png extension"):
        d = tmp_path / "default_ext"
        slmsuite.configure_plotting(mode="save", save_dir=d)
        _draw()
        _plotting._slmsuite_plt_show(name="demo")
        assert (d / "demo_00000.png").exists()

    with subtests.test("extension controls the saved file format"):
        d = tmp_path / "extension"
        slmsuite.configure_plotting(mode="save", save_dir=d, extension="pdf")
        _draw()
        _plotting._slmsuite_plt_show(name="demo")
        path = d / "demo_00000.pdf"
        assert path.read_bytes()[:4] == b"%PDF"

    with subtests.test("savefig_kwargs override the merged defaults"):
        d = tmp_path / "savefig_kwargs"
        slmsuite.configure_plotting(mode="save", save_dir=d, savefig_kwargs={"dpi": 72})
        _draw()
        _plotting._slmsuite_plt_show(name="demo")
        assert Image.open(d / "demo_00000.png").info["dpi"][0] == pytest.approx(72, abs=1)

    with subtests.test("mode='save' closes figures after writing"):
        d = tmp_path / "closes"
        slmsuite.configure_plotting(mode="save", save_dir=d)
        _draw()
        _plotting._slmsuite_plt_show(name="demo")
        assert plt.get_fignums() == []

    with subtests.test("a lone figure gets no index suffix"):
        d = tmp_path / "single"
        slmsuite.configure_plotting(mode="save", save_dir=d)
        _draw()
        _plotting._slmsuite_plt_show(name="demo")
        assert (d / "demo_00000.png").exists()
        assert not (d / "demo_0_00000.png").exists()

    with subtests.test("multiple figures from one call get index suffixes"):
        d = tmp_path / "multiple"
        slmsuite.configure_plotting(mode="save", save_dir=d)
        _draw()
        _draw()
        _plotting._slmsuite_plt_show(name="demo")
        assert (d / "demo_0_00000.png").exists()
        assert (d / "demo_1_00000.png").exists()

    with subtests.test("repeated calls do not clobber earlier files"):
        d = tmp_path / "repeated"
        slmsuite.configure_plotting(mode="save", save_dir=d)
        _draw()
        _plotting._slmsuite_plt_show(name="demo")
        _draw()
        _plotting._slmsuite_plt_show(name="demo")
        assert (d / "demo_00000.png").exists()
        assert (d / "demo_00001.png").exists()

    with subtests.test("a callable mode is installed as the handler verbatim"):
        handler = lambda name=None, **kwargs: None
        slmsuite.configure_plotting(mode=handler)
        assert _plotting._current_handler is handler

    with subtests.test("an unrecognized mode raises ValueError"):
        with pytest.raises(ValueError):
            slmsuite.configure_plotting(mode="bogus")


def test_slmsuite_plt_show(restore_handler, monkeypatch, subtests):
    """Test _slmsuite_plt_show() routing to the active plot handler."""
    with subtests.test("no handler installed defers to plt.show"):
        calls = []
        monkeypatch.setattr(plt, "show", lambda *args, **kwargs: calls.append((args, kwargs)))
        _plotting._current_handler = None
        _plotting._slmsuite_plt_show(name="site", block=False)
        assert calls == [((), {"block": False})]

    with subtests.test("an installed handler receives the name and forwarded kwargs"):
        calls = []
        _plotting._current_handler = lambda name=None, **kwargs: calls.append((name, kwargs))
        _plotting._slmsuite_plt_show(name="site", block=False)
        assert calls == [("site", {"block": False})]
