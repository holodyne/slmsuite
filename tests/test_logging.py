"""
Unit tests for slmsuite._logging: _Loggable, configure_logging, get_log.

The logging layer uses a single "slmsuite" logger with per-class descendant loggers and one
shared, bounded capture buffer. Handlers live only on the root; per-instance isolation is by
a uid tag on each record (filtered out of the shared buffer on read).
"""
import logging

import pytest

import slmsuite
from slmsuite._logging import (
    _BufferHandler,
    _Loggable,
    _package_logger,
    configure_logging,
    get_log,
)


def _console_handlers():
    return [h for h in _package_logger.handlers if getattr(h, "_slmsuite_console", False)]


class _Device(_Loggable):
    """Minimal _Loggable subclass for testing."""
    _pickle = ["value"]
    _pickle_data = []

    def __init__(self, name="dev"):
        self.name = name
        self.value = 0
        _Loggable.__init__(self)

    def __str__(self):
        return self.name


@pytest.fixture
def _clean_console_handler():
    """Give configure_logging() a clean console-handler slate, and restore it after."""
    configure_logging(level=None)
    yield
    configure_logging(level=None)


def test_configure_logging(subtests, _clean_console_handler):
    """Test configure_logging() console handler management."""
    with subtests.test("a level installs exactly one console handler"):
        configure_logging("INFO")
        assert len(_console_handlers()) == 1

    with subtests.test("re-calling with a level does not stack handlers"):
        configure_logging("INFO")
        assert len(_console_handlers()) == 1

    with subtests.test("None removes the console handler"):
        configure_logging(level=None)
        assert _console_handlers() == []

    with subtests.test("a string level sets the exact numeric level"):
        configure_logging("DEBUG")
        assert _console_handlers()[0].level == logging.DEBUG

    with subtests.test("an int level sets the exact numeric level"):
        configure_logging(logging.WARNING)
        assert _console_handlers()[0].level == logging.WARNING


def test_get_log(subtests):
    """Test package-level get_log() session-wide record capture."""
    with subtests.test("captures a _Loggable instance's records"):
        dev = _Device(name="pkg_log_test")
        dev.logger.info("hello from device")
        assert any("hello from device" in r for r in get_log())

    with subtests.test("captures records logged directly on the package logger"):
        slmsuite.logger.info("package-level message")
        assert any("package-level message" in r for r in get_log())

    with subtests.test("captured by a _BufferHandler installed on the package logger"):
        assert any(isinstance(h, _BufferHandler) for h in _package_logger.handlers)


class TestLoggable:

    @pytest.fixture(autouse=True)
    def _dev(self):
        self.dev = _Device(name="test_dev")

    def test_init(self, subtests):
        """Test _Loggable.__init__() logger setup."""
        with subtests.test("per-class child logger carries no handlers of its own"):
            child = logging.getLogger(f"slmsuite.{type(self.dev).__name__}")
            assert child.handlers == []
            assert child.level == logging.NOTSET

        with subtests.test("logs an Initialized record on construction"):
            assert any("Initialized" in r for r in self.dev.get_log())

        with subtests.test("a lone instance's index is 1"):
            solo = _Device(name="solo_init")
            assert solo._log_index == 1

        with subtests.test("repeated names get sequential per-name indices"):
            dup_a = _Device(name="dup_init")
            dup_b = _Device(name="dup_init")
            assert dup_a._log_index == 1 and dup_b._log_index == 2

    def test_get_log(self, subtests, capsys):
        """Test _Loggable.get_log() instance-scoped record retrieval."""
        with subtests.test("returns this instance's own records"):
            self.dev.logger.info("mine")
            assert any("mine" in r for r in self.dev.get_log())

        with subtests.test("excludes another instance's records, even sharing a class logger"):
            other = _Device(name="test_dev")
            other.logger.info("theirs")
            assert not any("theirs" in r for r in self.dev.get_log())
            assert any("theirs" in r for r in other.get_log())

        with subtests.test("a lone instance's records carry no index suffix"):
            solo = _Device(name="solo_get_log")
            solo.logger.info("only one")
            assert any("solo_get_log" in r and "solo_get_log #" not in r for r in solo.get_log())

        with subtests.test("duplicate names disambiguate records with a 1-based index"):
            a = _Device(name="dup_get_log")
            b = _Device(name="dup_get_log")
            a.logger.info("from a")
            b.logger.info("from b")
            assert any("dup_get_log #1" in r for r in a.get_log())
            assert any("dup_get_log #2" in r for r in b.get_log())
            assert not any("dup_get_log #2" in r for r in a.get_log())

        with subtests.test("verbose=True echoes records to stdout"):
            capsys.readouterr()
            self.dev.logger.info("echo me")
            self.dev.get_log(verbose=True)
            assert "echo me" in capsys.readouterr().out

    def test_log_state(self):
        """Test _Loggable.log_state() tracked-attribute reporting."""
        self.dev.value = 99
        self.dev.log_state()
        assert any("value" in r and "99" in r for r in self.dev.get_log())
