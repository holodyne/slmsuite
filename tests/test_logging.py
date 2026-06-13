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


class TestTopology:

    def test_silent_by_default_then_opt_in(self):
        """No console handler should exist until configure_logging() adds one."""
        configure_logging(level=None)           # ensure none
        assert _console_handlers() == []
        configure_logging("INFO")
        assert len(_console_handlers()) == 1
        configure_logging(level=None)           # restore silent

    def test_only_root_has_handlers(self):
        """Descendant (per-class) loggers must carry no handlers of their own."""
        dev = _Device(name="topology")
        child = logging.getLogger(f"slmsuite.{type(dev).__name__}")
        assert child.handlers == []
        assert child.level == logging.NOTSET    # inherits from root

    def test_buffer_attached_to_root(self):
        assert any(isinstance(h, _BufferHandler) for h in _package_logger.handlers)


class TestConfigureLogging:

    def test_idempotent(self):
        """Calling configure_logging twice must not stack console handlers."""
        configure_logging("INFO")
        configure_logging("INFO")
        assert len(_console_handlers()) == 1
        configure_logging(level=None)

    def test_level_none_removes_console(self):
        configure_logging("INFO")
        configure_logging(level=None)
        assert _console_handlers() == []

    def test_level_accepts_string_and_int(self, subtests):
        with subtests.test("string level"):
            configure_logging("DEBUG")
            assert _console_handlers()[0].level == logging.DEBUG
        with subtests.test("int level"):
            configure_logging(logging.WARNING)
            assert _console_handlers()[0].level == logging.WARNING
        configure_logging(level=None)


class TestPackageLog:

    def test_get_log_captures_instance_records(self):
        """Package get_log() must include records from _Loggable instances."""
        before = len(get_log())
        dev = _Device(name="pkg_log_test")
        dev.logger.info("hello from device")
        assert len(get_log()) > before
        assert any("hello from device" in r for r in get_log())

    def test_package_logger_captured(self):
        """Direct calls to slmsuite.logger must appear in get_log()."""
        slmsuite.logger.info("package-level message")
        assert any("package-level message" in r for r in get_log())


class TestLoggable:

    @pytest.fixture(autouse=True)
    def _dev(self):
        self.dev = _Device(name="test_dev")

    def test_init_record_in_instance_log(self):
        """Instance log must contain the 'Initialized' record."""
        assert any("Initialized" in r for r in self.dev.get_log())

    def test_log_state_reports_tracked_attrs(self):
        """log_state() must emit the current value of tracked attributes."""
        self.dev.value = 99
        self.dev.log_state()
        assert any("value" in r and "99" in r for r in self.dev.get_log())

    def test_plain_assignment_has_no_logging_side_effect(self):
        """Assigning a tracked attribute must NOT log on its own (no __setattr__ magic)."""
        before = len(self.dev.get_log())
        self.dev.value = 42
        assert len(self.dev.get_log()) == before

    def test_instance_log_isolated(self, subtests):
        """Each instance's get_log() must only contain its own records, even when two
        instances share a name (and thus the same per-class logger)."""
        other = _Device(name="test_dev")        # SAME name -> same class logger
        self.dev.logger.info("mine")
        other.logger.info("theirs")

        with subtests.test("own log excludes the other's message"):
            assert any("mine" in r for r in self.dev.get_log())
            assert not any("theirs" in r for r in self.dev.get_log())

        with subtests.test("other's log excludes ours"):
            assert any("theirs" in r for r in other.get_log())
            assert not any("mine" in r for r in other.get_log())

        with subtests.test("package log contains both"):
            assert any("mine" in r for r in get_log())
            assert any("theirs" in r for r in get_log())

    def test_duplicate_names_show_index(self, subtests):
        """A lone instance prints just its name; once a name is shared, records append a
        per-name index (#1, #2, ... in creation order), not the global uid."""
        solo = _Device(name="solo_dev")
        solo.logger.info("only one")
        with subtests.test("single instance: no index suffix"):
            assert any("solo_dev" in r and "solo_dev #" not in r for r in solo.get_log())

        a = _Device(name="dup_dev")
        b = _Device(name="dup_dev")
        a.logger.info("from a")
        b.logger.info("from b")
        with subtests.test("duplicates: 1-based per-name index in creation order"):
            assert a._log_index == 1 and b._log_index == 2
            assert any("dup_dev #1" in r for r in a.get_log())
            assert any("dup_dev #2" in r for r in b.get_log())
            assert not any("dup_dev #2" in r for r in a.get_log())

    def test_get_log_verbose_prints(self, capsys):
        """get_log(verbose=True) echoes records to stdout."""
        self.dev.logger.info("echo me")
        self.dev.get_log(verbose=True)
        assert "echo me" in capsys.readouterr().out
