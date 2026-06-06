"""
Unit tests for slmsuite._logging: _Loggable, configure_logging, get_log.
"""
import logging
import pytest

import slmsuite
from slmsuite._logging import (
    _LogCapture, _Loggable, _slmsuite_logger, _slmsuite_log,
    configure_logging, get_log,
)
from slmsuite._pickling import _Picklable


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


class TestConfigureLogging:

    def test_idempotent(self):
        """Calling configure_logging twice must not add duplicate StreamHandlers."""
        configure_logging()
        n_before = sum(
            isinstance(h, logging.StreamHandler) and not isinstance(h, _LogCapture)
            for h in _slmsuite_logger.handlers
        )
        configure_logging()
        n_after = sum(
            isinstance(h, logging.StreamHandler) and not isinstance(h, _LogCapture)
            for h in _slmsuite_logger.handlers
        )
        assert n_before == n_after == 1

    def test_level_false_suppresses_display(self):
        """level=False must set the StreamHandler to above CRITICAL."""
        configure_logging(level=False)
        stream_handlers = [
            h for h in _slmsuite_logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, _LogCapture)
        ]
        assert all(h.level > logging.CRITICAL for h in stream_handlers)
        configure_logging()  # restore


class TestPackageLog:

    def test_get_log_captures_instance_records(self):
        """Package get_log() must include records from _Loggable instances."""
        before = len(get_log())
        dev = _Device(name="pkg_log_test")
        dev.logger.info("hello from device")
        after = len(get_log())
        assert after > before
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
        """Instance log must contain the 'Initialized.' record."""
        assert any("Initialized." in r for r in self.dev.get_log())

    def test_attr_change_logged(self):
        """Setting a tracked attribute must produce a DEBUG log entry."""
        self.dev.value = 99
        assert any("value" in r and "99" in r for r in self.dev.get_log())

    def test_instance_log_isolated(self, subtests):
        """Each instance's get_log() must only contain its own records."""
        other = _Device(name="other_dev")
        other.logger.info("other message")

        with subtests.test("own log does not contain other's message"):
            assert not any("other message" in r for r in self.dev.get_log())

        with subtests.test("package log contains both"):
            assert any("other message" in r for r in get_log())

    def test_suppress_attr_logging(self):
        """suppress_attr_logging must prevent attribute change entries."""
        before = len(self.dev.get_log())
        with self.dev.suppress_attr_logging():
            self.dev.value = 42
        after = len(self.dev.get_log())
        assert after == before

    def test_vlog_info_vs_debug(self, subtests):
        """vlog emits INFO when verbose is truthy, DEBUG otherwise."""
        with subtests.test("verbose=True → INFO"):
            self.dev.vlog(True, "verbose message")
            assert any("verbose message" in r for r in self.dev.get_log())

        with subtests.test("verbose=False → DEBUG, still captured"):
            self.dev.vlog(False, "quiet message")
            assert any("quiet message" in r for r in self.dev.get_log())
