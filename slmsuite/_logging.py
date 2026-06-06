"""Class logging."""
import contextlib
import functools
import logging
import sys

from slmsuite._pickling import _Picklable

# Handle default appearance of the logs.

_LOGGER_COLORS = {
    "grey" :         "\033[38;20m",
    "red" :          "\033[31m",
    "orange" :       "\033[38;5;208m",
    "yellow" :       "\033[33m",
    "green" :        "\033[32m",
    "blue" :         "\033[34m",
    "cyan" :         "\033[36m",
    "magenta" :      "\033[35m",
    "bold_grey" :    "\033[1;38;20m",
    "bold_red" :     "\033[1;31m",
    "bold_orange" :  "\033[1;38;5;208m",
    "bold_yellow" :  "\033[1;33m",
    "bold_green" :   "\033[1;32m",
    "bold_blue" :    "\033[1;34m",
    "bold_cyan" :    "\033[1;36m",
    "bold_magenta" : "\033[1;35m",
    "reset" :        "\033[0m",
}

_SLMSUITE_COLORS = {
    "camera" : "bold_yellow",
    "cameraslm" : "bold_green",
    "slm" : "bold_blue",
    "hologram" : "bold_cyan",
    "slmsuite" : "bold_magenta",
    "default" : "reset",
}

def _attr_repr(value):
    if hasattr(value, "shape"):
        if len(value.shape) <= 1:
            return f"{value}"
        elif hasattr(value, "dtype"):
            return f"<{type(value).__name__} shape={value.shape} dtype={value.dtype}>"
        else:
            return f"<{type(value).__name__} shape={value.shape}>"
    elif isinstance(value, dict):
        return f"<dict keys={tuple(value.keys())}>"
    return repr(value)

class _LogCapture(logging.Handler):
    """Accumulates plain-text log records for saving to .h5 files."""

    _FMT = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(self._FMT.format(record))

    def get_log(self):
        """Return accumulated records as a list of strings."""
        return list(self.records)

class _ColorFormatter(logging.Formatter):
    _LEVEL_COLORS = {
        logging.DEBUG:    _LOGGER_COLORS["grey"],
        logging.INFO:     _LOGGER_COLORS["reset"],
        logging.WARNING:  _LOGGER_COLORS["bold_yellow"],
        logging.ERROR:    _LOGGER_COLORS["bold_red"],
        logging.CRITICAL: _LOGGER_COLORS["bold_red"],
    }
    _FMT = (
        "{grey}%(asctime)s{reset} "
        "{device}%(name)s{reset} "
        "{level}%(levelname)s:{reset} %(message)s"
    )

    def format(self, record):
        grey = _LOGGER_COLORS["grey"]
        reset = _LOGGER_COLORS["reset"]
        device_color = getattr(record, "device_color", reset)
        level_color = self._LEVEL_COLORS.get(record.levelno, reset)
        fmt = self._FMT.format(grey=grey, reset=reset, device=device_color, level=level_color)
        record = logging.makeLogRecord(record.__dict__)
        record.name = record.name.removeprefix("slmsuite.")
        return logging.Formatter(fmt).format(record)

# Make the exposed methods and loggers.

_slmsuite_logger = logging.getLogger("slmsuite")
_slmsuite_logger.setLevel(logging.DEBUG)
_slmsuite_logger.handlers = [h for h in _slmsuite_logger.handlers if not isinstance(h, _LogCapture)]
_slmsuite_log = _LogCapture()
_slmsuite_logger.addHandler(_slmsuite_log)
logger = logging.LoggerAdapter(
    _slmsuite_logger,
    extra={"device_color": _LOGGER_COLORS[_SLMSUITE_COLORS["slmsuite"]]},
)

def get_log():
    """Return all log records emitted by any slmsuite object this session."""
    return _slmsuite_log.get_log()

def configure_logging(level=logging.INFO, stream=None):
    """Configure slmsuite console logging.

    Called automatically at import with INFO level. Call again to change
    the level or output stream. Pass ``level=False`` to suppress display
    output while keeping capture active via :func:`get_log`.

    Args:
        level (int or bool): Logging level for the console handler.
            ``True`` → INFO, ``False`` → suppress display output entirely.
        stream: Output stream (default: ``sys.stdout``).
    """
    if level is True:
        level = logging.INFO
    elif level is False:
        level = logging.CRITICAL + 1

    # Remove existing display handlers so re-calls don't duplicate output.
    _slmsuite_logger.handlers = [
        h for h in _slmsuite_logger.handlers
        if isinstance(h, (_LogCapture, logging.FileHandler))
    ]
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_ColorFormatter())
    _slmsuite_logger.addHandler(handler)

# Superclass for objects we want to log.

def _wrap_with_logging(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            if hasattr(self, "logger"):
                self.logger.exception(f"{fn.__qualname__} raised:")
            raise
    wrapper._log_wrapped = True
    return wrapper


class _Loggable(_Picklable):

    # TODO: decide whether to enable this error logging.
    # def __init_subclass__(cls, **kwargs):
    #     """Adds any errors raise in methods of the subclass to the log."""
    #     super().__init_subclass__(**kwargs)
    #     for name, obj in list(cls.__dict__.items()):
    #         if name.startswith("_") or getattr(obj, "_log_wrapped", False):
    #             continue
    #         if callable(obj):
    #             setattr(cls, name, _wrap_with_logging(obj))

    def __init__(self, logger_attributes=None, logger_color=None):
        """
        Parameters
        ----------
        logger_attributes : list of str, optional
            Attributes logged at DEBUG on every ``__setattr__``. Defaults to
            ``_pickle + _pickle_data``, so any attribute added to the pickle
            list is automatically logged too. Pass an explicit list to override.
        logger_color : str, optional
            Key into :data:`_LOGGER_COLORS`. Set to defaults based on class when ``None``.
        """
        if logger_attributes is None:
            logger_attributes = self._pickle + self._pickle_data

        logger_name = f"slmsuite.{self.name}"

        if logger_color is None:
            mro_names = {c.__name__ for c in type(self).__mro__}
            if "CameraSLM" in mro_names:
                logger_color = _SLMSUITE_COLORS["cameraslm"]
            elif "Camera" in mro_names:
                logger_color = _SLMSUITE_COLORS["camera"]
            elif "SLM" in mro_names:
                logger_color = _SLMSUITE_COLORS["slm"]
            elif "Hologram" in mro_names:
                logger_color = _SLMSUITE_COLORS["hologram"]
            else:
                logger_color = _SLMSUITE_COLORS["default"]

        _logger = logging.getLogger(logger_name)
        # Set to DEBUG so all records reach the capture handler; display handler
        # applies its own level filter (set by configure_logging).
        _logger.setLevel(logging.DEBUG)
        # Replace any existing capture handler (handles re-init of same-named objects).
        _logger.handlers = [h for h in _logger.handlers if not isinstance(h, _LogCapture)]
        self._log_capture = _LogCapture()
        _logger.addHandler(self._log_capture)

        self.logger = logging.LoggerAdapter(
            _logger,
            extra={"device_color": _LOGGER_COLORS.get(logger_color, _LOGGER_COLORS["reset"])},
        )

        self._logger_attributes = logger_attributes

        self.logger.info(f"Initialized {self.__class__.__name__}.")

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        attrs = self.__dict__.get("_logger_attributes")
        if (attrs is not None
                and name in attrs
                and "logger" in self.__dict__
                and not self.__dict__.get("_attr_logging_suppressed")
                and self.logger.isEnabledFor(logging.DEBUG)):
            self.logger.debug(f"Set {name}: {_attr_repr(value)}")

    def get_log(self, verbose=False):
        """Return accumulated log records as a list of plain-text strings.

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, also print the log to the console.
        """
        log = self._log_capture.get_log()

        if verbose:
            for record in log:
                print(record)

        return log

    def set_log_level(self, level):
        """Set the logging level for this device only."""
        self.logger.logger.setLevel(level)

    def log_state(self, level=logging.DEBUG):
        """Log current values of all tracked attributes."""
        for name in self._logger_attributes:
            if hasattr(self, name):
                self.logger.log(level, f"{name}: {_attr_repr(getattr(self, name))}")

    @contextlib.contextmanager
    def log_at(self, level):
        """Context manager that temporarily changes the log level for this device."""
        prev = self.logger.logger.level
        self.logger.logger.setLevel(level)
        try:
            yield
        finally:
            self.logger.logger.setLevel(prev)

    @contextlib.contextmanager
    def suppress_attr_logging(self):
        """Context manager that suppresses attribute change logging.

        Useful when wrapping tight loops (e.g. hologram optimization) where
        tracked arrays are updated every iteration but per-iteration log lines
        are not desired.
        """
        object.__setattr__(self, "_attr_logging_suppressed", True)
        try:
            yield
        finally:
            object.__setattr__(self, "_attr_logging_suppressed", False)

    def vlog(self, verbose, msg, *args, **kwargs):
        """Log at INFO if verbose is ``True``, DEBUG otherwise.

        Replaces the old ``if verbose: print(msg)`` pattern.
        """
        self.logger.log(logging.INFO if verbose else logging.DEBUG, msg, *args, **kwargs)
