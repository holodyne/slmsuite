"""
slmsuite logging.

A single package-level logger owns all handlers (a :class:`_BufferHandler`
for :func:`get_log`/h5 capture, and an opt-in console handler via
:func:`configure_logging`. Each :class:`_Loggable` instance logs through a
:class:`logging.LoggerAdapter` tagged with a unique ``log_uid``, so
:meth:`_Loggable.get_log` returns only that instance's records.
"""
import collections
import itertools
import logging
import logging.config
import sys

_DEFAULT_LEVEL = logging.INFO
_BUFFER_CAPACITY = 10000

_LOGGER_COLORS = {
    "reset":                  "\033[0m",
    "grey":                   "\033[90m",
    "red":                    "\033[31m",
    "green":                  "\033[32m",
    "yellow":                 "\033[33m",
    "blue":                   "\033[34m",
    "magenta":                "\033[35m",
    "cyan":                   "\033[36m",
    "bold_red":               "\033[1;31m",
    "bold_green":             "\033[1;32m",
    "bold_yellow":            "\033[1;33m",
    "bold_blue":              "\033[1;34m",
    "bold_magenta":           "\033[1;35m",
    "bold_cyan":              "\033[1;36m",
    "bold_italic_bright_red": "\033[1;3;91m",
}

_SLMSUITE_COLORS = {
    "Camera":    "bold_blue",
    "CameraSLM": "bold_cyan",
    "SLM":       "bold_green",
    "Hologram":  "bold_yellow",
    "slmsuite":  "bold_magenta",
    "default":   "reset",
}
_LOGGER_COLORS.update({k: _LOGGER_COLORS[v] for k, v in _SLMSUITE_COLORS.items()})


def print_colors():
    """Print all entries in :data:`_LOGGER_COLORS` rendered in their own color."""
    reset = _LOGGER_COLORS["reset"]
    for name, code in _LOGGER_COLORS.items():
        print(f"{code}{name}{reset}")


def _attr_repr(value):
    """Log-friendly representation of an attribute value."""
    shape = getattr(value, "shape", None)
    if shape is not None and isinstance(shape, tuple):
        if len(shape) <= 1:
            return f"{value}"
        elif hasattr(value, "dtype"):
            return f"<{type(value).__name__} shape={shape} dtype={value.dtype}>"
        else:
            return f"<{type(value).__name__} shape={shape}>"
    elif isinstance(value, dict):
        return f"<dict keys={tuple(value.keys())}>"
    return repr(value)

#  Abbreviations to prepend onto log messages
_LEVEL_ABBR = {
    logging.DEBUG:    "DBG",
    logging.INFO:     "INF",
    logging.WARNING:  "WRN",
    logging.ERROR:    "ERR",
    logging.CRITICAL: "CRT",
}
def _level_tag(record):
    """Fixed-width ``[LVL]`` tag (always 5 chars) so class names stay aligned."""
    abbr = _LEVEL_ABBR.get(record.levelno, record.levelname[:3].upper())
    return f"[{abbr:>3.3}]"

# Global counters to keep track of _Loggable instances
_uid_counter = itertools.count()
_name_counts = {}

def _display_name(record):
    """Display name for a logging record.

    When several instances share a name, a per-name index is appended (e.g.
    ``Hologram #2`` for the second ``Hologram``) to disambiguate them. Falls back to the
    (trimmed) logger name for records logged outside a :class:`_Loggable`.
    """
    name = getattr(record, "log_name", "") or record.name.removeprefix("slmsuite.")
    index = getattr(record, "log_index", None)
    if index is not None and _name_counts.get(name, 0) > 1:
        return f"{name} #{index}"
    return name

def _infer_color(cls):
    """Pick a default device color from the first class in the MRO with a known color."""
    for c in cls.__mro__:
        if c.__name__ in _SLMSUITE_COLORS:
            return _SLMSUITE_COLORS[c.__name__]
    return _SLMSUITE_COLORS["default"]

class _PlainFormatter(logging.Formatter):
    """Uncolored format used for in-memory capture and h5 export."""

    def __init__(self):
        super().__init__("%(leveltag)s %(asctime)s %(display)s %(message)s")

    def format(self, record):
        record.display = _display_name(record)
        record.leveltag = _level_tag(record)
        return super().format(record)


class _ColorFormatter(logging.Formatter):
    """Colorized console format, with one sub-formatter built per level."""

    _LEVEL_COLORS = {
        logging.DEBUG:    _LOGGER_COLORS["grey"],
        logging.INFO:     _LOGGER_COLORS["reset"],
        logging.WARNING:  _LOGGER_COLORS["red"],
        logging.ERROR:    _LOGGER_COLORS["bold_red"],
        logging.CRITICAL: _LOGGER_COLORS["bold_italic_bright_red"],
    }

    def __init__(self):
        super().__init__()
        grey = _LOGGER_COLORS["grey"]
        reset = _LOGGER_COLORS["reset"]
        self._formatters = {
            level: logging.Formatter(
                f"{grey}%(leveltag)s{reset} {grey}%(asctime)s{reset} "
                f"%(logcolor)s%(display)s{reset} {color}%(message)s{reset}",
                "%H:%M:%S"
            )
            for level, color in self._LEVEL_COLORS.items()
        }

    def format(self, record):
        record.display = _display_name(record)
        record.leveltag = _level_tag(record)
        record.logcolor = getattr(record, "log_color", _LOGGER_COLORS["reset"])
        formatter = self._formatters.get(record.levelno, self._formatters[logging.INFO])
        return formatter.format(record)

class _BufferHandler(logging.Handler):
    """Stores all plain-text log messages for file output."""

    def __init__(self, capacity=_BUFFER_CAPACITY):
        super().__init__(level=logging.DEBUG)
        self.buffer = collections.deque(maxlen=capacity)
        self.setFormatter(_PlainFormatter())

    # Overwrite emit to store log_uids
    def emit(self, record):
        try:
            self.buffer.append((getattr(record, "log_uid", None), self.format(record)))
        except Exception:
            self.handleError(record)

# slmsuite logger: capture everything; let handlers filter by level
_package_logger = logging.getLogger("slmsuite")
_package_logger.setLevel(logging.DEBUG)
_package_logger.propagate = False
_package_logger.addHandler(logging.NullHandler())
_BUFFER = _BufferHandler()
_package_logger.addHandler(_BUFFER)

logger = logging.LoggerAdapter(
    _package_logger,
    extra={"log_name": "slmsuite", "log_color": _LOGGER_COLORS[_SLMSUITE_COLORS["slmsuite"]]},
)

def get_log():
    """Return all records emitted by any slmsuite object this session, as plain strings."""
    return [s for (_uid, s) in _BUFFER.buffer]


def configure_logging(level: "int | str | None" = _DEFAULT_LEVEL, stream=None):
    """Enable or adjust slmsuite console logging.

    slmsuite is silent by default (records are still captured for :func:`get_log` and
    pickling). Call this once to turn on colored console output.

    Parameters
    ----------
    level : int or str or None
        Console verbosity. Accepts a :mod:`logging` level (``logging.DEBUG``) or its name
        (``"DEBUG"``). ``None`` removes the console handler (capture stays active).
    stream : stream, optional
        Output stream. Defaults to ``sys.stdout``.
    """

    # Remove existing console handlers on re-call
    for handler in [h for h in _package_logger.handlers if getattr(h, "_slmsuite_console", False)]:
        _package_logger.removeHandler(handler)

    if level is None:
        return

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(level.upper() if isinstance(level, str) else level)
    handler.setFormatter(_ColorFormatter())
    handler._slmsuite_console = True
    _package_logger.addHandler(handler)


def make_logger(name, color="default"):
    """Return a colorized :class:`logging.LoggerAdapter` for use outside :class:`_Loggable`.

    Parameters
    ----------
    name : str
        Logger name, prefixed with ``"slmsuite."`` automatically. Passing a module
        ``__name__`` works too: a leading ``"slmsuite."`` is stripped to avoid doubling.
    color : str, optional
        Key into :data:`_LOGGER_COLORS` (e.g. ``"bold_cyan"``, ``"Hologram"``).
        Defaults to uncolored.
    """
    name = name.removeprefix("slmsuite.")
    return logging.LoggerAdapter(
        logging.getLogger(f"slmsuite.{name}"),
        extra={"log_name": name, "log_color": _LOGGER_COLORS.get(color, _LOGGER_COLORS["reset"])},
    )

# Imported here to avoid a circular import: _pickling pulls in
# misc/analysis/toolbox, which import make_logger above.
from slmsuite._pickling import _Picklable


class _Loggable(_Picklable):
    """Gives an object a colorized logger and an isolated slice of the shared log buffer."""

    def __init__(self, logger_attributes=None, logger_color=None):
        """Initialize logging for this object.

        Parameters
        ----------
        logger_attributes : list of str, optional
            Attributes reported by :meth:`log_state`. Defaults to ``_pickle + _pickle_data``.
        logger_color : str, optional
            Key into :data:`_LOGGER_COLORS`. Inferred from the class MRO when ``None``.
        """
        cls = type(self).__name__
        name = getattr(self, "name", "") or cls

        if logger_attributes is None:
            logger_attributes = self._pickle + self._pickle_data
        self._logger_attributes = logger_attributes

        if logger_color is None:
            logger_color = _infer_color(type(self))

        self._log_uid = next(_uid_counter)
        self._log_index = _name_counts.get(name, 0) + 1
        _name_counts[name] = self._log_index
        self.logger = logging.LoggerAdapter(
            logging.getLogger(f"slmsuite.{cls}"),
            extra={
                "log_uid": self._log_uid,
                "log_index": self._log_index,
                "log_name": name,
                "log_color": _LOGGER_COLORS.get(logger_color, _LOGGER_COLORS["reset"]),
            },
        )

        detail = self._log_detail()
        self.logger.info(f"Initialized {cls}." if detail is None else f"Initialized {cls} {detail}.")

    def _log_detail(self):
        """
        Extra phrase for the ``"Initialized ..."`` message, e.g. ``"on display 1"``.

        Override to identify which piece of hardware an instance is attached to.
        Any attribute used here must be set before :meth:`_Loggable.__init__` runs.

        Returns
        -------
        str OR None
            Phrase to append, or ``None`` for the bare message.
        """
        return None

    # def _log_setattr(self, enabled=True):
    #     """Activates or deactivates automatic logging of tracked attributes on assignment.

    #     Parameters
    #     ----------
    #     enabled : bool, optional
    #         If ``True``, any assignment to an attribute in ``_logger_attributes`` will be
    #         logged at DEBUG level. Defaults to ``True``.
    #     """
    #     self.logger.warning(f"{'Enabling' if enabled else 'Disabling'} logging of tracked attributes upon assignment.")
    #     def __setattr__(name, value):
    #         print(f"__setattr__ called with name={name}, value={value}")
    #         object.__setattr__(self, name, value)
    #         if "logger" in self.__dict__:
    #             attrs = self.__dict__.get("_logger_attributes")
    #             print(attrs)
    #             if attrs is not None and name in attrs:
    #                 self.logger.debug(f"Set {name} = {_attr_repr(value)}")

    #     __setattr__("test", "test value")

    #     super(self.__class__).__setattr__ = __setattr__ if enabled else _Loggable.__setattr__

    def get_log(self, verbose=False):
        """Return this object's log records (only), as a list of plain-text strings.

        Parameters
        ----------
        verbose : bool, optional
            If ``True``, also print the records to the console.
        """
        log = [s for (uid, s) in _BUFFER.buffer if uid == self._log_uid]
        if verbose:
            for record in log:
                print(record)
        return log

    def log_state(self, level=logging.DEBUG):
        """Log the current values of all tracked attributes.

        Parameters
        ----------
        level : int, optional
            Logging level. Defaults to ``logging.DEBUG``.
        """
        for name in self._logger_attributes:
            if hasattr(self, name):
                self.logger.log(level, "%s = %s", name, _attr_repr(getattr(self, name)))
