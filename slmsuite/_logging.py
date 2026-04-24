"""Class logging."""
import sys
import logging

_LOGGER_COLORS = {
    "grey" : "\033[38;20m",
    "red" : "\033[31m",
    "orange" : "\033[38;5;208m",
    "yellow" : "\033[33m",
    "green" : "\033[32m",
    "blue" : "\033[34m",
    "cyan" : "\033[36m",
    "magenta" : "\033[35m",
    "bold_grey" : "\033[1;38;20m",
    "bold_red" : "\033[1;31m",
    "bold_orange" : "\033[1;38;5;208m",
    "bold_yellow" : "\033[1;33m",
    "bold_green" : "\033[1;32m",
    "bold_blue" : "\033[1;34m",
    "bold_cyan" : "\033[1;36m",
    "bold_magenta" : "\033[1;35m",
    "reset" : "\033[0m",
}

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class _Loggable(object):

    def __init__(self, logger_attributes=[], logger_methods=[], logger_color="reset"):
        name = self.__class__.__name__
        if hasattr(self, "name") and len(self.name) > 0:
            name += "-" + self.name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        self._logger_attributes = logger_attributes
        self._logger_methods = logger_methods

        grey = _LOGGER_COLORS["grey"]
        color = _LOGGER_COLORS[logger_color]
        reset = _LOGGER_COLORS["reset"]

        # self.logger.formatter = logging.Formatter(
        #     fmt=(
        #         f"{grey}%(asctime)s{reset} "
        #         f"{color}%(name)s{reset} "
        #         f"{grey}%(levelname)s:{reset} %(message)s"
        #     ),
        #     # datefmt="%Y-%m-%d %H:%M:%S"
        # )

        self.logger.warning(f"Initialized.")

    # def __getattribute__(self, name):
    #     attr = super().__getattribute__(name)

    #     if callable(attr) and name in self._logger_methods:
    #         def logged_method(*args, **kwargs):
    #             self.logger.info(f"Called {name} with args={args} kwargs={kwargs}.")
    #             return attr(*args, **kwargs)
    #         return logged_method
    #     else:
    #         return attr

    # def __setattr__(self, name, value):
    #     super().__setattr__(name, value)

    #     level = self.logger.getEffectiveLevel()
    #     if level <= logging.INFO:
    #         if name in self._logger_attributes:
    #             self.logger.info(f"Set {name} to {hash(value)}.")
    #         elif level <= logging.DEBUG:
    #             self.logger.debug(f"Set {name} to {hash(value)}.")