__version__ = '0.4.1'

from slmsuite._logging import configure_logging, get_log, logger, make_logger

configure_logging()
logger.debug(f"Activated version {__version__}")

from slmsuite._plotting import configure_plotting, capture_plots
