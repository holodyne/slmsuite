__version__ = '0.4.1'

from slmsuite._logging import configure_logging, get_log, logger, make_logger

logger.debug("Activated slmsuite version %s", __version__)

from slmsuite._plotting import capture_plots, configure_plotting
