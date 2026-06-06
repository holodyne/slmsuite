__version__ = '0.4.1'

from slmsuite._logging import configure_logging, get_log, logger

configure_logging()

from slmsuite._plotting import configure_plotting, capture_plots
