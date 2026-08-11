__version__ = '0.5.0'

import logging

from slmsuite._logging import configure_logging, get_log, logger, make_logger

# This is non-idiomatic (vs configuring a NullHandler), but we expect more people
# will be confused by having to configure logging to get print statements
# vs those who will be be configuring slmsuite logging externally. So we'll configure
# to INFO level here.
configure_logging(logging.INFO)
logger.debug("Activated version %s", __version__)

from slmsuite._plotting import configure_plotting
