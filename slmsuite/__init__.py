__version__ = '0.5.0'

# (1) Handle logging defaults
import logging

from slmsuite._logging import configure_logging, get_log, logger, make_logger

# Log at INFO by default (rather than a NullHandler) so messages show without setup.
configure_logging(logging.INFO)
logger.debug("Activated version %s", __version__)

# (2) Handle plotting defaults
from slmsuite._plotting import configure_plotting

# (3) Handle tqdm defaults
import os

if os.environ.get("SLMSUITE_TQDM_ASCII", "").lower() not in ("", "0"):
    from tqdm import tqdm   # This is used to force ascii tqdm bars when running example notebooks.
else:
    from tqdm.auto import tqdm
