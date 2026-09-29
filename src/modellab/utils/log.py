import logging
import sys

LOGGER_NAME = "modellab"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

_handler: logging.Handler | None = None


def setup_logging(level: str = "INFO") -> logging.Logger:
    global _handler
    logger = logging.getLogger(LOGGER_NAME)
    if _handler is None:
        _handler = logging.StreamHandler(sys.stderr)
        _handler.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        logger.addHandler(_handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
