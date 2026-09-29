import io
import logging

from modellab.utils import get_logger, setup_logging


def test_setup_is_idempotent():
    setup_logging("INFO")
    setup_logging("DEBUG")
    logger = logging.getLogger("modellab")
    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG


def test_level_filters_messages():
    logger = setup_logging("WARNING")
    buf = io.StringIO()
    extra = logging.StreamHandler(buf)
    logger.addHandler(extra)
    try:
        get_logger("x").info("hidden")
        get_logger("x").warning("shown")
    finally:
        logger.removeHandler(extra)
    out = buf.getvalue()
    assert "shown" in out
    assert "hidden" not in out
