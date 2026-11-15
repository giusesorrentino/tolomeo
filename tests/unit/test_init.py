import contextlib
import io
import logging

import pytest

import tolomeo


@pytest.fixture(autouse=True)
def restore_log_level():
    logger = logging.getLogger("tolomeo")
    level = logger.level
    yield
    logger.setLevel(level)


# -----------------------------------------------------------------------------
# logger of the package
# -----------------------------------------------------------------------------


def test_logger_defaults():
    logger = logging.getLogger("tolomeo")

    assert logger.level == logging.INFO
    assert not logger.propagate


def test_debug_and_info():
    logger = logging.getLogger("tolomeo")

    tolomeo.debug()
    assert logger.level == logging.DEBUG

    tolomeo.info()
    assert logger.level == logging.INFO


def test_messages_go_to_the_current_stdout(capsys):
    # Check that a module logger writes the bare message on stdout, as print does.
    logging.getLogger("tolomeo.mapmaking").info("hello")

    out, err = capsys.readouterr()
    assert out == "hello\n"
    assert err == ""


def test_messages_follow_a_redirection():
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        logging.getLogger("tolomeo.io").info("redirected")

    assert buffer.getvalue() == "redirected\n"


def test_debug_messages_hidden_by_default(capsys):
    logging.getLogger("tolomeo.processing").debug("details")

    out, _ = capsys.readouterr()
    assert out == ""
