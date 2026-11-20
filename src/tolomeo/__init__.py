import logging
import sys


class _StdoutHandler(logging.StreamHandler):
    """
    ``StreamHandler`` that writes on the current ``sys.stdout``, like ``print``.

    ``StreamHandler(sys.stdout)`` would keep the stream of the moment of the import, and
    miss a later redirection (``contextlib.redirect_stdout``, the capture of pytest).
    """

    @property
    def stream(self):
        """The current ``sys.stdout``, looked up at every message."""
        return sys.stdout

    @stream.setter
    def stream(self, value):
        # StreamHandler.__init__ assigns the stream: ignored, it is always sys.stdout
        pass


# Progress and timings go to the logger "tolomeo"; every module logs on its child,
# logging.getLogger(__name__). They are shown by default, on stdout and without a
# prefix, like print. Problems the user can fix are warnings (_lib/messages.py), errors
# are exceptions.
logger = logging.getLogger("tolomeo")
_handler = _StdoutHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
# not passed on to the root logger: with a handler there too (logging.basicConfig) every
# message would be printed twice
logger.propagate = False


def debug() -> None:
    """
    Show also the detailed messages of tolomeo (level ``DEBUG``).

    For example the time of every cleaning step and the numbers behind a warning on
    the baseline fits. :func:`info` goes back to the default.
    """
    logger.setLevel(logging.DEBUG)


def info() -> None:
    """
    Show the progress messages of tolomeo (level ``INFO``, the default).

    To see only the warnings: ``logging.getLogger("tolomeo").setLevel("WARNING")``.
    """
    logger.setLevel(logging.INFO)


from . import beammap, coords, io, map, mapmaking, plotting, processing, tod

__all__ = [
    "beammap",
    "coords",
    "debug",
    "info",
    "io",
    "map",
    "mapmaking",
    "plotting",
    "processing",
    "tod",
]
