import logging

logger = logging.getLogger("tolomeo")
logger.addHandler(logging.NullHandler())


def debug() -> None:
    """Abilita i log di debug per tolomeo."""
    logger.setLevel(logging.DEBUG)
    _ensure_handler()


def info() -> None:
    """Imposta il livello di log su INFO per tolomeo."""
    logger.setLevel(logging.INFO)


def _ensure_handler() -> None:
    """Aggiunge un StreamHandler se l'utente non ha configurato alcun handler."""
    if not any(not isinstance(h, logging.NullHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)


from . import coords, io, map, mapmaking, plotting, processing, tod

__all__ = [
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
