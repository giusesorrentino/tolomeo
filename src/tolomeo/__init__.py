try:
    from tolomeo.__config__ import show_config  # pyright: ignore[reportMissingImports]
except ImportError as e:
    msg = """Error importing tolomeo: you cannot import tolomeo while
    being in tolomeo source directory; please exit the tolomeo source
    tree first and relaunch your Python interpreter."""
    raise ImportError(msg) from e

# version
try:
    from .version import __version__
except ImportError:
    __version__ = ""
