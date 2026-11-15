import argparse
import logging
import sys
import traceback
import warnings
from pathlib import Path

from yaml import YAMLError

from .config import RunConfig
from .runner import RunMapmaker


def main(argv: list[str] | None = None) -> int:
    """
    Run the command ``tolomeo``: read the YAML, make the map, return the exit code.

    The exit code is 0 if the map was written, 2 if the configuration cannot be read
    and 1 if the run fails. The progress messages are shown by default; ``-v`` adds
    the detailed ones and the traceback of an error, ``-q`` leaves only the warnings
    and the errors.
    """
    parser = argparse.ArgumentParser(
        prog="tolomeo", description="Make a map from a YAML configuration."
    )
    parser.add_argument("config", type=Path, help="Path to the YAML file.")
    # parser.add_argument("--validate", action="store_true", help="Check and exit.")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show also the detailed messages and, on an error, the traceback.",
    )
    verbosity.add_argument(
        "-q", "--quiet", action="store_true", help="Show only warnings and errors."
    )
    args = parser.parse_args(argv)

    # the progress messages are on the logger of the package, at INFO by default
    if args.verbose:
        logging.getLogger("tolomeo").setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger("tolomeo").setLevel(logging.WARNING)

    # a warning on one line, like the errors: the file and line of the code that issued
    # it mean nothing to the user of the command
    warnings.showwarning = _show_warning

    try:
        config = RunConfig.from_yaml(args.config)
    except (ValueError, YAMLError, FileNotFoundError, NotADirectoryError) as e:
        _report(e, verbose=args.verbose)
        return 2

    # TODO: qui deve esserci anche:
    #   * output già esistente con overwrite: false
    #   * ... idk ?

    # if args.validate:
    #     n_samples, n_det = config.scan.tod.signal.shape
    #     print(
    #         f"OK: {n_det} detectors, {n_samples} samples, "
    #         f"map {config.geometry.nx}x{config.geometry.ny}, output {config.output.path}"
    #     )
    #     return 0

    try:
        RunMapmaker(config).run()
    except Exception as e:  # noqa: BLE001  any failure of the run: message and exit code 1
        _report(e, verbose=args.verbose)
        return 1

    return 0


def _report(error: Exception, *, verbose: bool) -> None:
    """Print ``error: <message>`` on stderr, after the traceback if ``verbose``."""
    if verbose:
        traceback.print_exception(error)
    print(f"error: {error}", file=sys.stderr)


def _show_warning(message, category, filename, lineno, file=None, line=None) -> None:
    """Print a warning as ``warning: <message>`` on stderr (see ``main``)."""
    print(f"warning: {message}", file=sys.stderr)
