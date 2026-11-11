import argparse
import sys
from pathlib import Path

from yaml import YAMLError

from .config import RunConfig
from .runner import RunMapmaker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tolomeo", description="Make a map from a YAML configuration."
    )
    parser.add_argument("config", type=Path, help="Path to the YAML file.")
    # parser.add_argument("--validate", action="store_true", help="Check and exit.")
    args = parser.parse_args(argv)

    try:
        config = RunConfig.from_yaml(args.config)
    except (ValueError, YAMLError, FileNotFoundError, NotADirectoryError) as e:
        print(f"error: {e}", file=sys.stderr)
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
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0
