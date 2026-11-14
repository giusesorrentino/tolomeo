"""
Parallel work of the package: the number of threads of a ``workers`` argument, and a
pool of processes over the skydip files.
"""

import os
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


def _resolve_workers(workers: int | None) -> int:
    """
    Number of threads for a ``workers`` argument, with the convention of SciPy.

    * ``None``: one thread, no parallelism;
    * a positive integer: that many threads;
    * a negative integer: counted back from the number of cores, so ``-1`` is all the
      cores, ``-2`` all but one, and so on (``workers + n_cores + 1``);
    * ``0``, or less than ``-n_cores``: ``ValueError``.
    """
    n_cores = os.cpu_count() or 1

    if workers is None:
        return 1

    if workers == 0:
        raise ValueError("`workers` must not be zero: use None for a single thread.")

    if workers < -n_cores:
        raise ValueError(
            f"`workers` = {workers} is out of range: it must not be less than "
            f"{-n_cores} (minus the number of cores)."
        )

    if workers < 0:
        return workers + n_cores + 1

    return workers


def _run_on_files(
    gain_dir: str | Path,
    detector_ids: Sequence[int],
    func: Callable[..., Any],
    *args: Any,
) -> list[Any]:
    """
    Run ``func(file, detector_ids, *args)`` on every skydip file, in a process pool.

    It is what :func:`~tolomeo.io.read_skydip` does with ``func=read_skydip_file``;
    the calibration passes a function that also fits the file, so that the signal
    stays in the worker.

    Parameters
    ----------
    gain_dir : str or :class:`~pathlib.Path`
        Directory with the ``GAIN_CAL`` FITS files, or the path of a single FITS file.
    detector_ids : sequence of int
        Detectors to read, passed to ``func``.
    func : callable
        A function defined at the top level of a module (it is sent to the workers).
        It returns None for a file that has to be skipped.
    *args
        Further arguments of ``func``.

    Returns
    -------
    list
        The results of ``func`` that are not None, in the order of the file names.

    Raises
    ------
    NotADirectoryError
        If ``gain_dir`` is neither a directory nor a file.
    FileNotFoundError
        If every file is skipped.
    """
    gain_dir = Path(gain_dir).resolve()

    if gain_dir.is_file():
        # a single skydip, named explicitly: its extension is not checked
        files = [gain_dir]
    elif gain_dir.is_dir():
        # sorting by name keeps the files in time order
        files = sorted(gain_dir.glob("*.fit*", case_sensitive=False))
    else:
        raise NotADirectoryError(f"{gain_dir} is neither a directory nor a file.")

    workers = os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(func, file, detector_ids, *args) for file in files]
        results = [future.result() for future in futures]

    # a file that cannot be used comes back as None
    skipped = [file.name for file, result in zip(files, results) if result is None]
    used = [result for result in results if result is not None]

    if skipped:
        # a warning and not a log: the logger of the package has a NullHandler;
        # stacklevel 3 points at the caller of the public function
        warnings.warn(
            f"{len(skipped)} of {len(files)} skydip files skipped: "
            f"{', '.join(skipped)}.",
            RuntimeWarning,
            stacklevel=3,
        )

    if not used:
        raise FileNotFoundError(f"No usable skydip file in {gain_dir}.")

    return used
