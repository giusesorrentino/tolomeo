"""Number of threads for the ``workers`` arguments of the package."""

import os


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
