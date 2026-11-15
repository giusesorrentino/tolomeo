import os

import pytest

from tolomeo._lib.parallel import _resolve_workers

N_CORES = os.cpu_count() or 1


# -----------------------------------------------------------------------------
# _resolve_workers
# -----------------------------------------------------------------------------


def test_resolve_workers_none_is_one_thread():
    assert _resolve_workers(None) == 1


@pytest.mark.parametrize("workers", [1, 2, 7])
def test_resolve_workers_positive(workers):
    # Check that a positive number is that many threads.
    assert _resolve_workers(workers) == workers


def test_resolve_workers_negative_counts_back_from_the_cores():
    # Check SciPy's convention: -1 is all the cores, -2 all but one.
    assert _resolve_workers(-1) == N_CORES
    assert _resolve_workers(-N_CORES) == 1


def test_resolve_workers_zero_raises():
    with pytest.raises(ValueError, match=r"must not be zero"):
        _resolve_workers(0)


def test_resolve_workers_too_negative_raises():
    with pytest.raises(ValueError, match=r"out of range"):
        _resolve_workers(-N_CORES - 1)
