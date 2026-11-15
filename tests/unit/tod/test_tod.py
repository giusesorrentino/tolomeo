import numpy as np
import pytest

from tests.common import MJD_START, SAMPLE_RATE
from tolomeo.tod import TimeOrderedData


def _tod(n_samples=10, n_det=2):
    """A TOD whose signal is the sample index, in two subscans of equal length."""
    signal = np.repeat(np.arange(n_samples, dtype=float)[:, None], n_det, axis=1)
    return TimeOrderedData(
        signal=signal,
        time=MJD_START + np.arange(n_samples) / SAMPLE_RATE / 86400,
        flags=np.ones(n_samples, dtype=bool),
        segments=np.array([0, n_samples // 2, n_samples], dtype=np.int64),
    )


# -----------------------------------------------------------------------------
# sample_rate
# -----------------------------------------------------------------------------


def test_sample_rate():
    # Check that the sampling frequency comes from the time step, in Hz. A MJD of
    # ~60000 days in float64 resolves about 1 microsecond, so the rate computed from
    # steps of 4 ms is exact only to ~1e-3.
    assert np.isclose(_tod().sample_rate, SAMPLE_RATE, rtol=1e-3)


def test_sample_rate_ignores_gaps():
    # Check that the median time step is not moved by one gap between subscans.
    tod = _tod(20)
    tod.time[10:] += 5.0 / 86400

    assert np.isclose(tod.sample_rate, SAMPLE_RATE, rtol=1e-3)


# -----------------------------------------------------------------------------
# select
# -----------------------------------------------------------------------------


def test_select_keeps_the_masked_samples():
    tod = _tod()
    mask = np.zeros(10, dtype=bool)
    mask[[1, 2, 7]] = True

    selected = tod.select(mask)

    np.testing.assert_array_equal(selected.signal[:, 0], [1.0, 2.0, 7.0])
    assert selected.time.shape == (3,)
    assert selected.flags.shape == (3,)


def test_select_moves_the_segments():
    # Check that the subscan boundaries follow the kept samples: two samples of the
    # first subscan and one of the second.
    tod = _tod()
    mask = np.zeros(10, dtype=bool)
    mask[[1, 2, 7]] = True

    selected = tod.select(mask)

    np.testing.assert_array_equal(selected.segments, [0, 2, 3])


def test_select_keeps_passband_and_unit():
    tod = _tod().model_copy(update={"passband": (1.0, 30.0), "unit": "K"})

    selected = tod.select(np.ones(10, dtype=bool))

    assert selected.passband == (1.0, 30.0)
    assert selected.unit == "K"


def test_select_non_boolean_mask_raises():
    with pytest.raises(TypeError):
        _tod().select(np.ones(10, dtype=int))


def test_select_wrong_shape_raises():
    with pytest.raises(ValueError):
        _tod().select(np.ones(9, dtype=bool))
