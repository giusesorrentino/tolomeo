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


def test_tod_default_unit_is_rad():
    # The raw TOD of MISTRAL is the phase of the KIDs.
    assert _tod().unit == "rad"


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


# -----------------------------------------------------------------------------
# to
# -----------------------------------------------------------------------------


def _tod_in(unit):
    return _tod().model_copy(update={"unit": unit})


def test_tod_to_mk():
    # Check that only the signal is multiplied, the other arrays are shared, and the
    # original TOD does not change.
    tod = _tod_in("K")

    converted = tod.to("mK")

    np.testing.assert_allclose(converted.signal, 1000 * tod.signal)
    assert converted.unit == "mK"
    assert converted.time is tod.time
    assert converted.flags is tod.flags
    assert tod.unit == "K"
    np.testing.assert_array_equal(tod.signal, _tod().signal)


def test_tod_to_jy_per_beam():
    # 1 K is 0.954 Jy/beam at 90 GHz with a FWHM of 12 arcsec, as the docstring says.
    converted = _tod_in("K").to("Jy/beam")

    np.testing.assert_allclose(converted.signal, 0.954 * _tod().signal, rtol=1e-3)
    assert converted.unit == "Jy beam-1"


@pytest.mark.parametrize(
    ("unit", "target", "match"),
    [
        (None, "mK", "not known"),
        ("K", "blah", "did not parse"),
        ("K", "s", "not convertible"),
        ("rad", "K", "calibrate"),
    ],
)
def test_tod_to_invalid_raises(unit, target, match):
    with pytest.raises(ValueError, match=match):
        _tod_in(unit).to(target)


def test_tod_to_mrad():
    # A phase can still change its angular unit.
    np.testing.assert_allclose(_tod().to("mrad").signal, 1000 * _tod().signal)


@pytest.mark.parametrize("kwargs", [{"frequency": 0.0}, {"beam_fwhm": "-12 arcsec"}])
def test_tod_to_non_positive_beam_raises(kwargs):
    with pytest.raises(ValueError, match="must be positive"):
        _tod_in("K").to("mJy/beam", **kwargs)
