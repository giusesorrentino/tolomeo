import numpy as np
import pytest

from tolomeo._lib.spectral import (
    _check_nyquist,
    _narrow_passband,
    _prepare_for_spectrum,
)

# -----------------------------------------------------------------------------
# _check_nyquist
# -----------------------------------------------------------------------------


def test_check_nyquist_below_passes():
    _check_nyquist(100.0, 10.0, 49.9)


@pytest.mark.parametrize("fc", [50.0, 60.0])
def test_check_nyquist_at_or_above_raises(fc):
    with pytest.raises(ValueError, match=r"Nyquist"):
        _check_nyquist(100.0, 10.0, fc)


# -----------------------------------------------------------------------------
# _narrow_passband
# -----------------------------------------------------------------------------


def test_narrow_passband_from_none():
    # Check that the first filter gives its own band.
    assert _narrow_passband(None, f_lower=1.0) == (1.0, np.inf)
    assert _narrow_passband(None, f_upper=30.0) == (0.0, 30.0)


def test_narrow_passband_intersection():
    # Check that one more filter keeps only the frequencies that both keep.
    assert _narrow_passband((1.0, 30.0), 5.0, 50.0) == (5.0, 30.0)


def test_narrow_passband_empty_raises():
    with pytest.raises(ValueError, match=r"no frequency band"):
        _narrow_passband((1.0, 5.0), f_lower=10.0)


# -----------------------------------------------------------------------------
# _prepare_for_spectrum
# -----------------------------------------------------------------------------


def test_prepare_for_spectrum_removes_the_mean():
    tod = np.arange(10.0) + 3.0

    out = _prepare_for_spectrum(tod, None)

    assert np.isclose(out.mean(), 0.0)


def test_prepare_for_spectrum_interpolates_the_masked_samples():
    # Check that the masked samples become a straight line through their neighbours,
    # and that the mean is computed on the good samples only.
    tod = np.array([0.0, 1.0, 100.0, 100.0, 4.0, 5.0])
    background = np.array([True, True, False, False, True, True])

    out = _prepare_for_spectrum(tod, background)

    good_mean = tod[background].mean()
    np.testing.assert_allclose(out, np.arange(6.0) - good_mean)


def test_prepare_for_spectrum_does_not_modify_the_input():
    tod = np.array([0.0, 1.0, 100.0, 3.0])
    original = tod.copy()

    _prepare_for_spectrum(tod, np.array([True, True, False, True]))

    np.testing.assert_array_equal(tod, original)


def test_prepare_for_spectrum_all_masked_raises():
    with pytest.raises(ValueError, match=r"Every sample is masked"):
        _prepare_for_spectrum(np.ones(5), np.zeros(5, dtype=bool))


def test_prepare_for_spectrum_wrong_shape_raises():
    with pytest.raises(ValueError, match=r"same shape"):
        _prepare_for_spectrum(np.ones(5), np.ones(4, dtype=bool))
