import numpy as np
import pytest

from tests.common import MJD_START, SAMPLE_RATE, make_scan
from tolomeo._signal.filters import gain
from tolomeo.mapmaking import InverseNoise, estimate_inverse_noise
from tolomeo.mapmaking._noise import (
    _FLOOR_FACTOR,
    _STOP_FACTOR,
    _auto_nperseg,
    _filter_inverse,
    _find_segments,
    _inverse_with_floor,
    _marginalize,
    _noise_background,
)
from tolomeo.processing import Bandpass, Highpass, Lowpass
from tolomeo.processing.masks import EdgeMask

rng = np.random.default_rng(23)

FREQS = np.fft.rfftfreq(16, d=1 / SAMPLE_RATE)  # 9 bins, 15.25 Hz apart


def _long_scan(noise_rms=2.0, **kwargs):
    """White noise scan of 32768 samples: enough Welch blocks for a stable spectrum."""
    return make_scan(n_subscans=8, subscan_len=4096, noise_rms=noise_rms, **kwargs)


def _inverse_noise(n_det=4, nperseg=16, sample_rate=SAMPLE_RATE):
    return InverseNoise(
        inv_noiseps=np.ones((nperseg // 2 + 1, n_det)),
        det_pairs=np.array([(j, j) for j in range(n_det)], dtype=np.uintp),
        lag_zero=np.ones(n_det),
        sample_rate=sample_rate,
        detector_ids=[3, 7, 12, 20][:n_det],
    )


# -----------------------------------------------------------------------------
# InverseNoise
# -----------------------------------------------------------------------------


def test_inverse_noise_properties():
    inv_noise = _inverse_noise(nperseg=16)

    assert inv_noise.nperseg == 16
    assert inv_noise.n_det == 4
    np.testing.assert_allclose(inv_noise.freqs, FREQS)


def test_inverse_noise_check_scan_accepts_its_scan():
    scan = make_scan()

    _inverse_noise(sample_rate=scan.tod.sample_rate).check_scan(scan)


@pytest.mark.parametrize(
    "update, match",
    [
        ({"detector_ids": [3, 7, 12]}, r"has 3 detectors and the scan 4"),
        (
            {"detector_ids": [3, 12, 7, 20]},
            r"at position 1 the model has 12 and the scan 7",
        ),
        ({"sample_rate": 100.0}, r"estimated at 100.0 Hz"),
    ],
    ids=["n_det", "order", "sample_rate"],
)
def test_inverse_noise_check_scan_refuses_another_scan(update, match):
    scan = make_scan()
    inv_noise = _inverse_noise(sample_rate=scan.tod.sample_rate).model_copy(
        update=update
    )

    with pytest.raises(ValueError, match=match):
        inv_noise.check_scan(scan)


# -----------------------------------------------------------------------------
# estimate_inverse_noise
# -----------------------------------------------------------------------------


def test_estimate_inverse_noise_white_noise_level():
    # The one-sided PSD of white noise of variance sigma^2 is 2 sigma^2 / fs, so the
    # inverse spectrum is fs / (2 sigma^2) at every frequency but 0 (set to zero) and
    # Nyquist (one-sided Welch does not double it).
    sigma = 2.0
    scan = _long_scan(noise_rms=sigma)
    fs = scan.tod.sample_rate

    inv_noise = estimate_inverse_noise(scan, nperseg=256, workers=1)

    assert inv_noise.inv_noiseps.shape == (129, 4)
    np.testing.assert_array_equal(inv_noise.inv_noiseps[0], 0.0)
    level = np.median(inv_noise.inv_noiseps[1:-1], axis=0)
    np.testing.assert_allclose(level, fs / (2 * sigma**2), rtol=0.03)


def test_estimate_inverse_noise_outputs():
    scan = _long_scan()

    inv_noise = estimate_inverse_noise(scan, nperseg=128, workers=2)

    assert inv_noise.nperseg == 128
    assert inv_noise.detector_ids == scan.detector_ids
    assert inv_noise.sample_rate == scan.tod.sample_rate
    np.testing.assert_array_equal(inv_noise.det_pairs, [(j, j) for j in range(4)])
    np.testing.assert_allclose(
        inv_noise.lag_zero, inv_noise.inv_noiseps.sum(axis=0) / 128
    )
    assert inv_noise.passband is None
    assert inv_noise.filters == ()


def test_estimate_inverse_noise_filters_and_marginalization():
    # Check that the filters are applied to the inverse spectra, and that outside
    # their band the spectra are _STOP_FACTOR times their maximum inside it.
    scan = _long_scan()
    filters = [Highpass(fc=5.0), Lowpass(fc=60.0)]

    inv_noise = estimate_inverse_noise(scan, nperseg=128, filters=filters, workers=1)
    plain = estimate_inverse_noise(scan, nperseg=128, workers=1)

    freqs = inv_noise.freqs
    inside = (freqs >= 5.0) & (freqs <= 60.0)
    total_gain = gain(freqs, 5.0, scan.tod.sample_rate, 4, btype="highpass") * gain(
        freqs, 60.0, scan.tod.sample_rate, 4, btype="lowpass"
    )
    expected = plain.inv_noiseps * total_gain[:, None]

    assert inv_noise.passband == (5.0, 60.0)
    assert inv_noise.filters == tuple(filters)
    np.testing.assert_allclose(inv_noise.inv_noiseps[inside], expected[inside])
    stopped = inv_noise.inv_noiseps[~inside & (freqs > 0)]
    np.testing.assert_allclose(
        stopped,
        np.broadcast_to(_STOP_FACTOR * expected[inside].max(axis=0), stopped.shape),
    )


def test_estimate_inverse_noise_uses_the_passband_of_the_tod():
    scan = _long_scan()
    scan = scan.model_copy(
        update={"tod": scan.tod.model_copy(update={"passband": (2.0, 50.0)})}
    )

    inv_noise = estimate_inverse_noise(scan, nperseg=128, filters=[Lowpass(fc=40.0)])

    assert inv_noise.passband == (2.0, 40.0)


def test_estimate_inverse_noise_automatic_nperseg():
    # Subscans of 4096 samples with a gap in time between them: blocks of 4096.
    scan = _long_scan(gap=1.0)

    inv_noise = estimate_inverse_noise(scan, nperseg=None, workers=1)

    assert inv_noise.nperseg == 4096


@pytest.mark.parametrize("nperseg", [0, 127])
def test_estimate_inverse_noise_invalid_nperseg_raises(nperseg):
    with pytest.raises(ValueError, match=r"must be even and at least 2"):
        estimate_inverse_noise(make_scan(), nperseg=nperseg)


def test_estimate_inverse_noise_nperseg_longer_than_the_tod_raises():
    with pytest.raises(ValueError, match=r"fewer than nperseg"):
        estimate_inverse_noise(make_scan(), nperseg=4096)


def test_estimate_inverse_noise_cutoff_above_nyquist_raises():
    with pytest.raises(ValueError, match=r"Nyquist"):
        estimate_inverse_noise(make_scan(), nperseg=64, filters=[Lowpass(fc=500.0)])


def test_estimate_inverse_noise_cross_spectra_not_implemented():
    with pytest.raises(NotImplementedError):
        estimate_inverse_noise(make_scan(), nperseg=64, cross_spectra=True)


def test_estimate_inverse_noise_nan_raises():
    scan = make_scan()
    scan.tod.signal[10, 2] = np.nan

    with pytest.raises(ValueError, match=r"NaN or inf"):
        estimate_inverse_noise(scan, nperseg=64)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def test_noise_background_both_given_raises():
    scan = make_scan()

    with pytest.raises(ValueError, match=r"either `background` or `mask`"):
        _noise_background(scan, EdgeMask(), None, scan.tod.flags)


def test_noise_background_default_is_the_flags():
    scan = make_scan()
    scan.tod.flags[:7] = False

    np.testing.assert_array_equal(
        _noise_background(scan, None, None, None), scan.tod.flags
    )


def test_noise_background_mask_and_flags():
    scan = make_scan(n_subscans=2, subscan_len=100)
    scan.tod.flags[0] = False

    background = _noise_background(scan, EdgeMask(fraction=0.1), None, None)

    assert not background[0]
    assert background[1:10].all()
    assert not background[10:90].any()


def test_inverse_with_floor():
    psd = np.array([5.0, 2.0, -4.0, 1.0, 0.0])

    inverse = _inverse_with_floor(psd)

    floor = np.median(np.abs(psd)) * _FLOOR_FACTOR
    np.testing.assert_allclose(inverse, [0.0, 0.5, -0.25, 1.0, 1 / floor])


def test_marginalize():
    inv = rng.uniform(1, 2, (FREQS.size, 3))
    inv[0] = 0.0
    original = inv.copy()

    out = _marginalize(inv, FREQS, (20.0, 80.0))

    inside = (FREQS >= 20.0) & (FREQS <= 80.0)
    np.testing.assert_array_equal(out[inside], inv[inside])
    stopped = out[~inside & (FREQS > 0)]
    np.testing.assert_allclose(
        stopped, np.broadcast_to(_STOP_FACTOR * inv[inside].max(axis=0), stopped.shape)
    )
    np.testing.assert_array_equal(out[0], 0.0)
    np.testing.assert_array_equal(inv, original)  # input untouched


def test_marginalize_without_passband_does_nothing():
    inv = rng.random((FREQS.size, 2))

    assert _marginalize(inv, FREQS, None) is inv


@pytest.mark.parametrize(
    "filt, fc",
    [
        (Lowpass(fc=30.0, order=2), 30.0),
        (Highpass(fc=30.0), 30.0),
        (Bandpass(f_lower=20.0, f_upper=60.0), (20.0, 60.0)),
    ],
    ids=["lowpass", "highpass", "bandpass"],
)
def test_filter_inverse(filt, fc):
    inv = rng.random((FREQS.size, 2))

    out = _filter_inverse(inv, FREQS, [filt], SAMPLE_RATE)

    expected = gain(FREQS, fc, SAMPLE_RATE, filt.order, btype=filt.type)[:, None] * inv
    np.testing.assert_allclose(out, expected)


def test_filter_inverse_without_filters_does_nothing():
    inv = rng.random((FREQS.size, 2))

    assert _filter_inverse(inv, FREQS, [], SAMPLE_RATE) is inv


def _times(n):
    return MJD_START + np.arange(n) / SAMPLE_RATE / 86400


def test_find_segments_contiguous():
    segments = _find_segments(_times(100), np.ones(100, dtype=bool), SAMPLE_RATE)

    np.testing.assert_array_equal(segments, [[0, 100]])


def test_find_segments_flags_and_gaps():
    # Check that a run of flagged samples and a jump in time both end a segment, and
    # that the indices refer to the full TOD.
    time = _times(100)
    time[60:] += 1.0 / 86400  # one second of nothing before sample 60
    ok = np.ones(100, dtype=bool)
    ok[:5] = False
    ok[20:30] = False

    segments = _find_segments(time, ok, SAMPLE_RATE)

    np.testing.assert_array_equal(segments, [[5, 20], [30, 60], [60, 100]])


def test_find_segments_no_good_sample_raises():
    with pytest.raises(ValueError, match=r"No good sample"):
        _find_segments(_times(10), np.zeros(10, dtype=bool), SAMPLE_RATE)


@pytest.mark.parametrize(
    "lengths, expected",
    [
        ([1000], 512),
        ([1024], 1024),
        ([1000, 50], 512),  # 50 of 1050 samples left out: below 10%
        ([300, 1000], 256),  # with 512, 300 of 1300 samples left out: above 10%
        ([3, 3], 2),
    ],
)
def test_auto_nperseg(lengths, expected):
    stops = np.cumsum(lengths)
    segments = np.column_stack([stops - lengths, stops])

    assert _auto_nperseg(segments) == expected
