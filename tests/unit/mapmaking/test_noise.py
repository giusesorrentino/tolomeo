import numpy as np
import pytest
from scipy.signal import welch

from tests.common import (
    MJD_START,
    N_SUBSCANS,
    SAMPLE_RATE,
    SUBSCAN_LEN,
    make_scan,
    make_two_passes,
    pixelized_source,
)
from tolomeo._signal.filters import gain
from tolomeo.map import MapGeometry
from tolomeo.mapmaking import InverseNoise, estimate_inverse_noise
from tolomeo.mapmaking._noise import (
    _FLOOR_FACTOR,
    _STOP_FACTOR,
    LeaveOneOutError,
    _auto_nperseg,
    _inverse_spectra,
    _shape_inverse,
)
from tolomeo.processing import Bandpass, EdgeMask, Highpass, Lowpass

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
        channel_ids=[3, 7, 12, 20][:n_det],
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
        ({"channel_ids": [3, 7, 12]}, r"has 3 detectors and the scan 4"),
        (
            {"channel_ids": [3, 12, 7, 20]},
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
    assert inv_noise.channel_ids == scan.channel_ids
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


def _two_passes_on_a_source(sigma=1.0, amplitude=100.0):
    """
    Two passes over a bright source, sampled at the centres of the pixels of the
    returned geometry, with white noise of rms ``sigma``.

    The source is constant inside every pixel, so a binned map models it exactly. The
    TOD is marked as calibrated, as the leave-one-out wants.
    """
    scan = make_two_passes(subscan_len=4096, noise_rms=sigma)
    geometry = MapGeometry.from_scan(scan, pixel_size=10.0)
    _, source = pixelized_source(scan, geometry, amplitude)
    tod = scan.tod.model_copy(update={"signal": scan.tod.signal + source, "unit": "K"})
    return scan.model_copy(update={"tod": tod}), geometry


def test_estimate_inverse_noise_leave_one_out_removes_the_source():
    # Check that the leave-one-out gives back the level of the white noise under a
    # bright source, fs / (2 sigma^2), while the plain estimate sees the source at the
    # low frequencies. The model of each half adds sigma^2 / hits to the variance,
    # about 0.6% with ~170 samples per pixel: within the tolerance.
    sigma = 1.0
    scan, geometry = _two_passes_on_a_source(sigma=sigma)
    white = scan.tod.sample_rate / (2 * sigma**2)

    plain = estimate_inverse_noise(scan, nperseg=256, workers=1)
    loo = estimate_inverse_noise(
        scan, nperseg=256, geometry=geometry, leave_one_out=True, workers=1
    )

    level = np.median(loo.inv_noiseps[1:-1], axis=0)
    np.testing.assert_allclose(level, white, rtol=0.03)
    assert np.all(np.median(plain.inv_noiseps[1:9], axis=0) < 0.2 * white)


def test_estimate_inverse_noise_leave_one_out_single_subscan_raises():
    scan = make_scan(n_subscans=1, subscan_len=1024)
    geometry = MapGeometry.from_scan(scan, pixel_size=10.0)

    with pytest.raises(LeaveOneOutError, match="single subscan"):
        estimate_inverse_noise(
            scan, nperseg=128, geometry=geometry, leave_one_out=True, workers=1
        )


@pytest.mark.parametrize("unit", ["rad", None])
def test_estimate_inverse_noise_leave_one_out_uncalibrated_warns(unit):
    # "rad" is the raw TOD of MISTRAL, None a TOD of unknown unit: neither is
    # calibrated.
    scan, geometry = _two_passes_on_a_source()
    raw = scan.model_copy(update={"tod": scan.tod.model_copy(update={"unit": unit})})

    with pytest.warns(UserWarning, match="not calibrated"):
        estimate_inverse_noise(
            raw, nperseg=256, geometry=geometry, leave_one_out=True, workers=1
        )


def test_estimate_inverse_noise_leave_one_out_one_raster_raises():
    # Check that one raster cut in two is refused: the halves see different rows.
    scan = make_scan()
    geometry = MapGeometry.from_scan(scan, pixel_size=10.0)

    with pytest.raises(ValueError, match="at least two scans"):
        estimate_inverse_noise(
            scan, nperseg=128, geometry=geometry, leave_one_out=True, workers=1
        )


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"mask": EdgeMask()}, "a mask"),
        ({"background": np.ones(2 * N_SUBSCANS * SUBSCAN_LEN, dtype=bool)}, "a mask"),
        ({"geometry": None}, "needs `geometry`"),
    ],
    ids=["mask", "background", "no-geometry"],
)
def test_estimate_inverse_noise_leave_one_out_invalid_arguments_raise(update, match):
    scan = make_two_passes()
    arguments = {
        "nperseg": 128,
        "geometry": MapGeometry.from_scan(scan, pixel_size=10.0),
        "leave_one_out": True,
        "workers": 1,
    }

    with pytest.raises(ValueError, match=match):
        estimate_inverse_noise(scan, **(arguments | update))


def test_estimate_inverse_noise_mask_and_background_raises():
    scan = make_scan()

    with pytest.raises(ValueError, match=r"either `background` or `mask`"):
        estimate_inverse_noise(
            scan, nperseg=64, mask=EdgeMask(), background=scan.tod.flags
        )


def test_estimate_inverse_noise_samples_left_out(capsys):
    # Check the samples left out of the spectra, from the message of the log: the
    # flags alone, then an EdgeMask combined with the flags.
    scan = make_scan(n_subscans=2, subscan_len=100)
    scan.tod.flags[0] = False

    estimate_inverse_noise(scan, nperseg=64, workers=1)
    flags_only = capsys.readouterr().out
    estimate_inverse_noise(scan, nperseg=64, mask=EdgeMask(fraction=0.1), workers=1)
    with_mask = capsys.readouterr().out

    # 4 detectors: sample 0 of each; then also the middle 80 samples of each subscan
    assert "4 of 800 detector samples are masked" in flags_only
    assert "644 of 800 detector samples are masked" in with_mask


# -----------------------------------------------------------------------------
# _auto_nperseg
# -----------------------------------------------------------------------------


def _times(n):
    return MJD_START + np.arange(n) / SAMPLE_RATE / 86400


def _segmented_times(lengths):
    """Time of a TOD made of segments of the given lengths, one second apart."""
    times = []
    start = MJD_START
    for length in lengths:
        times.append(start + np.arange(length) / SAMPLE_RATE / 86400)
        start = times[-1][-1] + 1.0 / 86400
    return np.concatenate(times)


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
    time = _segmented_times(lengths)

    assert _auto_nperseg(time, np.ones(time.size, dtype=bool), SAMPLE_RATE) == expected


def test_auto_nperseg_flags_and_gaps_end_the_segments():
    # Check that a run of flagged samples and a jump in time both end a segment: if
    # either were missed, the 100 samples would be one segment, and nperseg 64.
    ok = np.ones(100, dtype=bool)
    ok[20:30] = False  # segments of 20 and 70 samples: 16
    time = _times(100)
    time[60:] += 1.0 / 86400  # one second of nothing: segments of 60 and 40, 32

    assert _auto_nperseg(_times(100), np.ones(100, dtype=bool), SAMPLE_RATE) == 64
    assert _auto_nperseg(_times(100), ok, SAMPLE_RATE) == 16
    assert _auto_nperseg(time, np.ones(100, dtype=bool), SAMPLE_RATE) == 32


def test_auto_nperseg_no_good_sample_raises():
    with pytest.raises(ValueError, match=r"No good sample"):
        _auto_nperseg(_times(10), np.zeros(10, dtype=bool), SAMPLE_RATE)


# -----------------------------------------------------------------------------
# _inverse_spectra
# -----------------------------------------------------------------------------


def test_inverse_spectra_against_welch_with_the_floor():
    # Check against scipy: the inverse of the Welch spectrum of every detector, zero at
    # f = 0, with the floor where the TOD has almost no power. The band from 40 to
    # 80 Hz is removed from the TOD, so that the floor is used there.
    n, nperseg = 8192, 256
    spectrum = np.fft.rfft(rng.standard_normal((n, 2)), axis=0)
    f = np.fft.rfftfreq(n, d=1 / SAMPLE_RATE)
    spectrum[(f > 40) & (f < 80)] = 0.0
    tod = np.fft.irfft(spectrum, n=n, axis=0)

    inverse = _inverse_spectra(tod, SAMPLE_RATE, nperseg)

    _, psd = welch(tod, fs=SAMPLE_RATE, window="hann", nperseg=nperseg, axis=0)
    floor = np.median(psd, axis=0) * _FLOOR_FACTOR
    assert np.all((psd < floor).any(axis=0))  # the floor is used in every detector
    expected = 1 / np.maximum(psd, floor)
    expected[0] = 0.0
    np.testing.assert_allclose(inverse, expected, rtol=1e-9)


def test_inverse_spectra_background_of_all_the_detectors():
    # Check that a background of shape (n_samples,) is used for every detector.
    tod = rng.standard_normal((2048, 3))
    background = np.ones(2048, dtype=bool)
    background[500:700] = False

    one = _inverse_spectra(tod, SAMPLE_RATE, 128, background)
    every = _inverse_spectra(tod, SAMPLE_RATE, 128, np.tile(background[:, None], 3))

    np.testing.assert_array_equal(one, every)


def test_inverse_spectra_1d_tod_raises():
    with pytest.raises(ValueError, match=r"must be 2D"):
        _inverse_spectra(np.zeros(256), SAMPLE_RATE, 64)


# -----------------------------------------------------------------------------
# _shape_inverse
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filt, fc",
    [
        (Lowpass(fc=30.0, order=2), 30.0),
        (Highpass(fc=30.0), 30.0),
        (Bandpass(f_lower=20.0, f_upper=60.0), (20.0, 60.0)),
    ],
    ids=["lowpass", "highpass", "bandpass"],
)
def test_shape_inverse_filters(filt, fc):
    inv = rng.random((FREQS.size, 2))

    out = _shape_inverse(inv, FREQS, [filt], None, SAMPLE_RATE)

    expected = gain(FREQS, fc, SAMPLE_RATE, filt.order, btype=filt.type)[:, None] * inv
    np.testing.assert_allclose(out, expected)


def test_shape_inverse_marginalization():
    inv = rng.uniform(1, 2, (FREQS.size, 3))
    inv[0] = 0.0
    original = inv.copy()

    out = _shape_inverse(inv, FREQS, [], (20.0, 80.0), SAMPLE_RATE)

    inside = (FREQS >= 20.0) & (FREQS <= 80.0)
    np.testing.assert_array_equal(out[inside], inv[inside])
    stopped = out[~inside & (FREQS > 0)]
    np.testing.assert_allclose(
        stopped, np.broadcast_to(_STOP_FACTOR * inv[inside].max(axis=0), stopped.shape)
    )
    np.testing.assert_array_equal(out[0], 0.0)
    np.testing.assert_array_equal(inv, original)  # input untouched


def test_shape_inverse_filters_before_marginalization():
    # Check that the level outside the band is relative to the filtered weights.
    inv = rng.uniform(1, 2, (FREQS.size, 2))
    inv[0] = 0.0
    filt = Lowpass(fc=60.0, order=2)

    out = _shape_inverse(inv, FREQS, [filt], (0.0, 60.0), SAMPLE_RATE)

    filtered = gain(FREQS, 60.0, SAMPLE_RATE, 2, btype="lowpass")[:, None] * inv
    inside = (FREQS > 0) & (FREQS <= 60.0)
    np.testing.assert_allclose(out[inside], filtered[inside])
    np.testing.assert_allclose(
        out[FREQS > 60.0],
        np.broadcast_to(
            _STOP_FACTOR * filtered[inside].max(axis=0), out[FREQS > 60.0].shape
        ),
    )


def test_shape_inverse_without_filters_and_band_does_nothing():
    inv = rng.random((FREQS.size, 2))

    assert _shape_inverse(inv, FREQS, [], None, SAMPLE_RATE) is inv
