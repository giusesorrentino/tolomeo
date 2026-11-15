import numpy as np
import pytest

from tolomeo._signal.filters import bandpass, gain, highpass, lowpass

SAMPLE_RATE = 100.0
N_SAMPLES = 4096
T = np.arange(N_SAMPLES) / SAMPLE_RATE

# the central samples, away from the transients at the two ends
CENTRE = slice(N_SAMPLES // 4, 3 * N_SAMPLES // 4)


def _tone(freq):
    return np.sin(2 * np.pi * freq * T)


def _impulse():
    impulse = np.zeros(N_SAMPLES)
    impulse[N_SAMPLES // 2] = 1.0
    return impulse


# -----------------------------------------------------------------------------
# lowpass, highpass, bandpass
# -----------------------------------------------------------------------------


def test_lowpass_keeps_low_and_removes_high_frequencies():
    out_low = lowpass(_tone(0.5), fc=5.0, sample_rate=SAMPLE_RATE)
    out_high = lowpass(_tone(30.0), fc=5.0, sample_rate=SAMPLE_RATE)

    np.testing.assert_allclose(out_low[CENTRE], _tone(0.5)[CENTRE], atol=1e-3)
    assert np.abs(out_high[CENTRE]).max() < 1e-3


def test_highpass_keeps_high_and_removes_low_frequencies():
    out_low = highpass(_tone(0.5), fc=5.0, sample_rate=SAMPLE_RATE)
    out_high = highpass(_tone(30.0), fc=5.0, sample_rate=SAMPLE_RATE)

    assert np.abs(out_low[CENTRE]).max() < 1e-3
    np.testing.assert_allclose(out_high[CENTRE], _tone(30.0)[CENTRE], atol=1e-3)


def test_bandpass_keeps_only_the_band():
    out_in = bandpass(_tone(5.0), 2.0, 10.0, sample_rate=SAMPLE_RATE)
    out_out = bandpass(_tone(40.0), 2.0, 10.0, sample_rate=SAMPLE_RATE)

    np.testing.assert_allclose(out_in[CENTRE], _tone(5.0)[CENTRE], atol=1e-2)
    assert np.abs(out_out[CENTRE]).max() < 1e-3


def test_filters_have_zero_phase():
    # Check that a forward and backward filter does not delay the signal: the
    # response to a centred impulse is symmetric.
    out = lowpass(_impulse(), fc=5.0, sample_rate=SAMPLE_RATE)

    np.testing.assert_allclose(out[1:], out[1:][::-1], atol=1e-12)


def test_filters_along_axis():
    # Check that the filter runs along the given axis, detector by detector.
    data = np.column_stack([_tone(0.5), _tone(30.0)])

    out = lowpass(data, fc=5.0, sample_rate=SAMPLE_RATE, axis=0)

    np.testing.assert_allclose(out[:, 0], lowpass(_tone(0.5), 5.0, SAMPLE_RATE))
    np.testing.assert_allclose(out[:, 1], lowpass(_tone(30.0), 5.0, SAMPLE_RATE))


# -----------------------------------------------------------------------------
# gain
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func, fc, btype",
    [
        (lambda x: lowpass(x, 5.0, SAMPLE_RATE, order=4), 5.0, "lowpass"),
        (lambda x: highpass(x, 5.0, SAMPLE_RATE, order=4), 5.0, "highpass"),
        (
            lambda x: bandpass(x, 2.0, 10.0, SAMPLE_RATE, order=3),
            (2.0, 10.0),
            "bandpass",
        ),
    ],
    ids=["lowpass", "highpass", "bandpass"],
)
def test_gain_is_the_response_of_the_filter(func, fc, btype):
    # Check that gain() is the modulus of the response of the forward and backward
    # filter to an impulse, at the FFT frequencies.
    order = 3 if btype == "bandpass" else 4
    freqs = np.fft.rfftfreq(N_SAMPLES, d=1 / SAMPLE_RATE)

    response = np.abs(np.fft.rfft(func(_impulse())))

    np.testing.assert_allclose(
        gain(freqs, fc, SAMPLE_RATE, order, btype), response, atol=1e-10
    )


@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
def test_gain_at_the_cutoff_is_one_half(btype):
    # Check that a Butterworth filter applied twice has gain |H|^2 = 1/2 at the cutoff.
    g = gain(np.array([5.0]), 5.0, SAMPLE_RATE, 4, btype)

    np.testing.assert_allclose(g, 0.5)
