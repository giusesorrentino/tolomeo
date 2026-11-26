import numpy as np
import pytest
from scipy.signal import welch

from tolomeo._signal.realization import (
    _apply_precision,
    draw_noise,
    fill_gaps,
    rfft_variances,
)

SAMPLE_RATE = 244.0


def _one_over_f(n_samples, knee=1.0, alpha=2.0, white=1.0):
    """S_k of white noise plus a 1/f with ``knee`` in Hz, flat below bin 1."""
    f = np.fft.rfftfreq(n_samples, d=1 / SAMPLE_RATE)
    f[0] = f[1]
    return white * (1 + (knee / f) ** alpha)


def _gaps(n_samples, starts, width):
    """True everywhere but ``width`` samples from every one of ``starts``."""
    good = np.ones(n_samples, dtype=bool)
    for start in starts:
        good[start : start + width] = False
    return good


# -----------------------------------------------------------------------------
# rfft_variances
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [1024, 1025])
def test_rfft_variances_white_noise(n_samples):
    # A flat one-sided PSD P gives S_k = fs P / 2 at every frequency, bin 0 included.
    freqs = np.fft.rfftfreq(64, d=1 / SAMPLE_RATE)
    psd = np.full(freqs.size, 3.0)

    variances = rfft_variances(freqs, psd, n_samples, SAMPLE_RATE)

    assert variances.shape == (n_samples // 2 + 1,)
    np.testing.assert_allclose(variances, SAMPLE_RATE * 3.0 / 2, rtol=1e-12)


def test_rfft_variances_on_the_frequencies_of_the_estimate():
    # Where a frequency of the TOD is one of the estimate, S_k is fs/2 times the PSD
    # there; below the first positive frequency it is the first positive value, and
    # bin 0 is bin 1.
    nperseg, n_samples = 64, 64 * 16
    freqs = np.fft.rfftfreq(nperseg, d=1 / SAMPLE_RATE)
    psd = 1 + (2.0 / np.maximum(freqs, freqs[1])) ** 2

    variances = rfft_variances(freqs, psd, n_samples, SAMPLE_RATE)

    np.testing.assert_allclose(variances[::16], SAMPLE_RATE * psd / 2, rtol=1e-12)
    np.testing.assert_allclose(variances[:16], SAMPLE_RATE * psd[1] / 2, rtol=1e-12)


# -----------------------------------------------------------------------------
# draw_noise
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n_samples", [2**16, 2**16 + 1])
def test_draw_noise_white_noise_rms(n_samples):
    # With a constant S_k = S the realization is white noise of rms sqrt(S).
    variances = np.full(n_samples // 2 + 1, 4.0)

    noise = draw_noise(variances, n_samples, np.random.default_rng(0))

    assert noise.shape == (n_samples,)
    assert noise.std() == pytest.approx(2.0, rel=0.02)


def test_draw_noise_has_the_spectrum_it_is_given():
    # The Welch spectrum of a long realization is the PSD that rfft_variances started
    # from: the normalization of S_k = fs P / 2 and of the FFTs agree.
    nperseg, n_samples = 512, 2**18
    freqs = np.fft.rfftfreq(nperseg, d=1 / SAMPLE_RATE)
    psd = 1e-2 * (1 + (1.0 / np.maximum(freqs, freqs[1])) ** 2)
    variances = rfft_variances(freqs, psd, n_samples, SAMPLE_RATE)

    noise = draw_noise(variances, n_samples, np.random.default_rng(1))

    _, estimate = welch(noise, fs=SAMPLE_RATE, window="hann", nperseg=nperseg)
    # bin 1 is left out: the Hann window leaks the flat part below it into it
    ratio = estimate[2:-1] / psd[2:-1]
    assert np.median(ratio) == pytest.approx(1.0, abs=0.02)


# -----------------------------------------------------------------------------
# fill_gaps
# -----------------------------------------------------------------------------


def test_fill_gaps_against_the_dense_solution():
    # Check against numpy on a short TOD: Q built column by column, the correction
    # in the gaps y = C_go C_oo^-1 delta_o with the covariance C = Q^-1 (the
    # Hoffman-Ribak formula), and the free realization drawn with the same seed.
    n_samples = 101
    variances = _one_over_f(n_samples)
    good = _gaps(n_samples, [20, 70], 12)
    tod = draw_noise(variances, n_samples, np.random.default_rng(2))

    filled = fill_gaps(tod, good, variances, np.random.default_rng(3), tol=1e-12)

    noise = draw_noise(variances, n_samples, np.random.default_rng(3))
    Q = np.column_stack([_apply_precision(e, variances) for e in np.eye(n_samples)])
    C = np.linalg.inv(Q)
    gaps = ~good
    delta = tod[good] - noise[good]
    y = C[np.ix_(gaps, good)] @ np.linalg.solve(C[np.ix_(good, good)], delta)
    np.testing.assert_allclose(filled[gaps], noise[gaps] + y, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(filled[good], tod[good])


def test_fill_gaps_does_not_modify_the_tod():
    n_samples = 512
    variances = _one_over_f(n_samples)
    good = _gaps(n_samples, [100], 50)
    tod = np.random.default_rng(4).standard_normal(n_samples)
    original = tod.copy()

    filled = fill_gaps(tod, good, variances, np.random.default_rng(5))

    np.testing.assert_array_equal(tod, original)
    assert not np.shares_memory(filled, tod)


def test_fill_gaps_without_gaps_returns_a_copy():
    tod = np.arange(16.0)

    filled = fill_gaps(
        tod, np.ones(16, dtype=bool), np.ones(9), np.random.default_rng()
    )

    np.testing.assert_array_equal(filled, tod)
    assert filled is not tod


def test_fill_gaps_white_noise_is_a_free_realization():
    # With white noise Q is proportional to the identity, Q_go = 0: the good samples
    # say nothing about the gaps, which get the free realization as it is.
    n_samples = 1000
    variances = np.full(n_samples // 2 + 1, 2.0)
    good = _gaps(n_samples, [100, 600], 80)
    tod = np.random.default_rng(6).standard_normal(n_samples)

    filled = fill_gaps(tod, good, variances, np.random.default_rng(7))

    noise = draw_noise(variances, n_samples, np.random.default_rng(7))
    np.testing.assert_allclose(filled[~good], noise[~good], rtol=0, atol=1e-10)


def test_fill_gaps_not_converged_warns():
    n_samples = 4096
    variances = _one_over_f(n_samples, knee=10.0)
    good = _gaps(n_samples, [1000], 500)
    tod = draw_noise(variances, n_samples, np.random.default_rng(8))

    with pytest.warns(RuntimeWarning, match="did not reach"):
        fill_gaps(tod, good, variances, np.random.default_rng(9), maxiter=1)
