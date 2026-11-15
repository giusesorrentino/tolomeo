import numpy as np
import pytest

from tolomeo._signal.common_mode import decompose, remove_common_mode

rng = np.random.default_rng(3)

N_SAMPLES = 500
N_DET = 6

# a slow drift shared by all the detectors
COMMON = np.cumsum(rng.standard_normal(N_SAMPLES))
METHODS = ["median", "mean", "corr"]


# -----------------------------------------------------------------------------
# remove_common_mode
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_remove_common_mode_identical_detectors(method):
    # Check that a signal seen in the same way by every detector is removed exactly.
    D = np.repeat(COMMON[:, None], N_DET, axis=1)

    out = remove_common_mode(D, np.ones(N_SAMPLES, dtype=bool), method=method)

    np.testing.assert_allclose(out, 0.0, atol=1e-10)


def test_remove_common_mode_corr_follows_the_gains():
    # Check that "corr" removes a common mode seen with a different gain by every
    # detector, which "median" cannot do.
    gains = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
    noise = 0.01 * rng.standard_normal((N_SAMPLES, N_DET))
    D = COMMON[:, None] * gains[None, :] + noise
    background = np.ones(N_SAMPLES, dtype=bool)

    out_corr = remove_common_mode(D, background, method="corr")
    out_median = remove_common_mode(D, background, method="median")

    assert out_corr.std() < 0.1 * out_median.std()


def test_remove_common_mode_background_keeps_the_source():
    # Check that the samples outside the background are not used for the estimate:
    # a source seen by one detector is kept, the common mode is removed.
    source = np.zeros((N_SAMPLES, N_DET))
    source[200:220, 0] = 50.0
    D = COMMON[:, None] + source
    background = np.ones((N_SAMPLES, N_DET), dtype=bool)
    background[200:220, 0] = False

    out = remove_common_mode(D, background, method="mean")

    np.testing.assert_allclose(out, source, atol=1e-10)


def test_remove_common_mode_nothing_usable_subtracts_nothing():
    # Check that at the samples where no detector can be used nothing is subtracted.
    D = np.repeat(COMMON[:, None], N_DET, axis=1)
    background = np.ones(N_SAMPLES, dtype=bool)
    background[:10] = False

    out = remove_common_mode(D, background, method="median")

    np.testing.assert_allclose(out[:10], D[:10])


def test_remove_common_mode_ignores_non_finite():
    # Check that a NaN in one detector does not spread to the others.
    D = np.repeat(COMMON[:, None], N_DET, axis=1)
    D[5, 2] = np.nan

    out = remove_common_mode(D, np.ones(N_SAMPLES, dtype=bool), method="median")

    assert np.isfinite(np.delete(out, 2, axis=1)).all()


def test_remove_common_mode_does_not_modify_the_input():
    D = np.repeat(COMMON[:, None], N_DET, axis=1)
    original = D.copy()

    remove_common_mode(D, np.ones(N_SAMPLES, dtype=bool))

    np.testing.assert_array_equal(D, original)


def test_remove_common_mode_1d_tod_raises():
    with pytest.raises(ValueError, match=r"D must have shape"):
        remove_common_mode(COMMON, np.ones(N_SAMPLES, dtype=bool))


def test_remove_common_mode_wrong_background_raises():
    D = np.zeros((N_SAMPLES, N_DET))

    with pytest.raises(ValueError, match=r"`background` must have shape"):
        remove_common_mode(D, np.ones(N_DET, dtype=bool))


def test_remove_common_mode_unknown_method_raises():
    D = np.zeros((N_SAMPLES, N_DET))

    with pytest.raises(ValueError, match=r"Unknown method"):
        remove_common_mode(D, np.ones(N_SAMPLES, dtype=bool), method="mode")


# -----------------------------------------------------------------------------
# decompose
# -----------------------------------------------------------------------------


def test_decompose_rank_one():
    # Check that one mode rebuilds a TOD made of a single mode.
    weights_true = rng.uniform(0.5, 2.0, N_DET)
    D = np.outer(COMMON, weights_true)

    modes, weights = decompose(D, k=1)

    assert modes.shape == (N_SAMPLES, 1)
    assert weights.shape == (1, N_DET)
    np.testing.assert_allclose(modes @ weights, D, atol=1e-8 * np.abs(D).max())


def test_decompose_modes_are_sorted():
    # Check that the strongest mode comes first.
    strong = np.sin(np.linspace(0, 20, N_SAMPLES))
    weak = np.cos(np.linspace(0, 7, N_SAMPLES))
    D = 10 * np.outer(strong, np.ones(N_DET)) + np.outer(
        weak, np.linspace(-1, 1, N_DET)
    )

    _, weights = decompose(D, k=2)

    assert np.abs(weights[0]).sum() > np.abs(weights[1]).sum()


def test_decompose_unknown_norm_raises():
    with pytest.raises(ValueError, match=r"Invalid norm"):
        decompose(np.outer(COMMON, np.ones(N_DET)), k=1, norm="max")
