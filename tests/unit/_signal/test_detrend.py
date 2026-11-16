import numpy as np
import pytest

from tolomeo._signal.detrend import remove_baseline, remove_spline
from tolomeo._signal.splines import bspline_basis_from_knots, bspline_knots

rng = np.random.default_rng(5)

N_SAMPLES = 300
N_DET = 3

# time of a subscan, in days as in the TOD
T = 60600.0 + np.arange(N_SAMPLES) / 244.0 / 86400


def _polynomials(deg):
    """One random polynomial of degree ``deg`` in time per detector."""
    # the time mapped to [-1, 1] as remove_baseline does: a MJD in float64 resolves
    # about 1 microsecond, so it is not exactly a linspace
    x = 2 * (T - T.min()) / np.ptp(T) - 1
    coeffs = rng.normal(size=(deg + 1, N_DET))
    return np.polynomial.polynomial.polyval(x, coeffs).T


def _hole_per_detector(width):
    """One mask per detector, with a hole of ``width`` samples in a different place."""
    mask = np.ones((N_SAMPLES, N_DET), dtype=bool)
    for j, start in enumerate(rng.integers(0, N_SAMPLES - width, N_DET)):
        mask[start : start + width, j] = False
    return mask


def _lstsq_fit(basis, D, mask):
    """The fit of every detector on its own samples, one numpy lstsq per detector."""
    coeffs = [
        np.linalg.lstsq(basis[mask[:, j]], D[mask[:, j], j])[0]
        for j in range(D.shape[1])
    ]
    return basis @ np.column_stack(coeffs)


# -----------------------------------------------------------------------------
# remove_baseline
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("deg", [0, 1, 2, 5])
def test_remove_baseline_removes_a_polynomial(deg):
    # Check that a polynomial of the same degree is removed exactly.
    D = _polynomials(deg)

    out = remove_baseline(D, T, deg=deg)

    np.testing.assert_allclose(out, 0.0, atol=1e-8)


def test_remove_baseline_degree_zero_is_the_mean():
    D = rng.normal(size=(N_SAMPLES, N_DET)) + 4.0

    out = remove_baseline(D, T, deg=0)

    np.testing.assert_allclose(out, D - D.mean(axis=0), atol=1e-12)


@pytest.mark.parametrize("per_detector", [False, True], ids=["1d", "2d"])
def test_remove_baseline_mask_keeps_the_source(per_detector):
    # Check that the fit uses only the background samples: the line is removed and
    # the source, outside the background, is kept.
    line = _polynomials(1)
    source = np.zeros((N_SAMPLES, N_DET))
    source[140:160] = 10.0
    mask = np.ones(N_SAMPLES, dtype=bool)
    mask[140:160] = False
    if per_detector:
        mask = np.repeat(mask[:, None], N_DET, axis=1)

    out = remove_baseline(line + source, T, deg=1, mask=mask)

    np.testing.assert_allclose(out, source, atol=1e-8)


@pytest.mark.parametrize("deg", [2, 4])
def test_remove_baseline_mask_per_detector_matches_lstsq(deg):
    # Check the normal equations solved for all the detectors at once against one
    # lstsq per detector. With deg = 2 there are 3 coefficients and 3 detectors, the
    # case where a transposed result would have the right shape. With Legendre
    # polynomials cond(G) is about 10: the two fits agree to rounding.
    D = rng.normal(size=(N_SAMPLES, N_DET)).cumsum(axis=0)
    mask = _hole_per_detector(60)
    x = 2 * (T - T.min()) / np.ptp(T) - 1
    basis = np.polynomial.legendre.legvander(x, deg)

    out = remove_baseline(D, T, deg=deg, mask=mask)

    expected = D - _lstsq_fit(basis, D, mask)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-10 * np.abs(D).max())


def test_remove_baseline_too_few_samples_per_detector_warns():
    # Check that only the detector with fewer background samples than coefficients is
    # skipped (nothing subtracted, a warning), and the others are fitted.
    D = _polynomials(2)
    mask = np.ones((N_SAMPLES, N_DET), dtype=bool)
    mask[2:, 1] = False

    with pytest.warns(UserWarning, match=r"fits were skipped"):
        out = remove_baseline(D, T, deg=2, mask=mask)

    np.testing.assert_array_equal(out[:, 1], D[:, 1])
    np.testing.assert_allclose(out[:, [0, 2]], 0.0, atol=1e-8)


def test_remove_baseline_too_few_samples_warns():
    # Check that a fit with fewer background samples than coefficients is skipped:
    # a warning, and nothing is subtracted.
    D = _polynomials(2)
    mask = np.zeros(N_SAMPLES, dtype=bool)
    mask[:2] = True

    with pytest.warns(UserWarning, match=r"fits were skipped"):
        out = remove_baseline(D, T, deg=2, mask=mask)

    np.testing.assert_array_equal(out, D)


def test_remove_baseline_does_not_modify_the_input():
    D = _polynomials(1)
    original = D.copy()

    remove_baseline(D, T, deg=1)

    np.testing.assert_array_equal(D, original)


# -----------------------------------------------------------------------------
# remove_spline
# -----------------------------------------------------------------------------


def test_remove_spline_removes_a_spline():
    # Check that a spline on the same knots is removed exactly.
    knots = bspline_knots(T, spacing=(T[-1] - T[0]) / 4, order=3)
    basis = bspline_basis_from_knots(T, knots, order=3)
    D = basis.T @ rng.normal(size=(basis.shape[0], N_DET))

    out = remove_spline(D, T, knots, order=3)

    np.testing.assert_allclose(out, 0.0, atol=1e-8)


def test_remove_spline_knot_over_the_hole_warns():
    # A hole in the mask of detector 0 wider than the support of a basis function
    # leaves its coefficient without data: G is singular, the fit of that detector goes
    # to lstsq, which warns. The other detectors are fitted as usual, and detector 0
    # still fits its own background samples exactly.
    knots = bspline_knots(T, spacing=np.ptp(T) / 8, order=3)
    basis = bspline_basis_from_knots(T, knots, order=3)
    D = basis.T @ rng.normal(size=(basis.shape[0], N_DET))
    mask = np.ones((N_SAMPLES, N_DET), dtype=bool)
    mask[N_SAMPLES // 8 : 7 * N_SAMPLES // 8, 0] = False

    with pytest.warns(UserWarning, match=r"not fully constrained"):
        out = remove_spline(D, T, knots, order=3, mask=mask)

    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:, 1:], 0.0, atol=1e-8)
    np.testing.assert_allclose(out[mask[:, 0], 0], 0.0, atol=1e-8)
