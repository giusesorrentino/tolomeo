# Adapted from:
# https://github.com/thomaswmorris/maria/blob/master/maria/tod/processing.py
# Original implementation by Thomas W. Morris (Maria).

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lstsq

from .._lib.messages import _warn
from .splines import bspline_basis_from_knots

logger = logging.getLogger(__name__)

# the texts of the warnings are the same for every subscan, so that Python shows each
# of them once; the numbers of every subscan are debug messages
_SKIPPED = (
    "Some baseline or spline fits were skipped: in some subscans a detector has "
    "fewer background samples than coefficients, and nothing was subtracted there. "
    "tolomeo.debug() (or `tolomeo -v`) shows the numbers."
)
_DEFICIENT = (
    "Some baseline or spline fits are not fully constrained by the background "
    "samples: the coefficients with no data were set to 0. tolomeo.debug() (or "
    "`tolomeo -v`) shows the numbers."
)

# largest condition number of the normal equations of a fit with one mask per detector:
# their error on the coefficients is about eps * cond, 2e-8 here. A fit above it (a
# coefficient with little data, as a spline knot over the hole of the mask) is done by
# lstsq on its samples. A choice, not a standard value
_MAX_GRAM_COND = 1e8


def remove_baseline(
    D: NDArray[np.float64],
    t: NDArray[np.float64],
    deg: int = 1,
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """
    Subtract from each detector the polynomial in time that best fits it.

    For every detector the polynomial of degree ``deg`` is found with a least
    squares fit on the samples of ``mask``, and it is subtracted from all the
    samples. With ``deg=1`` it is a straight line: the slow linear drift of the
    detector.

    Parameters
    ----------
    D : :class:`~numpy.ndarray`
        TOD of one subscan, one column per detector. It must be finite.
        Shape ``(n_samples, n_det)``.
    t : :class:`~numpy.ndarray`
        Time of every sample, in any unit: it is mapped to [0, 1] before the fit.
        Shape ``(n_samples,)``.
    deg : int
        Degree of the polynomial, smaller than ``n_samples``. With 0 only the mean
        is removed.
        Default ``1``.
    mask : :class:`~numpy.ndarray` of bool, optional
        True for the samples used in the fit (for example the ones far from the
        source): the same for every detector with shape ``(n_samples,)``, one per
        detector with ``(n_samples, n_det)``. None means all the samples.

    Returns
    -------
    :class:`~numpy.ndarray`
        New array with the polynomial removed. ``D`` is not modified.
        Shape ``(n_samples, n_det)``.
    """
    time_poly = np.polynomial.legendre.legvander(2 * _normalize(t) - 1, deg)
    coeff = _masked_lstsq(time_poly, D, mask=mask)

    # evaluate each polynomial at every sample and subtract it
    return D - time_poly @ coeff


def remove_spline(
    D: NDArray[np.float64],
    t: NDArray[np.float64],
    knots: NDArray[np.float64],
    order: int = 3,
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """
    Subtract from each detector the B-spline in time that best fits it.

    For every detector the spline on ``knots`` is found with a least squares fit on
    the samples of ``mask``, and it is subtracted from all the samples.

    Parameters
    ----------
    D : :class:`~numpy.ndarray`
        TOD of one subscan, one column per detector. It must be finite.
        Shape ``(n_samples, n_det)``.
    t : :class:`~numpy.ndarray`
        Time of every sample, in the unit of ``knots``.
        Shape ``(n_samples,)``.
    knots : :class:`~numpy.ndarray`
        Knots of the spline, for example from
        :func:`~tolomeo._signal.splines.bspline_knots`. They give
        ``len(knots) - order - 1`` basis functions, which must be fewer than
        ``n_samples``.
    order : int
        Order (degree) of the spline.
        Default ``3``.
    mask : :class:`~numpy.ndarray` of bool, optional
        True for the samples used in the fit (for example the ones far from the
        source): the same for every detector with shape ``(n_samples,)``, one per
        detector with ``(n_samples, n_det)``. None means all the samples.

    Returns
    -------
    :class:`~numpy.ndarray`
        New array with the spline removed. ``D`` is not modified.
        Shape ``(n_samples, n_det)``.
    """
    # spline basis functions evaluated at each sample: shape (n_basis, n_samples)
    B = bspline_basis_from_knots(t, knots, order)

    coeff = _masked_lstsq(B.T, D, mask=mask)
    return D - B.T @ coeff


def _normalize(a: NDArray) -> NDArray:
    """Normalize an array to [0, 1], handling constant arrays."""
    a = np.asarray(a, dtype=np.float64)
    span = np.ptp(a)

    # a constant array has no range to divide by: return all zeros
    if span == 0:
        return np.zeros_like(a)

    return (a - a.min()) / span


def _masked_lstsq(
    basis: NDArray[np.float64],
    block: NDArray[np.float64],
    mask: NDArray[np.bool_] | None,
) -> NDArray[np.float64]:
    """
    Fit every detector of a subscan on ``basis``, using only the samples of ``mask``.

    Parameters
    ----------
    basis : :class:`~numpy.ndarray`
        Basis functions evaluated at the samples of the subscan, one per column
        (the polynomial in time, or the transposed spline basis).
        Shape ``(n_t, n_coef)``.
    block : :class:`~numpy.ndarray`
        Signal of the subscan, one column per detector.
        Shape ``(n_t, n_det)``.
    mask : :class:`~numpy.ndarray` of bool or None
        True for the samples used in the fit: the same for every detector with
        shape ``(n_t,)``, one per detector with ``(n_t, n_det)``. None means all.

    Returns
    -------
    :class:`~numpy.ndarray`
        Coefficients of every detector. The caller evaluates the fit on ALL the
        samples (``basis @ coeff``), not only on the ones used to find it, so that
        the source is removed from the baseline but not from the signal.
        Shape ``(n_coef, n_det)``.

    Notes
    -----
    With one mask per detector, the fits of all the detectors are solved together
    through their normal equations ``G_j c_j = r_j``, with
    ``G_j = basis^T diag(m_j) basis`` and ``r_j = basis^T (m_j * d_j)`` (``m_j`` the
    mask of detector ``j`` as 0 and 1, ``d_j`` its signal): the same solution as a
    ``lstsq`` for each detector, about ten times faster. The normal equations square
    the condition number of the fit, which is small for the Legendre polynomials of
    :func:`remove_baseline`; the detectors with ``cond(G_j) > _MAX_GRAM_COND`` are
    fitted by ``lstsq`` on their samples, as those with a single mask.
    """
    n_t, n_coef = basis.shape
    n_det = block.shape[1]

    # no mask: every sample, and all the detectors in a single solve
    if mask is None:
        return lstsq(basis, block)[0]

    # the same samples for every detector: still a single solve, on those rows
    if mask.ndim == 1:
        n_usable = int(mask.sum())

        # fewer samples than coefficients: the fit is skipped, nothing is subtracted
        if n_usable < n_coef:
            _warn(_SKIPPED)
            logger.debug(
                "%d of %d fits skipped: %d background samples for %d coefficients; "
                "nothing was subtracted.",
                n_det,
                n_det,
                n_usable,
                n_coef,
            )
            return np.zeros((n_coef, n_det))

        coeff, _, rank, _ = lstsq(basis[mask], block[mask])

        # the samples leave a coefficient without data (a spline knot over the hole)
        if rank < n_coef:
            _warn(_DEFICIENT)
            logger.debug(
                "%d of %d fits are not fully constrained by the background samples "
                "(rank %d of %d): the coefficients with no data are 0.",
                n_det,
                n_det,
                rank,
                n_coef,
            )

        return coeff

    # one mask per detector: the normal equations of all the detectors at once.
    # G[j, k, l] = sum_t m[t, j] basis[t, k] basis[t, l] is one matrix product, m^T
    # times the products of the columns of the basis, (n_t, n_coef**2) values (40 MB
    # for a spline of 28 functions on 6000 samples); r[j, k] = sum_t m[t, j] d[t, j]
    # basis[t, k]
    weights = mask.astype(np.float64)
    products = (basis[:, :, None] * basis[:, None, :]).reshape(n_t, n_coef * n_coef)
    gram = (weights.T @ products).reshape(n_det, n_coef, n_coef)
    rhs = (weights * block).T @ basis  # (n_det, n_coef)

    # fewer samples than coefficients: the fit is skipped, nothing is subtracted
    skipped = mask.sum(axis=0) < n_coef

    # cond(G_j) from the singular values (in decreasing order): only the well
    # conditioned systems are solved here, the others go to lstsq below
    sv = np.linalg.svd(gram, compute_uv=False)  # (n_det, n_coef)
    solved = ~skipped & (sv[:, -1] * _MAX_GRAM_COND > sv[:, 0])

    coeff = np.zeros((n_coef, n_det))
    coeff[:, solved] = np.linalg.solve(gram[solved], rhs[solved][..., None])[..., 0].T

    # lstsq also finds the rank: a coefficient without data is set to 0
    n_deficient = 0
    for j in np.flatnonzero(~skipped & ~solved):
        rows = mask[:, j]
        coeff[:, j], _, rank, _ = lstsq(basis[rows], block[rows, j])
        n_deficient += int(rank < n_coef)

    n_skipped = int(skipped.sum())

    # one message per call, not one per detector
    if n_skipped:
        _warn(_SKIPPED)
        logger.debug(
            "%d of %d fits skipped: fewer background samples than the %d "
            "coefficients; nothing was subtracted there.",
            n_skipped,
            n_det,
            n_coef,
        )

    if n_deficient:
        _warn(_DEFICIENT)
        logger.debug(
            "%d of %d fits are not fully constrained by the background samples: "
            "the coefficients with no data are 0.",
            n_deficient,
            n_det,
        )

    return coeff
