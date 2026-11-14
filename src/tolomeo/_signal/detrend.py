# Adapted from:
# https://github.com/thomaswmorris/maria/blob/master/maria/tod/processing.py
# Original implementation by Thomas W. Morris (Maria).

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lstsq

from .splines import bspline_basis_from_knots

logger = logging.getLogger("tolomeo")


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
    # shape (n_samples, deg + 1): column i is t**i, with t mapped to [0, 1]
    time_poly = _normalize(t)[:, None] ** np.arange(deg + 1)

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


# TODO: use ThreadPoolExecutor
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
    """
    n_coef = basis.shape[1]
    n_det = block.shape[1]

    # no mask: every sample, and all the detectors in a single solve
    if mask is None:
        return lstsq(basis, block)[0]

    # the same samples for every detector: still a single solve, on those rows
    if mask.ndim == 1:
        n_usable = int(mask.sum())

        # fewer samples than coefficients: the fit is skipped, nothing is subtracted
        if n_usable < n_coef:
            logger.warning(
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
            logger.warning(
                "%d of %d fits are not fully constrained by the background samples "
                "(rank %d of %d): the coefficients with no data are 0.",
                n_det,
                n_det,
                rank,
                n_coef,
            )

        return coeff

    # one mask per detector: each has its own samples, so one solve for each
    coeff = np.zeros((n_coef, n_det))
    n_skipped = n_deficient = 0

    for j in range(n_det):
        rows = mask[:, j]

        if rows.sum() < n_coef:
            n_skipped += 1
            continue

        coeff[:, j], _, rank, _ = lstsq(basis[rows], block[rows, j])
        n_deficient += int(rank < n_coef)

    # one message per call, not one per detector
    if n_skipped:
        logger.warning(
            "%d of %d fits skipped: fewer background samples than the %d "
            "coefficients; nothing was subtracted there.",
            n_skipped,
            n_det,
            n_coef,
        )

    if n_deficient:
        logger.warning(
            "%d of %d fits are not fully constrained by the background samples: "
            "the coefficients with no data are 0.",
            n_deficient,
            n_det,
        )

    return coeff
