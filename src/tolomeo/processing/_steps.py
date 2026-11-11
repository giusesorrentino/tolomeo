import logging
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lstsq
from scipy.signal import windows

from .._signal.common_mode import remove_common_mode
from .._signal.decompose import decompose
from .._signal.splines import bspline_basis_from_knots, bspline_knots

logger = logging.getLogger("tolomeo")


def _normalize(a: NDArray) -> NDArray:
    """Normalize an array to [0, 1], handling constant arrays."""
    a = np.asarray(a, dtype=np.float64)
    span = np.ptp(a)

    # a constant array has no range to divide by: return all zeros
    if span == 0:
        return np.zeros_like(a)

    return (a - a.min()) / span


def _common_mode(signal, tod, step, background):
    for start, stop in pairwise(tod.segments):
        if stop <= start:
            continue

        # no background given: every sample can be used
        if background is None:
            usable = np.ones(stop - start, dtype=bool)
        else:
            usable = background[start:stop]

        signal[start:stop] = remove_common_mode(
            signal[start:stop], usable, method=step.method
        )
    return signal


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


def _window(signal, step):
    try:
        window_function = getattr(windows, step.name)
    except AttributeError as exc:
        raise ValueError(f"Unknown scipy window: {step.name!r}") from exc

    window = window_function(signal.shape[0], **step.kwargs)
    signal *= window[:, None]
    return signal


def _baseline(signal, tod, step, background):
    mask = None

    for k, (start, stop) in enumerate(pairwise(tod.segments)):
        if stop <= start:
            continue

        n_t = stop - start

        if n_t <= step.deg:
            raise ValueError(
                f"A polynomial of degree {step.deg} needs more than {step.deg} "
                f"samples, subscan {k} has {n_t}."
            )

        if background is not None:
            mask = background[start:stop]

        # shape (n_t, deg + 1): column i is t**i, with the time t of this subscan
        # mapped to [0, 1]
        time_poly = _normalize(tod.time[start:stop])[:, None] ** np.arange(step.deg + 1)

        coeff = _masked_lstsq(time_poly, signal[start:stop], mask=mask)

        # evaluate each polynomial at every sample and subtract it
        signal[start:stop] -= time_poly @ coeff
    return signal


def _spline(signal, tod, step, background):
    mask = None

    for k, (start, stop) in enumerate(pairwise(tod.segments)):
        if stop <= start:
            continue

        t = tod.time[start:stop]

        # the knots are cheap; the basis is not (it needs about n_knots * order *
        # n_t floats), so check the number of basis functions before building it
        knots = bspline_knots(t, step.knot_spacing, step.order)
        n_basis = len(knots) - step.order - 1

        if n_basis >= stop - start:
            raise ValueError(
                f"The spline has {n_basis} basis functions but subscan {k} has "
                f"only {stop - start} samples: use a larger knot_spacing "
                "(in days)."
            )

        if background is not None:
            mask = background[start:stop]

        # spline basis functions evaluated at each sample of the subscan: shape
        # (n_basis, n_t)
        B = bspline_basis_from_knots(t, knots, step.order)

        coeff = _masked_lstsq(B.T, signal[start:stop], mask=mask)
        signal[start:stop] -= B.T @ coeff

    return signal


def _remove_modes(signal, step):
    modes_to_remove = step.modes_to_remove

    # nothing to remove (the SVD does not accept k = 0)
    if modes_to_remove == 0:
        return signal

    # the SVD needs k strictly smaller than both sizes of the TOD
    if modes_to_remove >= min(signal.shape):
        raise ValueError(
            f"Too many modes requested ({modes_to_remove}) for a TOD of shape "
            f"{signal.shape}."
        )

    # modes (n_samples, k) @ weights (k, n_det) is the part of the
    # TOD explained by the k strongest modes
    modes, weights = decompose(signal, k=modes_to_remove)
    signal -= modes @ weights

    return signal
