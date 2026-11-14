# Adapted from:
# https://github.com/thomaswmorris/maria/blob/master/maria/tod/processing.py
# Original implementation by Thomas W. Morris (Maria).

import numpy as np
from numpy.typing import NDArray


def bspline_knots(
    t: NDArray[np.float64], spacing: float, order: int
) -> NDArray[np.float64]:
    """
    Uniform knots that cover the range of ``t``, padded for a spline of ``order``.

    Parameters
    ----------
    t : :class:`~numpy.ndarray` of float64
        Abscissa of the data (for example the time of every sample), in any unit.
        Shape ``(n_t,)``.
    spacing : float
        Distance between two knots, in the unit of ``t``.
    order : int
        Order (degree) of the spline: 3 is a cubic spline.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        The knots: ``n_bins = max(floor((t.max() - t.min()) / spacing), 1)`` inner knots
        centred on the range of ``t``, plus ``order + 1`` knots added on each side.
        Shape ``(n_bins + 2 * (order + 1),)``.
    """
    tmin = t.min()
    tmax = t.max()
    n_bins = int(
        np.maximum((tmax - tmin) // spacing, 1)
    )  # how many bins straddle the data?

    k = spacing * np.arange(
        n_bins, dtype=float
    )  # so that it straddles the whole domain
    k += float(tmax + tmin) / 2 - k.mean()
    k = np.r_[
        k[0] + spacing * np.arange(-order - 1, 0),
        k,
        k[-1] + spacing * np.arange(1, order + 2),
    ]  # pad the edges

    return k


def bspline_basis_from_knots(
    t: NDArray[np.float64], k: NDArray[np.float64], order: int
) -> NDArray[np.float64]:
    """
    B-spline basis functions on the knots ``k``, evaluated at ``t``.

    Uses the Cox-de Boor recursion, starting from the indicator of each knot interval.

    Parameters
    ----------
    t : :class:`~numpy.ndarray` of float64
        Points where the basis is evaluated.
        Shape ``(n_t,)``.
    k : :class:`~numpy.ndarray` of float64
        Knots, increasing (for example from :func:`bspline_knots`).
        Shape ``(n_knots,)``.
    order : int
        Order (degree) of the spline.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        Row ``i`` is the basis function ``i`` at every point of ``t``.
        Shape ``(n_knots - order - 1, n_t)``.
    """
    n_basis = len(k) - order - 1
    B = np.zeros((len(k) + 1, order + 1, len(t)))
    B[np.digitize(t, k) - 1, 0, np.arange(len(t))] = 1

    for p in range(1, order + 1):
        for i in range(len(k) - p - 1):
            B[i, p] = B[i, p - 1] * (t - k[i]) / (k[i + p] - k[i]) + B[i + 1, p - 1] * (
                k[i + p + 1] - t
            ) / (k[i + p + 1] - k[i + 1])

    return B[:n_basis, -1]
