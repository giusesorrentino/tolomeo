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


def bspline_basis(
    t: NDArray[np.float64], spacing: float, order: int
) -> NDArray[np.float64]:
    """
    B-spline basis on uniform knots of the given ``spacing``, evaluated at ``t``.

    Parameters
    ----------
    t : :class:`~numpy.ndarray` of float64
        Points where the basis is evaluated.
        Shape ``(n_t,)``.
    spacing : float
        Distance between two knots, in the unit of ``t``.
    order : int
        Order (degree) of the spline.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        One row per basis function, ``n_basis = n_bins + order + 1`` (see
        :func:`bspline_knots`).
        Shape ``(n_basis, n_t)``.
    """
    k = bspline_knots(t, spacing, order)
    return bspline_basis_from_knots(t, k, order)


def fit_bspline(
    y: NDArray[np.float64], x: NDArray[np.float64], spacing: float, order: int = 3
) -> NDArray[np.float64]:
    """
    Least-squares B-spline fit of ``y(x)``, done along the last axis of ``y``.

    Parameters
    ----------
    y : :class:`~numpy.ndarray` of float64
        Data to fit; every row (every index of the leading axes) is fitted separately.
        Shape ``(..., n_x)``.
    x : :class:`~numpy.ndarray` of float64
        Abscissa of the data.
        Shape ``(n_x,)``.
    spacing : float
        Distance between two knots, in the unit of ``x``.
    order : int
        Order (degree) of the spline.
        Default ``3``.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        The fitted spline evaluated at ``x``.
        Shape ``(..., n_x)``.
    """
    B = bspline_basis(x, spacing=spacing, order=order)
    A = y @ (np.linalg.inv(B @ B.T) @ B).T
    return A @ B


def cross_basis(
    X: list[NDArray[np.float64]], spacing: list[float], order: list[int]
) -> NDArray[np.float64]:
    """
    Tensor-product B-spline basis in several dimensions, evaluated at the same points.

    Parameters
    ----------
    X : list of :class:`~numpy.ndarray` of float64
        One coordinate per dimension, all with the same length.
        Each of shape ``(n_x,)``.
    spacing : list of float
        Knot spacing of every dimension, in the unit of that coordinate.
    order : list of int
        Spline order of every dimension.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        The products of the one-dimensional basis functions; products that are zero at
        every point are dropped, so ``n_basis`` depends on the data.
        Shape ``(n_basis, n_x)``.
    """
    basis = np.ones((1, 1))

    for dim, x in enumerate(X):
        x_basis = bspline_basis(x, spacing[dim], order[dim])
        basis = (x_basis[:, None] * basis).reshape(-1, len(x))
        basis = basis[basis.sum(axis=-1) > 0]

    return basis
