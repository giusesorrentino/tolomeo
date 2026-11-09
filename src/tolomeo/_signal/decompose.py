# Adapted from:
# https://github.com/thomaswmorris/maria/blob/master/maria/tod/processing.py
# Original implementation by Thomas W. Morris (Maria).

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import svds


def decompose(
    D: NDArray[np.float64], k: int = 64, norm: Literal["var", "sum"] = "var"
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Split a TOD into its ``k`` strongest modes, with a truncated SVD.

    A mode is a time stream shared, with a different weight, by all the
    detectors (for example the atmosphere). The first mode explains most of
    the signal, the second most of what is left, and so on.

    Parameters
    ----------
    D : :class:`~numpy.ndarray`
        TOD, one column per detector. It must be finite: the SVD does not
        accept NaN.
        Shape ``(n_samples, n_det)``.
    k : int
        Number of modes. It must be smaller than both ``n_samples`` and
        ``n_det``.
    norm : {"var", "sum"}
        Normalization of the time modes: ``"var"`` gives each mode unit
        variance, ``"sum"`` gives each mode unit sum of squares.

    Returns
    -------
    modes : :class:`~numpy.ndarray`
        The time modes, from the strongest to the weakest.
        Shape ``(n_samples, k)``.
    weights : :class:`~numpy.ndarray`
        How much of each mode is in each detector, in the units of ``D``.
        Shape ``(k, n_det)``.

    Notes
    -----
    ``modes @ weights`` is the part of ``D`` explained by the ``k`` modes, so
    ``D - modes @ weights`` is the TOD with those modes removed.
    """
    n_samples, _n_det = D.shape

    # the SVD is done on the TOD with every detector scaled to unit norm, so
    # that a detector with a large signal does not dominate the modes
    dnorm = np.sqrt(np.sum(np.square(D), axis=0))  # shape (n_det,)
    dnorm = np.where(dnorm > 0, dnorm, 1)  # dead detector: avoid dividing by 0

    # u: time modes (n_samples, k), s: strengths (k,), v: detectors (k, n_det)
    u, s, v = svds(D / dnorm, k=k)

    # scale of the time modes (1 for an exact SVD, divided by sqrt(n_samples)
    # for the unit-variance normalization)
    unorm = np.sqrt(np.sum(np.square(u), axis=0))  # shape (k,)

    if norm == "var":
        unorm /= np.sqrt(n_samples)
    elif norm != "sum":
        raise ValueError(f"Invalid norm '{norm}'.")

    # svds does not return the modes in a fixed order: strongest first
    mode_sort = np.argsort(-s)

    modes = (u / unorm)[:, mode_sort]

    # undo the two scalings, so that modes @ weights approximates D
    weights = (s[:, None] * unorm[:, None] * v * dnorm)[mode_sort]

    return modes, weights
