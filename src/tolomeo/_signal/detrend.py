# Adapted from:
# https://github.com/thomaswmorris/maria/blob/master/maria/tod/processing.py
# Original implementation by Thomas W. Morris (Maria).

import numpy as np
from numpy.typing import NDArray


def remove_slope(D: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Subtract from each detector the line through its first and last sample.

    Parameters
    ----------
    D : :class:`~numpy.ndarray`
        TOD, one column per detector.
        Shape ``(n_samples, n_det)``.

    Returns
    -------
    :class:`~numpy.ndarray`
        New array in which the first and the last sample of every detector
        are 0. ``D`` is not modified.
        Shape ``(n_samples, n_det)``.

    Notes
    -----
    Only two samples define each line, so their noise ends up in the whole
    TOD as a random offset and slope. To remove a drift from noisy data use
    :func:`remove_baseline`, which uses all the samples.
    """
    # linspace between two rows gives, for every detector, the line that goes
    # from its first to its last sample: shape (n_samples, n_det)
    return D - np.linspace(D[0], D[-1], D.shape[0])


def remove_baseline(D: NDArray[np.float64], deg: int = 1) -> NDArray[np.float64]:
    """
    Subtract from each detector the polynomial that best fits it.

    For every detector the polynomial of degree ``deg`` is found with a least
    squares fit on all its samples, using the sample index as time. With
    ``deg=1`` it is a straight line: the slow linear drift of the detector.

    Parameters
    ----------
    D : :class:`~numpy.ndarray`
        TOD of one subscan, one column per detector. It must be finite.
        Shape ``(n_samples, n_det)``.
    deg : int
        Degree of the polynomial. With 0 only the mean is removed.

    Returns
    -------
    :class:`~numpy.ndarray`
        New array with the baseline removed, so that every detector has zero
        mean. ``D`` is not modified.
        Shape ``(n_samples, n_det)``.

    Notes
    -----
    Every sample is used, the source included. A bright and narrow source is
    therefore slightly attenuated, because the fit follows it a little.
    """
    D = np.asarray(D, dtype=np.float64)

    if D.ndim != 2:
        raise ValueError("D must have shape (n_samples, n_det).")

    if deg < 0:
        raise ValueError("deg must be a non-negative integer.")

    n_samples = D.shape[0]

    if n_samples <= deg:
        raise ValueError(
            f"A polynomial of degree {deg} needs more than {deg} samples, "
            f"got {n_samples}."
        )

    if not np.isfinite(D).all():
        raise ValueError("D must not contain NaN or inf.")

    # time of each sample, from -1 to 1: keeps the fit well conditioned
    t = np.linspace(-1, 1, n_samples)

    # polyfit fits every column of D separately: coeff is (deg + 1, n_det),
    # one polynomial per detector, highest power first
    coeff = np.polyfit(t, D, deg)

    # vander has one column per power of t, so vander @ coeff evaluates the
    # polynomial of every detector at every sample: shape (n_samples, n_det)
    baseline = np.vander(t, deg + 1) @ coeff

    return D - baseline
