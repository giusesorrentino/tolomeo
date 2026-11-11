# Adapted from:
# https://github.com/thomaswmorris/maria/blob/master/maria/tod/processing.py
# Original implementation by Thomas W. Morris (Maria).

import warnings

import numpy as np
from numpy.typing import NDArray

from .._typing import CommonModeMethod

__all__ = ["remove_common_mode"]


def remove_common_mode(
    D: NDArray[np.float64],
    background: NDArray[np.bool_],
    method: CommonModeMethod = "median",
) -> NDArray[np.float64]:
    """
    Subtract the common mode from the TOD of one subscan.

    With ``"median"`` and ``"mean"``, at each sample the common mode is the median
    (or the mean) over the detectors, and the same value is subtracted from all of
    them. With ``"corr"`` every detector has its own common mode, the mean of the
    other detectors weighted by their correlation with it, multiplied by a gain fitted
    on that detector (see :func:`_correlated_common_mode`). Only the detectors and
    samples with ``background`` set to True are used to estimate it, so that a source
    seen by the telescope does not bias it.

    Parameters
    ----------
    D : :class:`~numpy.ndarray`
        TOD of one subscan, one column per detector.
        Shape ``(n_samples, n_det)``.
    background : :class:`~numpy.ndarray` of bool
        True for the samples far from the source, which can be used.
        Shape ``(n_samples,)``, the same mask for every detector, or
        ``(n_samples, n_det)``, one mask per detector (for example the one built
        from :class:`~tolomeo.processing.RadiusMask`).
    method : :class:`~tolomeo._typing.CommonModeMethod`
        How the common mode is estimated. The median is not moved by a noisy
        detector or by one that sees the source; the mean is. ``"corr"`` follows
        detectors that see different common modes (for example groups on different
        readout lines) and detectors with different gains; it needs the baseline
        already removed. Non-finite values in ``D`` are never used.

    Returns
    -------
    :class:`~numpy.ndarray`
        New array with the common mode removed. ``D`` is not modified.
        At the samples where no detector can be used nothing is subtracted.
        Shape ``(n_samples, n_det)``.
    """
    D = np.asarray(D, dtype=np.float64)
    background = np.asarray(background, dtype=bool)

    if D.ndim != 2:
        raise ValueError("D must have shape (n_samples, n_det).")

    if background.shape == (D.shape[0],):
        # one value per sample (row): [:, None] makes it apply to every detector
        usable = np.broadcast_to(background[:, None], D.shape)
    elif background.shape == D.shape:
        usable = background
    else:
        raise ValueError(
            "`background` must have shape (n_samples,) or (n_samples, n_det)."
        )

    # a non-finite value is never used to estimate the common mode
    usable = usable & np.isfinite(D)

    if method == "corr":
        # one common mode per detector, built from the detectors correlated with it,
        # and a gain of its own
        common_mode, gain = _correlated_common_mode(D, usable)
        return D - common_mode * gain

    # samples near the source are ignored when estimating the common mode
    masked = np.where(usable, D, np.nan)

    # pick the estimator: the nan versions skip the masked samples
    if method == "median":
        estimate = np.nanmedian
    elif method == "mean":
        estimate = np.nanmean
    else:
        raise ValueError(f"Unknown method {method!r}: use 'median', 'mean' or 'corr'.")

    # one value per sample: the median (or mean) over the detectors (axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        common_mode = estimate(masked, axis=1)

    # where every sample was masked there is no estimate: subtract nothing
    common_mode = np.nan_to_num(common_mode, nan=0.0)

    # subtract the same value from all the detectors of each sample
    return D - common_mode[:, None]


# The "corr" method is adapted from the custom common mode of MIRTOS:
# https://github.com/EleBarbavara/mirtos/blob/main/src/mirtos/cleaner_class.py#L31


def _correlated_common_mode(
    D: NDArray[np.float64], usable: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Common mode of every detector, from the detectors correlated with it, and its gain.

    1. Correlation matrix ``C`` between the detectors, from the usable samples: for
       every pair, the samples usable for both.
    2. Weights ``W``: ``C`` with the diagonal set to 0, so that a detector does not enter
       its own common mode.
    3. Common mode of detector ``i`` at sample ``t``: ``sum_j W[i, j] * D[t, j]``
       divided by ``sum_j |W[i, j]|``, both over the detectors ``j`` usable at ``t``.
       An anticorrelated detector (negative weight) enters with its sign flipped.
    4. Gain of detector ``i``: least squares slope of ``D[:, i]`` on its common mode,
       through zero, on its usable samples.

    Adapted from ``corr_matrix`` and ``custom_cm`` of MIRTOS, with two differences:
    the masked samples are left out of the normalization of step 3 (MIRTOS uses
    ``np.average`` on a masked array, which keeps their weight in the denominator) and
    out of the fit of step 4 (``statsmodels.OLS`` ignores the mask of a masked array).

    Parameters
    ----------
    D : :class:`~numpy.ndarray` of float64
        TOD of one subscan, one column per detector, with the baseline already
        removed: the common mode and the gain go through zero. Shape ``(n_samples, n_det)``.
    usable : :class:`~numpy.ndarray` of bool
        True for the samples that can be used (far from the source, finite).
        Shape ``(n_samples, n_det)``.

    Returns
    -------
    common_mode : :class:`~numpy.ndarray` of float64
        Column ``i`` is the common mode of detector ``i``; 0 at the samples where no
        other correlated detector is usable. Shape ``(n_samples, n_det)``.
    gain : :class:`~numpy.ndarray` of float64
        What to multiply the common mode by before subtracting it; 0 for a detector
        whose common mode is 0 on all its usable samples. Shape ``(n_det,)``.
    """
    m = usable.astype(np.float64)
    # the unusable samples count as 0 in every sum below
    Dz = np.where(usable, D, 0.0)

    # mean and standard deviation of every detector, on its own usable samples
    n = m.sum(axis=0)  # (n_det,)
    mean = np.divide(Dz.sum(axis=0), n, out=np.zeros(n.shape), where=n > 0)
    var = np.divide(
        (m * (Dz - mean) ** 2).sum(axis=0), n, out=np.zeros(n.shape), where=n > 0
    )
    std = np.sqrt(var)

    # a detector with fewer than two usable samples, or a constant signal, has no
    # correlation with the others: z = 0 makes its row and column of C zero
    ok = (n >= 2) & (std > 0)
    z = np.where(usable & ok, (Dz - mean) / np.where(ok, std, 1.0), 0.0)

    # pairwise correlation: sum of z_i * z_j over the samples usable for both, divided
    # by how many they are. counts[i, j] is that number
    counts = m.T @ m  # (n_det, n_det)
    C = np.divide(z.T @ z, counts, out=np.zeros(counts.shape), where=counts > 0)
    np.clip(C, -1.0, 1.0, out=C)

    # weights: the correlations, without the detector itself
    np.fill_diagonal(C, 0.0)

    # weighted mean of the other detectors, sample by sample. The numerator uses the
    # signed weights; the denominator adds the |weights| of the detectors usable at that
    # sample only, so that a masked detector does not lower the mean
    numerator = Dz @ C.T  # (n_samples, n_det)
    denominator = m @ np.abs(C).T  # (n_samples, n_det)
    common_mode = np.divide(
        numerator, denominator, out=np.zeros(numerator.shape), where=denominator > 0
    )

    # gain: slope through zero of D[:, i] against common_mode[:, i], on the usable
    # samples of detector i (the sums run over time, axis 0)
    cm_used = common_mode * m
    top = (Dz * cm_used).sum(axis=0)
    bottom = (common_mode * cm_used).sum(axis=0)
    gain = np.divide(top, bottom, out=np.zeros(top.shape), where=bottom > 0)

    return common_mode, gain
