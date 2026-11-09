# Adapted from:
# https://github.com/thomaswmorris/maria/blob/master/maria/tod/processing.py
# Original implementation by Thomas W. Morris (Maria).

import warnings

import numpy as np
from numpy.typing import NDArray

from .._typing import CommonModeMethod

__all__ = ["remove_common_mode"]

# TODO: common mode di MIRTOS (https://github.com/EleBarbavara/mirtos/blob/main/src/mirtos/cleaner_class.py#L31)
def remove_common_mode(
    D: NDArray[np.float64],
    background: NDArray[np.bool_],
    method: CommonModeMethod = "median",
) -> NDArray[np.float64]:
    """
    Subtract the common mode from the TOD of one subscan.

    At each sample the common mode is the median (or the mean) over the
    detectors. Only the detectors and samples with ``background`` set to True
    are used to estimate it, so that a source seen by the telescope does not
    bias it.

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
        How the common mode is estimated at each sample. The median is not
        moved by a noisy detector or by one that sees the source; the mean
        is. Non-finite values in ``D`` are ignored in both cases.

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
        # samples near the source are ignored when estimating the common mode.
        # background has one value per sample (row), so [:, None] makes it apply
        # to every detector (column)
        masked = np.where(background[:, None], D, np.nan)
    elif background.shape == D.shape:
        masked = np.where(background, D, np.nan)
    else:
        raise ValueError(
            "`background` must have shape (n_samples,) or (n_samples, n_det)."
        )

    # pick the estimator: the nan versions skip the masked samples
    if method == "median":
        estimate = np.nanmedian
    elif method == "mean":
        estimate = np.nanmean
    else:
        raise ValueError(f"Unknown method {method!r}: use 'median' or 'mean'.")

    # one value per sample: the median (or mean) over the detectors (axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
        common_mode = estimate(masked, axis=1)

    # where every sample was masked there is no estimate: subtract nothing
    common_mode = np.nan_to_num(common_mode, nan=0.0)

    # subtract the same value from all the detectors of each sample
    return D - common_mode[:, None]
