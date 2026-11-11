"""
Frequency bands of the filters, and the preparation of a TOD for a spectral estimate.

Used by the cleaning of the TOD (``processing``), by the noise model (``mapmaking``)
and by the plots of the spectra (``plotting``).
"""

import numpy as np
from numpy.typing import NDArray


def _check_nyquist(sample_rate, *cutoffs):
    """Raise if a cutoff frequency is not below the Nyquist frequency."""
    nyquist = sample_rate / 2

    for fc in cutoffs:
        if fc >= nyquist:
            raise ValueError(
                f"A cutoff of {fc} Hz is not below the Nyquist frequency of "
                f"{nyquist:.2f} Hz (sample rate {sample_rate:.2f} Hz)."
            )


def _narrow_passband(
    passband: tuple[float, float] | None, f_lower: float = 0.0, f_upper: float = np.inf
) -> tuple[float, float]:
    """
    Band left by one more filter: the intersection of ``passband`` with ``[f_lower, f_upper]``.

    ``passband`` is the band of the filters already applied (None if there is none yet).
    With more filters in the same config, or ``process_tod`` called more than once, the
    band is the frequencies that all of them keep.

    Raises
    ------
    ValueError
        If the filters together leave no frequency.
    """
    lo, hi = passband if passband is not None else (0.0, np.inf)
    lo, hi = max(lo, f_lower), min(hi, f_upper)
    if lo >= hi:
        raise ValueError(
            f"The filters leave no frequency band: [{lo}, {hi}] Hz is empty."
        )
    return (lo, hi)


def _prepare_for_spectrum(
    tod: NDArray[np.float64],
    background: NDArray[np.bool_] | None,
) -> NDArray[np.float64]:
    """
    Return a copy of ``tod`` ready for the spectral estimate.

    The masked samples (``background`` False, for instance the ones on a bright source)
    are replaced by a straight line through the good samples, and the mean of the good
    samples is subtracted. ``tod`` itself is not modified.

    Parameters
    ----------
    tod : :class:`~numpy.ndarray`
        TOD of one detector.
        Shape ``(n_samples,)``.
    background : :class:`~numpy.ndarray` of bool, optional
        True for the good samples. None means all the samples are good.
        Shape ``(n_samples,)``.
    """
    if background is None:
        background = np.ones(tod.shape, dtype=bool)

    if background.shape != tod.shape:
        raise ValueError("`background` must have the same shape as `tod`.")

    if not background.any():
        raise ValueError("Every sample is masked: cannot estimate the noise spectrum.")

    # work on a copy: the array we received must not change
    tod = tod.copy()

    source = ~background

    # np.interp returns only the values at the masked samples (the first argument),
    # so we write them back into tod, in the masked positions
    t = np.arange(tod.size)
    tod[source] = np.interp(t[source], t[background], tod[background])

    # the mean is computed on the good samples only
    tod -= tod[background].mean()

    return tod
