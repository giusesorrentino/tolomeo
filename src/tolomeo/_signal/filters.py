# Adapted from:
# https://github.com/thomaswmorris/maria/blob/master/maria/tod/processing.py
# Original implementation by Thomas W. Morris (Maria).

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

__all__ = ["bandpass", "highpass", "lowpass"]


def _apply(
    data: NDArray[np.float64],
    fc: float | list[float],
    sample_rate: float,
    order: int,
    axis: int,
    btype: Literal["lowpass", "highpass", "bandpass"],
) -> NDArray[np.float64]:
    """
    Design a Butterworth filter and apply it forward and backward along ``axis``.

    ``fc`` is the cutoff in Hz, or ``[f_lower, f_upper]`` for ``"bandpass"``. See
    :func:`lowpass`, :func:`highpass` and :func:`bandpass`.
    """
    # second-order sections ("sos") stay numerically stable at high order, unlike
    # the (b, a) form
    sos = butter(order, fc, btype=btype, fs=sample_rate, output="sos")
    # forward and backward: zero phase, so the signal is not delayed
    return sosfiltfilt(sos, data, axis=axis)


def lowpass(
    data: NDArray[np.float64],
    fc: float,
    sample_rate: float,
    order: int = 4,
    axis: int = -1,
) -> NDArray[np.float64]:
    """
    Low-pass filter ``data`` along ``axis``.

    Parameters
    ----------
    data : :class:`~numpy.ndarray` of float64
        Signal to filter, time along ``axis``. For a TOD of shape
        ``(n_samples, n_det)`` pass ``axis=0``.
        Any shape.
    fc : float
        Cutoff frequency in Hz; it must be below the Nyquist frequency
        ``sample_rate / 2``.
    sample_rate : float
        Sampling frequency in Hz.
    order : int
        Order of the Butterworth filter. It is applied forward and backward,
        so the roll-off is twice as steep.
        Default ``4``.
    axis : int
        Time axis of ``data``.
        Default ``-1``.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        The filtered signal.
        Same shape as ``data``.

    Notes
    -----
    The filter is a Butterworth one, in second-order sections, applied with
    :func:`scipy.signal.sosfiltfilt`: once forward and once backward in time. It has
    zero phase, so the output is not delayed and a source does not shift along the
    scan. The magnitude response is squared: the gain at ``fc`` is 0.5 (-6 dB), not
    ``1 / sqrt(2)``. Edge transients can remain near the two ends of ``data``.
    """
    return _apply(
        data=data,
        fc=fc,
        sample_rate=sample_rate,
        order=order,
        axis=axis,
        btype="lowpass",
    )


def highpass(
    data: NDArray[np.float64],
    fc: float,
    sample_rate: float,
    order: int = 4,
    axis: int = -1,
) -> NDArray[np.float64]:
    """
    High-pass filter ``data`` along ``axis``.

    Parameters
    ----------
    data : :class:`~numpy.ndarray` of float64
        Signal to filter, time along ``axis``. For a TOD of shape
        ``(n_samples, n_det)`` pass ``axis=0``.
        Any shape.
    fc : float
        Cutoff frequency in Hz; it must be below the Nyquist frequency
        ``sample_rate / 2``.
    sample_rate : float
        Sampling frequency in Hz.
    order : int
        Order of the Butterworth filter. It is applied forward and backward,
        so the roll-off is twice as steep.
        Default ``4``.
    axis : int
        Time axis of ``data``.
        Default ``-1``.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        The filtered signal.
        Same shape as ``data``.

    Notes
    -----
    The filter is a Butterworth one, in second-order sections, applied with
    :func:`scipy.signal.sosfiltfilt`: once forward and once backward in time. It has
    zero phase, so the output is not delayed and a source does not shift along the
    scan. The magnitude response is squared: the gain at ``fc`` is 0.5 (-6 dB), not
    ``1 / sqrt(2)``. Edge transients can remain near the two ends of ``data``.
    """
    return _apply(
        data=data,
        fc=fc,
        sample_rate=sample_rate,
        order=order,
        axis=axis,
        btype="highpass",
    )


def bandpass(
    data: NDArray[np.float64],
    f_lower: float,
    f_upper: float,
    sample_rate: float,
    order: int = 4,
    axis: int = -1,
) -> NDArray[np.float64]:
    """
    Band-pass filter ``data`` along ``axis``.

    Parameters
    ----------
    data : :class:`~numpy.ndarray` of float64
        Signal to filter, time along ``axis``.
        Any shape.
    f_lower : float
        Lower edge of the band in Hz.
    f_upper : float
        Upper edge of the band in Hz; it must be larger than ``f_lower`` and below
        the Nyquist frequency ``sample_rate / 2``.
    sample_rate : float
        Sampling frequency in Hz.
    order : int
        Order of the Butterworth filter at each edge of the band; the
        band-pass filter itself has order ``2 * order``. As in
        :func:`lowpass`, it is applied forward and backward.
        Default ``4``.
    axis : int
        Time axis of ``data``.
        Default ``-1``.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        The filtered signal.
        Same shape as ``data``.

    Notes
    -----
    Same as :func:`lowpass`: Butterworth, applied forward and backward, so with zero
    phase and no delay. The gain is 0.5 (-6 dB) at both ``f_lower`` and ``f_upper``.
    """
    return _apply(
        data=data,
        fc=[f_lower, f_upper],
        sample_rate=sample_rate,
        order=order,
        axis=axis,
        btype="bandpass",
    )
