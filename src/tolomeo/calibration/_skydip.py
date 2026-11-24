from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .._lib.messages import _warn
from ..io._skydip import SkyDip

__all__ = ["skydip_responsivity"]


def skydip_responsivity(
    skydips: Sequence[SkyDip],
    *,
    T_atm: float = 267.0,
    tau_atm: float = 0.16,
) -> NDArray[np.float64]:
    """
    Responsivity of every detector from skydips.

    In each skydip the signal of a detector is fitted with a straight line against the
    temperature of the sky, ``T_sky = T_atm * (1 - exp(-tau_atm * airmass))`` with
    ``airmass = 1 / sin(el)``. The responsivity is the slope, and the result is its
    median over the skydips.

    Parameters
    ----------
    skydips : sequence of :class:`~tolomeo.io.SkyDip`
        From :func:`~tolomeo.io.read_skydip`. The result is in the order of the
        ``channel_ids`` given to it, which for :func:`~tolomeo.calibration.calibrate`
        must be ``scan.channel_ids``.
    T_atm : float
        Temperature of the atmosphere, in kelvin. Defaults to ``267``.
    tau_atm : float
        Zenith opacity of the atmosphere. Defaults to ``0.16``.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        Responsivity in signal units per kelvin; it can be negative. Shape ``(n_det,)``.

    Raises
    ------
    ValueError
        If no skydip can be used. A skydip is skipped, with a warning that gives its
        position in ``skydips``, if its ``T_sky`` does not change.
    """
    slopes = []
    skipped = []

    for k, skydip in enumerate(skydips):
        airmass = 1 / np.sin(skydip.el)
        T_sky = T_atm * (1 - np.exp(-tau_atm * airmass))  # shape (n_samples,)

        # the elevation does not change: there is no line to fit
        if np.ptp(T_sky) < 1e-6:
            skipped.append(k)
            continue

        # slope of the line of every detector at once, shape (n_det,)
        slopes.append(np.polyfit(T_sky, skydip.signal, 1)[0])

    if skipped:
        _warn(
            f"{len(skipped)} of {len(skydips)} skydips skipped (positions {skipped}): "
            "their T_sky does not change.",
            RuntimeWarning,
        )

    if not slopes:
        raise ValueError(f"No usable skydip among the {len(skydips)} given.")

    # (n_skydips, n_det) -> (n_det,)
    return np.median(slopes, axis=0)
