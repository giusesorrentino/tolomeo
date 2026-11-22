"""Gain calibration of a ``Scan`` with a skydip."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .._lib.messages import _warn
from .._lib.units import _is_temperature
from ..io.skydip import SkyDip
from ..tod import Scan

__all__ = ["calibrate", "skydip_responsivity"]


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
    skydips : sequence of :class:`~tolomeo.io.skydip.SkyDip`
        From :func:`~tolomeo.io.read_skydip`. The result is in the order of the
        ``channel_ids`` given to it, which for :func:`calibrate` must be
        ``scan.channel_ids``.
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


def calibrate(
    scan: Scan,
    responsivity: NDArray[np.float64],
    *,
    tau_atm: float | None = None,
) -> Scan:
    """
    Divide the signal of every detector by its responsivity.

    The TOD must be cleaned first (:func:`~tolomeo.processing.process_tod`) and
    calibrated after.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        Data to calibrate. It is not modified.
    responsivity : :class:`~numpy.ndarray`
        From :func:`skydip_responsivity`, in the order of ``scan.channel_ids``: for
        MISTRAL in rad/K. Shape ``(n_det,)``.
    tau_atm : float, optional
        If given, the signal is also divided by ``exp(-tau_atm * airmass)``, the
        transmission of the atmosphere at the elevation of each sample, so that the
        result is the temperature of the source above the atmosphere. A sample whose
        elevation is not valid (not finite or not positive) is only divided by the
        responsivity.

    Returns
    -------
    :class:`~tolomeo.tod.Scan`
        A new scan, with the signal in kelvin. A negative responsivity flips the sign.

    Raises
    ------
    ValueError
        If the TOD is already in a temperature unit (calibrated twice), if
        ``responsivity`` does not have one value per detector, or if one of them is
        zero or not finite.
    """
    if _is_temperature(scan.tod.unit):
        raise ValueError(
            f"The TOD is already in {scan.tod.unit!r}: it has been calibrated."
        )

    n_det = scan.tod.signal.shape[1]
    responsivity = np.asarray(responsivity, dtype=np.float64)

    if responsivity.shape != (n_det,):
        raise ValueError(
            f"`responsivity` has shape {responsivity.shape}, expected ({n_det},)."
        )

    invalid = ~np.isfinite(responsivity) | (responsivity == 0)
    if invalid.any():
        ids = [scan.channel_ids[i] for i in np.flatnonzero(invalid)]
        raise ValueError(
            f"{invalid.sum()} of {n_det} detectors have a zero or non-finite "
            f"responsivity (ids: {ids[:10]}{'...' if len(ids) > 10 else ''})."
        )

    signal = scan.tod.signal / responsivity[None, :]

    if tau_atm is not None:
        el = scan.coordinates.el
        with np.errstate(divide="ignore", invalid="ignore"):
            transmission = np.exp(-tau_atm / np.sin(el))

        # where the elevation is not valid there is nothing to correct (NaN, or 0)
        transmission = np.where((el > 0) & (transmission > 0), transmission, 1.0)
        signal /= transmission[:, None]

    return scan.model_copy(
        update={"tod": scan.tod.model_copy(update={"signal": signal, "unit": "K"})}
    )
