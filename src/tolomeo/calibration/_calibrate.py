import numpy as np
from numpy.typing import NDArray

from .._lib.units import _is_temperature
from ..tod import Scan

__all__ = ["calibrate"]


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
        For example from :func:`~tolomeo.calibration.skydip_responsivity`, in the
        order of ``scan.channel_ids``: for MISTRAL in rad/K. Shape ``(n_det,)``.
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

    n_det = scan.tod.n_det
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
        signal /= _transmission(scan.coordinates.el, tau_atm)[:, None]

    return scan.model_copy(
        update={"tod": scan.tod.model_copy(update={"signal": signal, "unit": "K"})}
    )


def _transmission(el: NDArray[np.float64], tau_atm: float) -> NDArray[np.float64]:
    """
    Transmission of the atmosphere at every sample, ``exp(-tau_atm / sin(el))``.

    Where the elevation is not valid (not finite, or not positive) it is 1: there is
    nothing to correct. ``el`` is in radians. Shape ``(n_samples,)``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        transmission = np.exp(-tau_atm / np.sin(el))
    return np.where((el > 0) & (transmission > 0), transmission, 1.0)
