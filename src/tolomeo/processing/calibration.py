"""Gain calibration of a ``Scan`` with a skydip."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..io.skydip import _read_dip, _run_on_files
from ..tod import Scan

__all__ = ["calibrate", "skydip_responsivity"]


def _file_responsivity(
    file: Path, columns: list[str], T_atm: float, tau_atm: float
) -> NDArray[np.float64] | None:
    """
    Responsivity of every detector from one skydip file, or None to skip the file.

    Runs in a worker process of :func:`skydip_responsivity`: the signal of the file
    stays in the worker and only ``n_det`` numbers go back. The file is skipped if it
    cannot be read (see :func:`~tolomeo.io.skydip.read_skydip`) or if ``T_sky`` does
    not change. Shape ``(n_det,)``.
    """
    skydip = _read_dip(file, columns)
    if skydip is None:
        return None

    airmass = 1 / np.sin(skydip.el)
    T_sky = T_atm * (1 - np.exp(-tau_atm * airmass))  # shape (n_samples,)

    if np.ptp(T_sky) < 1e-6:
        return None

    # slope of the line of every detector at once, shape (n_det,)
    return np.polyfit(T_sky, skydip.signal, 1)[0]


def skydip_responsivity(
    gain_dir: str | Path,
    detector_ids: Sequence[int],
    *,
    T_atm: float = 267.0,
    tau_atm: float = 0.16,
) -> NDArray[np.float64]:
    """
    Responsivity of every detector from the skydip files of a directory (or from a
    single file).

    In each file the signal of a detector is fitted with a straight line against the
    temperature of the sky, ``T_sky = T_atm * (1 - exp(-tau_atm * airmass))`` with
    ``airmass = 1 / sin(el)``. The responsivity is the slope, and the result is its
    mean over the files. The files are read and fitted in a pool of processes, so
    that only ``n_det`` numbers per file are sent back (use
    :func:`~tolomeo.io.skydip.read_skydip` to get the signal of the files).

    Parameters
    ----------
    gain_dir : str or :class:`~pathlib.Path`
        Directory with the ``GAIN_CAL`` FITS files, or the path of a single FITS file.
    detector_ids : sequence of int
        Detectors to use, for example ``scan.detector_ids``. The result is in this
        order.
    T_atm : float
        Temperature of the atmosphere, in kelvin.
        Default ``267``.
    tau_atm : float
        Zenith opacity of the atmosphere.
        Default ``0.16``.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        Responsivity in signal units per kelvin; it can be negative.
        Shape ``(n_det,)``.

    Raises
    ------
    NotADirectoryError
        If ``gain_dir`` is neither a directory nor a file.
    FileNotFoundError
        If no file can be used. A file is skipped, with a warning that lists its
        name, if it cannot be read (see :func:`~tolomeo.io.skydip.read_skydip`) or
        if its ``T_sky`` does not change.
    """
    slopes = _run_on_files(gain_dir, detector_ids, _file_responsivity, T_atm, tau_atm)

    return np.mean(slopes, axis=0)


def calibrate(
    scan: Scan,
    responsivity: NDArray[np.float64],
    *,
    tau_atm: float | None = None,
) -> Scan:
    """
    Divide the signal of every detector by its responsivity.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        Data to calibrate. It is not modified.
    responsivity : :class:`~numpy.ndarray`
        From :func:`skydip_responsivity`, in the order of ``scan.detector_ids``.
        Shape ``(n_det,)``.
    tau_atm : float, optional
        If given, the signal is also divided by ``exp(-tau_atm * airmass)``, the
        transmission of the atmosphere at the elevation of each sample, so that the
        result is the temperature of the source above the atmosphere. A sample whose
        elevation is not valid (not finite or not positive) is only divided by the
        responsivity: it has no valid pointing and is dropped later by
        :func:`~tolomeo.processing.process_tod`. Default None: only the responsivity.

    Returns
    -------
    :class:`~tolomeo.tod.Scan`
        A new scan, with the signal in kelvin. A negative responsivity flips the sign.

    Raises
    ------
    ValueError
        If ``responsivity`` does not have one value per detector, or if one of them is
        zero or not finite.
    """
    n_det = scan.tod.signal.shape[1]
    responsivity = np.asarray(responsivity, dtype=np.float64)

    if responsivity.shape != (n_det,):
        raise ValueError(
            f"`responsivity` has shape {responsivity.shape}, expected ({n_det},)."
        )

    invalid = ~np.isfinite(responsivity) | (responsivity == 0)
    if invalid.any():
        ids = [scan.detector_ids[i] for i in np.flatnonzero(invalid)]
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
        update={"tod": scan.tod.model_copy(update={"signal": signal})}
    )