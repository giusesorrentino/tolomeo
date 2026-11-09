# Adapted from:
# https://github.com/thomaswmorris/maria/blob/master/maria/tod/processing.py
# Original implementation by Thomas W. Morris (Maria).

"""
Cleaning of the detector TODs, the step that comes before mapmaking.

The entry point is :func:`process_tod`. It works on a ``Scan`` whose
``tod.signal`` is shaped ``(n_samples, n_det)``: one row per time sample, one
column per detector, with all the subscans one after the other.
``tod.segments`` gives the first row of every subscan.

The cleaning is a list of steps (:class:`~tolomeo.processing.FilteringConfig`) run
in order. The common mode (what all the detectors see together: atmosphere and
electronics), the baseline and the spline work subscan by subscan, and they can
leave the source out of their estimate with a mask (``background``).
The slope, the window, the filters and the SVD modes work on the whole TOD and do
not use the mask. :func:`clean_subscans` is a shortcut for the common mode and the
baseline only.
"""

import logging
import time

import numpy as np
from numpy.typing import NDArray

from .._signal.detrend import remove_slope
from .._signal.filters import bandpass, highpass, lowpass
from ..map import MapGeometry
from ..tod import Scan
from ._mask import _build_mask
from ._steps import (
    _baseline,
    _check_nyquist,
    _common_mode,
    _remove_modes,
    _spline,
    _window,
)
from .config import (
    Bandpass,
    Baseline,
    CommonMode,
    FilteringConfig,
    Highpass,
    Lowpass,
    RemoveModes,
    RemoveSlope,
    RemoveSpline,
    Window,
)

logger = logging.getLogger("tolomeo")

__all__ = ["clean_subscans", "process_tod"]


def _drop_invalid_pointing(scan: Scan) -> Scan:
    """
    Keep only the samples whose ``ra``, ``dec`` and ``par_angle`` are all finite.

    Returns a new :class:`~tolomeo.tod.Scan` (see
    :meth:`~tolomeo.tod.Scan.select`).
    """
    coords = scan.coordinates
    ok = (
        np.isfinite(coords.ra) & np.isfinite(coords.dec) & np.isfinite(coords.par_angle)
    )
    return scan.select(ok)


def _check_scan(scan, background) -> None:
    n_samples = scan.tod.signal.shape[0]

    # tod.segments holds the first row of every subscan plus, at the end, the
    # total number of rows: subscan k is signal[segments[k]:segments[k + 1]]
    if scan.tod.segments[0] != 0 or scan.tod.segments[-1] != n_samples:
        raise ValueError("segments must start at 0 and end at n_samples.")

    if scan.tod.signal.ndim != 2:
        raise ValueError("Expected TOD data shaped (n_samples, n_detectors).")

    if background is not None and np.shape(background) not in {
        (n_samples,),
        scan.tod.signal.shape,
    }:
        raise ValueError(
            "`background` must have shape (n_samples,) or (n_samples, n_det)."
        )

    if scan.tod.time.shape != (n_samples,):
        raise ValueError("Timestamp length does not match the TOD.")

    if scan.tod.flags.shape != (n_samples,):
        raise ValueError("Flag length does not match the TOD.")


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


def _run_steps(scan, steps, background, overwrite_scan):
    _check_scan(scan, background)

    tod = scan.tod
    # None stays None: the steps then use every sample
    if background is not None:
        background = np.asarray(background, dtype=bool)

    signal = tod.signal if overwrite_scan else tod.signal.copy()
    passband = tod.passband
    sample_rate = tod.sample_rate

    windowed = False

    for step in steps:
        start = time.monotonic()

        if isinstance(step, (Lowpass, Highpass, Bandpass)) and not windowed:
            logger.warning("Filtering without windowing is not recommended.")

        match step:
            case RemoveSlope():
                signal = remove_slope(signal)
            case CommonMode():
                signal = _common_mode(signal, tod, step, background)
            case Baseline():
                signal = _baseline(signal, tod, step, background)
            case RemoveSpline():
                signal = _spline(signal, tod, step, background)
            case Window():
                signal = _window(signal, step)
                windowed = True
            case Lowpass():
                _check_nyquist(sample_rate, step.fc)
                signal = lowpass(
                    signal,
                    fc=step.fc,
                    sample_rate=sample_rate,
                    order=step.order,
                    axis=0,
                )
                passband = _narrow_passband(passband, f_upper=step.fc)
            case Highpass():
                _check_nyquist(sample_rate, step.fc)
                signal = highpass(
                    signal,
                    fc=step.fc,
                    sample_rate=sample_rate,
                    order=step.order,
                    axis=0,
                )
                passband = _narrow_passband(passband, f_lower=step.fc)
            case Bandpass():
                _check_nyquist(sample_rate, step.f_lower, step.f_upper)
                signal = bandpass(
                    signal,
                    f_lower=step.f_lower,
                    f_upper=step.f_upper,
                    sample_rate=sample_rate,
                    order=step.order,
                    axis=0,
                )
                passband = _narrow_passband(passband, step.f_lower, step.f_upper)
            case RemoveModes():
                signal = _remove_modes(signal, step)
            case _:
                raise NotImplementedError(f"No implementation for {step!r}.")

        logger.debug("Completed %s in %.3f s.", step.type, time.monotonic() - start)

        if np.isnan(signal).any():
            raise ValueError(f"{step.type} introduced NaNs.")

    cleaned = scan.model_copy(update={"tod": tod.model_copy(update={"signal": signal, "passband": passband})})
    return _drop_invalid_pointing(cleaned)


# defaults of `clean_subscans`; the steps are frozen, so sharing them between calls is safe
_DEFAULT_COMMON_MODE = CommonMode()
_DEFAULT_BASELINE = Baseline()


def clean_subscans(
    scan: Scan,
    *,
    common_mode=_DEFAULT_COMMON_MODE,
    baseline=_DEFAULT_BASELINE,
    background: NDArray[np.bool_] | None = None,
    overwrite_scan: bool = False,
) -> Scan:
    """
    Remove the common mode and the baseline from every subscan.

    The two steps are applied subscan by subscan, never on the whole TOD:
    every subscan has its own offset, and between two subscans there is a gap
    of several seconds, so the signal is not continuous across them.

    The common mode is removed first, because the baseline of each detector
    is estimated on what is left after that.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        Data to clean. It is not modified.
    common_mode : :class:`~tolomeo.processing.CommonMode` or None, optional
        Parameters of the common mode removal. None skips the step.
        Default ``CommonMode()``.
    baseline : :class:`~tolomeo.processing.Baseline` or None, optional
        Parameters of the baseline removal. None skips the step.
        Default ``Baseline()``.
    background : :class:`~numpy.ndarray` of bool, optional
        True for the samples far from the source, the only ones used to
        estimate the common mode and the baseline. With shape ``(n_samples,)``
        the same samples are used for every detector; with ``(n_samples, n_det)``
        each detector has its own. None means all the samples.
    overwrite_scan : bool, optional
        If True, ``scan.tod.signal`` is modified in place instead of copied.
        Default ``False``.

    Returns
    -------
    :class:`~tolomeo.tod.Scan`
        New scan with the cleaned signal. Times, flags, segments,
        coordinates and offsets are the same as the input. The signal is
        always a new array, even when both steps are skipped, so the caller
        can modify it in place.
    """
    steps = [s for s in (common_mode, baseline) if s is not None]
    return _run_steps(scan, steps, background, overwrite_scan)


def process_tod(
    scan: Scan,
    config: FilteringConfig | None = None,
    background: NDArray[np.bool_] | None = None,
    overwrite_scan: bool = False,
    *,
    geometry: MapGeometry | None = None,
) -> Scan:
    """
    Clean the detector TODs of a scan.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        Scan to clean. It is not modified. ``scan.tod.signal`` is shaped
        ``(n_samples, n_det)``.
    config : :class:`~tolomeo.processing.FilteringConfig`, optional
        Steps to run and their parameters, and the optional ``mask`` that leaves the
        source out of the fits. Default None, which is ``FilteringConfig()``: the common
        mode and a linear baseline of each subscan.
    background : :class:`~numpy.ndarray` of bool, optional
        True for the samples far from the source, the only ones used by the steps
        that work subscan by subscan. With shape ``(n_samples,)`` the same samples
        are used for every detector; with ``(n_samples, n_det)`` each detector has
        its own. It is an alternative to ``config.mask``: giving both is an error.
        None (and no ``config.mask``) means all the samples.
    overwrite_scan : bool, optional
        If True, ``scan.tod.signal`` is modified in place instead of copied.
        Default ``False``.
    geometry : :class:`~tolomeo.map.MapGeometry`, optional
        Grid of the map that will be made from the cleaned scan. It is needed only
        with a :class:`~tolomeo.processing.RadiusMask`: the distance from the source
        is measured on the plane of the projection of the map, so that the mask is
        where the source is in the map. Build it before cleaning; it only needs
        ``scan.center``. The other masks do not use it.

    Returns
    -------
    :class:`~tolomeo.tod.Scan`
        New scan with the cleaned signal. Times, flags, segments,
        coordinates, offsets and the order of the detectors are the same as
        the input.

    Notes
    -----
    The steps of ``config.steps`` run in order. ``common_mode``, ``baseline`` and
    ``remove_spline`` work subscan by subscan and, with a mask, estimate what to
    subtract from the background samples only (it is then subtracted from every
    sample). ``remove_slope``, ``window``, the filters and ``remove_modes`` work on
    the whole TOD and ignore the mask.
    """
    if config is None:
        config = FilteringConfig()

    # the mask of the config becomes a background array, unless the caller gave one
    if config.mask is not None:
        if background is not None:
            raise ValueError("Give either `background` or `config.mask`, not both.")
        background = _build_mask(scan, config.mask, geometry)

    return _run_steps(scan, config.steps, background, overwrite_scan)
