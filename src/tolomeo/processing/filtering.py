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
The window, the filters and the SVD modes work on the whole TOD and do not use the
mask. :func:`clean_subscans` is a shortcut for the common mode and the
baseline only.
"""

import logging
import time
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy.signal import windows

from .._lib.messages import _warn
from .._lib.spectral import _check_nyquist, _narrow_passband
from .._signal.common_mode import decompose, remove_common_mode
from .._signal.detrend import remove_baseline, remove_spline
from .._signal.filters import bandpass, highpass, lowpass
from .._signal.splines import bspline_knots
from ..map import MapGeometry
from ..tod import Scan
from .config import (
    Bandpass,
    Baseline,
    CommonMode,
    FilteringConfig,
    Highpass,
    Lowpass,
    RemoveModes,
    RemoveSpline,
    Window,
)
from .masks import RadiusMask

logger = logging.getLogger(__name__)

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


def _common_mode(signal, tod, step, background):
    for start, stop in pairwise(tod.segments):
        if stop <= start:
            continue

        # no background given: every sample can be used
        if background is None:
            usable = np.ones(stop - start, dtype=bool)
        else:
            usable = background[start:stop]

        signal[start:stop] = remove_common_mode(
            signal[start:stop], usable, method=step.method
        )
    return signal


def _window(signal, step):
    try:
        window_function = getattr(windows, step.name)
    except AttributeError as exc:
        raise ValueError(f"Unknown scipy window: {step.name!r}") from exc

    window = window_function(signal.shape[0], **step.kwargs)
    signal *= window[:, None]
    return signal


def _baseline(signal, tod, step, background):
    mask = None

    for k, (start, stop) in enumerate(pairwise(tod.segments)):
        if stop <= start:
            continue

        n_t = stop - start

        if n_t <= step.deg:
            raise ValueError(
                f"A polynomial of degree {step.deg} needs more than {step.deg} "
                f"samples, subscan {k} has {n_t}."
            )

        if background is not None:
            mask = background[start:stop]

        signal[start:stop] = remove_baseline(
            signal[start:stop], tod.time[start:stop], deg=step.deg, mask=mask
        )
    return signal


def _spline(signal, tod, step, background):
    mask = None

    for k, (start, stop) in enumerate(pairwise(tod.segments)):
        if stop <= start:
            continue

        t = tod.time[start:stop]

        # the knots are cheap; the basis is not (it needs about n_knots * order *
        # n_t floats), so check the number of basis functions before building it
        knots = bspline_knots(t, step.knot_spacing, step.order)
        n_basis = len(knots) - step.order - 1

        if n_basis >= stop - start:
            raise ValueError(
                f"The spline has {n_basis} basis functions but subscan {k} has "
                f"only {stop - start} samples: use a larger knot_spacing "
                "(in days)."
            )

        if background is not None:
            mask = background[start:stop]

        signal[start:stop] = remove_spline(
            signal[start:stop], t, knots, order=step.order, mask=mask
        )

    return signal


def _remove_modes(signal, step):
    modes_to_remove = step.modes_to_remove

    # nothing to remove (the SVD does not accept k = 0)
    if modes_to_remove == 0:
        return signal

    # the SVD needs k strictly smaller than both sizes of the TOD
    if modes_to_remove >= min(signal.shape):
        raise ValueError(
            f"Too many modes requested ({modes_to_remove}) for a TOD of shape "
            f"{signal.shape}."
        )

    # modes (n_samples, k) @ weights (k, n_det) is the part of the
    # TOD explained by the k strongest modes
    modes, weights = decompose(signal, k=modes_to_remove)
    signal -= modes @ weights

    return signal


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
            _warn(
                "Filtering without windowing is not recommended: put a `window` step "
                "before the filters."
            )

        match step:
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

    cleaned = scan.model_copy(
        update={"tod": tod.model_copy(update={"signal": signal, "passband": passband})}
    )
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
    Remove the baseline first and then the common mode from every subscan.

    The two steps are applied subscan by subscan, never on the whole TOD:
    every subscan has its own offset, and between two subscans there is a gap
    of several seconds, so the signal is not continuous across them.

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
    steps = [s for s in (baseline, common_mode) if s is not None]
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
        ``scan.source``. The other masks do not use it.

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
    sample). ``window``, the filters and ``remove_modes`` work on the whole TOD
    and ignore the mask.
    """
    if config is None:
        config = FilteringConfig()

    # the mask of the config becomes a background array, unless the caller gave one
    if config.mask is not None:
        if background is not None:
            raise ValueError("Give either `background` or `config.mask`, not both.")

        args = (scan, geometry) if isinstance(config.mask, RadiusMask) else (scan,)
        background = config.mask.build(*args)

    return _run_steps(scan, config.steps, background, overwrite_scan)
