from itertools import pairwise

import numpy as np
from scipy.signal import windows

from .._signal.common_mode import remove_common_mode
from .._signal.decompose import decompose
from .._signal.detrend import remove_baseline, remove_spline
from .._signal.splines import bspline_knots


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
