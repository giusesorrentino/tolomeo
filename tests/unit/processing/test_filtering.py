from itertools import pairwise

import numpy as np
import pytest

from tests.common import DEC0, PIXEL_SIZE, RA0, make_scan, with_signal
from tolomeo.map import MapGeometry
from tolomeo.processing import (
    Bandpass,
    Baseline,
    CommonMode,
    FilteringConfig,
    Highpass,
    Lowpass,
    RemoveModes,
    Window,
    clean_subscans,
    process_tod,
)
from tolomeo.processing.masks import RadiusMask

rng = np.random.default_rng(11)

GEOMETRY = MapGeometry(nx=60, ny=40, pixel_size=PIXEL_SIZE, center=(RA0, DEC0))


def _drifting_scan():
    """Scan with a random walk shared by the detectors and a line per subscan."""
    scan = make_scan(noise_rms=0.01)
    n_samples, n_det = scan.tod.signal.shape
    common = np.cumsum(rng.standard_normal(n_samples))[:, None]
    lines = np.zeros((n_samples, n_det))
    for start, stop in pairwise(scan.tod.segments):
        x = np.linspace(-1, 1, stop - start)[:, None]
        lines[start:stop] = rng.normal(size=n_det) + rng.normal(size=n_det) * x
    return with_signal(scan, scan.tod.signal + common + lines)


# -----------------------------------------------------------------------------
# process_tod
# -----------------------------------------------------------------------------


def test_process_tod_default_removes_drifts():
    # Check that the default steps (baseline, then common mode) remove a shared drift
    # and a baseline per subscan, down to the white noise.
    scan = _drifting_scan()

    clean = process_tod(scan)

    assert clean.tod.signal.std() < 0.05


def test_process_tod_does_not_modify_the_input():
    scan = _drifting_scan()
    original = scan.tod.signal.copy()

    process_tod(scan)

    np.testing.assert_array_equal(scan.tod.signal, original)


def test_process_tod_overwrite_scan():
    # Check that the input signal is cleaned in place.
    scan = _drifting_scan()

    clean = process_tod(scan, overwrite_scan=True)

    np.testing.assert_array_equal(scan.tod.signal, clean.tod.signal)


def test_process_tod_drops_samples_without_pointing():
    scan = make_scan()
    scan.coordinates.ra[10:15] = np.nan

    clean = process_tod(scan, FilteringConfig(steps=[]))

    assert clean.tod.signal.shape[0] == scan.tod.signal.shape[0] - 5
    assert np.isfinite(clean.coordinates.ra).all()


def test_process_tod_filters_narrow_the_passband():
    scan = make_scan()
    steps = [
        Window(name="tukey", kwargs={"alpha": 0.1}),
        Highpass(fc=1.0),
        Lowpass(fc=30.0),
    ]

    clean = process_tod(scan, FilteringConfig(steps=steps))

    assert clean.tod.passband == (1.0, 30.0)


def test_process_tod_filter_without_window_warns():
    with pytest.warns(UserWarning, match=r"without windowing"):
        process_tod(
            make_scan(), FilteringConfig(steps=[Bandpass(f_lower=1, f_upper=10)])
        )


def test_process_tod_cutoff_above_nyquist_raises():
    steps = [Window(name="hann"), Lowpass(fc=200.0)]

    with pytest.raises(ValueError, match=r"Nyquist"):
        process_tod(make_scan(), FilteringConfig(steps=steps))


def test_process_tod_mask_keeps_the_source():
    # Check that with a radius mask the baseline is fitted only away from the
    # source, so that a source on top of a constant offset survives the cleaning.
    scan = make_scan(noise_rms=0.0)
    background = RadiusMask(radius=0.5).build(scan, GEOMETRY)
    source = np.where(background, 0.0, 1.0)
    scan = with_signal(scan, source + 3.0)

    config = FilteringConfig(steps=[Baseline(deg=0)], mask=RadiusMask(radius=0.5))
    clean = process_tod(scan, config, geometry=GEOMETRY)

    np.testing.assert_allclose(clean.tod.signal, source, atol=1e-10)


def test_process_tod_mask_and_background_raises():
    scan = make_scan()
    config = FilteringConfig(mask=RadiusMask(radius=0.5))

    with pytest.raises(ValueError, match=r"either `background` or `config.mask`"):
        process_tod(scan, config, np.ones(scan.tod.signal.shape[0], dtype=bool))


def test_process_tod_wrong_background_shape_raises():
    with pytest.raises(ValueError, match=r"`background` must have shape"):
        process_tod(make_scan(), background=np.ones(3, dtype=bool))


def test_process_tod_remove_modes():
    # Check that removing one mode removes a mode shared by the detectors.
    scan = make_scan(noise_rms=0.0)
    mode = np.sin(np.linspace(0, 30, scan.tod.signal.shape[0]))
    scan = with_signal(scan, np.outer(mode, [1.0, 2.0, -1.0, 0.5]))

    clean = process_tod(scan, FilteringConfig(steps=[RemoveModes(modes_to_remove=1)]))

    np.testing.assert_allclose(clean.tod.signal, 0.0, atol=1e-8)


def test_process_tod_too_many_modes_raises():
    config = FilteringConfig(steps=[RemoveModes(modes_to_remove=4)])

    with pytest.raises(ValueError, match=r"Too many modes"):
        process_tod(make_scan(), config)


# -----------------------------------------------------------------------------
# clean_subscans
# -----------------------------------------------------------------------------


def test_clean_subscans_is_baseline_then_common_mode():
    scan = _drifting_scan()

    clean = clean_subscans(scan)
    reference = process_tod(scan, FilteringConfig(steps=[Baseline(), CommonMode()]))

    np.testing.assert_allclose(clean.tod.signal, reference.tod.signal)


def test_clean_subscans_skips_a_step_given_as_none():
    scan = _drifting_scan()

    clean = clean_subscans(scan, common_mode=None)
    reference = process_tod(scan, FilteringConfig(steps=[Baseline()]))

    np.testing.assert_allclose(clean.tod.signal, reference.tod.signal)
