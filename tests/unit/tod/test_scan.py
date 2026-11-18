from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from pydantic import ValidationError

from tests.common import (
    DEC0,
    N_SUBSCANS,
    RA0,
    SAMPLE_RATE,
    SCAN_HALF_WIDTH,
    SUBSCAN_LEN,
    make_scan,
)
from tolomeo.tod import ScanSpec

# -----------------------------------------------------------------------------
# ScanSpec
# -----------------------------------------------------------------------------


def test_scan_spec_path_is_absolute(tmp_path, monkeypatch):
    # Check that a relative path is turned into an absolute one.
    monkeypatch.chdir(tmp_path)

    spec = ScanSpec(path="data")

    assert spec.path.is_absolute()
    assert spec.path == (tmp_path / "data").resolve()


def test_scan_spec_default_offsets():
    spec = ScanSpec(path=Path("data"))

    assert spec.ra_offset == 0.0
    assert spec.dec_offset == 0.0


@pytest.mark.parametrize("value", [90.0, "1.5 arcmin", 1.5 * u.arcmin])
def test_scan_spec_offsets_in_arcsec(value):
    # Check that the offsets are in arcsec, whatever unit they are given in.
    spec = ScanSpec(path="data", ra_offset=value)

    assert np.isclose(spec.ra_offset, 90.0)


def test_scan_spec_unknown_key_raises():
    with pytest.raises(ValidationError):
        ScanSpec(path="data", ra_ofset=1.0)


# -----------------------------------------------------------------------------
# Scan.select
# -----------------------------------------------------------------------------


def test_scan_select_keeps_tod_and_pointing_together():
    scan = make_scan()
    mask = np.zeros(scan.tod.signal.shape[0], dtype=bool)
    mask[::3] = True

    selected = scan.select(mask)

    np.testing.assert_array_equal(selected.tod.signal, scan.tod.signal[mask])
    np.testing.assert_array_equal(selected.coordinates.ra, scan.coordinates.ra[mask])
    assert selected.channel_ids == scan.channel_ids


# -----------------------------------------------------------------------------
# Scan.center
# -----------------------------------------------------------------------------


def test_scan_center_is_the_field_centre():
    ra, dec = make_scan().center

    assert np.isclose(ra, RA0)
    assert np.isclose(dec, DEC0)


def test_scan_center_uses_only_good_samples():
    # Check that a flagged sample far away does not move the centre.
    scan = make_scan()
    scan.coordinates.ra[0] = RA0 + 1.0
    scan.tod.flags[0] = False

    ra, _ = scan.center

    assert np.isclose(ra, RA0, atol=1e-9)


# -----------------------------------------------------------------------------
# Scan.scan_speed
# -----------------------------------------------------------------------------


def test_scan_speed():
    # Check the speed of the synthetic raster: 2 * SCAN_HALF_WIDTH on the sky in
    # (SUBSCAN_LEN - 1) / SAMPLE_RATE seconds, in arcmin/s.
    expected = 2 * SCAN_HALF_WIDTH / ((SUBSCAN_LEN - 1) / SAMPLE_RATE)
    expected = np.rad2deg(expected) * 60

    assert np.isclose(make_scan().scan_speed, expected, rtol=1e-3)


def test_scan_speed_no_valid_subscan_raises():
    scan = make_scan(n_subscans=N_SUBSCANS)
    scan.tod.flags[:] = False

    with pytest.raises(ValueError, match=r"No valid subscan"):
        _ = scan.scan_speed
