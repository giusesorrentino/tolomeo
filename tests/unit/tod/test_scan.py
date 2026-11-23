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
# Scan.source
# -----------------------------------------------------------------------------


def _with_source(scan, source_ra, source_dec):
    """Copy of ``scan`` with another position of the source (None: unknown)."""
    coords = scan.coordinates.model_copy(
        update={"source_ra": source_ra, "source_dec": source_dec}
    )
    return scan.model_copy(update={"coordinates": coords})


def test_scan_source_is_the_position_of_the_source():
    ra, dec = make_scan().source

    assert ra == RA0
    assert dec == DEC0


def test_scan_source_of_a_moving_source_is_the_middle_sample():
    # Check that for a planet, whose position changes along the scan, the position
    # is the one of the sample in the middle.
    scan = make_scan()
    n_samples = scan.coordinates.ra.size
    track = RA0 + np.linspace(0.0, 1e-4, n_samples)

    ra, _ = _with_source(scan, track, scan.coordinates.source_dec).source

    assert ra == track[n_samples // 2]


def test_scan_source_unknown_raises():
    scan = _with_source(make_scan(), None, None)

    with pytest.raises(ValueError, match=r"no position of the source"):
        _ = scan.source


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


# -----------------------------------------------------------------------------
# Scan.to
# -----------------------------------------------------------------------------


def test_scan_to_converts_the_tod():
    # Check that the result is a scan with the TOD of TimeOrderedData.to, and the
    # pointing of this one.
    scan = make_scan()
    scan = scan.model_copy(update={"tod": scan.tod.model_copy(update={"unit": "K"})})

    converted = scan.to("mJy/beam", frequency=150.0, beam_fwhm="18 arcsec")

    expected = scan.tod.to("mJy/beam", frequency=150.0, beam_fwhm="18 arcsec")
    np.testing.assert_array_equal(converted.tod.signal, expected.signal)
    assert converted.tod.unit == expected.unit == "mJy beam-1"
    assert converted.coordinates is scan.coordinates
    assert converted.channel_ids == scan.channel_ids
    assert scan.tod.unit == "K"
