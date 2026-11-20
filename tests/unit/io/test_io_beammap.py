import numpy as np
import pytest

from tests.common import write_beammap as write_plain_beammap
from tolomeo.beammap import BeamMap, ChannelFlag
from tolomeo.io import read_beammap, write_beammap

ARCSEC = np.deg2rad(1 / 3600)

# the header and the rows of an older MISTRAL file: Tcal in the columns 4 and 5, the
# flag 0 or 1, the offsets of the reference channel
OLD_FILE = """\
# Feed list created: Feb 19 17:59:39 2025
# nfeeds: 3
# Col.1: id
# Col.2: lon-offset (deg)
# Col.3: lat-offset (deg)
# Col.4: Tcal pol1 (Kelvin)
# Col.5: Tcal pol2 (Kelvin)
# Col.6: flag (0=>Ok, 1=>bad)
# chp8 az_off (deg): 1.722222222e-05
# chp8 el_off (deg): -0.0003638888889
 3     0.007735    -0.000947     0.000000     0.000000  0
 8     0.000000     0.000000     0.000000     0.000000  0
12    -0.005203     0.005192     0.000000     0.000000  1
"""


def _beammap():
    """A BeamMap with every field, a failed fit and a channel with two reasons."""
    nan = np.nan
    return BeamMap(
        channel_ids=np.array([3, 7, 12, 20]),
        flags=np.array([0, ChannelFlag.FIT_FAILED, 0, 10]),
        lat=np.array([15.0, nan, 0.0, 5.0]) * ARCSEC,
        lon=np.array([-20.0, nan, 0.0, 25.0]) * ARCSEC,
        lat_error=np.array([0.03, nan, 0.02, 0.4]) * ARCSEC,
        lon_error=np.array([0.04, nan, 0.02, 0.5]) * ARCSEC,
        ref_channel=12,
        ref_lat=-5.3 * ARCSEC,
        ref_lon=1.0 * ARCSEC,
        snr_threshold=10.0,
    )


# -----------------------------------------------------------------------------
# write_beammap and read_beammap
# -----------------------------------------------------------------------------


def test_write_and_read_beammap_round_trip(tmp_path):
    # Check that every field comes back, the NaN of the failed fit too.
    beammap = _beammap()
    path = tmp_path / "beammap.dat"

    write_beammap(beammap, path)
    read = read_beammap(path)

    np.testing.assert_array_equal(read.channel_ids, beammap.channel_ids)
    np.testing.assert_array_equal(read.flags, beammap.flags)
    for name in ["lat", "lon"]:  # written to 1e-6 deg
        np.testing.assert_allclose(
            getattr(read, name), getattr(beammap, name), atol=np.deg2rad(1e-6)
        )
    for name in ["lat_error", "lon_error"]:  # written with 4 digits
        np.testing.assert_allclose(
            getattr(read, name), getattr(beammap, name), rtol=1e-3
        )
    assert read.ref_channel == beammap.ref_channel
    assert read.ref_lat == pytest.approx(beammap.ref_lat, rel=1e-11)
    assert read.ref_lon == pytest.approx(beammap.ref_lon, rel=1e-11)
    assert read.snr_threshold == beammap.snr_threshold


def test_write_beammap_header(tmp_path):
    # Check that the header lists every reason, with the threshold of LOW_SNR.
    path = tmp_path / "beammap.dat"

    write_beammap(_beammap(), path)

    header = [line for line in path.read_text().splitlines() if line.startswith("#")]
    for flag in ChannelFlag:
        assert any(f"{flag.value} {flag.name}" in line for line in header)
    assert any("LOW_SNR (SNR < 10)" in line for line in header)
    assert any(line.startswith("# chp12 az_off (deg)") for line in header)


def test_write_beammap_without_errors_and_reference(tmp_path):
    # Check that what is not known is written as zeros, and the reference not at all.
    beammap = _beammap().model_copy(
        update={"lat_error": None, "lon_error": None, "ref_channel": None}
    )
    path = tmp_path / "beammap.dat"

    write_beammap(beammap, path)
    read = read_beammap(path)

    np.testing.assert_array_equal(np.loadtxt(path)[:, 3:5], 0)
    assert read.ref_channel is None
    assert read.ref_lat is None


def test_read_beammap_old_file(tmp_path):
    # Check that a bad channel of an old file has an unknown reason, and that the
    # Tcal are not read as errors.
    path = tmp_path / "old.dat"
    path.write_text(OLD_FILE)

    read = read_beammap(path)

    np.testing.assert_array_equal(read.channel_ids, [3, 8, 12])
    np.testing.assert_array_equal(read.flags, [0, 0, ChannelFlag.UNKNOWN])
    np.testing.assert_allclose(read.lon, np.deg2rad([0.007735, 0.0, -0.005203]))
    assert read.lat_error is None
    assert read.lon_error is None
    assert read.snr_threshold is None
    assert read.ref_channel == 8
    assert read.ref_lon == pytest.approx(np.deg2rad(1.722222222e-05), rel=1e-12)
    assert read.ref_lat == pytest.approx(np.deg2rad(-0.0003638888889), rel=1e-12)


def test_read_beammap_old_file_written_again(tmp_path):
    # Check that the reason stays unknown, and the offsets are kept.
    old = tmp_path / "old.dat"
    old.write_text(OLD_FILE)
    new = tmp_path / "new.dat"

    write_beammap(read_beammap(old), new)
    read = read_beammap(new)

    np.testing.assert_array_equal(read.flags, [0, 0, ChannelFlag.UNKNOWN])
    np.testing.assert_allclose(read.lon, read_beammap(old).lon, atol=np.deg2rad(1e-6))


def test_read_beammap_without_header(tmp_path):
    path = tmp_path / "plain.dat"
    write_plain_beammap(path, [3, 7], np.zeros(2), np.ones(2) * ARCSEC, [0, 1])

    read = read_beammap(path)

    np.testing.assert_array_equal(read.flags, [0, ChannelFlag.UNKNOWN])
    np.testing.assert_allclose(read.lat, ARCSEC)
    assert read.ref_channel is None


def test_read_beammap_too_few_columns_raises(tmp_path):
    path = tmp_path / "short.dat"
    np.savetxt(path, np.ones((2, 5)))

    with pytest.raises(ValueError, match=r"at least six columns"):
        read_beammap(path)


def test_read_beammap_duplicate_ids_raises(tmp_path):
    path = tmp_path / "twice.dat"
    write_plain_beammap(path, [3, 3], np.zeros(2), np.zeros(2))

    with pytest.raises(ValueError, match=r"duplicate detector ids"):
        read_beammap(path)
