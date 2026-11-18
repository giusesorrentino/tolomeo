import numpy as np
import pytest
from astropy.io import fits

from tests.common import (
    ARCSEC,
    DETECTOR_IDS,
    OFFSETS_LAT,
    OFFSETS_LON,
    make_scan,
    write_beammap,
    write_scan_dir,
    write_subscan,
)
from tolomeo.io import read_scan
from tolomeo.tod import ScanSpec

# small scans: every call of read_scan starts a pool of processes
N_SUBSCANS = 3
SUBSCAN_LEN = 64


def _scan(seed=0):
    return make_scan(n_subscans=N_SUBSCANS, subscan_len=SUBSCAN_LEN, seed=seed)


# -----------------------------------------------------------------------------
# read_scan
# -----------------------------------------------------------------------------


def test_read_scan_round_trip(tmp_path):
    # Check that a scan written as MISTRAL files is read back as it was.
    scan = _scan()
    scan.tod.flags[5:9] = False
    tod_dir, beammap = write_scan_dir(tmp_path, scan)

    read = read_scan(tod_dir, beammap)

    np.testing.assert_array_equal(read.tod.signal, scan.tod.signal)
    np.testing.assert_array_equal(read.tod.time, scan.tod.time)
    np.testing.assert_array_equal(read.tod.flags, scan.tod.flags)
    np.testing.assert_array_equal(read.tod.segments, scan.tod.segments)
    for name in ["ra", "dec", "par_angle", "az", "el"]:
        np.testing.assert_array_equal(
            getattr(read.coordinates, name), getattr(scan.coordinates, name)
        )
    assert read.channel_ids == scan.channel_ids
    np.testing.assert_allclose(read.offsets["lon"], scan.offsets["lon"], rtol=1e-14)
    np.testing.assert_allclose(read.offsets["lat"], scan.offsets["lat"], rtol=1e-14)


def test_read_scan_accepts_a_str(tmp_path):
    tod_dir, beammap = write_scan_dir(tmp_path, _scan())

    read = read_scan(str(tod_dir), str(beammap))

    assert read.tod.signal.shape == (N_SUBSCANS * SUBSCAN_LEN, len(DETECTOR_IDS))


def test_read_scan_several_directories(tmp_path):
    # Check that the scans are joined in the given order, one segment per subscan.
    first, second = _scan(seed=1), _scan(seed=2)
    first_dir, beammap = write_scan_dir(tmp_path / "first", first)
    second_dir, _ = write_scan_dir(tmp_path / "second", second)

    read = read_scan([second_dir, first_dir], beammap)

    np.testing.assert_array_equal(
        read.tod.signal, np.concatenate([second.tod.signal, first.tod.signal])
    )
    np.testing.assert_array_equal(
        read.tod.segments, np.arange(2 * N_SUBSCANS + 1) * SUBSCAN_LEN
    )


def test_read_scan_subscans_sorted_by_name(tmp_path):
    # Check that the files are read in the order of their names, whatever the case of
    # the suffix.
    scan = _scan()
    tod_dir = tmp_path / "tod"
    tod_dir.mkdir()
    write_subscan(tod_dir / "b.FITS", scan, SUBSCAN_LEN, 2 * SUBSCAN_LEN)
    write_subscan(tod_dir / "a.fits", scan, 0, SUBSCAN_LEN)
    beammap = tmp_path / "beammap.dat"
    write_beammap(beammap, DETECTOR_IDS, OFFSETS_LON, OFFSETS_LAT)

    read = read_scan(tod_dir, beammap)

    np.testing.assert_array_equal(read.tod.signal, scan.tod.signal[: 2 * SUBSCAN_LEN])


def test_read_scan_spec_shifts_the_boresight(tmp_path):
    # Check that the offsets of a ScanSpec, in arcsec, are added to the boresight:
    # Dec moves by the offset, RA by the offset divided by cos(dec).
    scan = _scan()
    tod_dir, beammap = write_scan_dir(tmp_path, scan)
    spec = ScanSpec(path=tod_dir, ra_offset=10.0, dec_offset="-0.1 arcmin")

    read = read_scan(spec, beammap)

    dec = scan.coordinates.dec
    np.testing.assert_allclose(
        read.coordinates.ra, scan.coordinates.ra + 10 * ARCSEC / np.cos(dec), atol=1e-10
    )
    np.testing.assert_allclose(read.coordinates.dec, dec - 6 * ARCSEC, atol=1e-10)


def test_read_scan_beammap_flags_and_order(tmp_path):
    # Check that the flagged detectors are dropped and the others sorted by id, with
    # their offsets and their columns in the same order.
    scan = _scan()
    tod_dir, _ = write_scan_dir(tmp_path, scan)
    beammap = tmp_path / "unsorted.dat"
    order = [2, 0, 3, 1]  # ids 12, 3, 20, 7
    flags = [0, 0, 1, 0]  # id 20 is a bad detector
    write_beammap(
        beammap,
        np.array(DETECTOR_IDS)[order],
        OFFSETS_LON[order],
        OFFSETS_LAT[order],
        flags,
    )

    read = read_scan(tod_dir, beammap)

    assert read.channel_ids == [3, 7, 12]
    np.testing.assert_allclose(read.offsets["lon"], OFFSETS_LON[:3], rtol=1e-14)
    np.testing.assert_allclose(read.offsets["lat"], OFFSETS_LAT[:3], rtol=1e-14)
    np.testing.assert_array_equal(read.tod.signal, scan.tod.signal[:, :3])


def test_read_scan_beammap_with_one_detector(tmp_path):
    scan = _scan()
    tod_dir, _ = write_scan_dir(tmp_path, scan)
    beammap = tmp_path / "one.dat"
    write_beammap(beammap, [7], OFFSETS_LON[1:2], OFFSETS_LAT[1:2])

    read = read_scan(tod_dir, beammap)

    assert read.channel_ids == [7]
    np.testing.assert_array_equal(read.tod.signal, scan.tod.signal[:, 1:2])


def test_read_scan_not_a_directory_raises(tmp_path):
    with pytest.raises(NotADirectoryError, match=r"is not a directory"):
        read_scan(tmp_path / "missing", tmp_path / "beammap.dat")


def test_read_scan_no_fits_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"No FITS subscans"):
        read_scan(tmp_path, tmp_path / "beammap.dat")


@pytest.mark.parametrize(
    "ids, flags, match",
    [
        ([3, 7], [1, 1], r"no good detectors"),
        ([3, 3], [0, 0], r"duplicate detector ids"),
    ],
    ids=["all_flagged", "duplicates"],
)
def test_read_scan_invalid_beammap_raises(tmp_path, ids, flags, match):
    tod_dir, _ = write_scan_dir(tmp_path, _scan())
    beammap = tmp_path / "bad.dat"
    write_beammap(beammap, ids, OFFSETS_LON[:2], OFFSETS_LAT[:2], flags)

    with pytest.raises(ValueError, match=match):
        read_scan(tod_dir, beammap)


def test_read_scan_beammap_too_few_columns_raises(tmp_path):
    tod_dir, _ = write_scan_dir(tmp_path, _scan())
    beammap = tmp_path / "short.dat"
    np.savetxt(beammap, np.ones((2, 5)))

    with pytest.raises(ValueError, match=r"at least six columns"):
        read_scan(tod_dir, beammap)


def test_read_scan_missing_detector_column_raises(tmp_path):
    tod_dir, _ = write_scan_dir(tmp_path, _scan())
    beammap = tmp_path / "other.dat"
    write_beammap(beammap, [3, 99], OFFSETS_LON[:2], OFFSETS_LAT[:2])

    with pytest.raises(ValueError, match=r"'chp_099' missing"):
        read_scan(tod_dir, beammap)


def test_read_scan_angles_not_in_radians_raise(tmp_path):
    scan = _scan()
    tod_dir = tmp_path / "tod"
    tod_dir.mkdir()
    write_subscan(tod_dir / "a.fits", scan, 0, SUBSCAN_LEN, angle_unit="deg")
    beammap = tmp_path / "beammap.dat"
    write_beammap(beammap, DETECTOR_IDS, OFFSETS_LON, OFFSETS_LAT)

    with pytest.raises(ValueError, match=r"expected radians"):
        read_scan(tod_dir, beammap)


def test_read_scan_tables_of_different_length_raise(tmp_path):
    tod_dir, beammap = write_scan_dir(tmp_path, _scan())
    with fits.open(tod_dir / "subscan_001.fits", mode="update") as hdul:
        hdul[2] = fits.BinTableHDU(hdul[2].data[:-1], name="PH TABLE")

    with pytest.raises(ValueError, match=r"length mismatch in subscan_001.fits"):
        read_scan(tod_dir, beammap)
