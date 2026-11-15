import numpy as np
import pytest

from tests.common import DETECTOR_IDS, write_skydip
from tolomeo.io import read_skydip

rng = np.random.default_rng(5)

N_SAMPLES = 50


def _skydip_data(n_samples=N_SAMPLES):
    """Elevation (radians) and signal ``(n_samples, n_det)`` of a skydip."""
    el = np.linspace(np.deg2rad(20), np.deg2rad(80), n_samples)
    signal = rng.standard_normal((n_samples, len(DETECTOR_IDS)))
    return el, signal


# -----------------------------------------------------------------------------
# read_skydip
# -----------------------------------------------------------------------------


def test_read_skydip_directory(tmp_path):
    # Check that every file is read, in the order of the names, with the columns in
    # the order of `detector_ids`.
    first, second = _skydip_data(), _skydip_data()
    write_skydip(tmp_path / "b.fits", *second, DETECTOR_IDS)
    write_skydip(tmp_path / "a.fits", *first, DETECTOR_IDS)

    skydips = read_skydip(tmp_path, [12, 3])

    assert len(skydips) == 2
    np.testing.assert_array_equal(skydips[0].el, first[0])
    np.testing.assert_array_equal(skydips[0].signal, first[1][:, [2, 0]])
    np.testing.assert_array_equal(skydips[1].signal, second[1][:, [2, 0]])


def test_read_skydip_single_file(tmp_path):
    el, signal = _skydip_data()
    path = (
        tmp_path / "skydip.dat"
    )  # the suffix of a file named explicitly is not checked
    write_skydip(path, el, signal, DETECTOR_IDS)

    (skydip,) = read_skydip(path, DETECTOR_IDS)

    np.testing.assert_array_equal(skydip.signal, signal)


def test_read_skydip_keeps_only_valid_samples(tmp_path):
    # Check that the samples with flag_track unset or an elevation outside
    # (0, pi/2] are dropped.
    el, signal = _skydip_data()
    el[0] = np.nan
    el[1] = -0.1
    el[2] = 2.0
    flags = np.ones(N_SAMPLES, dtype=np.int16)
    flags[3] = 0
    write_skydip(tmp_path / "a.fits", el, signal, DETECTOR_IDS, flags)

    (skydip,) = read_skydip(tmp_path, DETECTOR_IDS)

    np.testing.assert_array_equal(skydip.el, el[4:])
    np.testing.assert_array_equal(skydip.signal, signal[4:])


def test_read_skydip_skips_unusable_files(tmp_path):
    # Check that an empty file and a file with too few valid samples are skipped,
    # with a warning that names them.
    write_skydip(tmp_path / "a.fits", *_skydip_data(), DETECTOR_IDS)
    write_skydip(tmp_path / "b.fits", *_skydip_data(n_samples=10), DETECTOR_IDS)
    (tmp_path / "c.fits").touch()

    with pytest.warns(
        RuntimeWarning, match=r"2 of 3 skydip files skipped: b.fits, c.fits"
    ):
        skydips = read_skydip(tmp_path, DETECTOR_IDS)

    assert len(skydips) == 1


def test_read_skydip_no_usable_file_raises(tmp_path):
    (tmp_path / "a.fits").touch()

    with (
        pytest.warns(RuntimeWarning, match=r"1 of 1 skydip files skipped"),
        pytest.raises(FileNotFoundError, match=r"No usable skydip file"),
    ):
        read_skydip(tmp_path, DETECTOR_IDS)


def test_read_skydip_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"No usable skydip file"):
        read_skydip(tmp_path, DETECTOR_IDS)


def test_read_skydip_missing_path_raises(tmp_path):
    with pytest.raises(NotADirectoryError, match=r"neither a directory nor a file"):
        read_skydip(tmp_path / "missing", DETECTOR_IDS)
