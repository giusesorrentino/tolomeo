import numpy as np
import pytest
from astropy.io import fits

from tests.common import DEC0, DETECTOR_IDS, RA0, make_scan
from tolomeo._lib.roma import write_roma_fits
from tolomeo.map import MapGeometry
from tolomeo.mapmaking import estimate_inverse_noise
from tolomeo.mapmaking._pointing import _pointing_matrix

NPERSEG = 128  # a multiple of 4: 65 bins, an odd number as ROMA wants

# smaller than the scan, so that some samples fall outside the map; nx != ny
GEOMETRY = MapGeometry(nx=12, ny=10, pixel_size=10.0, center=(RA0, DEC0))


def _scan():
    """Scan of ``make_scan`` with some flagged samples and the TOD in kelvin."""
    scan = make_scan()
    flags = scan.tod.flags.copy()
    flags[100:150] = False
    tod = scan.tod.model_copy(update={"flags": flags, "unit": "K"})
    return scan.model_copy(update={"tod": tod})


def _column(path):
    """The table of a file for ROMA, its header, and the primary HDU."""
    with fits.open(path) as hdul:
        assert len(hdul) == 2
        return hdul[1].data.copy(), hdul[1].header.copy(), hdul[0].data


# -----------------------------------------------------------------------------
# write_roma_fits
# -----------------------------------------------------------------------------


def test_write_roma_fits_files_of_every_detector(tmp_path):
    # Check every file of every detector: one column with the data, the inverted
    # flags (also outside the map), and the keywords of the header.
    scan = _scan()
    inv_noise = estimate_inverse_noise(scan, nperseg=NPERSEG, workers=1)
    pixels, inside = _pointing_matrix(scan.coordinates, scan.offsets, GEOMETRY)
    assert not inside.all()  # some samples are outside the map

    write_roma_fits(scan, GEOMETRY, inv_noise, tmp_path, workers=1)

    for j, channel in enumerate(DETECTOR_IDS):
        n = j + 1
        tod, tod_header, primary = _column(tmp_path / f"tod_{n}.fits")
        pix, pix_header, _ = _column(tmp_path / f"pix_{n}.fits")
        flags, _, _ = _column(tmp_path / f"flags_{n}.fits")
        invps, _, _ = _column(tmp_path / f"invps_{n}_{n}.fits")

        assert primary is None  # empty primary HDU
        for table in (tod, pix, flags, invps):
            assert len(table.columns) == 1
        np.testing.assert_array_equal(tod["TOD"], scan.tod.signal[:, j])
        np.testing.assert_array_equal(pix["PIXEL"], pixels[:, j])
        expected = np.where(scan.tod.flags & inside[:, j], 0, 1)
        np.testing.assert_array_equal(flags["FLAG"], expected)
        np.testing.assert_array_equal(invps["INVPS"], inv_noise.inv_noiseps[:, j])

        assert tod_header["CHANNEL"] == channel
        assert tod_header["FSAMP"] == pytest.approx(scan.tod.sample_rate)
        assert tod_header["BUNIT"] == "K"
        assert pix_header["MAPNX"] == 12
        assert pix_header["MAPNY"] == 10
        assert pix_header["CTYPE1"].startswith("RA---")

    assert not (tmp_path / f"tod_{len(DETECTOR_IDS) + 1}.fits").exists()


def test_write_roma_fits_even_number_of_bins_raises(tmp_path):
    scan = _scan()
    inv_noise = estimate_inverse_noise(scan, nperseg=130, workers=1)  # 66 bins

    with pytest.raises(ValueError, match="odd number"):
        write_roma_fits(scan, GEOMETRY, inv_noise, tmp_path, workers=1)


def test_write_roma_fits_signal_with_nan_raises(tmp_path):
    scan = _scan()
    inv_noise = estimate_inverse_noise(scan, nperseg=NPERSEG, workers=1)
    signal = scan.tod.signal.copy()
    signal[5, 1] = np.nan
    tod = scan.tod.model_copy(update={"signal": signal})

    with pytest.raises(ValueError, match="NaN"):
        write_roma_fits(
            scan.model_copy(update={"tod": tod}), GEOMETRY, inv_noise, tmp_path
        )


def test_write_roma_fits_noise_of_another_scan_raises(tmp_path):
    inv_noise = estimate_inverse_noise(make_scan(n_det=3), nperseg=NPERSEG, workers=1)

    with pytest.raises(ValueError, match="detectors"):
        write_roma_fits(_scan(), GEOMETRY, inv_noise, tmp_path)
