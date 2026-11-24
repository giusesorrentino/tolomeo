import numpy as np
import pytest
from astropy.io import fits

from tests.common import DEC0, DETECTOR_IDS, RA0, make_scan
from tolomeo._lib.roma import _detector_radec, write_roma_fits
from tolomeo.coords import Coordinates
from tolomeo.coords._transforms import radec2xy
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


# -----------------------------------------------------------------------------
# write_roma_fits with pointing="radec"
# -----------------------------------------------------------------------------


def test_write_roma_fits_radec_files_of_every_detector(tmp_path):
    # Check the files of pointing="radec": RA and DEC in degrees instead of the
    # pixels, with no grid, the flags of the scan alone, and the same other files.
    scan = _scan()
    inv_noise = estimate_inverse_noise(scan, nperseg=NPERSEG, workers=1)

    write_roma_fits(scan, None, inv_noise, tmp_path, pointing="radec")

    for j, channel in enumerate(DETECTOR_IDS):
        n = j + 1
        radec, header, primary = _column(tmp_path / f"radec_{n}.fits")
        flags, _, _ = _column(tmp_path / f"flags_{n}.fits")
        tod, _, _ = _column(tmp_path / f"tod_{n}.fits")
        invps, _, _ = _column(tmp_path / f"invps_{n}_{n}.fits")

        assert primary is None  # empty primary HDU
        assert radec.columns.names == ["RA", "DEC"]
        assert radec.columns["RA"].unit == radec.columns["DEC"].unit == "deg"
        assert header["CHANNEL"] == channel
        assert header["RADESYS"] == "FK5"
        assert header["EQUINOX"] == 2000.0
        np.testing.assert_array_equal(flags["FLAG"], np.where(scan.tod.flags, 0, 1))
        np.testing.assert_array_equal(tod["TOD"], scan.tod.signal[:, j])
        np.testing.assert_array_equal(invps["INVPS"], inv_noise.inv_noiseps[:, j])
        assert not (tmp_path / f"pix_{n}.fits").exists()


def test_write_roma_fits_radec_of_a_detector_on_the_boresight(tmp_path):
    # Check that the detector with no offset (channel 12) is on the boresight.
    scan = _scan()
    inv_noise = estimate_inverse_noise(scan, nperseg=NPERSEG, workers=1)
    j = DETECTOR_IDS.index(12)
    assert scan.offsets["lon"][j] == scan.offsets["lat"][j] == 0.0

    write_roma_fits(scan, None, inv_noise, tmp_path, pointing="radec")

    radec, _, _ = _column(tmp_path / f"radec_{j + 1}.fits")
    np.testing.assert_allclose(radec["RA"], np.rad2deg(scan.coordinates.ra))
    np.testing.assert_allclose(radec["DEC"], np.rad2deg(scan.coordinates.dec))


def test_write_roma_fits_radec_in_the_pixels_of_tolomeo(tmp_path):
    # Check the signs of the offsets and of the parallactic angle against the
    # pointing of gls_map: the RA and Dec of every sample, put on the grid with its
    # WCS, fall in the pixel of _pointing_matrix. The two add the offsets on planes
    # tangent at different points, the boresight and the centre of the map: here
    # they differ by less than 0.02 arcsec, 1% of a pixel.
    scan = make_scan(par_angle=0.4)
    geometry = MapGeometry(nx=120, ny=100, pixel_size=2.0, center=(RA0, DEC0))
    inv_noise = estimate_inverse_noise(scan, nperseg=NPERSEG, workers=1)
    pixels, inside = _pointing_matrix(scan.coordinates, scan.offsets, geometry)
    assert inside.mean() > 0.9

    write_roma_fits(scan, None, inv_noise, tmp_path, pointing="radec")

    wcs = geometry.to_wcs()
    for j in range(len(DETECTOR_IDS)):
        radec, _, _ = _column(tmp_path / f"radec_{j + 1}.fits")
        x, y = wcs.world_to_pixel_values(radec["RA"], radec["DEC"])
        ok = inside[:, j]
        # 0-based pixel coordinates: pixel i spans [i - 0.5, i + 0.5]
        ix, iy = pixels[ok, j] % geometry.nx, pixels[ok, j] // geometry.nx
        assert np.abs(x[ok] - ix).max() <= 0.51
        assert np.abs(y[ok] - iy).max() <= 0.51


@pytest.mark.parametrize(
    ("ra0", "dec0"), [(1.0, 0.3), (2 * np.pi - 1e-5, -0.5), (3.0, 1.5)]
)
def test_detector_radec_is_the_inverse_gnomonic_projection(ra0, dec0):
    # Check the inverse projection with the forward one of tolomeo: projected on the
    # plane tangent at the boresight, the position of the detector is its offset
    # turned by the parallactic angle, xi = -x_rot to the east and eta = y_rot to the
    # north. Offsets of half a degree, where the small-angle approximation would not
    # do, near RA = 0 (the RA must wrap into [0, 360)) and near the pole.
    par_angle = np.linspace(-1.0, 2.0, 7)
    n = par_angle.size
    coords = Coordinates(
        ra=np.full(n, ra0),
        dec=np.full(n, dec0),
        par_angle=par_angle,
        az=np.zeros(n),
        el=np.full(n, 0.8),
    )
    lon_offset, lat_offset = np.deg2rad(0.5), np.deg2rad(-0.3)

    ra, dec = _detector_radec(coords, lon_offset, lat_offset)

    assert np.all((ra >= 0) & (ra < 360))
    xi, eta = radec2xy(np.deg2rad(ra), np.deg2rad(dec), ra0, dec0, projection="tan")
    x_rot = lon_offset * np.cos(par_angle) - lat_offset * np.sin(par_angle)
    y_rot = lon_offset * np.sin(par_angle) + lat_offset * np.cos(par_angle)
    np.testing.assert_allclose(xi, -x_rot, rtol=0, atol=1e-12)
    np.testing.assert_allclose(eta, y_rot, rtol=0, atol=1e-12)


def test_write_roma_fits_pixels_without_geometry_raises(tmp_path):
    scan = _scan()
    inv_noise = estimate_inverse_noise(scan, nperseg=NPERSEG, workers=1)

    with pytest.raises(ValueError, match="needs `geometry`"):
        write_roma_fits(scan, None, inv_noise, tmp_path)


def test_write_roma_fits_unknown_pointing_raises(tmp_path):
    scan = _scan()
    inv_noise = estimate_inverse_noise(scan, nperseg=NPERSEG, workers=1)

    with pytest.raises(ValueError, match="`pointing` must be"):
        write_roma_fits(scan, GEOMETRY, inv_noise, tmp_path, pointing="azel")
