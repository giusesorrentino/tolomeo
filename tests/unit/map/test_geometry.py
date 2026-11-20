import astropy.units as u
import numpy as np
import pytest
from pydantic import ValidationError

from tests.common import DEC0, PIXEL_SIZE, RA0, make_scan
from tolomeo.map import MapGeometry
from tolomeo.mapmaking._pointing import _pointing_matrix


def _geometry(**kwargs):
    """A small geometry centred on the synthetic field."""
    return MapGeometry(
        **{"nx": 20, "ny": 10, "pixel_size": PIXEL_SIZE, "center": (RA0, DEC0)} | kwargs
    )


# -----------------------------------------------------------------------------
# MapGeometry
# -----------------------------------------------------------------------------


def test_n_pixels():
    assert _geometry().n_pixels == 200


@pytest.mark.parametrize("pixel_size", ["0.1 arcmin", 6 * u.arcsec, 6.0])
def test_pixel_size_in_arcsec(pixel_size):
    assert np.isclose(_geometry(pixel_size=pixel_size).pixel_size, 6.0)


def test_center_with_unit():
    # Check that the centre accepts quantities, converted to radians.
    geometry = _geometry(center=(52 * u.deg, "41.5 deg"))

    np.testing.assert_allclose(geometry.center, (RA0, DEC0))


@pytest.mark.parametrize("field, value", [("nx", 0), ("ny", -1), ("pixel_size", 0.0)])
def test_invalid_sizes_raise(field, value):
    with pytest.raises(ValidationError):
        _geometry(**{field: value})


def test_unknown_key_raises():
    with pytest.raises(ValidationError):
        _geometry(npix=10)


def test_geometry_is_frozen():
    with pytest.raises(ValidationError):
        _geometry().nx = 5


# -----------------------------------------------------------------------------
# to_wcs
# -----------------------------------------------------------------------------


def test_to_wcs_radec():
    # Check the celestial WCS: RA then Dec in FK5 J2000, CDELT1 negative (east on the
    # left), centred on the projection centre.
    wcs = _geometry().to_wcs()

    assert list(wcs.wcs.ctype) == ["RA---SIN", "DEC--SIN"]
    np.testing.assert_allclose(wcs.wcs.cdelt, [-PIXEL_SIZE / 3600, PIXEL_SIZE / 3600])
    np.testing.assert_allclose(wcs.wcs.crval, np.rad2deg([RA0, DEC0]))
    np.testing.assert_allclose(wcs.wcs.crpix, [20 / 2 + 0.5, 10 / 2 + 0.5])

    header = wcs.to_header()
    assert header["RADESYS"] == "FK5"
    assert header["EQUINOX"] == 2000.0


@pytest.mark.parametrize("projection", ["sin", "tan", "zea"])
def test_to_wcs_projection(projection):
    wcs = _geometry(projection=projection).to_wcs()

    assert wcs.wcs.ctype[0] == f"RA---{projection.upper()}"


def test_to_wcs_azel():
    # Check the linear WCS of an azel map: offsets from the centre, x growing with
    # the azimuth.
    wcs = _geometry(frame="azel").to_wcs()

    assert list(wcs.wcs.ctype) == ["AZOFFSET", "ELOFFSET"]
    np.testing.assert_allclose(wcs.wcs.cdelt, [PIXEL_SIZE / 3600, PIXEL_SIZE / 3600])
    np.testing.assert_allclose(wcs.wcs.crval, [0.0, 0.0])

    # the axes are not celestial: no equatorial frame
    header = wcs.to_header()
    assert "RADESYS" not in header
    assert "EQUINOX" not in header


# -----------------------------------------------------------------------------
# from_beam
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("beam_fwhm", [12.0, "12 arcsec", 0.2 * u.arcmin])
def test_from_beam_pixel_size(beam_fwhm):
    # Check that the pixel is the beam FWHM divided by pixels_per_beam.
    geometry = MapGeometry.from_beam(beam_fwhm, (RA0, DEC0), pixels_per_beam=4)

    assert np.isclose(geometry.pixel_size, 3.0)


# -----------------------------------------------------------------------------
# from_scan
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("frame", ["radec", "azel"])
@pytest.mark.parametrize("par_angle", [0.0, 0.6])
def test_from_scan_covers_every_detector(frame, par_angle):
    # Check that every sample of every detector falls inside the map, and inside the
    # margin of empty pixels along both axes: the pixels added on each side with
    # respect to the map with no margin.
    scan = make_scan(par_angle=par_angle)
    bare = MapGeometry.from_scan(scan, pixel_size=PIXEL_SIZE, frame=frame, margin=0)
    geometry = MapGeometry.from_scan(
        scan, pixel_size=PIXEL_SIZE, frame=frame, margin=0.2
    )
    border_x = (geometry.nx - bare.nx) // 2
    border_y = (geometry.ny - bare.ny) // 2

    pixels, valid = _pointing_matrix(scan.coordinates, scan.offsets, geometry)
    ix, iy = pixels % geometry.nx, pixels // geometry.nx

    assert valid.all()
    assert border_x > 0 and border_y > 0
    assert border_x <= ix.min() and ix.max() <= geometry.nx - border_x - 1
    assert border_y <= iy.min() and iy.max() <= geometry.ny - border_y - 1


def test_from_scan_default_center_is_the_source():
    scan = make_scan()

    geometry = MapGeometry.from_scan(scan, pixel_size=PIXEL_SIZE)

    np.testing.assert_array_equal(geometry.center, scan.source)


@pytest.mark.parametrize("margin", [0.05, 0.1, 0.5, 1.0])
def test_from_scan_margin_adds_a_fraction_of_the_half_size(margin):
    # Check that a fraction `margin` of the half-size, rounded up to whole pixels, is
    # added on each side, so that nx and ny stay even.
    scan = make_scan()

    small = MapGeometry.from_scan(scan, pixel_size=PIXEL_SIZE, margin=0)
    large = MapGeometry.from_scan(scan, pixel_size=PIXEL_SIZE, margin=margin)

    for n_small, n_large in ((small.nx, large.nx), (small.ny, large.ny)):
        half = n_small // 2
        assert n_large == 2 * (half + int(np.ceil(margin * half)))
        assert n_large % 2 == 0


@pytest.mark.parametrize("margin", [-0.1, 1.5])
def test_from_scan_margin_out_of_range_raises(margin):
    with pytest.raises(ValueError, match=r"`margin` must be between 0 and 1"):
        MapGeometry.from_scan(make_scan(), pixel_size=PIXEL_SIZE, margin=margin)


def test_from_scan_unknown_frame_raises():
    with pytest.raises(ValueError, match=r"Unknown frame"):
        MapGeometry.from_scan(make_scan(), pixel_size=PIXEL_SIZE, frame="galactic")
