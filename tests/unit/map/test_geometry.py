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
    # Check the celestial WCS: RA then Dec, CDELT1 negative (east on the left),
    # centred on the projection centre.
    wcs = _geometry().to_wcs()

    assert list(wcs.wcs.ctype) == ["RA---SIN", "DEC--SIN"]
    np.testing.assert_allclose(wcs.wcs.cdelt, [-PIXEL_SIZE / 3600, PIXEL_SIZE / 3600])
    np.testing.assert_allclose(wcs.wcs.crval, np.rad2deg([RA0, DEC0]))
    np.testing.assert_allclose(wcs.wcs.crpix, [20 / 2 + 0.5, 10 / 2 + 0.5])


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
    # margin of empty pixels along both axes.
    margin = 2
    scan = make_scan(par_angle=par_angle)
    geometry = MapGeometry.from_scan(
        scan, pixel_size=PIXEL_SIZE, frame=frame, margin=margin
    )

    pixels, valid = _pointing_matrix(scan.coordinates, scan.offsets, geometry)
    ix, iy = pixels % geometry.nx, pixels // geometry.nx

    assert valid.all()
    assert margin <= ix.min() and ix.max() <= geometry.nx - margin - 1
    assert margin <= iy.min() and iy.max() <= geometry.ny - margin - 1


def test_from_scan_default_center_is_the_scan_center():
    scan = make_scan()

    geometry = MapGeometry.from_scan(scan, pixel_size=PIXEL_SIZE)

    np.testing.assert_allclose(geometry.center, scan.center)


def test_from_scan_margin_adds_pixels():
    scan = make_scan()

    small = MapGeometry.from_scan(scan, pixel_size=PIXEL_SIZE, margin=0)
    large = MapGeometry.from_scan(scan, pixel_size=PIXEL_SIZE, margin=5)

    assert large.nx - small.nx == 10
    assert large.ny - small.ny == 10


def test_from_scan_unknown_frame_raises():
    with pytest.raises(ValueError, match=r"Unknown frame"):
        MapGeometry.from_scan(make_scan(), pixel_size=PIXEL_SIZE, frame="galactic")
