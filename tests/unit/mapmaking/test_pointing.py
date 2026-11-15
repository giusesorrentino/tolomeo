import numpy as np
import pytest

from tests.common import ARCSEC, DEC0, OFFSETS_LAT, OFFSETS_LON, RA0, make_scan
from tolomeo.coords import Coordinates
from tolomeo.map import MapGeometry
from tolomeo.mapmaking._pointing import _pointing_matrix

rng = np.random.default_rng(13)

NX, NY = 30, 20
PIXEL_SIZE = 10.0  # arcsec
NO_OFFSETS = {"lon": np.zeros(1), "lat": np.zeros(1)}


def _geometry(frame="radec", projection="sin"):
    return MapGeometry(
        nx=NX,
        ny=NY,
        pixel_size=PIXEL_SIZE,
        center=(RA0, DEC0),
        projection=projection,
        frame=frame,
    )


def _coordinates(ra, dec, par_angle=0.0):
    n = np.size(ra)
    return Coordinates(
        ra=np.asarray(ra, dtype=float),
        dec=np.asarray(dec, dtype=float),
        par_angle=np.full(n, par_angle),
        az=np.zeros(n),
        el=np.full(n, 0.8),
    )


def _random_coordinates(n=2000, half_size=180 * ARCSEC):
    """Boresight positions spread over a square of side ``2 * half_size``."""
    ra = RA0 + rng.uniform(-half_size, half_size, n) / np.cos(DEC0)
    dec = DEC0 + rng.uniform(-half_size, half_size, n)
    return _coordinates(ra, dec)


# -----------------------------------------------------------------------------
# _pointing_matrix
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("projection", ["sin", "tan", "zea"])
def test_pointing_matrix_matches_the_wcs(projection):
    # Check that the pixel of every sample is the pixel of the WCS of the map that
    # contains it: astropy gives 0-based pixel coordinates, with the centre of a pixel
    # at an integer, so the pixel is the rounded coordinate.
    geometry = _geometry(projection=projection)
    coords = _random_coordinates()

    pixels, valid = _pointing_matrix(coords, NO_OFFSETS, geometry)

    px, py = geometry.to_wcs().world_to_pixel_values(
        np.rad2deg(coords.ra), np.rad2deg(coords.dec)
    )
    ix, iy = np.floor(px + 0.5).astype(int), np.floor(py + 0.5).astype(int)
    inside = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)

    assert pixels.shape == valid.shape == (coords.ra.size, 1)
    assert inside.any() and not inside.all()
    np.testing.assert_array_equal(valid[:, 0], inside)
    np.testing.assert_array_equal(pixels[inside, 0], iy[inside] * NX + ix[inside])


def test_pointing_matrix_outside_is_invalid_with_pixel_zero():
    coords = _coordinates([RA0 + 0.01], [DEC0])

    pixels, valid = _pointing_matrix(coords, NO_OFFSETS, _geometry())

    assert not valid[0, 0]
    assert pixels[0, 0] == 0


def test_pointing_matrix_nan_is_invalid():
    coords = _coordinates([RA0, np.nan, RA0], [DEC0, DEC0, np.nan])

    pixels, valid = _pointing_matrix(coords, NO_OFFSETS, _geometry())

    np.testing.assert_array_equal(valid[:, 0], [True, False, False])
    np.testing.assert_array_equal(pixels[1:, 0], 0)


def test_pointing_matrix_center_pixel():
    # Check that the centre of the map is the corner of the four central pixels, and
    # that a point just up and to the east of it is in the pixel up and to the left
    # (east is on the left of the image).
    eps = 0.1 * ARCSEC
    coords = _coordinates([RA0 + eps / np.cos(DEC0)], [DEC0 + eps])

    pixels, _ = _pointing_matrix(coords, NO_OFFSETS, _geometry())

    assert pixels[0, 0] == (NY // 2) * NX + (NX // 2 - 1)


def test_pointing_matrix_azel_equals_radec_without_rotation():
    # With par_angle = 0 the two frames place every detector at the same pixel.
    scan = make_scan()

    radec = _pointing_matrix(scan.coordinates, scan.offsets, _geometry("radec"))
    azel = _pointing_matrix(scan.coordinates, scan.offsets, _geometry("azel"))

    np.testing.assert_array_equal(radec[0], azel[0])
    np.testing.assert_array_equal(radec[1], azel[1])


@pytest.mark.parametrize("par_angle", [0.3, 2.0, -1.0])
def test_pointing_matrix_parallactic_angle(par_angle):
    # With the boresight at the centre, a detector of offset (lon, lat) is at the
    # offset rotated by par_angle in radec (x growing toward the west), and at the
    # offset as it is in azel.
    offsets = {"lon": OFFSETS_LON.copy(), "lat": OFFSETS_LAT.copy()}
    coords = _coordinates([RA0], [DEC0], par_angle)
    pixel_size = PIXEL_SIZE * ARCSEC

    def expected_pixels(x, y):
        ix = np.floor(x / pixel_size + NX / 2).astype(int)
        iy = np.floor(y / pixel_size + NY / 2).astype(int)
        return iy * NX + ix

    c, s = np.cos(par_angle), np.sin(par_angle)
    x_rot = OFFSETS_LON * c - OFFSETS_LAT * s
    y_rot = OFFSETS_LON * s + OFFSETS_LAT * c

    radec, _ = _pointing_matrix(coords, offsets, _geometry("radec"))
    azel, _ = _pointing_matrix(coords, offsets, _geometry("azel"))

    np.testing.assert_array_equal(radec[0], expected_pixels(x_rot, y_rot))
    np.testing.assert_array_equal(azel[0], expected_pixels(OFFSETS_LON, OFFSETS_LAT))


@pytest.mark.parametrize("frame", ["radec", "azel"])
def test_pointing_matrix_does_not_depend_on_workers(frame):
    scan = make_scan(par_angle=0.7)
    geometry = _geometry(frame)

    one = _pointing_matrix(scan.coordinates, scan.offsets, geometry, workers=1)
    many = _pointing_matrix(scan.coordinates, scan.offsets, geometry, workers=3)

    np.testing.assert_array_equal(one[0], many[0])
    np.testing.assert_array_equal(one[1], many[1])
