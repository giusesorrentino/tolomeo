import numpy as np
import pytest
from astropy.wcs import WCS

from tests.common import ARCSEC, DEC0, RA0
from tolomeo.coords._transforms import (
    radec2latlon,
    radec2xy,
    rot,
    shift_boresight,
    xy2latlon,
)

PROJECTIONS = ["sin", "tan", "zea"]

# random points within 0.2 deg of the centre
_rng = np.random.default_rng(1)
RA = RA0 + np.deg2rad(_rng.uniform(-0.2, 0.2, 50))
DEC = DEC0 + np.deg2rad(_rng.uniform(-0.2, 0.2, 50))


# -----------------------------------------------------------------------------
# rot
# -----------------------------------------------------------------------------


def test_rot_quarter_turn():
    # Check that a rotation by pi / 2 is counter-clockwise.
    x, y = rot(1.0, 0.0, np.pi / 2)

    assert np.isclose(x, 0.0)
    assert np.isclose(y, 1.0)


def test_rot_inverse():
    # Check that rotating by theta and then by -theta gives back the input.
    x = np.array([1.0, -2.0, 0.5])
    y = np.array([0.0, 3.0, -1.5])

    x_rot, y_rot = rot(x, y, 0.7)
    x_back, y_back = rot(x_rot, y_rot, -0.7)

    np.testing.assert_allclose(x_back, x)
    np.testing.assert_allclose(y_back, y)


def test_rot_broadcasts():
    # Check that per-detector offsets and per-sample angles broadcast to a 2-D array.
    x_off = np.array([1.0, 2.0, 3.0])[:, None]
    theta = np.linspace(0, 1, 5)[None, :]

    x_rot, y_rot = rot(x_off, 0.0, theta)

    assert x_rot.shape == (3, 5)
    assert y_rot.shape == (3, 5)


# -----------------------------------------------------------------------------
# shift_boresight
# -----------------------------------------------------------------------------


def test_shift_boresight():
    # Check that an offset on the sky changes the RA by dra / cos(dec).
    ra = np.array([RA0])
    dec = np.array([DEC0])

    ra_new, dec_new = shift_boresight(ra, dec, 10 * ARCSEC, 5 * ARCSEC)

    np.testing.assert_allclose(ra_new, RA0 + 10 * ARCSEC / np.cos(DEC0))
    np.testing.assert_allclose(dec_new, DEC0 + 5 * ARCSEC)


# -----------------------------------------------------------------------------
# radec2xy
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_radec2xy_centre_is_origin(projection):
    x, y = radec2xy(np.array([RA0]), np.array([DEC0]), RA0, DEC0, projection=projection)

    np.testing.assert_allclose([x[0], y[0]], [0.0, 0.0], atol=1e-15)


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_radec2xy_matches_astropy_wcs(projection):
    # Check against the intermediate world coordinates of astropy: with
    # CDELT1 = -1 deg the pixel x grows toward the west, while x of radec2xy grows
    # toward the east, so x = -pixel_x (in radians).
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = [f"RA---{projection.upper()}", f"DEC--{projection.upper()}"]
    wcs.wcs.crval = [np.rad2deg(RA0), np.rad2deg(DEC0)]
    wcs.wcs.crpix = [1, 1]
    wcs.wcs.cdelt = [-1.0, 1.0]

    px, py = wcs.wcs_world2pix(np.rad2deg(RA), np.rad2deg(DEC), 0)
    x, y = radec2xy(RA, DEC, RA0, DEC0, projection=projection)

    np.testing.assert_allclose(x, np.deg2rad(-px), atol=1e-14)
    np.testing.assert_allclose(y, np.deg2rad(py), atol=1e-14)


def test_radec2xy_x_grows_toward_the_east():
    # Check that a larger RA gives a larger x, and a larger Dec a larger y.
    x, y = radec2xy(
        np.array([RA0 + ARCSEC, RA0]),
        np.array([DEC0, DEC0 + ARCSEC]),
        RA0,
        DEC0,
        projection="sin",
    )

    assert x[0] > 0
    assert y[1] > 0


def test_radec2xy_projection_is_case_insensitive():
    x_lower, _ = radec2xy(RA, DEC, RA0, DEC0, projection="sin")
    x_upper, _ = radec2xy(RA, DEC, RA0, DEC0, projection="SIN")

    np.testing.assert_array_equal(x_lower, x_upper)


@pytest.mark.parametrize("projection", ["sin", "tan"])
def test_radec2xy_far_hemisphere_is_nan(projection):
    # Check that SIN and TAN are not defined on the hemisphere opposite the centre.
    x, y = radec2xy(
        np.array([RA0 + np.pi]), np.array([0.0]), RA0, 0.0, projection=projection
    )

    assert np.isnan(x[0])
    assert np.isnan(y[0])


def test_radec2xy_zea_far_hemisphere_is_defined():
    # Check that ZEA is defined everywhere except the antipode of the centre.
    x, y = radec2xy(np.array([RA0 + 2.0]), np.array([0.0]), RA0, 0.0, projection="zea")

    assert np.isfinite(x[0])
    assert np.isfinite(y[0])


def test_radec2xy_unknown_projection_raises():
    with pytest.raises(ValueError, match=r"Unknown projection"):
        radec2xy(RA, DEC, RA0, DEC0, projection="car")


# -----------------------------------------------------------------------------
# xy2latlon
# -----------------------------------------------------------------------------

X = np.array([0.0, 1.0, -2.0]) * ARCSEC
Y = np.array([0.5, 0.0, 3.0]) * ARCSEC
X_OFF = np.array([10.0, -5.0]) * ARCSEC
Y_OFF = np.array([0.0, 7.0]) * ARCSEC


@pytest.mark.parametrize("frame", ["radec", "azel"])
def test_xy2latlon_zero_offsets_is_the_boresight(frame):
    # Check that a detector on the boresight is at the boresight, with no rotation.
    lat, lon = xy2latlon(X, Y, np.zeros(3), np.zeros(1), np.zeros(1), RA0, DEC0, frame)

    assert lat.shape == (1, 3)
    np.testing.assert_allclose(lat[0], Y)
    np.testing.assert_allclose(lon[0], X)


def test_xy2latlon_radec_rotates_the_offsets():
    # Check that in "radec" the offsets turn with the parallactic angle.
    par = np.full(3, 0.4)

    lat, lon = xy2latlon(X, Y, par, X_OFF, Y_OFF, RA0, DEC0, "radec")

    x_rot, y_rot = rot(X_OFF[:, None], Y_OFF[:, None], par[None, :])
    np.testing.assert_allclose(lat, Y[None, :] + y_rot)
    np.testing.assert_allclose(lon, X[None, :] - x_rot)


def test_xy2latlon_azel_rotates_the_boresight():
    # Check that in "azel" the boresight turns and the offsets are added as they are.
    par = np.full(3, 0.4)

    lat, lon = xy2latlon(X, Y, par, X_OFF, Y_OFF, RA0, DEC0, "azel")

    x_rot, y_rot = rot(X, Y, par)
    np.testing.assert_allclose(lat, y_rot[None, :] + Y_OFF[:, None])
    np.testing.assert_allclose(lon, x_rot[None, :] - X_OFF[:, None])


def test_xy2latlon_unknown_frame_raises():
    with pytest.raises(NotImplementedError, match=r"not available"):
        xy2latlon(X, Y, np.zeros(3), X_OFF, Y_OFF, RA0, DEC0, "galactic")


# -----------------------------------------------------------------------------
# radec2latlon
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("frame", ["radec", "azel"])
def test_radec2latlon_is_radec2xy_then_xy2latlon(frame):
    par = np.linspace(0, 0.5, RA.size)

    lat, lon = radec2latlon(RA, DEC, RA0, DEC0, "sin", par, X_OFF, Y_OFF, frame)

    x, y = radec2xy(RA, DEC, RA0, DEC0, projection="sin")
    lat_ref, lon_ref = xy2latlon(x, y, par, X_OFF, Y_OFF, RA0, DEC0, frame)
    np.testing.assert_allclose(lat, lat_ref)
    np.testing.assert_allclose(lon, lon_ref)
