import numpy as np
import pytest
from pydantic import ValidationError

from tests.common import ARCMIN, ARCSEC, DEC0, PIXEL_SIZE, RA0, make_scan
from tolomeo.coords._transforms import radec2latlon
from tolomeo.map import MapGeometry
from tolomeo.processing import EdgeMask, RadiusMask, SigmaClipMask

GEOMETRY = MapGeometry(nx=60, ny=40, pixel_size=PIXEL_SIZE, center=(RA0, DEC0))


def _distance(scan, center=(RA0, DEC0)):
    """Distance of every detector from ``center``, ``(n_samples, n_det)``."""
    coords = scan.coordinates
    lat, lon = radec2latlon(
        coords.ra,
        coords.dec,
        center[0],
        center[1],
        "sin",
        coords.par_angle,
        scan.offsets["lon"],
        scan.offsets["lat"],
        frame="radec",
    )
    return np.hypot(lat, lon).T


# -----------------------------------------------------------------------------
# RadiusMask
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [0.5, "30 arcsec", "0.5 arcmin"])
def test_radius_mask_radius_in_arcmin(radius):
    assert np.isclose(RadiusMask(radius=radius).radius, 0.5)


def test_radius_mask_build():
    # Check that the samples closer than the radius to the centre are left out, for
    # every detector separately.
    scan = make_scan()
    mask = RadiusMask(radius=0.5)

    background = mask.build(scan, GEOMETRY)

    assert background.shape == scan.tod.signal.shape
    np.testing.assert_array_equal(background, _distance(scan) >= 0.5 * ARCMIN)


def test_radius_mask_explicit_center():
    # Check that a centre given in degrees moves the masked region.
    scan = make_scan()
    center = (RA0 + 40 * ARCSEC / np.cos(DEC0), DEC0)
    mask = RadiusMask(radius=0.5, center=tuple(np.rad2deg(center)))

    background = mask.build(scan, GEOMETRY)

    np.testing.assert_array_equal(background, _distance(scan, center) >= 0.5 * ARCMIN)


def test_radius_mask_needs_geometry():
    with pytest.raises(ValueError, match=r"needs `geometry`"):
        RadiusMask(radius=0.5).build(make_scan(), None)


def test_radius_mask_nan_pointing_is_not_background():
    scan = make_scan()
    scan.coordinates.ra[:5] = np.nan

    background = RadiusMask(radius=0.5).build(scan, GEOMETRY)

    assert not background[:5].any()


def test_radius_mask_invalid_radius_raises():
    with pytest.raises(ValidationError):
        RadiusMask(radius=0)


# -----------------------------------------------------------------------------
# SigmaClipMask
# -----------------------------------------------------------------------------


def test_sigma_clip_mask_finds_a_spike():
    # Check that a sample far out of the noise is left out, only in its detector.
    scan = make_scan(noise_rms=1.0)
    scan.tod.signal[100, 1] = 100.0

    background = SigmaClipMask(sigma=6).build(scan)

    assert background.shape == scan.tod.signal.shape
    assert not background[100, 1]
    assert background[100, [0, 2, 3]].all()


# The flagged samples are passed to astropy as NaN, and astropy warns about them.
@pytest.mark.filterwarnings("ignore:Input data contains invalid values")
def test_sigma_clip_mask_flagged_samples_are_not_background():
    scan = make_scan()
    scan.tod.flags[10:20] = False

    background = SigmaClipMask().build(scan)

    assert not background[10:20].any()


# -----------------------------------------------------------------------------
# EdgeMask
# -----------------------------------------------------------------------------


def test_edge_mask_keeps_the_ends_of_every_subscan():
    scan = make_scan(n_subscans=2, subscan_len=100)

    background = EdgeMask(fraction=0.1).build(scan)

    expected = np.zeros(100, dtype=bool)
    expected[:10] = expected[-10:] = True
    np.testing.assert_array_equal(background, np.tile(expected, 2))


def test_edge_mask_flagged_samples_are_not_background():
    scan = make_scan(n_subscans=2, subscan_len=100)
    scan.tod.flags[0] = False

    background = EdgeMask(fraction=0.1).build(scan)

    assert not background[0]


@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.7])
def test_edge_mask_invalid_fraction_raises(fraction):
    with pytest.raises(ValidationError):
        EdgeMask(fraction=fraction)
