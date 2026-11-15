import numpy as np
import pytest

from tolomeo.coords import Coordinates


def _coordinates(ra_deg, dec_deg):
    """Coordinates with the given RA and Dec (degrees) and zeros elsewhere."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    zeros = np.zeros_like(ra)
    return Coordinates(ra=ra, dec=dec, par_angle=zeros, az=zeros, el=zeros)


# -----------------------------------------------------------------------------
# center
# -----------------------------------------------------------------------------


def test_center_is_the_mean():
    ra, dec = _coordinates([10.0, 20.0, 30.0], [-1.0, 0.0, 4.0]).center

    assert np.isclose(np.rad2deg(ra), 20.0)
    assert np.isclose(np.rad2deg(dec), 1.0)


def test_center_ra_is_a_circular_mean_across_zero():
    # Check that the mean of 359 deg and 3 deg is 1 deg, not 181 deg.
    ra, _ = _coordinates([359.0, 3.0], [0.0, 0.0]).center

    assert np.isclose(np.rad2deg(ra), 1.0)


@pytest.mark.parametrize("ra_deg", [5.0, 179.0, 181.0, 275.0, 359.0])
def test_center_ra_is_in_zero_two_pi(ra_deg):
    # Check that the RA of the centre is always in [0, 2 pi), also above 180 deg.
    ra, _ = _coordinates([ra_deg - 0.5, ra_deg + 0.5], [0.0, 0.0]).center

    assert 0.0 <= ra < 2 * np.pi
    assert np.isclose(np.rad2deg(ra), ra_deg)


def test_center_ignores_nan():
    ra, dec = _coordinates([10.0, np.nan, 30.0], [0.0, 5.0, np.nan]).center

    # only the first sample has both coordinates finite
    assert np.isclose(np.rad2deg(ra), 10.0)
    assert np.isclose(np.rad2deg(dec), 0.0)


def test_center_all_nan_raises():
    with pytest.raises(ValueError, match=r"No finite boresight"):
        _ = _coordinates([np.nan, np.nan], [0.0, 0.0]).center


# -----------------------------------------------------------------------------
# select
# -----------------------------------------------------------------------------


def test_select_keeps_the_masked_samples():
    coords = _coordinates([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
    mask = np.array([True, False, True, False])

    selected = coords.select(mask)

    np.testing.assert_allclose(np.rad2deg(selected.ra), [1.0, 3.0])
    np.testing.assert_allclose(np.rad2deg(selected.dec), [5.0, 7.0])
    for name in ("par_angle", "az", "el"):
        assert getattr(selected, name).shape == (2,)


def test_select_does_not_modify_the_original():
    coords = _coordinates([1.0, 2.0], [5.0, 6.0])

    selected = coords.select(np.array([True, False]))
    selected.ra[0] = 0.0

    assert np.isclose(np.rad2deg(coords.ra[0]), 1.0)


def test_select_non_boolean_mask_raises():
    with pytest.raises(TypeError):
        _coordinates([1.0, 2.0], [5.0, 6.0]).select(np.array([1, 0]))


def test_select_wrong_shape_raises():
    with pytest.raises(ValueError):
        _coordinates([1.0, 2.0], [5.0, 6.0]).select(np.array([True]))
