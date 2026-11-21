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


def test_select_keeps_the_position_of_the_source():
    coords = _coordinates([1.0, 2.0, 3.0], [5.0, 6.0, 7.0]).model_copy(
        update={
            "source_ra": np.array([0.1, 0.2, 0.3]),
            "source_dec": np.ones(3),
            "source_name": "MARS",
        }
    )

    selected = coords.select(np.array([True, False, True]))

    np.testing.assert_array_equal(selected.source_ra, [0.1, 0.3])
    np.testing.assert_array_equal(selected.source_dec, [1.0, 1.0])
    assert selected.source_name == "MARS"


def test_select_without_the_position_of_the_source():
    selected = _coordinates([1.0, 2.0], [5.0, 6.0]).select(np.array([True, False]))

    assert selected.source_ra is None
    assert selected.source_dec is None
    assert selected.source_name is None


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
