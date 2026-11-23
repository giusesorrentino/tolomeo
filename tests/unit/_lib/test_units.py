import astropy.units as u
import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from tolomeo._lib.units import (
    Arcmin,
    Arcsec,
    Days,
    Degrees,
    Hz,
    Kelvin,
    _conversion_factor,
    _is_temperature,
    _to_unit,
)


class _Model(BaseModel):
    """Model with one field per implicit unit, to test the pydantic aliases."""

    pixel: Arcsec = 3.0
    radius: Arcmin = 1.0
    angle: Degrees = 0.0
    fc: Hz = 1.0
    spacing: Days = 1.0
    temperature: Kelvin = 1.0


# -----------------------------------------------------------------------------
# _to_unit
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", [4, 4.0, np.float64(4.0)], ids=["int", "float", "numpy"]
)
def test_to_unit_number_is_already_in_unit(value):
    # Check that a plain number is returned as it is, in the implicit unit.
    assert _to_unit(value, u.arcsec) == 4.0


@pytest.mark.parametrize(
    "value, expected",
    [
        ("4", 4.0),
        ("4 arcsec", 4.0),
        ("1 arcmin", 60.0),
        ("0.5 deg", 1800.0),
        ("  2   arcmin ", 120.0),
    ],
)
def test_to_unit_string(value, expected):
    # Check that a string with a number and a unit is converted to the implicit unit.
    assert np.isclose(_to_unit(value, u.arcsec), expected)


def test_to_unit_quantity():
    # Check that an astropy Quantity is converted to the implicit unit.
    assert np.isclose(_to_unit(50 * u.mHz, u.Hz), 0.05)


@pytest.mark.parametrize("value", ["abc", "", "four arcsec"])
def test_to_unit_not_a_number_raises(value):
    with pytest.raises(ValueError, match=r"is not a number"):
        _to_unit(value, u.arcsec)


def test_to_unit_unknown_unit_raises():
    with pytest.raises(ValueError):
        _to_unit("4 parsnip", u.arcsec)


@pytest.mark.parametrize("value", ["3 Hz", 3 * u.Hz])
def test_to_unit_incompatible_unit_raises(value):
    with pytest.raises(ValueError, match=r"cannot be converted"):
        _to_unit(value, u.arcsec)


def test_to_unit_array_quantity_raises():
    with pytest.raises(ValueError, match=r"scalar"):
        _to_unit([1, 2] * u.arcsec, u.arcsec)


@pytest.mark.parametrize("value", [None, [1, 2], {"a": 1}])
def test_to_unit_wrong_type_raises(value):
    with pytest.raises(TypeError, match=r"expected a number"):
        _to_unit(value, u.arcsec)


# -----------------------------------------------------------------------------
# pydantic aliases
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("pixel", "0.1 arcmin", 6.0),
        ("radius", "120 arcsec", 2.0),
        ("angle", "1 hourangle", 15.0),
        ("fc", "50 mHz", 0.05),
        ("spacing", "43200 s", 0.5),
        ("temperature", 267, 267.0),
    ],
)
def test_aliases_convert_when_the_model_is_built(field, value, expected):
    # Check that every alias converts its value to the implicit unit, as a float.
    model = _Model(**{field: value})

    assert isinstance(getattr(model, field), float)
    assert np.isclose(getattr(model, field), expected)


def test_aliases_wrong_unit_raises():
    with pytest.raises(ValidationError):
        _Model(fc="3 arcsec")


# -----------------------------------------------------------------------------
# _is_temperature
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "expected"),
    [
        ("K", True),
        ("mK", True),
        ("uK", True),
        ("rad", False),
        ("Jy beam-1", False),
        (None, False),
        ("arbitrary", False),
    ],
)
def test_is_temperature(unit, expected):
    assert _is_temperature(unit) is expected


# -----------------------------------------------------------------------------
# _conversion_factor
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "target", "factor", "fits"),
    [
        ("K", "mK", 1000.0, "mK"),
        ("mK", u.K, 1e-3, "K"),
        ("rad", "mrad", 1000.0, "mrad"),  # a phase can change its angular unit
        ("K", "Jy/beam", 0.954, "Jy beam-1"),  # 90 GHz and a FWHM of 12 arcsec
        ("Jy/beam", "K", 1 / 0.954, "K"),
    ],
)
def test_conversion_factor(unit, target, factor, fits):
    result = _conversion_factor(unit, target, frequency=90.0, beam_fwhm=12.0)

    assert result[0] == pytest.approx(factor, rel=1e-3)
    assert result[1] == fits


@pytest.mark.parametrize(
    ("unit", "target", "kwargs", "match"),
    [
        (None, "mK", {}, "not known"),
        ("K", "blah", {}, "did not parse"),
        ("K", "s", {}, "not convertible"),
        ("K", "Jy", {}, "not convertible"),  # a flux density is per beam
        ("rad", "K", {}, "calibrate"),
        ("K", "Jy/beam", {"frequency": 0.0}, "must be positive"),
        ("K", "Jy/beam", {"beam_fwhm": "-12 arcsec"}, "must be positive"),
    ],
)
def test_conversion_factor_invalid_raises(unit, target, kwargs, match):
    arguments = {"frequency": 90.0, "beam_fwhm": 12.0} | kwargs
    with pytest.raises(ValueError, match=match):
        _conversion_factor(unit, target, **arguments)
