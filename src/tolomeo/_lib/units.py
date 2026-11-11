"""
Physical quantities given with or without a unit, in Python and in YAML files.

A parameter that has a physical dimension (a pixel size, a cutoff frequency, ...) keeps
one **implicit unit**, written in its docstring, and its name carries no unit. It accepts:

- a number: it is already in the implicit unit (``4`` for a pixel size means 4 arcsec);
- a string with a number and a unit, separated by a space (``"4 arcmin"``): the unit is
  read by :mod:`astropy.units` and the value is converted to the implicit unit;
- an :class:`~astropy.units.Quantity` (``4 * u.arcmin``): converted the same way.

The fields of the pydantic models are annotated with the aliases below, so the conversion
happens when the model is built and the attribute is always a plain ``float`` in the
implicit unit. A function that is not a model calls :func:`_to_unit`.
"""

import numbers
from typing import Annotated, Any

import astropy.units as u
from pydantic import BeforeValidator


def _to_unit(value: Any, unit: u.UnitBase) -> float:
    """
    Value of a quantity in a given unit, as a float.

    Parameters
    ----------
    value : float, str or :class:`~astropy.units.Quantity`
        A number is returned as it is, because it is already in ``unit``. A string is
        split once at the first whitespace: the first part is the number, the second
        one is a unit read by astropy (``"4 arcmin"``); a string with only the number
        (``"4"``) is in ``unit``. A scalar quantity is converted to ``unit``.
    unit : :class:`~astropy.units.UnitBase`
        The implicit unit of the parameter, and the unit of the result.

    Returns
    -------
    float
        ``value`` expressed in ``unit``.

    Raises
    ------
    ValueError
        If the string is malformed, if the unit is unknown to astropy, if the quantity
        is not a scalar, or if its unit cannot be converted to ``unit`` (for example
        ``"3 Hz"`` for an angle).
    """
    if isinstance(value, u.Quantity):
        quantity = value
    elif isinstance(value, str):
        parts = value.split(maxsplit=1)
        try:
            number = float(parts[0]) if parts else float("")
        except ValueError:
            raise ValueError(
                f"{value!r} is not a number, nor a number and a unit separated by a "
                f"space (for example '4 {unit}')."
            ) from None
        if len(parts) == 1:  # only the number: the implicit unit
            return number
        try:
            quantity = number * u.Unit(parts[1])
        except ValueError as error:  # astropy: the unit does not parse
            raise ValueError(f"{value!r}: {error}") from None
    elif isinstance(value, numbers.Real):
        return float(value)
    else:
        raise TypeError(
            "expected a number, a string such as "
            f"'4 {unit}' or an astropy Quantity, got {type(value).__name__}."
        )

    if not quantity.isscalar:
        raise ValueError(f"expected a scalar quantity, got shape {quantity.shape}.")
    try:
        return float(quantity.to_value(unit))
    except u.UnitsError:
        raise ValueError(f"{value!r} cannot be converted to {unit}.") from None


def _converter(unit: u.UnitBase) -> BeforeValidator:
    """Pydantic validator that runs :func:`_to_unit` before the type check of the field."""

    def convert(value: Any) -> float:
        return _to_unit(value, unit)

    return BeforeValidator(convert)


# One alias per implicit unit. The field keeps its own constraints, for example
# ``pixel_size: _Arcsec = Field(default=3.0, gt=0)``, applied after the conversion.
Arcsec = Annotated[float, _converter(u.arcsec)]
Arcmin = Annotated[float, _converter(u.arcmin)]
Degrees = Annotated[float, _converter(u.deg)]
Radians = Annotated[float, _converter(u.rad)]
Hz = Annotated[float, _converter(u.Hz)]
Days = Annotated[float, _converter(u.day)]
Kelvin = Annotated[float, _converter(u.K)]
