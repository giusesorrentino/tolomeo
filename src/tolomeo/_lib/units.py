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

import math
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


def _is_temperature(unit: str | None) -> bool:
    """
    Whether ``unit``, a FITS unit string, is a temperature (``"K"``, ``"mK"``, ...).

    None and a string that astropy cannot read are not: for a TOD they mean not
    calibrated.
    """
    if unit is None:
        return False
    try:
        return u.Unit(unit, format="fits").physical_type == "temperature"
    except ValueError:
        return False


def _conversion_factor(
    unit: str | None,
    target: str | u.UnitBase,
    *,
    frequency: Any,
    beam_fwhm: Any,
) -> tuple[float, str]:
    r"""
    Factor from ``unit`` to ``target``, and ``target`` as a FITS unit string.

    The common part of the ``to`` methods (:meth:`tolomeo.map.MapResult.to`,
    :meth:`tolomeo.tod.TimeOrderedData.to`), which multiply their data by the factor.
    Between a temperature and a flux density per beam (``"Jy/beam"``) the conversion
    is the Rayleigh-Jeans one, at ``frequency`` and for a Gaussian beam of FWHM
    :math:`\theta`, of solid angle :math:`\pi \theta^2 / (4 \ln 2)`; for the other
    units ``frequency`` and ``beam_fwhm`` are not used.

    Parameters
    ----------
    unit : str or None
        Unit of the data, a FITS unit string; None if it is not known.
    target : str or :class:`~astropy.units.UnitBase`
        The new unit. A string is read as a FITS unit string.
    frequency : float, str or :class:`~astropy.units.Quantity`
        Frequency of the conversion to or from ``Jy/beam``. A number is in GHz.
    beam_fwhm : float, str or :class:`~astropy.units.Quantity`
        FWHM of the beam, for the conversion to or from ``Jy/beam``. A number is in
        arcsec.

    Returns
    -------
    factor : float
        The value in ``target`` of one ``unit``.
    target : str
        ``target`` in the FITS syntax (``"Jy beam-1"`` for ``"Jy/beam"``).

    Raises
    ------
    ValueError
        If ``unit`` is None, if a unit is not a FITS unit, if ``frequency`` or
        ``beam_fwhm`` is not positive, if ``unit`` is an angle (the phase of the KIDs)
        and ``target`` is not, or if the two units cannot be converted into each
        other.
    """
    if unit is None:
        raise ValueError("The unit is not known (None): it cannot be converted.")
    source = u.Unit(unit, format="fits")
    new = u.Unit(target, format="fits")

    frequency = _to_unit(frequency, u.GHz)
    beam_fwhm = _to_unit(beam_fwhm, u.arcsec)
    if frequency <= 0 or beam_fwhm <= 0:
        raise ValueError(
            "`frequency` and `beam_fwhm` must be positive, got "
            f"{frequency} GHz and {beam_fwhm} arcsec."
        )

    # a phase is not a temperature: the responsivity of each detector converts it
    if source.is_equivalent(u.rad) and not new.is_equivalent(u.rad):
        raise ValueError(
            f"The unit is {unit!r}, the phase of the KIDs: "
            "tolomeo.calibration.calibrate converts a TOD to a temperature."
        )

    # the beam enters only between a temperature and a flux density per beam:
    # without it, "K" to "Jy" is refused instead of giving Jy per beam
    equivalencies = []
    per_beam = u.Jy / u.beam
    if source.is_equivalent(per_beam) or new.is_equivalent(per_beam):
        omega = math.pi * (beam_fwhm * u.arcsec) ** 2 / (4 * math.log(2))
        equivalencies = u.brightness_temperature(
            frequency * u.GHz, beam_area=omega.to(u.sr)
        )

    factor = source.to(new, equivalencies=equivalencies)
    return float(factor), new.to_string("fits")


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
