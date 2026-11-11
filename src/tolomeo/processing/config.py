from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .._typing import CommonModeMethod
from .._lib.units import Arcmin, Days, Degrees, Hz

__all__ = [
    "Bandpass",
    "Baseline",
    "CommonMode",
    "EdgeMask",
    "FilteringConfig",
    "FrequencyFilter",
    "Highpass",
    "Lowpass",
    "Mask",
    "RadiusMask",
    "RemoveModes",
    "RemoveSlope",
    "RemoveSpline",
    "SigmaClipMask",
    "Step",
    "Window",
]


class BaseStep(BaseModel):
    """
    Base of the cleaning steps: immutable, and an unknown field is an error.

    Every step has a ``type`` field with a fixed value, the name of the step. In a
    YAML file it tells which class to use for each entry of ``steps``; from Python
    it can be left out.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class RemoveSlope(BaseStep):
    """
    Subtract the line through the first and last sample of each detector.

    :func:`~tolomeo._signal.detrend.remove_slope`; it has no parameters.
    :class:`Baseline` is a better choice on noisy data.
    """

    type: Literal["remove_slope"] = "remove_slope"


class CommonMode(BaseStep):
    """
    Common mode removal, subscan by subscan.

    See :func:`~tolomeo._signal.common_mode.remove_common_mode`. With a mask, the
    common mode is estimated only from the detectors that are far from the source.

    Attributes
    ----------
    method : {"median", "mean", "corr"}
        How the common mode is estimated. ``"median"`` and ``"mean"`` subtract the
        same value from every detector. ``"corr"`` gives every detector its own common
        mode, the mean of the other detectors weighted by their correlation with it,
        times a gain fitted on that detector; put it after the baseline.
        Default ``"median"``.
    """

    type: Literal["common_mode"] = "common_mode"
    method: CommonModeMethod = "median"


class Baseline(BaseStep):
    """
    Baseline removal from each detector, subscan by subscan.

    A least squares polynomial in time (mapped to [0, 1] in each subscan) is fitted
    to each detector and subtracted. With a mask only the background samples are
    fitted.

    Attributes
    ----------
    deg : int
        Degree of the polynomial (1 is a straight line, 0 only the mean).
        Default ``1``.
    """

    type: Literal["baseline"] = "baseline"
    deg: int = Field(default=1, ge=0)


class RemoveSpline(BaseStep):
    """
    Removal of a B-spline in time from every detector, subscan by subscan.

    With a mask only the background samples are fitted, so ``knot_spacing``
    should be larger than the region left out by the mask.

    Attributes
    ----------
    knot_spacing : float
        Distance between two knots. A number is in **days**, the unit of ``tod.time``
        (modified Julian date); ``"30 s"`` or an :class:`~astropy.units.Quantity` is
        converted to days.
    order : int
        Order (degree) of the spline, for example 3.
        Default ``3``.
    """

    type: Literal["remove_spline"] = "remove_spline"
    knot_spacing: Days = Field(gt=0)  # implicit unit: days
    order: int = 3


class Window(BaseStep):
    """
    Window applied to the whole TOD, to be placed before the filters.

    Attributes
    ----------
    name : str
        Name of a function of :mod:`scipy.signal.windows` (for example ``"tukey"``).
    kwargs : dict
        Extra arguments of that function, after the window length (``n_samples``).
        Default empty.
    """

    type: Literal["window"] = "window"
    name: str
    kwargs: dict = Field(default_factory=dict)


class Lowpass(BaseStep):
    """
    Low-pass filter of every detector (along time).

    See :func:`~tolomeo._signal.filters.lowpass`.

    Attributes
    ----------
    fc : float
        Cutoff frequency. A number is in Hz; ``"50 mHz"`` or an
        :class:`~astropy.units.Quantity` is converted to Hz. It must be below the
        Nyquist frequency of the TOD.
    order : int
        Order of the Butterworth filter. It is applied forward and backward (zero
        phase), so the roll-off is twice as steep.
        Default ``4``.
    """

    type: Literal["lowpass"] = "lowpass"
    fc: Hz = Field(gt=0)  # implicit unit: Hz
    order: int = Field(default=4, ge=1)


class Highpass(BaseStep):
    """
    High-pass filter of every detector (along time).

    See :func:`~tolomeo._signal.filters.highpass`.

    Attributes
    ----------
    fc : float
        Cutoff frequency. A number is in Hz; ``"50 mHz"`` or an
        :class:`~astropy.units.Quantity` is converted to Hz. It must be below the
        Nyquist frequency of the TOD.
    order : int
        Order of the Butterworth filter. It is applied forward and backward (zero
        phase), so the roll-off is twice as steep.
        Default ``4``.
    """

    type: Literal["highpass"] = "highpass"
    fc: Hz = Field(gt=0)  # implicit unit: Hz
    order: int = Field(default=4, ge=1)


class Bandpass(BaseStep):
    """
    Band-pass filter of every detector (along time).

    See :func:`~tolomeo._signal.filters.bandpass`.

    Attributes
    ----------
    f_lower : float
        Lower edge of the band. A number is in Hz; ``"50 mHz"`` or an
        :class:`~astropy.units.Quantity` is converted to Hz.
    f_upper : float
        Upper edge of the band, same units as ``f_lower``; must be larger than
        ``f_lower``.
    order : int
        Order of the Butterworth filter at each edge of the band.
        Default ``4``.
    """

    type: Literal["bandpass"] = "bandpass"
    f_lower: Hz = Field(gt=0)  # implicit unit: Hz
    f_upper: Hz = Field(gt=0)  # implicit unit: Hz
    order: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def _check_band(self) -> Self:
        """Check that the band is not empty."""
        if self.f_lower >= self.f_upper:
            raise ValueError(
                f"f_lower ({self.f_lower}) must be smaller than f_upper ({self.f_upper})."
            )
        return self


class RemoveModes(BaseStep):
    """
    Removal of the strongest modes shared by the detectors (truncated SVD).

    See :func:`~tolomeo._signal.decompose.decompose`.

    Attributes
    ----------
    modes_to_remove : int
        Number of modes, from 0 to ``min(n_samples, n_det) - 1``. The upper limit
        depends on the data, so it is checked when the step runs.
    """

    type: Literal["remove_modes"] = "remove_modes"
    modes_to_remove: int = Field(ge=0)


Step = Annotated[
    RemoveSlope
    | CommonMode
    | Baseline
    | RemoveSpline
    | Window
    | Lowpass
    | Highpass
    | Bandpass
    | RemoveModes,
    Field(discriminator="type"),
]
"""Any cleaning step; the ``type`` field says which one."""


FrequencyFilter = Annotated[
    Lowpass | Highpass | Bandpass,
    Field(discriminator="type"),
]
"""
A frequency filter; the ``type`` field says which one.

The filters that :func:`~tolomeo.mapmaking.estimate_inverse_noise` applies to the
inverse noise spectra (not to the TOD). Same syntax as the steps of
:class:`FilteringConfig`.
"""


class RadiusMask(BaseModel):
    """
    Leave the source out of the fits of the cleaning.

    The samples closer than ``radius`` to the center of the source are not used by
    :class:`CommonMode`, :class:`Baseline` and :class:`RemoveSpline` to estimate
    what to subtract; the estimate is then subtracted from every sample. The distance
    is measured for every detector, each at its own position. The window, the filters,
    :class:`RemoveSlope` and :class:`RemoveModes` do not use the mask.

    Attributes
    ----------
    radius : float
        Radius around the source. A number is in arcmin; ``"20 arcsec"`` or an
        :class:`~astropy.units.Quantity` is converted to arcmin. It should contain the
        whole source plus a margin for the pointing error.
    center : "auto" or tuple of float
        Center of the source: ``"auto"`` is the mean boresight position of the scan,
        otherwise ``(ra, dec)``. Numbers are in degrees; ``"83.6 deg"`` or a
        quantity is converted to degrees.
        Default ``"auto"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["radius"] = "radius"
    radius: Arcmin = Field(gt=0)  # implicit unit: arcmin
    center: Literal["auto"] | tuple[Degrees, Degrees] = "auto"  # implicit unit: deg


# The next two masks do not need to know where the source is: they are what MIRTOS calls
# "mask without radius". In MIRTOS `radius: 0` silently switches to them; here the
# kind of mask is explicit (``type``), so a wrong radius cannot change the algorithm.
# "No mask" (MIRTOS mode "none") is ``mask: None``.
class SigmaClipMask(BaseModel):
    """
    Leave the source out of the fits of the cleaning, finding it by its amplitude.

    In every subscan, the samples of a detector that are more than ``sigma`` standard
    deviations away from its median are taken to be the source, and are not used by
    :class:`CommonMode`, :class:`Baseline` and :class:`RemoveSpline` to estimate what
    to subtract; the estimate is then subtracted from every sample. Each detector
    has its own mask. It does not need the position of the source, so
    :func:`~tolomeo.processing.process_tod` does not need ``geometry`` with it.

    Attributes
    ----------
    sigma : float
        Clipping threshold, in standard deviations. Default 6.
    maxiters : int
        Maximum number of clipping iterations. Default 10.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["sigma_clip"] = "sigma_clip"
    sigma: float = Field(default=6.0, gt=0)
    maxiters: int = Field(default=10, ge=1)


class EdgeMask(BaseModel):
    """
    Leave the source out of the fits of the cleaning, assuming it is in the middle.

    Only the samples at the two ends of every subscan are used by :class:`CommonMode`,
    :class:`Baseline` and :class:`RemoveSpline` to estimate what to subtract; the
    estimate is then subtracted from every sample. The same samples are used for
    every detector, so with :class:`CommonMode` no detector is left in the middle of
    the subscan: it is meant for :class:`Baseline` and :class:`RemoveSpline`.

    Attributes
    ----------
    fraction : float
        Part of each subscan, at each end, that is used: ``0.1`` is the first 10% and
        the last 10% of the samples. Between 0 and 0.5 (excluded). Default 0.1.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["edges"] = "edges"
    fraction: float = Field(default=0.1, gt=0, lt=0.5)


Mask = Annotated[RadiusMask | SigmaClipMask | EdgeMask, Field(discriminator="type")]
"""Any mask of the cleaning; the ``type`` field says which one."""


# andrebbe frozen se non sbaglio
class FilteringConfig(BaseModel):
    """
    The cleaning steps of :func:`~tolomeo.processing.process_tod`, in order.

    The steps run one after the other, in the order of ``steps``, and the same kind
    of step can appear more than once. :class:`CommonMode`, :class:`Baseline` and
    :class:`RemoveSpline` work subscan by subscan; the other steps work on the
    whole TOD.

    Attributes
    ----------
    steps : list of step
        Default: ``[CommonMode(), Baseline()]``. The list replaces the default, it
        does not extend it: to keep the common mode and the baseline, list them.
    mask : :class:`RadiusMask`, :class:`SigmaClipMask`, :class:`EdgeMask` or None
        Leaves the source out of the fits of the steps that work subscan by subscan.
        Default ``None``: every sample is used.

    Examples
    --------
    >>> FilteringConfig(steps=[CommonMode(), Baseline(), Highpass(fc=0.05)])

    The same in a YAML file::

        steps:
          - type: common_mode
          - type: baseline
          - type: highpass
            fc: 0.05       # Hz

    A value with a physical dimension is a number in the unit of its docstring (Hz,
    arcmin, days, ...), or a string with a unit that astropy reads; ``fc: 0.05`` and
    ``fc: 50 mHz`` are the same. From Python, ``Highpass(fc=50 * astropy.units.mHz)``
    works too.

    With a mask around the source::

        mask:
          type: radius
          radius: 4        # arcmin; "240 arcsec" is the same
          center: auto
        steps:
          - type: common_mode
          - type: baseline

    Without knowing where the source is (``geometry`` is not needed)::

        mask:
          type: sigma_clip
          sigma: 6

    """

    steps: list[Step] = Field(default_factory=lambda: [Baseline(), CommonMode()])
    mask: Mask | None = None

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> Self:
        """
        Read the config from a YAML file with a ``steps`` list.

        ``filepath`` is a str or :class:`~pathlib.Path`; an empty file gives the
        default config.
        """
        filepath = Path(filepath).resolve()

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Build the config from a dict with a ``steps`` list (validated)."""
        return cls.model_validate(data)
