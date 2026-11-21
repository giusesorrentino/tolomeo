"""
Configuration of the command line: the YAML file that ``tolomeo`` reads.

The sections follow the pipeline ``read_scan -> process_tod -> MapGeometry ->
estimate_inverse_noise -> gls_map -> write_map``; each key is an argument of one of those
functions. Only ``scan`` and
``output`` are required.

Example::

    scan:
      tod_dirs:                      # one or more scans, mapped together
        - path: ~/Developer/ROMA_DATA/JUPITER/20241214-204624-MISTRAL-JUPITER_TY-2.5_TZ-7
          ra_offset: 0.0             # arcsec, added to the boresight (optional)
          dec_offset: 0.0
        # a plain string is also accepted:
        # - ~/Developer/ROMA_DATA/JUPITER/another_scan
      beammap: ~/Developer/ROMA_DATA/JUPITER/chp_offset_rel8_14DEC24_matteo.dat

    geometry:                        # MapGeometry
      nx: 384
      ny: 384
      pixel_size: 4.0                # arcsec
      projection: sin                # sin | tan | zea
      center: auto                   # auto = the FITS headers, or [ra, dec] in degrees

    calibration:                     # optional, applied before `filtering`
      skydip_dir: ~/Developer/ROMA_DATA/JUPITER/CAL   # the GAIN_CAL FITS files, or a single FITS file
      T_atm: 267.0                   # K
      tau_atm: 0.16                  # zenith opacity
      correct_opacity: true          # also divide by exp(-tau_atm * airmass)

    filtering:                       # FilteringConfig, same keys as in its own YAML
      mask:
        type: radius
        radius: 3.5                  # arcmin
      steps:
        - type: baseline
        - type: common_mode

    noise:                           # arguments of estimate_inverse_noise
      nperseg: 8192                  # samples per block (even); optional, default: auto
      mask:                          # optional; default: the mask of filtering
        type: radius
        radius: 3.5                  # arcmin
      filters:                       # optional: filters of the inverse spectra, not of the TOD
        - type: bandpass             # lowpass | highpass | bandpass, as in `filtering`
          f_lower: 0.1               # Hz
          f_upper: 30
          order: 2

    gls:                             # arguments of gls_map
      solver: cg                     # cg | minres
      residual_norm: plain           # plain | preconditioned | m_inverse
      tol: 1.0e-3
      maxiter: 300
      min_hits_frac: 0.1             # optional: only pixels with >= 0.1 * max hits
      workers: -1                    # -1 = all the cores

    output:
      path: jupiter.fits             # FITS map (SCI) and hits (HITS)
      overwrite: false
      header:                        # more FITS keywords
        OBJECT: Jupiter
      plot: jupiter.png              # optional

Every relative path of the file is relative to the folder of the YAML file, not to the
directory where ``tolomeo`` is launched.

Units: the name of a key never carries a unit. A key with a physical dimension has an
implicit unit (written in the comments above: arcsec for ``pixel_size`` and the offsets,
arcmin for ``radius``, degrees for ``center``, kelvin for ``T_atm``, Hz for the
frequencies of the filters). A plain number is in that unit; a string with a number and a
unit, separated by a space, is converted to it with :mod:`astropy.units`. These lines are
the same::

    pixel_size: 4.0
    pixel_size: 4 arcsec
    pixel_size: 0.0666667 arcmin
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import astropy.units as u
import yaml
from numpy import deg2rad
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .._lib.units import Kelvin, _to_unit
from .._typing import ResidualNorm, Solver
from ..io import read_scan
from ..io.map import check_fits_suffix
from ..map import MapGeometry
from ..processing import FilteringConfig, FrequencyFilter
from ..processing.masks import Mask
from ..tod import Scan, ScanSpec


def _resolve(p: str, base: Path) -> Path:
    return (base / Path(p).expanduser()).resolve()


def _required(section: Any, key: str, name: str, yaml_path: str | Path) -> Any:
    """
    Return ``section[key]``; if it is missing or empty, raise a ``ValueError``
    that names the key (``name``, as written in the YAML) and the file.
    """
    if not isinstance(section, dict) or section.get(key) is None:
        raise ValueError(f"{yaml_path}: `{name}` is missing or empty.")
    return section[key]


class NoiseConfig(BaseModel):
    """
    The noise model (arguments of ``estimate_inverse_noise``).

    ``nperseg`` is the number of samples in one block (even), or ``auto`` to choose it
    from the segments of the TOD, the runs of good samples with no gap in time.
    Defaults to ``auto``.

    ``filters`` are applied to the inverse noise spectra, not to the TOD: the same
    ``lowpass``, ``highpass`` and ``bandpass`` steps as in ``filtering``, with the same
    keys. ``mask`` leaves samples out of the spectra, with the same keys as
    ``filtering.mask``; without it the mask of ``filtering`` is used. ``background``
    and ``cross_spectra`` are not exposed yet; the threads are ``gls.workers``.
    """

    model_config = ConfigDict(extra="forbid")

    nperseg: int | None = Field(default=None, ge=2)  # None: chosen automatically
    filters: list[FrequencyFilter] = Field(default_factory=list)
    mask: Mask | None = None

    @field_validator("nperseg", mode="before")
    @classmethod
    def _auto_nperseg(cls, nperseg: Any) -> Any:
        """Read ``auto`` (in any case) as None, the automatic choice."""
        if isinstance(nperseg, str) and nperseg.strip().lower() == "auto":
            return None
        return nperseg

    @field_validator("nperseg")
    @classmethod
    def _even_nperseg(cls, nperseg: int | None) -> int | None:
        """Refuse an odd length: the blocks of the kernel have an even length."""
        if nperseg is not None and nperseg % 2:
            raise ValueError(f"`nperseg` must be even, got {nperseg}.")
        return nperseg


class GLSConfig(BaseModel):
    """
    The solver (arguments of ``gls_map``).

    ``solver``, ``residual_norm``, ``tol``, ``maxiter`` and ``min_hits_frac`` have the
    meaning and the defaults of ``gls_map``. ``workers`` follows the convention of
    ``gls_map``: None is one thread, -1 is all the cores; ``estimate_inverse_noise``
    uses it too. ``initial_map`` is not exposed yet.
    """

    model_config = ConfigDict(extra="forbid")

    solver: Solver = "cg"
    residual_norm: ResidualNorm = "plain"
    tol: float = Field(default=1e-3, gt=0)
    maxiter: int = Field(default=300, gt=0)
    min_hits_frac: float | None = Field(default=None, ge=0, lt=1)
    workers: int | None = -1


class CalibrationConfig(BaseModel):
    """
    The skydip calibration (``skydip_responsivity`` and ``calibrate``).

    Attributes
    ----------
    skydip_dir : :class:`~pathlib.Path`
        Directory with the ``GAIN_CAL`` FITS files, or the path of a single FITS file.
    T_atm : float
        Temperature of the atmosphere. A number is in kelvin; ``"267 K"`` or an
        :class:`~astropy.units.Quantity` is converted to kelvin. Defaults to ``267``.
    tau_atm : float
        Zenith opacity of the atmosphere. Defaults to ``0.16``.
    correct_opacity : bool
        If True the signal is also divided by ``exp(-tau_atm * airmass)``, so that the
        map is the temperature of the source above the atmosphere; if False only by
        the responsivity. Defaults to ``True``.
    """

    model_config = ConfigDict(extra="forbid")

    skydip_dir: Path
    T_atm: Kelvin = Field(default=267.0, gt=0)  # implicit unit: kelvin
    tau_atm: float = Field(default=0.16, ge=0)
    correct_opacity: bool = True


class OutputConfig(BaseModel):
    """
    Where the results go.

    Attributes
    ----------
    path : :class:`~pathlib.Path`
        FITS file of the map (see ``write_map`` for the accepted suffixes).
    overwrite : bool
        Replace ``path`` if it exists.
    header : dict
        More FITS keywords, written to both HDUs (``extra_header`` of ``write_map``).
    plot : :class:`~pathlib.Path`, optional
        Image of the map (``plot_map`` with its defaults); no image if None.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    overwrite: bool = False
    header: dict[str, str | int | float | bool] = Field(default_factory=dict)
    plot: Path | None = None

    @field_validator("path")
    @classmethod
    def _fits_suffix(cls, path: Path) -> Path:
        """Refuse a suffix that ``write_map`` would refuse, now and not after the run."""
        check_fits_suffix(path)
        return path


class RunConfig(BaseModel):
    """The whole YAML file. Unknown keys are an error, so a typo does not go unnoticed.
    SCRIVERE CHE LA PATH (SE RELATIVE) E' RISOLTA A PARTIRE DALLA POSIZIONE DELLO YAML!
    """

    model_config = ConfigDict(extra="forbid")

    scan: Scan
    output: OutputConfig
    geometry: MapGeometry
    calibration: CalibrationConfig | None = None
    filtering: FilteringConfig = Field(default_factory=FilteringConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    gls: GLSConfig = Field(default_factory=GLSConfig)
    # text of the YAML file, written in the FITS map (None if not built from a file)
    yaml_text: str | None = None

    @classmethod
    def from_yaml(cls: type[RunConfig], path: str | Path) -> RunConfig:
        YAML_FOLDER = Path(path).expanduser().resolve().parent

        with open(path) as yaml_file:
            yaml_text = yaml_file.read()
            raw = yaml.safe_load(yaml_text)
            if raw is None:
                raise ValueError(
                    f"{path} is empty: it needs at least the `scan` and `output` sections."
                )

            if "cleaning" in raw:
                raise ValueError(
                    f"{path}: the `cleaning` section is now called `filtering`."
                )

            # old keys: n_freqs is now nperseg = 2 * (n_freqs - 1), in `noise`
            for section in ("gls", "noise"):
                if isinstance(raw.get(section), dict) and "n_freqs" in raw[section]:
                    raise ValueError(
                        f"{path}: `{section}.n_freqs` is now `noise.nperseg`, "
                        "with nperseg = 2 * (n_freqs - 1) (4097 -> 8192)."
                    )

            raw_scan = _required(raw, "scan", "scan", path)
            raw_specs = _required(raw_scan, "tod_dirs", "scan.tod_dirs", path)
            if not isinstance(raw_specs, list):  # a single entry, without the list
                raw_specs = [raw_specs]

            specs = []
            for raw_spec in raw_specs:
                if isinstance(raw_spec, str):
                    raw_spec = {"path": raw_spec}
                tod_path = _resolve(
                    _required(raw_spec, "path", "scan.tod_dirs[].path", path),
                    YAML_FOLDER,
                )

                # an empty `ra_offset` (or `dec_offset`) field is interpreted as zero
                ra_offset = raw_spec.get("ra_offset") or 0.0
                dec_offset = raw_spec.get("dec_offset") or 0.0

                spec = ScanSpec(
                    path=tod_path,
                    ra_offset=ra_offset,
                    dec_offset=dec_offset,
                )
                specs.append(spec)

            beammap = _resolve(
                _required(raw_scan, "beammap", "scan.beammap", path), YAML_FOLDER
            )

            # the `calibration` field is optional (without it: no calibration)
            calibration = None
            if raw.get("calibration"):
                raw_calibration = raw["calibration"]
                raw_calibration["skydip_dir"] = _resolve(
                    _required(
                        raw_calibration, "skydip_dir", "calibration.skydip_dir", path
                    ),
                    YAML_FOLDER,
                )
                calibration = CalibrationConfig(**raw_calibration)

            # the `filtering` field is optional (uses the default values)
            raw_filtering = raw.get("filtering") or {}
            filtering = FilteringConfig(**raw_filtering)

            # the `noise` field is optional (uses the default values)
            raw_noise = raw.get("noise") or {}
            noise = NoiseConfig(**raw_noise)

            # the `gls` field is optional (uses the default values)
            raw_gls = raw.get("gls") or {}
            gls = GLSConfig(**raw_gls)

            raw_output = _required(raw, "output", "output", path)

            raw_output["path"] = _resolve(
                _required(raw_output, "path", "output.path", path), YAML_FOLDER
            )
            if raw_output.get("plot"):
                raw_output["plot"] = _resolve(raw_output["plot"], YAML_FOLDER)

            output = OutputConfig(**raw_output)

            # the slow part (it reads the data) comes after what does not need it
            scan = read_scan(
                tod_dirs=specs, beammap_path=beammap, channels=raw_scan.get("channels")
            )

            # the `geometry` field is optional (uses the default values)
            raw_geometry = raw.get("geometry") or {}

            center = raw_geometry.get("center") or "auto"
            if isinstance(center, str) and center.lower() == "auto":
                center = scan.source
            else:
                if not isinstance(center, (list, tuple)) or len(center) != 2:
                    raise ValueError(
                        f"{path}: `geometry.center` must be `auto` or [ra, dec], "
                        f"got {center!r}."
                    )
                # numbers are degrees; "83.6 deg", "5.57 hourangle", ... are converted
                center = tuple(deg2rad(_to_unit(c, u.deg)) for c in center)

            raw_geometry["center"] = center

            # nx and ny: both numbers, or both missing / `auto` (fitted to the scan)
            nx = raw_geometry.pop("nx", None)
            ny = raw_geometry.pop("ny", None)
            nx_auto, ny_auto = (v is None or str(v).lower() == "auto" for v in (nx, ny))

            if nx_auto != ny_auto:
                raise ValueError(
                    f"{path}: `geometry.nx` and `geometry.ny` must be both numbers "
                    "or both `auto`."
                )

            if nx_auto:
                geometry = MapGeometry.from_scan(scan, **raw_geometry)
            else:
                raw_geometry.pop("margin", None)  # used only when nx and ny are auto
                geometry = MapGeometry(nx=nx, ny=ny, **raw_geometry)

        return cls(
            scan=scan,
            geometry=geometry,
            calibration=calibration,
            filtering=filtering,
            noise=noise,
            gls=gls,
            output=output,
            yaml_text=yaml_text,
        )
