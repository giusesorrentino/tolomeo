from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from .._lib.units import Arcsec
from ..coords import Coordinates
from ._tod import TimeOrderedData

__all__ = ["Scan", "ScanSpec"]

_RADIAN_TO_ARCMIN = 180 * 60 / np.pi


class ScanSpec(BaseModel):
    """
    Where a scan is on disk, and a pointing correction to apply when reading it.

    Attributes
    ----------
    path : str or :class:`~pathlib.Path`
        Directory with the FITS subscans; turned into an absolute ``Path``.
    ra_offset, dec_offset : float
        Correction added to the boresight RA and Dec (see
        :func:`tolomeo.coords._transforms.shift_boresight`). A number is in arcsec;
        ``"1.5 arcmin"`` or an :class:`~astropy.units.Quantity` is converted to arcsec.
        Defaults to ``0.0``.
    """

    model_config = ConfigDict(extra="forbid")

    path: Path
    ra_offset: Arcsec = 0.0  # implicit unit: arcsec
    dec_offset: Arcsec = 0.0  # implicit unit: arcsec

    @field_validator("path")
    @classmethod
    def _resolve_path(cls, path: Path) -> Path:
        """Turn the path into an absolute one (a str is converted to Path before)."""
        return path.resolve()


class Scan(BaseModel):
    """
    One observation: the TOD, the boresight pointing and the detector offsets.

    Attributes
    ----------
    tod : :class:`~tolomeo.tod.TimeOrderedData`
        Signal ``(n_samples, n_det)``, time, flags and subscan boundaries.
    coordinates : :class:`~tolomeo.coords.Coordinates`
        Boresight pointing, one value per sample (``(n_samples,)`` arrays, radians).
    offsets : dict of str to :class:`~numpy.ndarray` of float64
        ``"lon"`` and ``"lat"``: offset of every detector from the boresight in the
        focal plane, in radians. Each of shape ``(n_det,)``.
    channel_ids : list of int
        Beammap id of every detector, in the order of the columns of the signal.
        Length ``n_det``.
    telescope : str, optional
        Name of the telescope, written in the FITS map as ``TELESCOP``.
        Defaults to ``"SRT"``.
    instrument : str, optional
        Name of the instrument, written in the FITS map as ``INSTRUME``.
        Defaults to ``"MISTRAL"``.
    """

    # numpy arrays are not pydantic types: they are only checked with isinstance
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tod: TimeOrderedData
    coordinates: Coordinates
    offsets: dict[str, NDArray[np.float64]]
    channel_ids: list[int]
    telescope: str | None = "SRT"
    instrument: str | None = "MISTRAL"

    def select(self, mask: NDArray[np.bool_]) -> Scan:  # mask has shape (n_samples,)
        """
        Keep only the samples where ``mask`` is True, in the TOD and in the pointing.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` of bool
            True for the samples to keep. Shape ``(n_samples,)``.

        Returns
        -------
        :class:`~tolomeo.tod.Scan`
            A new scan; ``offsets`` and ``channel_ids`` are shared, not copied.
        """
        return self.model_copy(
            update={
                "tod": self.tod.select(mask),
                "coordinates": self.coordinates.select(mask),
            }
        )

    @property
    def source(self) -> tuple[float, float]:
        """
        Position ``(ra, dec)`` of the source, in radians, from the FITS headers.

        :func:`~tolomeo.io.read_scan` reads it from the header of every subscan
        (:attr:`~tolomeo.coords.Coordinates.source_ra` and ``source_dec``). It is
        the same at every sample for a fixed source; a planet moves, and the
        position is the one of the sample in the middle of the scan.

        Raises
        ------
        ValueError
            If the coordinates have no position of the source (a scan built by
            hand): then give the center explicitly.
        """
        ra, dec = self.coordinates.source_ra, self.coordinates.source_dec
        if ra is None or dec is None or ra.size == 0:
            raise ValueError(
                "The scan has no position of the source: give the center explicitly."
            )

        mid = ra.size // 2
        return float(ra[mid]), float(dec[mid])

    @property
    def scan_speed(self) -> float:
        """
        Median speed of the boresight on the sky, in arcmin/s.

        Computed subscan by subscan, from the good samples with a finite pointing;
        subscans with fewer than two such samples are skipped. Raise ``ValueError`` if
        none is left.
        """
        speed = []
        for start, stop in pairwise(self.tod.segments):
            if stop - start < 2:
                continue

            ra = self.coordinates.ra[start:stop]
            dec = self.coordinates.dec[start:stop]
            t = self.tod.time[start:stop] * 86400

            ok = self.tod.flags[start:stop] & np.isfinite(ra) & np.isfinite(dec)
            ra = np.unwrap(ra[ok])

            if ok.sum() < 2:
                continue

            v = np.hypot(
                np.cos(dec[ok]) * np.gradient(ra, t[ok]), np.gradient(dec[ok], t[ok])
            )
            speed.append(v)

        if not speed:
            raise ValueError("No valid subscan.")

        speed = np.concatenate(speed)
        return np.median(speed) * _RADIAN_TO_ARCMIN

    def to(
        self,
        unit: str | u.UnitBase,
        *,
        frequency: float | str | u.Quantity = 90.0,
        beam_fwhm: float | str | u.Quantity = 12.0,
    ) -> Scan:
        """
        The same scan with its TOD in another unit.

        :meth:`TimeOrderedData.to <tolomeo.tod.TimeOrderedData.to>` on ``tod``, with
        the same arguments; the coordinates and the offsets are shared with this scan,
        not copied. Convert the scan before
        :func:`~tolomeo.mapmaking.estimate_inverse_noise`: the map of
        :func:`~tolomeo.mapmaking.gls_map` does not depend on the unit of the noise
        model, but ``lag_zero``, and so the weights of the map, are an inverse
        variance in the unit of the scan the model was estimated on.

        Parameters
        ----------
        unit : str or :class:`~astropy.units.UnitBase`
            The new unit, for example ``"mK"`` or ``"mJy/beam"``. A string is read as a
            FITS unit string.
        frequency : float, str or :class:`~astropy.units.Quantity`, optional
            Frequency of the conversion to or from ``Jy/beam``. A number is in GHz.
            Defaults to ``90``.
        beam_fwhm : float, str or :class:`~astropy.units.Quantity`, optional
            FWHM of the beam of the instrument, for the conversion to or from
            ``Jy/beam``. A number is in arcsec. Defaults to ``12``.

        Returns
        -------
        :class:`~tolomeo.tod.Scan`
            A copy with the converted TOD.

        Raises
        ------
        ValueError
            As :meth:`TimeOrderedData.to <tolomeo.tod.TimeOrderedData.to>`.
        """
        tod = self.tod.to(unit, frequency=frequency, beam_fwhm=beam_fwhm)
        return self.model_copy(update={"tod": tod})
