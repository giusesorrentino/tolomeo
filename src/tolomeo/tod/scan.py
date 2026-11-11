from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator

from .._lib.units import Arcsec
from ..coords import Coordinates
from .tod import TimeOrderedData

__all__ = ["Scan", "ScanSpec"]

_RADIAN_TO_ARCSEC = 180 * 3600 / np.pi


class ScanSpec(BaseModel):
    """
    Where a scan is on disk, and a pointing correction to apply when reading it.

    Attributes
    ----------
    path : str or :class:`~pathlib.Path`
        Directory with the FITS subscans; turned into an absolute ``Path``.
    ra_offset, dec_offset : float
        Correction added to the boresight RA and Dec (see
        :func:`tolomeo.coords.transforms.shift_boresight`). A number is in arcsec;
        ``"1.5 arcmin"`` or an :class:`~astropy.units.Quantity` is converted to arcsec.
        Default ``0.0``.
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
    detector_ids : list of int
        Beammap id of every detector, in the order of the columns of the signal.
        Length ``n_det``.
    telescope : str, optional
        Name of the telescope, written in the FITS map as ``TELESCOP``.
        Default ``"SRT"``.
    instrument : str, optional
        Name of the instrument, written in the FITS map as ``INSTRUME``.
        Default ``"MISTRAL"``.
    """

    # numpy arrays are not pydantic types: they are only checked with isinstance
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tod: TimeOrderedData
    coordinates: Coordinates
    offsets: dict[str, NDArray[np.float64]]
    detector_ids: list[int]
    telescope: str | None = "SRT"
    instrument: str | None = "MISTRAL"

    def select(self, mask: NDArray[np.bool_]) -> Scan:  # mask has shape (n_samples,)
        """
        Keep only the samples where ``mask`` is True, in the TOD and in the pointing.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` of bool
            True for the samples to keep.
            Shape ``(n_samples,)``.

        Returns
        -------
        :class:`~tolomeo.tod.Scan`
            A new scan; ``offsets`` and ``detector_ids`` are shared, not copied.
        """
        return self.model_copy(
            update={
                "tod": self.tod.select(mask),
                "coordinates": self.coordinates.select(mask),
            }
        )

    @property
    def center(self) -> tuple[float, float]:
        """
        Mean boresight position ``(ra, dec)`` in radians, over the good samples.

        Only the samples with ``tod.flags`` True are used (see
        :attr:`~tolomeo.coords.Coordinates.center`).
        """
        coord = self.coordinates.select(self.tod.flags)
        return coord.center

    @property
    def scan_speed(self) -> float:
        """
        Median speed of the boresight on the sky, in arcsec/s.

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
        return np.median(speed) * _RADIAN_TO_ARCSEC
