from __future__ import annotations

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

from .._lib.units import _conversion_factor

__all__ = ["TimeOrderedData"]


class TimeOrderedData(BaseModel):
    """
    Time-ordered data (TOD) of all the detectors of one observation.

    Attributes
    ----------
    signal : :class:`~numpy.ndarray` of float64
        One row per time sample, one column per detector, in ``unit``.
        Shape ``(n_samples, n_det)``.
    time : :class:`~numpy.ndarray` of float64
        Time of every sample, modified Julian date (days). Shape ``(n_samples,)``.
    flags : :class:`~numpy.ndarray` of bool
        True for a good sample, the same for all the detectors. Shape ``(n_samples,)``.
    segments : :class:`~numpy.ndarray` of int64
        First row of every subscan, followed by ``n_samples``: subscan ``k`` is
        ``signal[segments[k]:segments[k + 1]]``. Shape ``(n_subscans + 1,)``.
    chunks : :class:`~numpy.ndarray` of int64, optional
        First row of every chunk, followed by ``n_samples``: chunk ``k`` is
        ``signal[chunks[k]:chunks[k + 1]]``, made of whole subscans. A chunk is a part
        of the TOD with its own noise, independent of the others, as one directory of
        the ``tod_dirs`` of :func:`~tolomeo.io.read_scan`: nothing is filled across
        its boundaries. Shape ``(n_chunks + 1,)``. Defaults to ``[0, n_samples]``:
        the whole TOD is one chunk.
    patched : :class:`~numpy.ndarray` of bool, optional
        True for a sample whose signal is synthetic, written by
        :meth:`Scan.patch <tolomeo.tod.Scan.patch>`, the same for all the detectors.
        A patched sample is flagged, so that the map leaves it out, but the noise
        spectra use it (see :attr:`spectrum_flags`). Shape ``(n_samples,)``.
        Defaults to ``None``: no sample is patched.
    passband : 2-tuple of float, optional
        Band, in Hz, left by the frequency filters already applied to ``signal`` by
        :func:`~tolomeo.processing.process_tod`: the intersection of their bands.
        :func:`~tolomeo.mapmaking.estimate_inverse_noise` uses it for the
        marginalization. Defaults to ``None``: no filter applied yet.
    unit : str, optional
        Unit of ``signal``, as a FITS unit string: ``"rad"`` for the raw TOD of
        MISTRAL, the phase shift of the KIDs, and ``"K"`` after
        :func:`~tolomeo.calibration.calibrate`. For another instrument give the unit
        of its signal, or None if it is not known. Defaults to ``"rad"``.
    """

    # numpy arrays are not pydantic types: they are only checked with isinstance
    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: NDArray[np.float64]  # shape (n_samples, n_det)
    time: NDArray[np.float64]  # shape (n_samples,)
    flags: NDArray[np.bool_]  # shape (n_samples,)
    segments: NDArray[np.int64]
    chunks: NDArray[np.int64] | None = None  # None: one chunk, see _one_chunk
    patched: NDArray[np.bool_] | None = None  # shape (n_samples,)
    passband: tuple[float, float] | None = None
    unit: str | None = "rad"

    @model_validator(mode="after")
    def _one_chunk(self) -> TimeOrderedData:
        """Without ``chunks`` the whole TOD is one chunk: ``[0, n_samples]``."""
        if self.chunks is None:
            self.chunks = np.array([0, self.n_samples], dtype=np.int64)
        return self

    def select(
        self, mask: NDArray[np.bool_]
    ) -> TimeOrderedData:  # mask has shape (n_samples,)
        """
        Keep only the samples where ``mask`` is True.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` of bool
            True for the samples to keep. Shape ``(n_samples,)``.

        Returns
        -------
        :class:`~tolomeo.tod.TimeOrderedData`
            A new object with ``n_samples = mask.sum()``; ``segments`` and ``chunks``
            are moved to the positions of the kept samples. The arrays are copies.

        Raises
        ------
        TypeError
            If ``mask`` is not boolean.
        ValueError
            If ``mask`` does not have shape ``(n_samples,)``.
        """
        if mask.dtype != bool:
            raise TypeError(f"`mask` must be boolean, got dtype {mask.dtype}.")
        if mask.shape != (self.n_samples,):
            raise ValueError(
                f"`mask` must have shape ({self.n_samples},), got {mask.shape}."
            )

        # for every original row i, the number of kept rows before it: a boundary at
        # row i (of a subscan or of a chunk) moves to new_index[i]
        new_index = np.concatenate(([0], np.cumsum(mask)))
        return self.model_copy(
            update={
                "signal": self.signal[mask, :],
                "time": self.time[mask],
                "flags": self.flags[mask],
                "segments": new_index[self.segments],
                "chunks": new_index[self.chunks],
                "patched": None if self.patched is None else self.patched[mask],
            }
        )

    @property
    def sample_rate(self) -> float:
        """Sampling frequency in Hz, from the median time step between samples."""
        dt = np.diff(
            self.time * 86400, axis=0
        )  # remember that time is in MJD, so we convert to seconds
        return 1 / np.median(dt)

    @property
    def spectrum_flags(self) -> NDArray[np.bool_]:
        """
        True for the samples that the noise spectra can use: the good and the patched.

        ``flags`` says which samples enter a map. A patched sample is flagged there,
        but its signal is synthetic noise with no sky, so the spectra use it.
        Shape ``(n_samples,)``.
        """
        if self.patched is None:
            return self.flags
        return self.flags | self.patched

    @property
    def n_samples(self) -> int:
        """Number of time samples: the rows of ``signal``."""
        return self.signal.shape[0]

    @property
    def n_det(self) -> int:
        """Number of detectors: the columns of ``signal``."""
        return self.signal.shape[1]

    def to(
        self,
        unit: str | u.UnitBase,
        *,
        frequency: float | str | u.Quantity = 90.0,
        beam_fwhm: float | str | u.Quantity = 12.0,
    ) -> TimeOrderedData:
        """
        The same TOD in another unit.

        ``signal`` is multiplied by the conversion factor; the other fields do not
        change. Between a temperature and a flux density per beam (``"Jy/beam"``) the
        conversion is the Rayleigh-Jeans one, at ``frequency`` and for a Gaussian beam
        of FWHM ``beam_fwhm`` (see Notes); they are not used for the other units. A
        TOD in ``rad``, the phase of the KIDs, becomes a temperature with
        :func:`~tolomeo.calibration.calibrate`, not with this method.

        Parameters
        ----------
        unit : str or :class:`~astropy.units.UnitBase`
            The new unit, for example ``"mK"``, ``"uK"`` or ``"mJy/beam"``. A string is
            read as a FITS unit string.
        frequency : float, str or :class:`~astropy.units.Quantity`, optional
            Frequency of the conversion to or from ``Jy/beam``. A number is in GHz.
            Defaults to ``90``.
        beam_fwhm : float, str or :class:`~astropy.units.Quantity`, optional
            FWHM of the beam of the instrument, for the conversion to or from
            ``Jy/beam``. A number is in arcsec. Defaults to ``12``.

        Returns
        -------
        :class:`~tolomeo.tod.TimeOrderedData`
            A copy with the new ``signal`` and ``unit``, the last one in the FITS
            syntax (``"Jy beam-1"`` for ``"Jy/beam"``). The other arrays are shared
            with this TOD, not copied.

        Raises
        ------
        ValueError
            If the unit of this TOD is not known (None), if ``unit`` is not a FITS
            unit, if the two units cannot be converted into each other (as ``rad``
            and ``K``), or if ``frequency`` or ``beam_fwhm`` is not positive.

        Notes
        -----
        The conversion to and from ``Jy/beam`` is the one of
        :meth:`MapResult.to <tolomeo.map.MapResult.to>`: see its Notes for the formula
        and for a TOD in :math:`T_A^*`.
        """
        factor, unit = _conversion_factor(
            self.unit, unit, frequency=frequency, beam_fwhm=beam_fwhm
        )
        return self.model_copy(update={"signal": factor * self.signal, "unit": unit})
