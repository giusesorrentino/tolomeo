from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Literal

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, field_validator
from scipy.interpolate import make_smoothing_spline
from scipy.signal import welch
from scipy.stats import median_abs_deviation

from .._lib.units import Arcsec
from .._signal.realization import fill_gaps, rfft_variances
from ..coords import Coordinates
from ._tod import TimeOrderedData

__all__ = ["Scan", "ScanSpec"]

_RADIAN_TO_ARCMIN = 180 * 60 / np.pi
# a jump in time longer than this many sampling steps is a gap
_JUMP_FACTOR = 1.5
# samples fitted on each side of a gap, in units of the length of the gap
_GAP_FIT_WINDOW_FACTOR = 2


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

    # TODO(patch):
    # - where="flag" and "both": fill the flagged samples in place, without inserting
    #   new ones (with "constrained": fill_gaps with the flagged samples as the gaps);
    # - a maximum gap length (a long stop of the telescope would take GB); a fit window
    #   of fewer than 5 samples, next to a very short stretch of data, makes
    #   make_smoothing_spline fail.
    # See also the gap filling of sotodlib:
    # https://sotodlib.readthedocs.io/en/latest/_modules/sotodlib/tod_ops/gapfill.html
    def patch(
        self,
        mode: Literal["constrained", "spline"] = "constrained",
        where: Literal["gap", "flag", "both"] = "gap",
        side: Literal["previous", "next"] = "previous",
        seed: int | None = 0,
    ) -> Scan:
        r"""
        The same scan with synthetic samples in its time gaps.

        A gap is a jump in ``tod.time`` longer than 1.5 sampling steps, as between two
        subscans. It is filled with the samples that are missing, one sampling step
        apart, so that the spectral estimates and the filters in the time domain see a
        continuous TOD.

        With ``mode="spline"`` the signal of a detector in a gap is a smoothing spline
        (:func:`~scipy.interpolate.make_smoothing_spline`, with the smoothing chosen
        by generalized cross-validation) fitted on the samples on the two sides of the
        gap: twice as many samples per side as the gap has, and at least a quarter of
        the shortest stretch of contiguous samples, but never past the next gap. On
        top of the spline goes white noise, with the standard deviation of the
        residuals of the fit (from their median absolute deviation, robust to a
        glitch or to the source in the window). This is enough for gaps short
        compared to the time scale of the :math:`1/f` noise, :math:`1/f_{\rm knee}`:
        over longer gaps the spline adds power at low frequency.

        With ``mode="constrained"`` the new samples are a constrained realization
        (:func:`~tolomeo._signal.realization.fill_gaps`, Hoffman & Ribak 1991):
        Gaussian noise with the spectrum of the detector, consistent with all its
        original samples, the flagged ones too, so that the TOD stays continuous at the
        edges of every gap. The spectrum comes from Welch's method on the stretches of
        original samples only, with blocks chosen as in
        :func:`~tolomeo.mapmaking.estimate_inverse_noise`, so that no fill enters it.
        Below its resolution the spectrum is taken flat: there the data do not
        constrain it.

        Every chunk of the TOD (``tod.chunks``, one per directory of the scan) is
        patched by itself: the jump between two chunks is not filled, the fit windows
        stop at their boundaries, and the constrained realization of a chunk has the
        spectrum of that chunk only.

        The new samples are flagged (``tod.flags`` False), so that the map leaves them
        out, and patched (``tod.patched`` True), so that the noise spectra use them
        (see :attr:`TimeOrderedData.spectrum_flags
        <tolomeo.tod.TimeOrderedData.spectrum_flags>`). They have no pointing (NaN);
        the position of the source is copied from the sample before the gap, because
        :attr:`source` reads it at one sample.

        Parameters
        ----------
        mode : {"constrained", "spline"}, optional
            How the signal in a gap is made: a constrained realization of the noise,
            or the faster smoothing spline with white noise.
            Defaults to ``"constrained"``.
        where : {"gap", "flag", "both"}, optional
            What is filled: the time gaps, the flagged samples, or both. Only
            ``"gap"`` is implemented. Defaults to ``"gap"``.
        side : {"previous", "next"}, optional
            Subscan that the new samples of a gap between two subscans join: the one
            before the gap, at its end, or the one after it, at its start.
            Defaults to ``"previous"``.
        seed : int or None, optional
            Seed of the noise of the fill; every detector gets its own generator
            from it. With the same seed the patch is the same. Defaults to ``0``;
            None gives a different noise at every call.

        Returns
        -------
        :class:`~tolomeo.tod.Scan`
            A new scan with the samples of the gaps added, and ``tod.segments`` and
            ``tod.chunks`` moved accordingly; ``offsets`` and ``channel_ids`` are
            shared, not copied. A copy of this scan if it has no gap inside a chunk.

        Raises
        ------
        ValueError
            If ``mode``, ``where`` or ``side`` is not one of the accepted values.
        NotImplementedError
            If ``where`` is ``"flag"`` or ``"both"``: they are not implemented yet.

        Warns
        -----
        RuntimeWarning
            With ``mode="constrained"``, if the conjugate gradient of the constrained
            realization of a detector does not converge.
        """
        mode, where, side = mode.lower(), where.lower(), side.lower()
        if mode not in ("constrained", "spline"):
            raise ValueError(f"`mode` must be 'constrained' or 'spline', got {mode!r}.")
        if where not in ("gap", "flag", "both"):
            raise ValueError(f"`where` must be 'gap', 'flag' or 'both', got {where!r}.")
        if side not in ("previous", "next"):
            raise ValueError(f"`side` must be 'previous' or 'next', got {side!r}.")

        # one generator per detector: a Generator must not be shared between threads
        rngs = np.random.default_rng(seed).spawn(self.tod.n_det)

        # every kind of patch returns a new scan. With "both" the flagged samples are
        # filled first: the samples inserted in the gaps are flagged too, and would
        # otherwise be filled twice
        scan = self
        if where in ("flag", "both"):
            scan = scan._patch_flags(mode, rngs)
        if where in ("gap", "both"):
            scan = scan._patch_gaps(mode, side, rngs)
        return scan

    def _patch_flags(self, mode: str, rngs: list[np.random.Generator]) -> Scan:
        """Fill the flagged samples in place, as :meth:`patch` with ``where="flag"``."""
        # TODO(patch): find the runs of flagged samples and fill every run with
        # _spline_fill, fitted on the good samples around it as _patch_gaps does for
        # the gaps, then with "constrained" call fill_gaps with the flagged samples as
        # the gaps; the samples stay flagged and become patched, and no sample is
        # inserted
        raise NotImplementedError("`where='flag'` is not implemented yet.")

    def _patch_gaps(
        self, mode: str, side: str, rngs: list[np.random.Generator]
    ) -> Scan:
        """Insert the missing samples of the time gaps, as :meth:`patch` does."""
        tod, coords = self.tod, self.coordinates
        sample_rate = tod.sample_rate

        # time from the first sample, in seconds (tod.time is in MJD). A jump is a
        # time step longer than _JUMP_FACTOR sampling steps; jump_end is the first
        # sample after it
        seconds = (tod.time - tod.time[0]) * 86400
        steps = np.diff(seconds)
        jump_end = np.flatnonzero(steps > _JUMP_FACTOR / sample_rate) + 1

        # first row of every stretch of contiguous samples, and n_samples at the end:
        # a stretch ends at every jump and at every boundary of chunks, also one with
        # no jump in time (sorted and unique, with 0 and n_samples from tod.chunks)
        edges = np.union1d(jump_end, tod.chunks)

        # only the jumps inside a chunk are filled: two chunks are independent, and
        # the time between them (minutes to days) is not a gap of the TOD
        jump_end = jump_end[~np.isin(jump_end, tod.chunks)]
        if jump_end.size == 0:
            return self.model_copy()
        jump_start = jump_end - 1  # the last sample before the jump

        # number of missing samples in every gap, and for every new sample the index
        # of the original sample it goes before. np.insert reads all the indices on
        # the original array, and inserts the values of a repeated index in order
        gap_len = np.round(steps[jump_start] * sample_rate).astype(int) - 1
        insert_index = np.repeat(jump_end, gap_len)

        # times of the new samples of every gap, one sampling step apart, in seconds
        gap_seconds = [
            seconds[start] + np.arange(1, length + 1) / sample_rate
            for start, length in zip(jump_start, gap_len, strict=True)
        ]

        # the original times are kept as they are; the new ones go back to MJD
        new_times = tod.time[0] + np.concatenate(gap_seconds) / 86400
        time = np.insert(tod.time, insert_index, new_times)

        # the new samples are flagged, so that the map leaves them out, and patched,
        # so that the spectra use them
        flags = np.insert(tod.flags, insert_index, False)
        patched = (
            np.zeros(tod.n_samples, dtype=bool) if tod.patched is None else tod.patched
        )
        patched = np.insert(patched, insert_index, True)

        # True for the original samples of the new TOD, False for the inserted ones
        original = np.insert(np.ones(tod.n_samples, dtype=bool), insert_index, False)

        # every boundary moves by the number of new samples before it, which
        # searchsorted counts. For a gap right on a boundary of subscans,
        # side="right" counts its samples as before the boundary (they join the
        # previous subscan), "left" as after it (they join the next one). No sample
        # goes on a boundary of chunks, so there the side does not matter
        n_before = np.searchsorted(
            insert_index, tod.segments, side="right" if side == "previous" else "left"
        )
        segments = tod.segments + n_before
        chunks = tod.chunks + np.searchsorted(insert_index, tod.chunks)

        if mode == "spline":
            # samples fitted on each side of every gap: at least _GAP_FIT_WINDOW_FACTOR
            # times the gap and a quarter of the shortest stretch, but at most the
            # shorter of the two stretches next to the gap, so that the window never
            # reaches another gap or another chunk (- 1: on the left the window also
            # takes the sample jump_start itself). edges[k] is jump_end, so the two
            # stretches are edges[k - 1]:jump_end and jump_end:edges[k + 1]
            k = np.searchsorted(edges, jump_end)
            room = np.minimum(jump_end - edges[k - 1], edges[k + 1] - jump_end) - 1
            window = np.maximum(
                np.diff(edges).min() // 4, _GAP_FIT_WINDOW_FACTOR * gap_len
            )
            window = np.minimum(window, room)
            fits = [
                slice(start - w, stop + w)
                for start, stop, w in zip(jump_start, jump_end, window, strict=True)
            ]
        else:
            # imported here because mapmaking imports tod: at the top of the module
            # the import would be circular
            from ..mapmaking._noise import _auto_nperseg

            # Welch blocks of every chunk, chosen as in estimate_inverse_noise on the
            # stretches of original samples of that chunk
            npersegs = [
                _auto_nperseg(time[a:b], original[a:b], sample_rate)
                for a, b in pairwise(chunks)
            ]

        # for every detector, with its own generator, the column with the new samples
        columns = []
        for idet, rng in enumerate(rngs):
            detector = tod.signal[:, idet]
            if mode == "spline":
                # the fill of every gap, from the samples of its window
                fills = [
                    _spline_fill(seconds[fit], detector[fit], new_seconds, rng)
                    for fit, new_seconds in zip(fits, gap_seconds, strict=True)
                ]
                column = np.insert(detector, insert_index, np.concatenate(fills))
            else:
                # fill_gaps does not read the new samples: zeros until it draws them.
                # One chunk at a time, with its own spectrum; a chunk with no new
                # sample is left as it is
                column = np.insert(detector, insert_index, 0.0)
                for (a, b), nperseg in zip(pairwise(chunks), npersegs, strict=True):
                    if not original[a:b].all():
                        column[a:b] = _constrained_fill(
                            column[a:b], original[a:b], sample_rate, nperseg, rng
                        )
            columns.append(column)
        signal = np.column_stack(columns)

        # no pointing for the new samples (NaN: they fall outside every map); the
        # position of the source, when there is one, is copied from the sample before
        # the gap
        pointing = {
            name: np.insert(getattr(coords, name), insert_index, np.nan)
            for name in ("ra", "dec", "az", "el", "par_angle")
        }
        for name in ("source_ra", "source_dec"):
            values = getattr(coords, name)
            if values is not None:
                pointing[name] = np.insert(
                    values, insert_index, values[insert_index - 1]
                )

        new_tod = tod.model_copy(
            update={
                "signal": signal,
                "time": time,
                "flags": flags,
                "segments": segments,
                "chunks": chunks,
                "patched": patched,
            }
        )
        return self.model_copy(
            update={"tod": new_tod, "coordinates": coords.model_copy(update=pointing)}
        )


def _spline_fill(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    x_new: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Signal at the times ``x_new``, from the samples ``(x, y)`` around them.

    A smoothing spline fitted on ``(x, y)``, with the smoothing chosen by generalized
    cross-validation, plus white noise with the standard deviation of the residuals
    of the fit: the spline gives the slow part of the signal, the noise its
    fluctuations.
    """
    spline = make_smoothing_spline(x, y)

    # scale="normal" turns the median absolute deviation of the residuals into a
    # standard deviation for Gaussian noise (x 1.4826)
    sigma = median_abs_deviation(y - spline(x), scale="normal")
    return spline(x_new) + sigma * rng.standard_normal(x_new.size)


def _constrained_fill(
    column: NDArray[np.float64],
    original: NDArray[np.bool_],
    sample_rate: float,
    nperseg: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    ``column`` with its inserted samples drawn as a constrained realization.

    The PSD comes from :func:`_welch_on_runs` on the original samples, with the floor
    of :func:`~tolomeo.mapmaking.estimate_inverse_noise` so that every variance is
    positive. :func:`~tolomeo._signal.realization.fill_gaps` then keeps the original
    samples and draws the inserted ones.
    """
    # imported here because mapmaking imports tod (see _patch_gaps)
    from ..mapmaking._noise import _FLOOR_FACTOR

    # fill_gaps wants a TOD with zero mean
    mean = column[original].mean()
    x = column - mean

    freqs, psd = _welch_on_runs(x, original, sample_rate, nperseg)
    psd = np.maximum(psd, np.median(psd) * _FLOOR_FACTOR)
    variances = rfft_variances(freqs, psd, x.size, sample_rate)
    filled = fill_gaps(x, original, variances, rng)

    # only the inserted samples are written: adding the mean back to the whole column
    # would move the original samples by a rounding error
    out = column.copy()
    out[~original] = filled[~original] + mean
    return out


def _welch_on_runs(
    x: NDArray[np.float64],
    good: NDArray[np.bool_],
    sample_rate: float,
    nperseg: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Welch's method on the runs of good samples only.

    Every run of contiguous good samples at least ``nperseg`` long gets its own
    estimate, with the blocks of :func:`scipy.signal.welch` (Hann window, half
    overlap); the estimates are averaged with the number of blocks of each run as
    weight, which is the average over all the blocks. No block crosses a bad sample,
    so the values of the bad samples never enter the spectrum.

    Returns
    -------
    freqs, psd : :class:`~numpy.ndarray` of float64
        Frequencies, in Hz, and one-sided PSD. Shape ``(nperseg // 2 + 1,)``.
    """
    # first and one-past-last sample of every run of good samples
    edges = np.flatnonzero(np.diff(np.r_[0, good.astype(np.int8), 0]))
    freqs = np.fft.rfftfreq(nperseg, 1 / sample_rate)
    total, n_blocks = np.zeros(freqs.size), 0
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        if stop - start < nperseg:
            continue
        # blocks of welch with noverlap = nperseg // 2; the tail that does not fill a
        # block is dropped
        n = (stop - start - nperseg) // (nperseg // 2) + 1
        _, psd = welch(x[start:stop], fs=sample_rate, window="hann", nperseg=nperseg)
        total += n * psd
        n_blocks += n
    return freqs, total / n_blocks
