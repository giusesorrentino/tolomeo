"""
Noise model of the GLS map maker: the estimate of ``N^-1``.

:func:`estimate_inverse_noise` estimates, from a cleaned scan, the inverse noise power
spectrum of every detector, applies the filters and the marginalization to it and
returns an :class:`InverseNoise`. ``InverseNoiseFilter``, in ``_operators.py``, applies
``N^-1``, built from those spectra, to the blocks of the TOD.

The noise model
---------------
The map-making kernel filters the TOD in the frequency domain: it multiplies the
Fourier transform of every detector by the *inverse* of its noise power spectrum, so
the noisier a frequency, the less weight it gets. ``gls_map`` passes three arrays to
the kernel:

``inv_noiseps``
    Shape ``(n_freqs, n_cols)``, one inverse spectrum per column, with
    ``n_freqs = nperseg // 2 + 1`` bins.
``det_pairs``
    Shape ``(n_cols, 2)``, the two detectors of every column: ``(j, j)`` for an
    auto-spectrum, ``(i, j)`` with ``i != j`` for a cross-spectrum.
``lag_zero``
    Shape ``(n_det,)``, used by the preconditioner of the solver.

Only auto-spectra are supported, so ``n_cols = n_det``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.signal import welch

from .._lib.parallel import _resolve_workers
from .._lib.spectral import _check_nyquist, _narrow_passband, _prepare_for_spectrum
from .._signal.filters import gain
from ..map import MapGeometry
from ..processing.config import Bandpass, FrequencyFilter, Highpass, Lowpass
from ..processing.masks import Mask, RadiusMask
from ..tod import Scan

logger = logging.getLogger(__name__)

# level of the inverse spectra outside the passband, relative to their maximum inside
_STOP_FACTOR = 1e-8

# floor of a power spectrum, relative to its median: without it a bin with almost no
# power would get an enormous weight
_FLOOR_FACTOR = 1e-6

# largest fraction of the good samples that the automatic block length may leave in
# segments shorter than one block. A choice, not a standard value
_MAX_DISCARDED = 0.1

__all__ = ["estimate_inverse_noise"]


class InverseNoise(BaseModel):
    """
    Noise model of the GLS map: the inverse noise spectra of the detectors.

    Output of :func:`estimate_inverse_noise` and input of
    :func:`~tolomeo.mapmaking.gls_map`. It belongs to the scan it was estimated from:
    ``gls_map`` checks, with :meth:`check_scan`, that the detectors and the sampling
    frequency are the same.

    Attributes
    ----------
    inv_noiseps : :class:`~numpy.ndarray` of float64
        One inverse spectrum per column: the ``n_det`` auto-spectra first, in the order
        of ``detector_ids``, then the cross-spectra (if any).
        Shape ``(n_freqs, n_cols)``, with ``n_freqs = nperseg // 2 + 1``.
    det_pairs : :class:`~numpy.ndarray` of uintp
        The two detectors of every column of ``inv_noiseps``.
        Shape ``(n_cols, 2)``.
    lag_zero : :class:`~numpy.ndarray` of float64
        Weight of every detector for the preconditioner: sum of the bins of its
        auto-spectrum divided by ``nperseg``.
        Shape ``(n_det,)``.
    sample_rate : float
        Sampling frequency of the TOD the spectra come from, in Hz.
    detector_ids : list of int
        Beammap id of every detector, in the order of the auto-spectra (the same as
        ``scan.detector_ids``).
    passband : 2-tuple of float or None
        Band of the marginalization in Hz: outside it the inverse spectra are almost
        zero. None means no marginalization.
    filters : tuple of :data:`~tolomeo.processing.FrequencyFilter`
        Filters applied to the inverse spectra (not to the TOD).
    """

    # numpy arrays are not pydantic types: they are only checked with isinstance
    model_config = ConfigDict(arbitrary_types_allowed=True)

    inv_noiseps: NDArray[np.float64]  # shape (n_freqs, n_cols)
    det_pairs: NDArray[np.uintp]  # shape (n_cols, 2)
    lag_zero: NDArray[np.float64]  # shape (n_det,)
    sample_rate: float  # [Hz]
    detector_ids: list[int]
    passband: tuple[float, float] | None = None  # [Hz]
    filters: tuple[FrequencyFilter, ...] = ()

    @property
    def nperseg(self) -> int:
        """
        Samples in one block, of the estimate and of the filter of ``gls_map``.

        The spectra have ``nperseg // 2 + 1`` bins, the rows of ``inv_noiseps``.
        """
        return 2 * (self.inv_noiseps.shape[0] - 1)

    @property
    def n_det(self) -> int:
        """Number of detectors (the length of ``detector_ids``)."""
        return len(self.detector_ids)

    @property
    def freqs(self) -> NDArray[np.float64]:
        """
        Frequency of every bin, in Hz.

        The bins of the real FFT of one block of ``nperseg`` samples, the same that
        ``welch`` gives with the same ``nperseg``.
        Shape ``(nperseg // 2 + 1,)``.
        """
        return np.fft.rfftfreq(self.nperseg, d=1 / self.sample_rate)

    def check_scan(self, scan: Scan) -> None:
        """
        Refuse a scan that this noise model does not belong to.

        A model of another scan would give a wrong map without any error: column ``j``
        of the spectra is used for column ``j`` of the TOD, and the bins of the spectra
        are at ``k * sample_rate / nperseg``.

        Raises
        ------
        ValueError
            If the detectors differ, also only in their order, or if the sampling
            frequencies differ.
        """
        scan_ids = list(scan.detector_ids)
        model_ids = list(self.detector_ids)

        if len(model_ids) != len(scan_ids):
            raise ValueError(
                f"The noise model has {len(model_ids)} detectors and the scan "
                f"{len(scan_ids)}: estimate it on this scan."
            )

        first = next(
            (
                j
                for j, (a, b) in enumerate(zip(scan_ids, model_ids, strict=True))
                if a != b
            ),
            None,
        )
        if first is not None:
            raise ValueError(
                "The detectors of the noise model are not those of the scan, in the "
                f"same order: at position {first} the model has {model_ids[first]} and "
                f"the scan {scan_ids[first]}."
            )

        # both come from the median time step of a TOD: equal for the same scan
        if not np.isclose(self.sample_rate, scan.tod.sample_rate, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"The noise model was estimated at {self.sample_rate} Hz and the scan "
                f"is sampled at {scan.tod.sample_rate} Hz."
            )


def estimate_inverse_noise(
    scan: Scan,
    *,
    nperseg: int | None = 8192,
    filters: Sequence[FrequencyFilter] = (),
    mask: Mask | None = None,
    geometry: MapGeometry | None = None,
    background: NDArray[np.bool_] | None = None,
    cross_spectra: bool = False,
    workers: int | None = -1,
) -> InverseNoise:
    """
    Noise model of a cleaned scan, for :func:`~tolomeo.mapmaking.gls_map`.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        Cleaned scan, the same that goes to ``gls_map``: ``scan.tod.signal`` without
        NaN or inf.
    nperseg : int or None, optional
        Samples in one block, even: the Welch blocks of the estimate and the filter
        blocks of ``gls_map``. The spectra have ``nperseg // 2 + 1`` bins, spaced by
        ``sample_rate / nperseg``. None chooses it from the segments of the TOD, see
        :func:`_auto_nperseg`.
        Default ``8192``.
    filters : sequence of :data:`~tolomeo.processing.FrequencyFilter`
        Filters applied to the inverse spectra, not to the TOD: ``Lowpass``,
        ``Highpass`` or ``Bandpass``, the same objects as the steps of ``process_tod``.
        Several filters are applied one after the other.
        Default none.
    mask : ``RadiusMask``, ``SigmaClipMask`` or ``EdgeMask``, optional
        Samples left out of the spectra (for example the source), the same objects as
        the ``mask`` of :class:`~tolomeo.processing.FilteringConfig`. The flagged
        samples (``scan.tod.flags`` False) are left out too. Give either ``mask`` or
        ``background``, not both.
    geometry : :class:`~tolomeo.map.MapGeometry`, optional
        Needed only by a ``RadiusMask``.
    background : :class:`~numpy.ndarray` of bool, optional
        ``True`` for the samples used to estimate the spectra (far from the source).
        None, and no ``mask``, means ``scan.tod.flags``.
        Shape ``(n_samples,)`` or ``(n_samples, n_det)``.
    cross_spectra : bool
        Also estimate the cross spectra of the detector pairs. Not implemented yet:
        True raises ``NotImplementedError``.
        Default ``False``.
    workers : int, optional
        Threads, with the convention of ``gls_map``: None is one thread, a positive
        number that many, ``-1`` all the cores.

    Returns
    -------
    :class:`InverseNoise`
    """
    sample_rate = scan.tod.sample_rate
    n_det = scan.tod.signal.shape[1]
    workers = _resolve_workers(workers)

    # the cutoffs of the filters are checked here, before the slow estimate
    passband = _noise_passband(scan.tod.passband, filters, sample_rate)
    background = _noise_background(scan, mask, geometry, background)

    # automatic block length: from the segments of the TOD, the runs of good samples
    # with no gap in time. The estimate below still uses the whole TOD
    if nperseg is None:
        segments = _find_segments(scan.tod.time, scan.tod.flags, sample_rate)
        nperseg = _auto_nperseg(segments)

    # the filter of gls_map needs blocks of an even number of samples
    if nperseg < 2 or nperseg % 2:
        raise ValueError(f"`nperseg` must be even and at least 2, got {nperseg}.")

    inv_noiseps, det_pairs = _build_noise_model(
        scan.tod.signal,
        sample_rate,
        nperseg=nperseg,
        background=background,
        cross_spectra=cross_spectra,
        workers=workers,
    )
    freqs = np.fft.rfftfreq(nperseg, d=1 / sample_rate)

    # the filters first: the level of the marginalization is relative to the maximum
    # of the weights that are really used
    inv_noiseps = _filter_inverse(inv_noiseps, freqs, filters, sample_rate)
    inv_noiseps = _marginalize(inv_noiseps, freqs, passband)

    # after the filters and the marginalization, so that the preconditioner sees the
    # same weights as N^-1
    lag_zero = inv_noiseps[:, :n_det].sum(axis=0) / nperseg

    return InverseNoise(
        inv_noiseps=inv_noiseps,
        det_pairs=det_pairs,
        lag_zero=lag_zero,
        sample_rate=sample_rate,
        detector_ids=scan.detector_ids,
        passband=passband,
        filters=tuple(filters),
    )


def _noise_passband(
    passband: tuple[float, float] | None,
    filters: Sequence[FrequencyFilter],
    sample_rate: float,
) -> tuple[float, float] | None:
    """
    Band of the marginalization: ``passband`` narrowed by every filter of ``filters``.

    Parameters
    ----------
    passband : 2-tuple of float or None
        Band of the filters already applied to the TOD (``scan.tod.passband``), in Hz.
        None if there is none.
    filters : sequence of :data:`~tolomeo.processing.FrequencyFilter`
        Filters of the inverse spectra. Every cutoff is checked against the Nyquist
        frequency.
    sample_rate : float
        Sampling frequency in Hz.

    Returns
    -------
    2-tuple of float or None
        The narrowed band in Hz, None if there is no filter at all.
    """
    for filt in filters:
        match filt:
            case Lowpass():
                _check_nyquist(sample_rate, filt.fc)
                passband = _narrow_passband(passband, f_upper=filt.fc)
            case Highpass():
                _check_nyquist(sample_rate, filt.fc)
                passband = _narrow_passband(passband, f_lower=filt.fc)
            case Bandpass():
                _check_nyquist(sample_rate, filt.f_lower, filt.f_upper)
                passband = _narrow_passband(passband, filt.f_lower, filt.f_upper)

    return passband


def _noise_background(
    scan: Scan,
    mask: Mask | None,
    geometry: MapGeometry | None,
    background: NDArray[np.bool_] | None,
) -> NDArray[np.bool_]:
    """
    Samples used to estimate the spectra: True far from the source and not flagged.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        The scan of the estimate.
    mask : ``RadiusMask``, ``SigmaClipMask`` or ``EdgeMask``, optional
        Samples to leave out (for example the source). The flagged samples are left
        out too.
    geometry : :class:`~tolomeo.map.MapGeometry`, optional
        Needed only by a ``RadiusMask``.
    background : :class:`~numpy.ndarray` of bool, optional
        Given directly instead of ``mask``, and used as it is.

    Returns
    -------
    :class:`~numpy.ndarray` of bool
        ``background`` if given, otherwise the mask combined with
        ``scan.tod.flags``, otherwise ``scan.tod.flags`` alone.
        Shape ``(n_samples,)`` or ``(n_samples, n_det)``, depending on the mask.

    Raises
    ------
    ValueError
        If both ``mask`` and ``background`` are given.
    """
    if mask is not None and background is not None:
        raise ValueError("Give either `background` or `mask`, not both.")

    if mask is not None:
        # (n_samples,) or (n_samples, n_det), depending on the mask
        args = (scan, geometry) if isinstance(mask, RadiusMask) else (scan,)
        background = mask.build(*args)

        flags = scan.tod.flags if background.ndim == 1 else scan.tod.flags[:, None]
        return background & flags

    if background is None:
        return scan.tod.flags

    return background


def _find_segments(
    time: NDArray[np.float64],
    ok: NDArray[np.bool_],
    sample_rate: float,
) -> NDArray[np.intp]:
    """
    Intervals of the TOD where the good samples are contiguous in time.

    Welch's method assumes that the samples of a block are taken every
    ``1 / sample_rate``. A segment is a run of good samples with no flagged sample
    and no gap in time inside it. Used to choose the block length, see
    :func:`_auto_nperseg`.

    Parameters
    ----------
    time : :class:`~numpy.ndarray`
        Time of every sample, in MJD.
        Shape ``(n_samples,)``.
    ok : :class:`~numpy.ndarray` of bool
        True for the good samples (``scan.tod.flags``).
        Shape ``(n_samples,)``.
    sample_rate : float
        Sampling frequency in Hz.

    Returns
    -------
    :class:`~numpy.ndarray` of intp
        One row per segment: the index of its first sample and the index after its
        last sample, so that segment ``k`` of a detector is ``x[start:stop]``. The
        indices refer to the full TOD, not to ``time[ok]``.
        Shape ``(n_segments, 2)``.

    Raises
    ------
    ValueError
        If no sample is good.

    Notes
    -----
    Two consecutive good samples are ``1 / sample_rate`` apart when they are
    contiguous, and at least ``2 / sample_rate`` apart when a flagged sample or a gap
    in time lies between them. The threshold ``1.5 / sample_rate`` separates the two
    cases and tolerates a small jitter of the time stamps.
    """
    # with no good sample there is no segment, and ok_idx[0] below would fail
    if not ok.any():
        raise ValueError("No good sample: cannot find the segments of the TOD.")

    # index in the full TOD of every good sample. Entry p is where the p-th good
    # sample sits, so this array converts positions in time[ok] into TOD indices
    ok_idx = np.flatnonzero(ok)

    # time between consecutive good samples, in seconds (time is in MJD, i.e. days):
    # dt[p] is the distance between the good samples p and p + 1
    dt = np.diff(time[ok]) * 86400

    # positions in time[ok] of the first good sample after every discontinuity:
    # where dt is longer than one sample, the good sample p + 1 starts a new segment
    after_jump = np.flatnonzero(dt > 1.5 / sample_rate) + 1

    # start of every segment but the first: the first good sample after a jump,
    # read in ok_idx to get its index in the TOD
    starts = ok_idx[after_jump]

    # stop of every segment but the last: the last good sample before a jump, plus
    # one because the stop is excluded, as in a slice
    stops = ok_idx[after_jump - 1] + 1

    # no jump marks the edges: the first segment starts at the first good sample and
    # the last one stops after the last good sample
    starts = np.insert(starts, 0, ok_idx[0])
    stops = np.append(stops, ok_idx[-1] + 1)

    # one row per segment: (start, stop)
    return np.column_stack((starts, stops))


def _auto_nperseg(segments: NDArray[np.integer]) -> int:
    r"""
    Block length chosen from the lengths of the segments of the TOD.

    Parameters
    ----------
    segments : :class:`~numpy.ndarray` of int
        Start and stop of every segment, as returned by :func:`_find_segments`.
        Shape ``(n_segments, 2)``.

    Returns
    -------
    int
        ``nperseg``, a power of two.

    Notes
    -----
    A block of ``nperseg`` samples fits in a segment only if the segment is at least
    that long; otherwise it crosses a flagged sample or a gap in time. ``nperseg``
    also sets the frequency resolution, ``sample_rate / nperseg``, which is the lowest
    frequency above zero in the spectra, and the longest time over which the filter
    of ``gls_map`` models the noise correlations. A longer block is therefore better,
    as long as most of the samples are in segments that can hold it.

    The rule is: the longest power of two such that the segments shorter than it
    hold at most a fraction :math:`\epsilon` (``_MAX_DISCARDED``) of the good samples.
    With :math:`\ell_k` the lengths of the segments,

    .. math::

        n_{\rm perseg} = \max \left\{ 2^n \;:\;
        \frac{\sum_{\ell_k < 2^n} \ell_k}{\sum_k \ell_k} \le \epsilon \right\}.

    Powers of two keep the FFTs fast. The value of :math:`\epsilon` is a choice, not
    a result from the literature. The segments only choose the length: the spectra
    are still estimated on the whole TOD.
    """
    # length of every segment, in samples
    lengths = segments[:, 1] - segments[:, 0]
    total = lengths.sum()

    # start from the longest power of two that fits in the longest segment: a longer
    # block would fit in no segment at all. bit_length gives it exactly, without the
    # rounding of a float log2
    nperseg = 1 << (int(lengths.max()).bit_length() - 1)

    # halve the block until the segments shorter than it hold at most a fraction
    # _MAX_DISCARDED of the good samples. The smallest block has 2 samples, the
    # shortest that gives n_freqs >= 2
    while nperseg > 2:
        discarded = lengths[lengths < nperseg].sum() / total
        if discarded <= _MAX_DISCARDED:
            break
        nperseg //= 2

    return nperseg


def _build_noise_model(
    tod: NDArray[np.float64],
    sample_rate: float,
    nperseg: int = 8192,
    background: NDArray[np.bool_] | None = None,
    cross_spectra: bool = False,
    workers: int = 1,  # TODO: a global setting in the config, not in every signature
) -> tuple[NDArray[np.float64], NDArray[np.uintp]]:
    """
    Inverse noise spectra of all the detectors, in the form the kernel expects.

    Only the estimate: the filters, the marginalization and ``lag_zero`` are added by
    :func:`estimate_inverse_noise`.

    Parameters
    ----------
    tod : :class:`~numpy.ndarray`
        Cleaned TOD, without NaN or inf (the FFT of the kernel does not accept them).
        Shape ``(n_samples, n_det)``.
    sample_rate : float
        Sampling frequency in Hz.
    nperseg : int
        Samples in one Welch block, as in the filter blocks of the kernel. Every
        spectrum has ``n_freqs = nperseg // 2 + 1`` bins.
        Default ``8192``.
    background : :class:`~numpy.ndarray` of bool, optional
        True for the samples far from the source: one column per detector, or one
        array for all of them.
        Shape ``(n_samples, n_det)`` or ``(n_samples,)``.
    cross_spectra : bool
        Add one cross-spectrum for every pair of detectors. Not implemented yet: True
        raises ``NotImplementedError``.
        Default ``False``.
    workers : int
        Threads: the detectors are estimated in parallel.
        Default ``1``.

    Returns
    -------
    inv_noiseps : :class:`~numpy.ndarray`
        One inverse spectrum per column, the auto-spectra of the detectors in order.
        Shape ``(n_freqs, n_det)``.
    det_pairs : :class:`~numpy.ndarray` of uintp
        Row ``j`` is ``(j, j)``: column ``j`` of ``inv_noiseps`` is the auto-spectrum
        of detector ``j``.
        Shape ``(n_det, 2)``.
    """
    # refuse inputs that the estimate cannot work on, before the slow part
    if tod.ndim != 2:
        raise ValueError(f"`tod` must be 2D (n_samples, n_det), got {tod.ndim}D.")

    if not np.all(np.isfinite(tod)):
        raise ValueError(
            "`tod` contains NaN or inf: the FFT of the kernel does not accept them."
        )

    if cross_spectra:
        raise NotImplementedError("The cross-spectra are not implemented yet.")

    n_samples, n_det = tod.shape

    # at least one Welch block must fit in the TOD
    if n_samples < nperseg:
        raise ValueError(
            f"The TOD has {n_samples} samples, fewer than nperseg={nperseg}."
        )

    n_freqs = nperseg // 2 + 1

    # one column of background per detector
    if background is not None and background.ndim == 1:
        background = np.broadcast_to(background[:, None], tod.shape)

    t0 = perf_counter()

    # samples left out of the estimate (background False), only for the log message
    n_total = n_samples * n_det
    n_masked = 0 if background is None else int((~background).sum())

    logger.info(
        f"[NOISE] Estimating the auto-spectra of {n_det} detectors "
        f"(n_freqs = {n_freqs}, nperseg = {nperseg})"
    )
    logger.info(
        f"[NOISE] {n_masked} of {n_total} detector samples are masked in the noise "
        f"estimate ({100 * n_masked / n_total:.1f}%)"
    )

    # one auto-spectrum per detector, in order, so that the first n_det columns of
    # inv_noiseps are the diagonal ones
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _welch_inverse_ps,
                tod[:, idet],
                sample_rate,
                nperseg,
                None if background is None else background[:, idet],
            )
            for idet in range(n_det)
        ]

    # (n_freqs, n_det): one column per detector, and its pair (j, j)
    inv_noiseps = np.column_stack([future.result() for future in futures])
    det_pairs = np.array([(idet, idet) for idet in range(n_det)], dtype=np.uintp)

    logger.info(f"[NOISE] Auto-spectra done in {perf_counter() - t0:.2f}s")
    logger.info(
        f"[NOISE] Estimation done in {perf_counter() - t0:.2f}s "
        f"(inv_noiseps shape = {inv_noiseps.shape}, "
        f"det_pairs shape = {det_pairs.shape})"
    )

    return inv_noiseps, det_pairs


def _welch_inverse_ps(
    x: NDArray[np.float64],
    sample_rate: float,
    nperseg: int,
    background: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """
    Inverse noise spectrum of one detector, with Welch's method on its whole TOD.

    The samples outside ``background`` (on the source or flagged) are replaced by a
    linear interpolation of the good ones, and the mean of the good samples is removed
    (:func:`~tolomeo._lib.spectral._prepare_for_spectrum`). Then Welch's method with
    Hann blocks of ``nperseg`` samples, and the inverse with
    :func:`_inverse_with_floor`.

    Parameters
    ----------
    x : :class:`~numpy.ndarray`
        TOD of one detector.
        Shape ``(n_samples,)``.
    sample_rate : float
        Sampling frequency in Hz.
    nperseg : int
        Samples in one Welch block.
    background : :class:`~numpy.ndarray` of bool, optional
        True for the samples used. None means all of them.
        Shape ``(n_samples,)``.

    Returns
    -------
    :class:`~numpy.ndarray`
        Inverse power spectral density, zero at ``f = 0``.
        Shape ``(nperseg // 2 + 1,)``.
    """
    x = _prepare_for_spectrum(x, background)
    _, psd = welch(x=x, fs=sample_rate, window="hann", nperseg=nperseg)

    return _inverse_with_floor(psd)


def _inverse_with_floor(psd: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Element-wise inverse of a power spectrum, with a floor and zero at ``f = 0``.

    The floor is ``_FLOOR_FACTOR`` times the median of ``|psd|``: without it a bin with
    almost no power would get an enormous weight. The sign is kept, because the real
    part of a cross power spectral density can be negative. Bin 0 is set to zero: the
    offset of the TOD gets no weight.

    ``psd`` is ``(n_freqs,)``; the result has the same shape.
    """
    floor = np.median(np.abs(psd)) * _FLOOR_FACTOR
    sign = np.where(psd < 0, -1.0, 1.0)
    inverse = sign / np.maximum(np.abs(psd), floor)
    inverse[0] = 0.0

    return inverse


def _filter_inverse(
    inv_noiseps: NDArray[np.float64],
    freqs: NDArray[np.float64],
    filters: Sequence[FrequencyFilter] | None,
    sample_rate: float,
) -> NDArray[np.float64]:
    """
    Apply ``filters`` to the inverse spectra.

    ``inv_noiseps`` is ``(n_freqs, n_cols)`` and ``freqs`` is ``(n_freqs,)``; the
    result has the shape of ``inv_noiseps``.
    """
    if not filters:
        return inv_noiseps

    total_gain = np.ones_like(freqs)
    for filt in filters:
        fc = (filt.f_lower, filt.f_upper) if isinstance(filt, Bandpass) else filt.fc
        total_gain *= gain(freqs, fc, sample_rate, filt.order, btype=filt.type)

    return inv_noiseps * total_gain[:, None]


def _marginalize(
    inv_noiseps: NDArray[np.float64],
    freqs: NDArray[np.float64],
    passband: tuple[float, float] | None,
) -> NDArray[np.float64]:
    """
    Make the inverse spectra almost zero outside ``passband``.

    Outside the band every column becomes ``_STOP_FACTOR`` times its maximum
    ``|inverse|`` inside the band (bin 0 excluded), and bin 0 becomes zero. With
    ``passband`` None nothing changes: bin 0 is already zero from
    :func:`_inverse_with_floor`.

    ``inv_noiseps`` is ``(n_freqs, n_cols)`` and ``freqs`` is ``(n_freqs,)``; the
    result has the shape of ``inv_noiseps``.
    """
    if passband is None:
        return inv_noiseps

    inv = inv_noiseps.copy()
    out = (freqs < passband[0]) | (freqs > passband[1])  # the bins the filters removed
    ref = ~out
    ref[0] = False  # bin 0 does not count for the maximum
    inv[out, :] = _STOP_FACTOR * np.abs(inv[ref, :]).max(axis=0)
    inv[0, :] = 0.0

    return inv
