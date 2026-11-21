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

from .._lib.messages import _warn
from .._lib.parallel import _resolve_workers
from .._lib.spectral import _check_nyquist, _narrow_passband, _prepare_for_spectrum
from .._signal.filters import gain
from ..map import MapGeometry
from ..processing.config import Bandpass, FrequencyFilter, Highpass, Lowpass
from ..processing.masks import Mask, RadiusMask
from ..tod import Scan
from ._pointing import _pointing_matrix

logger = logging.getLogger(__name__)

# level of the inverse spectra outside the passband, relative to their maximum inside
_STOP_FACTOR = 1e-8

# floor of a power spectrum, relative to its median: without it a bin with almost no
# power would get an enormous weight
_FLOOR_FACTOR = 1e-6

# largest fraction of the good samples that the automatic block length may leave in
# segments shorter than one block. A choice, not a standard value
_MAX_DISCARDED = 0.1

# smallest fraction of the good samples that the leave-one-out must be able to clean
# (in pixels seen by the other half): below it, most of the TOD would be interpolated,
# as with a mask. A choice, not a standard value
_MIN_COVERAGE = 0.5

__all__ = ["estimate_inverse_noise"]


class LeaveOneOutError(ValueError):
    """
    The leave-one-out cannot clean the TOD of a scan.

    Raised when the scan has a single subscan, or when its two halves do not see the
    same pixels: one scan, not two scans of the same field. The command line catches
    it and estimates the spectra without a mask.
    """


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
        of ``channel_ids``, then the cross-spectra (if any).
        Shape ``(n_freqs, n_cols)``, with ``n_freqs = nperseg // 2 + 1``.
    det_pairs : :class:`~numpy.ndarray` of uintp
        The two detectors of every column of ``inv_noiseps``. Shape ``(n_cols, 2)``.
    lag_zero : :class:`~numpy.ndarray` of float64
        Weight of every detector for the preconditioner: sum of the bins of its
        auto-spectrum divided by ``nperseg``. Shape ``(n_det,)``.
    sample_rate : float
        Sampling frequency of the TOD the spectra come from, in Hz.
    channel_ids : list of int
        Beammap id of every detector, in the order of the auto-spectra (the same as
        ``scan.channel_ids``).
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
    channel_ids: list[int]
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
        """Number of detectors (the length of ``channel_ids``)."""
        return len(self.channel_ids)

    @property
    def freqs(self) -> NDArray[np.float64]:
        """
        Frequency of every bin, in Hz.

        The bins of the real FFT of one block of ``nperseg`` samples, the same that
        ``welch`` gives with the same ``nperseg``. Shape ``(nperseg // 2 + 1,)``.
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
        scan_ids = list(scan.channel_ids)
        model_ids = list(self.channel_ids)

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
    nperseg: int | None = None,
    filters: Sequence[FrequencyFilter] = (),
    mask: Mask | None = None,
    geometry: MapGeometry | None = None,
    background: NDArray[np.bool_] | None = None,
    cross_spectra: bool = False,
    leave_one_out: bool = False,
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
        :func:`_auto_nperseg`. Defaults to ``None``.
    filters : sequence of :data:`~tolomeo.processing.FrequencyFilter`
        Filters applied to the inverse spectra, not to the TOD: ``Lowpass``,
        ``Highpass`` or ``Bandpass``, the same objects as the steps of ``process_tod``.
        Several filters are applied one after the other. Defaults to ``()``, no filter.
    mask : ``RadiusMask``, ``SigmaClipMask`` or ``EdgeMask``, optional
        Samples left out of the spectra (for example the source), the same objects as
        the ``mask`` of :class:`~tolomeo.processing.FilteringConfig`. The flagged
        samples (``scan.tod.flags`` False) are left out too. Give either ``mask`` or
        ``background``, not both.
    geometry : :class:`~tolomeo.map.MapGeometry`, optional
        Needed by a ``RadiusMask`` and by ``leave_one_out``.
    background : :class:`~numpy.ndarray` of bool, optional
        ``True`` for the samples used to estimate the spectra (far from the source),
        used as it is. None, and no ``mask``, means ``scan.tod.flags``. Shape
        ``(n_samples,)`` or ``(n_samples, n_det)``.
    cross_spectra : bool
        Also estimate the cross spectra of the detector pairs. Not implemented yet:
        True raises ``NotImplementedError``. Defaults to ``False``.
    leave_one_out : bool, optional
        Remove the source from the TOD instead of masking it: the TOD is cut in two
        halves at the boundary of subscans nearest to its middle, and each half is
        cleaned with the binned map of the other one on ``geometry`` (see
        :func:`_leave_one_out_residual`). It needs ``geometry``, at least two scans of
        the same field and a calibrated TOD (a warning if ``scan.tod.unit`` is None),
        and excludes ``mask`` and ``background``. Defaults to ``False``.
    workers : int, optional
        Maximum number of workers to use for parallel computation. If negative,
        the value wraps around from ``os.cpu_count()``.

    Returns
    -------
    :class:`InverseNoise`

    Raises
    ------
    ValueError
        If ``mask`` and ``background`` are both given, if ``leave_one_out`` is given
        with either of them or without ``geometry``, if a cutoff of ``filters`` is
        above the Nyquist frequency, if ``nperseg`` is odd, smaller than 2 or longer
        than the TOD, or if the TOD has NaN or inf.
    LeaveOneOutError
        If ``leave_one_out`` is True and the scan is not two scans of the same field
        (see :func:`_leave_one_out_residual`). A subclass of ``ValueError``.
    NotImplementedError
        If ``cross_spectra`` is True.
    """
    sample_rate = scan.tod.sample_rate
    n_det = scan.tod.signal.shape[1]
    workers = _resolve_workers(workers)

    # the arguments are checked here, before the slow estimate
    if cross_spectra:
        raise NotImplementedError("The cross-spectra are not implemented yet.")
    if mask is not None and background is not None:
        raise ValueError("Give either `background` or `mask`, not both.")
    if leave_one_out and (mask is not None or background is not None):
        raise ValueError(
            "`leave_one_out` removes the source without a mask: do not give `mask` "
            "or `background` with it."
        )
    if leave_one_out and geometry is None:
        raise ValueError(
            "`leave_one_out` needs `geometry`: the model of the sky is a binned map "
            "on that grid."
        )

    # band of the marginalization: the passband of the TOD, narrowed by every filter,
    # whose cutoffs must be below the Nyquist frequency
    passband = scan.tod.passband
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

    # samples of the estimate: the mask combined with the flags, the given background
    # as it is, or the flags alone. Shape (n_samples,) or (n_samples, n_det)
    flags = scan.tod.flags
    if mask is not None:
        args = (scan, geometry) if isinstance(mask, RadiusMask) else (scan,)
        background = mask.build(*args)
        background = background & (flags if background.ndim == 1 else flags[:, None])
    elif background is None:
        background = flags

    # automatic block length, from the runs of good samples with no gap in time. The
    # estimate below still uses the whole TOD
    if nperseg is None:
        nperseg = _auto_nperseg(scan.tod.time, flags, sample_rate)

    # the filter of gls_map needs blocks of an even number of samples
    if nperseg < 2 or nperseg % 2:
        raise ValueError(f"`nperseg` must be even and at least 2, got {nperseg}.")

    # the TOD of the estimate: the scan itself, or the scan without the source
    tod = scan.tod.signal
    if leave_one_out:
        tod, background = _leave_one_out_residual(scan, geometry, background, workers)
        if scan.tod.unit is None:
            _warn(
                "The TOD is not calibrated: the leave-one-out puts all the detectors "
                "in one map, and their different responsivities leave part of the "
                "source in the spectra. Calibrate the scan first."
            )

    inv_noiseps = _inverse_spectra(tod, sample_rate, nperseg, background, workers)
    freqs = np.fft.rfftfreq(nperseg, d=1 / sample_rate)
    inv_noiseps = _shape_inverse(inv_noiseps, freqs, filters, passband, sample_rate)

    # after the filters and the marginalization, so that the preconditioner sees the
    # same weights as N^-1
    lag_zero = inv_noiseps[:, :n_det].sum(axis=0) / nperseg

    return InverseNoise(
        inv_noiseps=inv_noiseps,
        # column j is the auto-spectrum of detector j
        det_pairs=np.array([(j, j) for j in range(n_det)], dtype=np.uintp),
        lag_zero=lag_zero,
        sample_rate=sample_rate,
        channel_ids=scan.channel_ids,
        passband=passband,
        filters=tuple(filters),
    )


def _auto_nperseg(
    time: NDArray[np.float64],
    ok: NDArray[np.bool_],
    sample_rate: float,
) -> int:
    r"""
    Block length chosen from the segments of the TOD.

    Welch's method assumes that the samples of a block are taken every
    ``1 / sample_rate``. A segment is a run of good samples with no flagged sample
    and no gap in time inside it, and a block of ``nperseg`` samples fits in a segment
    only if the segment is at least that long; otherwise it crosses a flagged sample
    or a gap in time.

    Parameters
    ----------
    time : :class:`~numpy.ndarray`
        Time of every sample, in MJD. Shape ``(n_samples,)``.
    ok : :class:`~numpy.ndarray` of bool
        True for the good samples (``scan.tod.flags``). Shape ``(n_samples,)``.
    sample_rate : float
        Sampling frequency in Hz.

    Returns
    -------
    int
        ``nperseg``, a power of two.

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

    ``nperseg`` sets the frequency resolution, ``sample_rate / nperseg``, which is the
    lowest frequency above zero in the spectra, and the longest time over which the
    filter of ``gls_map`` models the noise correlations. A longer block is therefore
    better, as long as most of the samples are in segments that can hold it.

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

    # start of every segment (the first good sample, then the first one after every
    # jump) and its stop, one after its last good sample as in a slice
    starts = np.insert(ok_idx[after_jump], 0, ok_idx[0])
    stops = np.append(ok_idx[after_jump - 1] + 1, ok_idx[-1] + 1)

    # length of every segment, in samples
    lengths = stops - starts
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


def _inverse_spectra(
    tod: NDArray[np.float64],
    sample_rate: float,
    nperseg: int,
    background: NDArray[np.bool_] | None = None,
    workers: int = 1,  # TODO: a global setting in the config, not in every signature
) -> NDArray[np.float64]:
    """
    Inverse noise spectrum of every detector, with Welch's method on its whole TOD.

    For every detector, in parallel: the samples outside ``background`` (on the source
    or flagged) are replaced by a linear interpolation of the good ones, and the mean
    of the good samples is removed
    (:func:`~tolomeo._lib.spectral._prepare_for_spectrum`); then Welch's method with
    Hann blocks of ``nperseg`` samples, and the inverse. The inverse has a floor,
    ``_FLOOR_FACTOR`` times the median of the spectrum: without it a bin with almost
    no power would get an enormous weight. Bin 0 is set to zero: the offset of the
    TOD gets no weight.

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
        Samples in one Welch block, as in the filter blocks of the kernel.
    background : :class:`~numpy.ndarray` of bool, optional
        True for the samples used: one column per detector, or one array for all of
        them. Shape ``(n_samples, n_det)`` or ``(n_samples,)``. Defaults to ``None``,
        all the samples.
    workers : int
        Threads: the detectors are estimated in parallel. Defaults to ``1``.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        One inverse spectrum per column, the detectors in order, zero at ``f = 0``.
        Shape ``(nperseg // 2 + 1, n_det)``.

    Raises
    ------
    ValueError
        If ``tod`` is not 2D, if it has NaN or inf, or if it is shorter than
        ``nperseg``.
    """
    # refuse inputs that the estimate cannot work on, before the slow part
    if tod.ndim != 2:
        raise ValueError(f"`tod` must be 2D (n_samples, n_det), got {tod.ndim}D.")

    if not np.all(np.isfinite(tod)):
        raise ValueError(
            "`tod` contains NaN or inf: the FFT of the kernel does not accept them."
        )

    n_samples, n_det = tod.shape

    # at least one Welch block must fit in the TOD
    if n_samples < nperseg:
        raise ValueError(
            f"The TOD has {n_samples} samples, fewer than nperseg={nperseg}."
        )

    # one column of background per detector
    if background is not None and background.ndim == 1:
        background = np.broadcast_to(background[:, None], tod.shape)

    t0 = perf_counter()

    # samples left out of the estimate (background False), only for the log message
    n_total = n_samples * n_det
    n_masked = 0 if background is None else int((~background).sum())

    logger.info(
        f"[NOISE] Estimating the auto-spectra of {n_det} detectors "
        f"(n_freqs = {nperseg // 2 + 1}, nperseg = {nperseg})"
    )
    logger.info(
        f"[NOISE] {n_masked} of {n_total} detector samples are masked in the noise "
        f"estimate ({100 * n_masked / n_total:.1f}%)"
    )

    def inverse_spectrum(j: int) -> NDArray[np.float64]:
        """Inverse spectrum of detector ``j``, with the floor and zero at f = 0."""
        x = _prepare_for_spectrum(
            tod[:, j], None if background is None else background[:, j]
        )
        _, psd = welch(x=x, fs=sample_rate, window="hann", nperseg=nperseg)

        # an auto-spectrum is positive: the floor keeps the inverse finite
        inverse = 1.0 / np.maximum(psd, np.median(psd) * _FLOOR_FACTOR)
        inverse[0] = 0.0
        return inverse

    # one column per detector, in order, so that column j is the spectrum of detector j
    with ThreadPoolExecutor(max_workers=workers) as executor:
        columns = list(executor.map(inverse_spectrum, range(n_det)))
    inv_noiseps = np.column_stack(columns)

    logger.info(
        f"[NOISE] Auto-spectra done in {perf_counter() - t0:.2f}s "
        f"(inv_noiseps shape = {inv_noiseps.shape})"
    )

    return inv_noiseps


def _shape_inverse(
    inv_noiseps: NDArray[np.float64],
    freqs: NDArray[np.float64],
    filters: Sequence[FrequencyFilter],
    passband: tuple[float, float] | None,
    sample_rate: float,
) -> NDArray[np.float64]:
    """
    Apply the filters, then make the inverse spectra almost zero outside the band.

    The filters multiply every column by the product of their gains. Then, outside
    ``passband``, every column becomes ``_STOP_FACTOR`` times its maximum ``|inverse|``
    inside the band (bin 0 excluded), and bin 0 becomes zero. The filters come first,
    so that the level of the marginalization is relative to the maximum of the weights
    that are really used. With no filters and ``passband`` None, ``inv_noiseps`` is
    returned as it is: bin 0 is already zero from :func:`_inverse_spectra`.

    Parameters
    ----------
    inv_noiseps : :class:`~numpy.ndarray` of float64
        Inverse spectra, one per column. Shape ``(n_freqs, n_cols)``.
    freqs : :class:`~numpy.ndarray` of float64
        Frequency of every bin, in Hz. Shape ``(n_freqs,)``.
    filters : sequence of :data:`~tolomeo.processing.FrequencyFilter`
        Filters of the inverse spectra; empty for none.
    passband : 2-tuple of float or None
        Band of the marginalization, in Hz; None for no marginalization.
    sample_rate : float
        Sampling frequency in Hz.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        The shaped spectra, a new array unless nothing changes.
        Shape ``(n_freqs, n_cols)``.
    """
    inv = inv_noiseps

    if filters:
        total_gain = np.ones_like(freqs)
        for filt in filters:
            fc = (filt.f_lower, filt.f_upper) if isinstance(filt, Bandpass) else filt.fc
            total_gain *= gain(freqs, fc, sample_rate, filt.order, btype=filt.type)
        inv = inv * total_gain[:, None]

    if passband is not None:
        if inv is inv_noiseps:
            inv = inv.copy()  # the input is not modified
        # the bins that the filters removed
        out = (freqs < passband[0]) | (freqs > passband[1])
        ref = ~out
        ref[0] = False  # bin 0 does not count for the maximum
        inv[out, :] = _STOP_FACTOR * np.abs(inv[ref, :]).max(axis=0)
        inv[0, :] = 0.0

    return inv


def _leave_one_out_residual(
    scan: Scan,
    geometry: MapGeometry,
    background: NDArray[np.bool_],
    workers: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """
    TOD without the source, for the spectra: each half cleaned with the other one.

    The TOD is cut at the boundary of subscans nearest to its middle, into A and B.
    The binned map of each half on ``geometry``, with all the detectors together, is a
    model of the sky without the noise of the other half: ``r_A = d_A - P_A m_B`` and
    ``r_B = d_B - P_B m_A`` keep the noise and lose the source. A sample is usable if
    it is in ``background``, inside the map and in a pixel that the other half sees;
    the others get 0 in the residual and False in the mask, and the spectral estimate
    interpolates them.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        Cleaned and calibrated scan: all the detectors go into the same maps, so they
        must have the same units and sign.
    geometry : :class:`~tolomeo.map.MapGeometry`
        Grid of the maps, in a frame where the sky does not move from one scan to the
        next.
    background : :class:`~numpy.ndarray` of bool
        True for the good samples. Shape ``(n_samples,)`` or ``(n_samples, n_det)``.
    workers : int
        Threads of the pointing. Defaults to ``1``.

    Returns
    -------
    residual : :class:`~numpy.ndarray` of float64
        The TOD minus the model of the other half, 0 where the sample is not usable.
        Shape ``(n_samples, n_det)``.
    usable : :class:`~numpy.ndarray` of bool
        True for the samples that the spectra can use. Shape ``(n_samples, n_det)``.

    Raises
    ------
    LeaveOneOutError
        If the scan has a single subscan, or if fewer than ``_MIN_COVERAGE`` of the
        good samples fall in pixels that the other half sees: the two halves are not
        two scans of the same field.
    """
    signal = scan.tod.signal
    n_samples = signal.shape[0]
    n_pixels = geometry.n_pixels

    # boundary of subscans nearest to the middle. The first and the last boundaries
    # are 0 and n_samples: one of them is the nearest only with a single subscan
    segments = scan.tod.segments
    half = int(segments[np.argmin(np.abs(segments - n_samples / 2))])
    if half in (0, n_samples):
        raise LeaveOneOutError(
            "The scan has a single subscan: the leave-one-out needs at least two "
            "scans of the same field."
        )

    pixels, inside = _pointing_matrix(
        scan.coordinates, scan.offsets, geometry, workers=workers
    )
    if background.ndim == 1:
        background = background[:, None]
    ok = inside & background

    pixels_a, pixels_b = pixels[:half], pixels[half:]
    signal_a, signal_b = signal[:half], signal[half:]
    ok_a, ok_b = ok[:half], ok[half:]

    # binned map of each half: the sum of its samples in every pixel over their
    # number, NaN where it has none
    hits_a = np.bincount(pixels_a[ok_a], minlength=n_pixels)
    hits_b = np.bincount(pixels_b[ok_b], minlength=n_pixels)
    sum_a = np.bincount(pixels_a[ok_a], weights=signal_a[ok_a], minlength=n_pixels)
    sum_b = np.bincount(pixels_b[ok_b], weights=signal_b[ok_b], minlength=n_pixels)
    map_a = np.divide(sum_a, hits_a, where=hits_a > 0, out=np.full_like(sum_a, np.nan))
    map_b = np.divide(sum_b, hits_b, where=hits_b > 0, out=np.full_like(sum_b, np.nan))

    # each half minus the map of the other one, read in its pixels; a sample whose
    # pixel the other half does not see has no model
    usable_a = ok_a & (hits_b[pixels_a] > 0)
    usable_b = ok_b & (hits_a[pixels_b] > 0)
    res_a = np.where(usable_a, signal_a - map_b[pixels_a], 0.0)
    res_b = np.where(usable_b, signal_b - map_a[pixels_b], 0.0)

    # about 1 for two scans of the same field, about 0 for one raster cut in two
    coverage = (usable_a.sum() + usable_b.sum()) / max(ok.sum(), 1)
    logger.info(
        f"[NOISE] Leave-one-out: {100 * coverage:.1f}% of the good samples are in "
        "pixels seen by the other half"
    )
    if coverage < _MIN_COVERAGE:
        raise LeaveOneOutError(
            f"Only {coverage:.0%} of the good samples fall in pixels seen by the other "
            "half: the leave-one-out needs at least two scans of the same field."
        )

    return np.concatenate([res_a, res_b]), np.concatenate([usable_a, usable_b])
