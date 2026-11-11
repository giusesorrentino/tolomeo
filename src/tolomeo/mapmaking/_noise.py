"""
Noise: the model estimated from the TOD and the filter ``N^-1`` built from it.

The module has two parts, in the order in which the pipeline uses them.
``estimate_inverse_noise`` estimates the inverse noise spectra of the detectors from the
cleaned TOD; ``InverseNoiseFilter`` applies ``N^-1``, built from those spectra, to the
blocks of the TOD.

The noise model
===============

Inverse noise power spectra: the noise model that the map-making kernel needs.

The kernel filters the TOD in the frequency domain, multiplying the Fourier transform of
every detector by the *inverse* of its noise power spectrum (PS): the noisier a frequency,
the less weight it gets. This module estimates those spectra from the cleaned TOD and
packs them in the three arrays that ``gls_map`` passes to the kernel:

``inv_noiseps``
    shape ``(n_freqs, n_cols)``, one column per spectrum.
``det_pairs``
    shape ``(n_cols, 2)``, the two detectors of every column.
``lag_zero``
    shape ``(n_det,)``, used by the preconditioner of the solver.

A column is an auto-spectrum when its pair is ``(j, j)`` and a cross-spectrum when it is
``(i, j)`` with ``i != j``. Without cross-spectra ``n_cols = n_det``.

The inverse noise filter
========================

Inverse noise filter ``N^-1`` applied to one block.

What the filter does to one block of all the detectors
-------------------------------------------------------
1. real FFT of every detector's ``block_len`` samples -> ``n_freqs`` complex bins;
2. multiplication by the inverse noise spectra. In general this mixes the detectors: for
   every column ``i`` of ``inv_noiseps`` (a detector pair ``(f, r)``) and every
   frequency ``k``:  ``out[f, k] += spectrum[r, k] * inv_noiseps[k, i]``  and, if
   ``f != r``,       ``out[r, k] += spectrum[f, k] * inv_noiseps[k, i]``;
3. inverse real FFT of ``out`` back to ``block_len`` samples per detector.

With only auto spectra (``f == r``, one column per detector) step 2 is just "multiply
each detector's spectrum by its own weights". This is the only case supported here.

Array shapes
------------
* ``inv_noiseps`` has shape ``(n_freqs, n_cols)``.
* ``det_pairs`` has shape ``(n_cols, 2)``.
* a block is ``(n_det, block_len)``: one row per detector.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.fft import irfft, rfft
from scipy.signal import welch

from .._lib.parallel import _resolve_workers
from .._lib.spectral import _check_nyquist, _narrow_passband, _prepare_for_spectrum
from .._signal.filters import _gain
from ..map import MapGeometry
from ..processing._mask import _build_mask
from ..processing.config import Bandpass, FrequencyFilter, Highpass, Lowpass, Mask
from ..tod import Scan
from ._blocks import BlockLayout

_STOP_FACTOR = 1e-8

__all__ = ["estimate_inverse_noise"]


def _build_noise_model(
    tod: NDArray[np.float64],
    sample_rate: float,
    n_freqs: int = 4097,
    background: NDArray[np.bool_] | None = None,
    cross_spectra: bool = False,
    workers: int = 1,  # TODO: in config, mi serve un parametro globale così che non devo aggiungerlo nella firma di ogni funzione
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
        Sampling frequency in Hz, see :func:`_welch_inverse_ps`.
    n_freqs : int
        Number of bins of every spectrum.
        Default ``4097``.
    background : :class:`~numpy.ndarray` of bool, optional
        True for the samples far from the source: one column per detector, or one
        array for all of them.
        Shape ``(n_samples, n_det)`` or ``(n_samples,)``.
    cross_spectra : bool
        If True, add one cross-spectrum for every pair of detectors, ``n_det * (n_det -
        1) / 2`` more columns. Computing them is slow for many detectors: with 236 it
        means 27730 cross-spectra and about 0.9 GB for ``inv_noiseps``.
        Default ``False``.

    Returns
    -------
    inv_noiseps : :class:`~numpy.ndarray`
        One inverse spectrum per column. The first ``n_det`` columns are the
        auto-spectra, one per detector in order. With ``cross_spectra`` the cross-spectra
        follow.
        Shape ``(n_freqs, n_cols)``.
    det_pairs : :class:`~numpy.ndarray` of uintp
        Row ``k`` is the pair of detectors of column ``k`` of ``inv_noiseps``:
        ``(j, j)`` for the auto-spectra, ``(i, j)`` with ``i > j`` for the cross ones.
        Shape ``(n_cols, 2)``.

    Notes
    -----
    A cross-spectrum is stored as the element-wise inverse of the real part of the
    cross power spectral density, ``1 / Re(csd)``.
    """
    # refuse inputs that the estimate cannot work on, before the slow part
    if tod.ndim != 2:
        raise ValueError(f"`tod` must be 2D (n_samples, n_det), got {tod.ndim}D.")

    if not np.all(np.isfinite(tod)):
        raise ValueError(
            "`tod` contains NaN or inf: the FFT of the kernel does not accept them."
        )

    n_samples, n_det = tod.shape

    # length of a Welch segment, which is also the length of the FFT blocks of the kernel;
    # at least one segment must fit in the TOD
    nperseg = 2 * (n_freqs - 1)

    if n_samples < nperseg:
        raise ValueError(
            f"The TOD has {n_samples} samples, fewer than the {nperseg} needed for "
            f"n_freqs={n_freqs}."
        )

    # one column of background per detector
    if background is not None and background.ndim == 1:
        background = np.broadcast_to(background[:, None], tod.shape)

    t0 = time.perf_counter()

    # samples left out of the estimate (background False), only for the log message
    n_total = n_samples * n_det
    n_masked = 0 if background is None else int((~background).sum())

    print(
        f"[NOISE] Estimating the auto-spectra of {n_det} detectors "
        f"(n_freqs = {n_freqs}, nperseg = {nperseg})"
    )
    print(
        f"[NOISE] {n_masked} of {n_total} detector samples are masked in the noise "
        f"estimate ({100 * n_masked / n_total:.1f}%)"
    )

    # one entry per spectrum: the column of inv_noiseps and the pair of detectors
    columns = []
    pairs = []

    def _welch_inverse_ps(
        tod: NDArray[np.float64],
        sample_rate: float,
        n_freqs: int = 4097,
        background: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float64]:
        nperseg = 2 * (n_freqs - 1)
        tod = _prepare_for_spectrum(tod, background)
        _, psd = welch(x=tod, fs=sample_rate, window="hann", nperseg=nperseg)

        floor = np.median(np.abs(psd)) * 1e-6
        sign = np.where(psd < 0, -1.0, 1.0)
        inverse = sign / np.maximum(np.abs(psd), floor)
        inverse[0] = 0.0

        return inverse

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idet in range(n_det):
            # auto-spectra: always, one per detector, so that the first n_det columns
            # are the diagonal ones
            futures.append(
                executor.submit(
                    _welch_inverse_ps,
                    tod[:, idet],
                    sample_rate,
                    n_freqs,
                    None if background is None else background[:, idet],
                )
            )
            pairs.append((idet, idet))

    for future in futures:
        columns.append(future.result())

    print(f"[NOISE] Auto-spectra done in {time.perf_counter() - t0:.2f}s")

    if cross_spectra:
        raise NotImplementedError
        # n_pairs = n_det * (n_det - 1) // 2
        # gb = n_freqs * (n_det + n_pairs) * 8 / 1e9
        # print(
        #     f"[NOISE] Estimating the cross-spectra: {n_pairs} pairs, "
        #     f"inv_noiseps will take about {gb:.2f} GB"
        # )
        # t1 = time.perf_counter()

        # # the cross-spectrum of a pair needs the same preparation as the auto-spectra,
        # # done once for all the detectors and not once for every pair
        # prepared = np.column_stack(
        #     [
        #         _prepare_for_spectrum(
        #             tod[:, j], None if background is None else background[:, j]
        #         )
        #         for j in range(n_det)
        #     ]
        # )

        # # progress: one message every ~10% of the pairs
        # k = 0
        # step = max(1, n_pairs // 10)

        # # every pair (i, j) with j < i, the lower triangle like the ROMA files
        # for i in range(n_det):
        #     for j in range(i):
        #         _, cross_psd = csd(
        #             prepared[:, i],
        #             prepared[:, j],
        #             fs=sample_rate,
        #             window="hann",
        #             nperseg=nperseg,
        #         )

        #         # csd is complex; the kernel uses a real weight, so we keep the real part
        #         columns.append(_inverse_with_floor(cross_psd.real))
        #         pairs.append((i, j))

        #         k += 1
        #         if k % step == 0:
        #             print(
        #                 f"[NOISE] cross-spectra {100 * k / n_pairs:3.0f}% "
        #                 f"({k}/{n_pairs} pairs)"
        #             )

        # print(f"[NOISE] Cross-spectra done in {time.perf_counter() - t1:.2f}s")

    # (n_freqs, n_cols): one column per spectrum
    inv_noiseps = np.column_stack(columns)
    det_pairs = np.array(pairs, dtype=np.uintp)

    print(
        f"[NOISE] Estimation done in {time.perf_counter() - t0:.2f}s "
        f"(inv_noiseps shape = {inv_noiseps.shape}, det_pairs shape = {det_pairs.shape})"
    )

    return inv_noiseps, det_pairs


class InverseNoise(BaseModel):
    """
    Noise model of the GLS map: the inverse noise spectra of the detectors.

    Output of :func:`estimate_inverse_noise` and input of
    :func:`~tolomeo.mapmaking.gls_map`. It belongs to the scan it was estimated from:
    ``gls_map`` checks that the detectors and the sampling frequency are the same.

    Attributes
    ----------
    inv_noiseps : :class:`~numpy.ndarray` of float64
        One inverse spectrum per column: the ``n_det`` auto-spectra first, in the order
        of ``detector_ids``, then the cross-spectra (if any).
        Shape ``(n_freqs, n_cols)``.
    det_pairs : :class:`~numpy.ndarray` of uintp
        The two detectors of every column of ``inv_noiseps``.
        Shape ``(n_cols, 2)``.
    lag_zero : :class:`~numpy.ndarray` of float64
        Weight of every detector for the preconditioner: sum of the bins of its
        auto-spectrum divided by ``2 * (n_freqs - 1)``.
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
    def n_freqs(self) -> int:
        """Number of bins of every spectrum (the rows of ``inv_noiseps``)."""
        return self.inv_noiseps.shape[0]

    @property
    def n_det(self) -> int:
        """Number of detectors (the length of ``detector_ids``)."""
        return len(self.detector_ids)

    @property
    def freqs(self) -> NDArray[np.float64]:
        """
        Frequency of every bin, in Hz.

        The bins of the real FFT of one block of ``2 * (n_freqs - 1)`` samples, the same
        that ``welch`` gives with ``nperseg`` equal to the block length.
        Shape ``(n_freqs,)``.
        """
        return np.fft.rfftfreq(2 * (self.n_freqs - 1), d=1 / self.sample_rate)


def estimate_inverse_noise(
    scan: Scan,
    *,
    n_freqs: int = 4097,
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
        Cleaned scan, the same that goes to ``gls_map``: ``scan.tod.signal`` without NaN
        or inf.
    n_freqs : int
        Bins of the noise spectra. The filter blocks of ``gls_map`` have
        ``2 * (n_freqs - 1)`` samples.
        Default ``4097``.
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
        Also estimate the cross spectra of the detector pairs. Keep False: their use in
        the operator is an open question, and the Python backend does not support them.
        Default ``False``.
    workers : int, optional
        Threads, with the convention of ``gls_map``: None is one thread, a positive
        number that many, ``-1`` all the cores.

    Returns
    -------
    :class:`InverseNoise`
    """
    sample_rate = scan.tod.sample_rate
    workers = _resolve_workers(workers)

    if mask is not None and background is not None:
        raise ValueError("Give either `background` or `mask`, not both.")
    if mask is not None:
        # (n_samples,) or (n_samples, n_det), depending on the mask
        background = _build_mask(scan, mask, geometry)
        flags = scan.tod.flags if background.ndim == 1 else scan.tod.flags[:, None]
        background = background & flags
    elif background is None:
        background = scan.tod.flags

    # band of the marginalization: the band of the filters already applied to the TOD
    # (None if there is none), narrowed by every filter of `filters`. The cutoffs are
    # checked here, before the slow estimate
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

    inv_noiseps, det_pairs = _build_noise_model(
        scan.tod.signal,
        sample_rate,
        n_freqs=n_freqs,
        background=background,
        cross_spectra=cross_spectra,
        workers=workers,
    )
    freqs = np.fft.rfftfreq(2 * (n_freqs - 1), d=1 / sample_rate)

    # the filters first: the level of the marginalization is relative to the maximum of
    # the weights that are really used
    inv_noiseps = _filter_inverse(inv_noiseps, freqs, filters, sample_rate)
    inv_noiseps = _marginalize(inv_noiseps, freqs, passband)

    # after the filters and the marginalization, so that the preconditioner sees the
    # same weights as N^-1
    n_det = scan.tod.signal.shape[1]
    lag_zero = inv_noiseps[:, :n_det].sum(axis=0) / (2 * (n_freqs - 1))

    return InverseNoise(
        inv_noiseps=inv_noiseps,
        det_pairs=det_pairs,
        lag_zero=lag_zero,
        sample_rate=sample_rate,
        detector_ids=scan.detector_ids,
        passband=passband,
        filters=tuple(filters),
    )


def _filter_inverse(
    inv_noiseps: NDArray[np.float64],
    freqs: NDArray[np.float64],
    filters: Sequence[FrequencyFilter] | None,
    sample_rate: float,
) -> NDArray[np.float64]:
    """
    Apply ``filters`` to the inverse spectra.

    ``inv_noiseps`` is ``(n_freqs, n_cols)`` and ``freqs`` is ``(n_freqs,)``; the result
    has the shape of ``inv_noiseps``.
    """
    if not filters:
        return inv_noiseps

    gain = np.ones_like(freqs)
    for filt in filters:
        fc = (filt.f_lower, filt.f_upper) if isinstance(filt, Bandpass) else filt.fc
        gain *= _gain(freqs, fc, sample_rate, filt.order, btype=filt.type)

    return inv_noiseps * gain[:, None]


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

    ``inv_noiseps`` is ``(n_freqs, n_cols)`` and ``freqs`` is ``(n_freqs,)``; the result
    has the shape of ``inv_noiseps``.
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


class InverseNoiseFilter:
    """``N^-1`` as an operator on blocks."""

    def __init__(
        self,
        layout: BlockLayout,
        weights: NDArray[np.float64],
    ) -> None:
        """
        Keep the block layout and the weights.

        Parameters
        ----------
        layout : :class:`~tolomeo.mapmaking._blocks.BlockLayout`
            Sizes of the TOD and of the blocks.
        weights : :class:`~numpy.ndarray` of float64
            Inverse noise spectrum of every detector (from ``_build_weights``).
            Shape ``(n_det, n_freqs)``.
        """
        self.layout = layout
        self.weights = weights  # (n_det, n_freqs), from _build_weights

    @classmethod
    def from_inv_noiseps(
        cls,
        inv_noiseps: NDArray[np.float64],
        det_pairs: NDArray[np.integer],
        layout: BlockLayout,
    ) -> InverseNoiseFilter:
        """
        Build the filter from the arrays that ``gls_map`` passes to the kernel.

        ``inv_noiseps`` is ``(n_freqs, n_cols)`` and ``det_pairs`` is ``(n_cols, 2)``;
        only auto spectra are supported (see ``_build_weights``).
        """
        n_det = layout.n_det
        n_freqs = inv_noiseps.shape[0]
        weights = np.empty((n_det, n_freqs))
        n_seen = np.zeros(n_det, dtype=int)  # how many columns claim each detector

        # every column of inv_noiseps is the spectrum of the pair (f, r) = det_pairs[i]
        for i, (f, r) in enumerate(det_pairs):
            if f != r:
                raise NotImplementedError(
                    f"column {i} is a cross spectrum (detectors {f} and {r}): "
                    "only auto spectra (j, j) are supported"
                )
            if f >= n_det:
                raise ValueError(
                    f"column {i} refers to detector {f}, but n_det = {n_det}"
                )
            weights[f] = inv_noiseps[:, i]
            n_seen[f] += 1

        # np.empty left the rows uninitialised: every detector must have been written once,
        # otherwise a row would hold garbage (missing) or the last column wins (repeated)
        if not np.all(n_seen == 1):
            missing = np.flatnonzero(n_seen == 0).tolist()
            repeated = np.flatnonzero(n_seen > 1).tolist()
            raise ValueError(
                f"Every detector needs exactly one auto spectrum: "
                f"missing = {missing}, repeated = {repeated}"
            )

        return cls(
            layout=layout,
            weights=weights,
        )

    def filter_block(self, time_buf: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply ``N^-1`` to one block, shape ``(n_det, block_len)`` -> same shape.

        Steps: ``scipy.fft.rfft`` along the time axis, multiplication by the weights,
        ``scipy.fft.irfft(..., n=block_len)``.

        * **Normalisation.** ``scipy.fft.irfft`` already divides by ``n``, so the weights
          are used as they are. With all the weights equal to 1 the filter returns its
          input.
        * **The input buffer is overwritten** (``overwrite_x=True``, to save memory): give
          a buffer that you own and do not use it afterwards.
        * **One FFT worker.** The parallelism is across blocks (threads in
          ``_filter_and_bin``); if every FFT also asked for all the cores, the threads would
          compete for them.
        """
        # TODO: raise ValueError if time_buf.shape is not (n_det, block_len).

        # spectrum of every detector: (n_det, block_len) -> (n_det, n_freqs), complex
        freq_buf = rfft(time_buf, axis=-1, workers=1, overwrite_x=True)

        # multiply each detector's spectrum by its own weights
        freq_buf *= self.weights

        # back to time: (n_det, n_freqs) -> (n_det, block_len). The length is given
        # explicitly, so that it never depends on how irfft guesses it from n_freqs
        return irfft(
            freq_buf, n=time_buf.shape[1], axis=-1, workers=1, overwrite_x=True
        )
