"""
Noise: the model estimated from the TOD and the filter ``N^-1`` built from it.

The module has two parts, in the order in which the pipeline uses them.
``_build_noise_model`` estimates the inverse noise spectra of the detectors from the
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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, rfft
from scipy.signal import csd, welch

from ._blocks import BlockLayout

_STOP_FACTOR = 1e-8


def _prepare_for_spectrum(
    tod: NDArray[np.float64],
    background: NDArray[np.bool_] | None,
) -> NDArray[np.float64]:
    """
    Return a copy of ``tod`` ready for the spectral estimate.

    The masked samples (``background`` False, for instance the ones on a bright source)
    are replaced by a straight line through the good samples, and the mean of the good
    samples is subtracted. ``tod`` itself is not modified.

    Parameters
    ----------
    tod : :class:`~numpy.ndarray`
        TOD of one detector.
        Shape ``(n_samples,)``.
    background : :class:`~numpy.ndarray` of bool, optional
        True for the good samples. None means all the samples are good.
        Shape ``(n_samples,)``.
    """
    if background is None:
        background = np.ones(tod.shape, dtype=bool)

    if background.shape != tod.shape:
        raise ValueError("`background` must have the same shape as `tod`.")

    if not background.any():
        raise ValueError("Every sample is masked: cannot estimate the noise spectrum.")

    # work on a copy: the array we received must not change
    tod = tod.copy()

    source = ~background

    # np.interp returns only the values at the masked samples (the first argument),
    # so we write them back into tod, in the masked positions
    t = np.arange(tod.size)
    tod[source] = np.interp(t[source], t[background], tod[background])

    # the mean is computed on the good samples only
    tod -= tod[background].mean()

    return tod


def _inverse_with_floor(spectrum: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Element-wise inverse of a spectrum, with a floor so that no value explodes.

    A bin of the spectrum that is almost zero would give a huge (or infinite) weight and
    spoil the filter. Every value whose modulus is below ``median(|spectrum|) * 1e-6`` is
    raised to that modulus, keeping its sign, and only then inverted. The sign matters
    for cross-spectra, whose real part can be negative. The bin 0 (the constant
    component) is set to zero: the filter ignores it.

    Parameters
    ----------
    spectrum : :class:`~numpy.ndarray`
        Auto-spectrum (always positive) or real part of a cross-spectrum.
        Shape ``(n_freqs,)``.
    """
    floor = np.median(np.abs(spectrum)) * 1e-6

    sign = np.where(spectrum < 0, -1.0, 1.0)
    inverse = sign / np.maximum(np.abs(spectrum), floor)

    inverse[0] = 0.0

    return inverse


def _welch_inverse_ps(
    tod: NDArray[np.float64],
    sample_rate: float,
    n_freqs: int = 4097,
    background: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """
    Inverse noise power spectrum of one detector, estimated with Welch's method.

    Parameters
    ----------
    tod : :class:`~numpy.ndarray`
        TOD of one detector, cleaned and without NaN.
        Shape ``(n_samples,)``.
    sample_rate : float
        Sampling frequency in Hz. It only changes the scale of the spectrum, which
        the solver does not care about, but pass 244.138766 to match ROMA.
    n_freqs : int
        Number of bins of the spectrum. The FFT blocks of the kernel are
        ``2 * (n_freqs - 1)`` samples long.
        Default ``4097``.
    background : :class:`~numpy.ndarray` of bool, optional
        True for the samples far from the source. The others are filled with a straight
        line before the estimate. None means all the samples are good.
        Shape ``(n_samples,)``.

    Returns
    -------
    :class:`~numpy.ndarray`
        The inverse spectrum, with the floor of :func:`_inverse_with_floor` and bin 0
        set to zero.
        Shape ``(n_freqs,)``.
    """
    # Welch averages the spectrum of segments of nperseg samples
    nperseg = 2 * (n_freqs - 1)

    if len(tod) < nperseg:
        raise ValueError(
            f"The TOD has {len(tod)} samples, fewer than the {nperseg} needed for "
            f"n_freqs={n_freqs}."
        )

    tod = _prepare_for_spectrum(tod, background)

    _, psd = welch(x=tod, fs=sample_rate, window="hann", nperseg=nperseg)

    return _inverse_with_floor(spectrum=psd)


def _build_noise_model(
    tod: NDArray[np.float64],
    sample_rate: float,
    n_freqs: int = 4097,
    background: NDArray[np.bool_] | None = None,
    cross_spectra: bool = False,
    workers: int = 1,  # TODO: in config, mi serve un parametro globale così che non devo aggiungerlo nella firma di ogni funzione
    passband: tuple[float, float] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.uintp], NDArray[np.float64]]:
    """
    Noise model of all the detectors, in the form the kernel expects.

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
        True for the samples far from the source, the same for all the detectors.
        Shape ``(n_samples,)``.
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
    lag_zero : :class:`~numpy.ndarray`
        Sum of the bins of the auto-spectrum of every detector divided by
        ``2 * (n_freqs - 1)``.
        Shape ``(n_det,)``.

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

    t0 = time.perf_counter()

    # samples left out of the estimate (background False), only for the log message
    n_masked = 0 if background is None else int((~background).sum())

    print(
        f"[NOISE] Estimating the auto-spectra of {n_det} detectors "
        f"(n_freqs = {n_freqs}, nperseg = {nperseg})"
    )
    print(
        f"[NOISE] {n_masked} of {n_samples} samples are masked in the noise estimate "
        f"({100 * n_masked / n_samples:.1f}%)"
    )

    # one entry per spectrum: the column of inv_noiseps and the pair of detectors
    columns = []
    pairs = []

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idet in range(n_det):
            # auto-spectra: always, one per detector, so that the first n_det columns
            # are the diagonal ones
            futures.append(
                executor.submit(
                    _welch_inverse_ps, tod[:, idet], sample_rate, n_freqs, background
                )
            )
            pairs.append((idet, idet))

    for future in futures:
        columns.append(future.result())

    # auto-spectra: always, one per detector, so that the first n_det columns are the
    # diagonal ones
    # for j in range(n_det):
    #     columns.append(_welch_inverse_ps(tod[:, j], sample_rate, n_freqs, background))
    #     pairs.append((j, j))

    print(f"[NOISE] Auto-spectra done in {time.perf_counter() - t0:.2f}s")

    if cross_spectra:
        n_pairs = n_det * (n_det - 1) // 2
        gb = n_freqs * (n_det + n_pairs) * 8 / 1e9
        print(
            f"[NOISE] Estimating the cross-spectra: {n_pairs} pairs, "
            f"inv_noiseps will take about {gb:.2f} GB"
        )
        t1 = time.perf_counter()

        # the cross-spectrum of a pair needs the same preparation as the auto-spectra,
        # done once for all the detectors and not once for every pair
        prepared = np.column_stack(
            [_prepare_for_spectrum(tod[:, j], background) for j in range(n_det)]
        )

        # progress: one message every ~10% of the pairs
        k = 0
        step = max(1, n_pairs // 10)

        # every pair (i, j) with j < i, the lower triangle like the ROMA files
        for i in range(n_det):
            for j in range(i):
                _, cross_psd = csd(
                    prepared[:, i],
                    prepared[:, j],
                    fs=sample_rate,
                    window="hann",
                    nperseg=nperseg,
                )

                # csd is complex; the kernel uses a real weight, so we keep the real part
                columns.append(_inverse_with_floor(cross_psd.real))
                pairs.append((i, j))

                k += 1
                if k % step == 0:
                    print(
                        f"[NOISE] cross-spectra {100 * k / n_pairs:3.0f}% "
                        f"({k}/{n_pairs} pairs)"
                    )

        print(f"[NOISE] Cross-spectra done in {time.perf_counter() - t1:.2f}s")

    # (n_freqs, n_cols): one column per spectrum, C-contiguous because the Rust kernel reads
    # the arrays as flat blocks of memory
    inv_noiseps = np.ascontiguousarray(np.column_stack(columns))
    det_pairs = np.ascontiguousarray(np.array(pairs, dtype=np.uintp))
    
    if passband is not None:
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / sample_rate)    # le frequenze dei bin, le stesse di welch
        out = (freqs < passband[0]) | (freqs > passband[1])      # i bin che il filtro ha tolto
        ref = ~out
        ref[0] = False                                           # il bin 0 non conta per il massimo
        inv_noiseps[out, :] = _STOP_FACTOR * np.abs(inv_noiseps[ref, :]).max(axis=0)
        inv_noiseps[0, :] = 0.0

    # one weight per detector, from its auto-spectrum only (the first n_det columns): the
    # sum of the bins divided by the block length. The diagonal preconditioner multiplies
    # the number of samples of each pixel by it
    lag_zero = np.ascontiguousarray(
        inv_noiseps[:, :n_det].sum(axis=0) / (2 * (n_freqs - 1))
    )

    print(
        f"[NOISE] Estimation done in {time.perf_counter() - t0:.2f}s "
        f"(inv_noiseps shape = {inv_noiseps.shape}, det_pairs shape = {det_pairs.shape})"
    )

    return inv_noiseps, det_pairs, lag_zero


def _build_weights(
    inv_noiseps: NDArray[np.float64],  # shape (n_freqs, n_cols)
    det_pairs: NDArray[np.integer],  # shape (n_cols, 2)
    n_det: int,
) -> NDArray[np.float64]:
    """
    Auto-spectra weights as one array of shape ``(n_det, n_freqs)``.

    Row ``j`` is the column of ``inv_noiseps`` whose pair is ``(j, j)``.

    Raise ``ValueError`` if a detector has no auto spectrum, or more than one, or if a
    pair refers to a detector that does not exist. Raise ``NotImplementedError`` for a
    cross spectrum (a pair with ``f != r``): supporting it means the symmetric update of
    step 2 above, which does not vectorise like the auto spectra, so it needs its own
    design.
    """
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
            raise ValueError(f"column {i} refers to detector {f}, but n_det = {n_det}")
        weights[f] = inv_noiseps[:, i]
        n_seen[f] += 1

    # np.empty left the rows uninitialised: every detector must have been written once,
    # otherwise a row would hold garbage (missing) or the last column wins (repeated)
    if not np.all(n_seen == 1):
        missing = np.flatnonzero(n_seen == 0).tolist()
        repeated = np.flatnonzero(n_seen > 1).tolist()
        raise ValueError(
            f"every detector needs exactly one auto spectrum: "
            f"missing = {missing}, repeated = {repeated}"
        )
    return weights


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
        weights = _build_weights(
            inv_noiseps=inv_noiseps, det_pairs=det_pairs, n_det=layout.n_det
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
