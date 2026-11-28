# Originally adapted from:
# https://github.com/thomaswmorris/maria/blob/master/maria/plotting/tod.py
# Original implementation by Thomas W. Morris (Maria).

from itertools import pairwise
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.fft import next_fast_len
from scipy.signal import periodogram, welch

from .._lib.messages import _warn
from .._lib.spectral import _prepare_for_spectrum
from ..mapmaking._noise import _auto_nperseg
from ..tod import Scan, TimeOrderedData
from ..tod._scan import _JUMP_FACTOR

__all__ = ["plot_psd"]

STYLE_FILE = Path(__file__).resolve().parent / "resources" / "default.mplstyle"

# channels per call of welch or periodogram: welch holds all the blocks of a stretch
# for every channel of the call, so a few at a time keep the memory bounded
_BATCH = 16


def _stretches(tod: TimeOrderedData) -> list[tuple[int, int]]:
    """
    Stretches of the TOD with no jump in time and inside one chunk, as ``(a, b)``.

    A jump is a time step longer than ``_JUMP_FACTOR`` sampling steps, as between two
    subscans of a TOD that is not patched. Every chunk boundary ends a stretch too.
    """
    seconds = (tod.time - tod.time[0]) * 86400
    jumps = np.flatnonzero(np.diff(seconds) > _JUMP_FACTOR / tod.sample_rate) + 1
    edges = np.union1d(jumps, tod.chunks)
    return [(int(a), int(b)) for a, b in pairwise(edges) if b > a]


def _prepared(
    signal: NDArray[np.float64], good: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """
    Columns of one stretch ready for the spectrum, and which columns can be used.

    Every column with at least one good sample goes through
    :func:`~tolomeo._lib.spectral._prepare_for_spectrum` (the other samples
    interpolated, the mean of the good ones removed); a column with none is left at
    zero and marked unusable.
    """
    usable = good.any(axis=0)
    out = np.zeros_like(signal)
    for c in np.flatnonzero(usable):
        out[:, c] = _prepare_for_spectrum(signal[:, c], good[:, c])
    return out, usable


def _welch_psd(
    signal: NDArray[np.float64],
    good: NDArray[np.bool_],
    stretches: list[tuple[int, int]],
    sample_rate: float,
    nperseg: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Welch spectrum of every column, on every stretch with at least ``nperseg``
    samples; the stretches averaged with their number of blocks as weight.
    """
    n_det = signal.shape[1]
    weighted = np.zeros((nperseg // 2 + 1, n_det))
    n_blocks = np.zeros(n_det)
    for a, b in stretches:
        if b - a < nperseg:
            continue  # no whole block in this stretch
        # blocks of welch in the stretch, with its default overlap of half a block
        k = (b - a - nperseg) // (nperseg // 2) + 1
        for first in range(0, n_det, _BATCH):
            cols = slice(first, first + _BATCH)
            x, usable = _prepared(signal[a:b, cols], good[a:b, cols])
            _, psd = welch(x, fs=sample_rate, window="hann", nperseg=nperseg, axis=0)
            weighted[:, cols] += k * psd * usable
            n_blocks[cols] += k * usable

    if not n_blocks.any():
        raise ValueError(
            f"No stretch of the TOD without jumps in time has nperseg={nperseg} "
            "samples with a good one."
        )
    freqs = np.fft.rfftfreq(nperseg, d=1 / sample_rate)
    with np.errstate(invalid="ignore", divide="ignore"):
        psd = weighted / n_blocks
    # bin 0 is the mean, removed in every stretch; the last one, at the Nyquist
    # frequency, is not doubled in the one-sided spectrum and would drop by half
    return freqs[1:-1], psd[1:-1]


def _periodogram(
    signal: NDArray[np.float64],
    good: NDArray[np.bool_],
    chunks: NDArray[np.int64],
    sample_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Periodogram of every column on every whole chunk (Hann window); the chunks zero
    padded to the length of the longest, and averaged with their length as weight.
    """
    n_det = signal.shape[1]
    lengths = np.diff(chunks)
    if lengths.max() < 2:
        raise ValueError("No chunk of the TOD has 2 samples.")

    # one frequency grid for all the chunks: the one of the longest (zero padding a
    # chunk interpolates its spectrum, with the same normalization), on a length
    # with small prime factors, fast for the FFT
    nfft = next_fast_len(int(lengths.max()), real=True)
    weighted = np.zeros((nfft // 2 + 1, n_det))
    weight = np.zeros(n_det)
    for a, b in pairwise(chunks):
        if b - a < 2:
            continue
        for first in range(0, n_det, _BATCH):
            cols = slice(first, first + _BATCH)
            x, usable = _prepared(signal[a:b, cols], good[a:b, cols])
            freqs, psd = periodogram(
                x, fs=sample_rate, window="hann", nfft=nfft, axis=0
            )
            weighted[:, cols] += (b - a) * psd * usable
            weight[cols] += (b - a) * usable

    with np.errstate(invalid="ignore", divide="ignore"):
        psd = weighted / weight
    # bin 0 is the mean, removed in every chunk
    return freqs[1:], psd[1:]


def plot_psd(
    data: Scan | TimeOrderedData,
    channel: int | Literal["all"] = "all",
    mode: Literal["welch", "fft", "both"] = "welch",
    nperseg: int | None = None,
    unit: str | None = None,
    background: NDArray[np.bool_] | None = None,
) -> Figure:
    r"""
    Power spectral density of the TOD, of one channel or of all of them.

    The samples that the spectra do not use (``tod.spectrum_flags`` False, or
    ``background`` False) are replaced by a straight line through the good ones, and
    the mean of the good samples is removed, as in
    :func:`~tolomeo.mapmaking.estimate_inverse_noise`.

    With ``mode="welch"`` it is Welch's method with Hann blocks of ``nperseg``
    samples, on every stretch of the TOD with no jump in time (as between two
    subscans, before :meth:`Scan.patch <tolomeo.tod.Scan.patch>`) and inside one
    chunk: no block crosses a gap or the boundary of two chunks. The stretches are
    averaged with their number of blocks as weight: the average of all the blocks,
    the estimate of the noise model. Its lowest frequency is
    ``sample_rate / nperseg``, and nothing is drawn below it (nor at the Nyquist
    frequency, whose bin is not doubled).

    With ``mode="fft"`` it is the periodogram, :math:`|\mathrm{FFT}|^2` with a Hann
    window, of every whole chunk, down to :math:`1/T` of the longest one; the
    chunks are zero padded to the length of the longest and averaged with their
    length as weight. The periodogram reads the samples of a chunk as contiguous:
    on a TOD with jumps in time, not patched, its frequencies are wrong, and a
    warning says so. With ``mode="both"`` Welch is drawn on top of the periodogram,
    to compare the two.

    Parameters
    ----------
    data : :class:`~tolomeo.tod.Scan` or :class:`~tolomeo.tod.TimeOrderedData`
        The TOD, or the scan that holds it.
    channel : int or "all", optional
        Column of the TOD to plot (the position, not the ID; with a scan the title
        shows its ID too), or ``"all"``: the mean spectrum of the channels.
        Defaults to ``"all"``.
    mode : {"welch", "fft", "both"}, optional
        Welch's method, the periodogram of the whole chunks, or both. Defaults to
        ``"welch"``.
    nperseg : int, optional
        Samples in one Welch block, only for Welch. Defaults to ``None``:
        the length that :func:`~tolomeo.mapmaking.estimate_inverse_noise` chooses.
    unit : str, optional
        Unit of the TOD in the label of the axis. Defaults to ``None``: the unit of
        the TOD.
    background : :class:`~numpy.ndarray` of bool, optional
        True for the samples that the spectra may use, on top of
        ``tod.spectrum_flags``: for instance False on a bright source. Shape
        ``(n_samples,)`` or ``(n_samples, n_det)``. Defaults to ``None``: all of
        them.

    Returns
    -------
    :class:`~matplotlib.figure.Figure`
        The figure, with log axes.

    Raises
    ------
    ValueError
        If ``channel`` or ``mode`` is not one of the accepted values, if ``nperseg``
        is below 2, or if no stretch or chunk is long enough for the estimate.

    Warns
    -----
    UserWarning
        With ``mode="fft"`` or ``"both"``, if a chunk has jumps in time.
    """
    tod = data.tod if isinstance(data, Scan) else data
    # None: the unit of the TOD (rad for the raw TOD of MISTRAL, K once calibrated)
    unit = tod.unit if unit is None else unit
    sample_rate = tod.sample_rate

    mode = mode.lower()
    if mode not in ("welch", "fft", "both"):
        raise ValueError(f"`mode` must be 'welch', 'fft' or 'both', got {mode!r}.")

    # the columns to estimate: one, or all
    if isinstance(channel, str):
        if channel.lower() != "all":
            raise ValueError(f"`channel` must be an integer or 'all', got {channel!r}.")
        columns = slice(None)
    else:
        if not -tod.n_det <= channel < tod.n_det:
            raise ValueError(
                f"`channel` must be between 0 and {tod.n_det - 1}, got {channel}."
            )
        columns = slice(channel % tod.n_det, channel % tod.n_det + 1)

    # samples the spectra may use: the good and the patched ones, and the background
    good = np.broadcast_to(tod.spectrum_flags[:, None], tod.signal.shape)
    if background is not None:
        background = np.asarray(background, dtype=bool)
        if background.ndim == 1:
            background = background[:, None]
        good = good & background
    signal, good = tod.signal[:, columns], good[:, columns]

    # the curves to draw, as (freqs, psd, label, line width): the periodogram first,
    # so that Welch goes on top of it
    curves = []
    stretches = _stretches(tod)
    if mode in ("fft", "both"):
        # more stretches than chunks: there are jumps in time inside a chunk
        if len(stretches) > len(tod.chunks) - 1:
            _warn(
                "The TOD has jumps in time inside a chunk: the periodogram reads its "
                "samples as contiguous, so its frequencies are wrong. Patch the scan "
                "first (Scan.patch)."
            )
        freqs, psd = _periodogram(signal, good, tod.chunks, sample_rate)
        curves.append((freqs, psd, "periodogram", 0.5))
    if mode in ("welch", "both"):
        if nperseg is None:
            nperseg = _auto_nperseg(tod.time, tod.spectrum_flags, sample_rate)
        if nperseg < 2:
            raise ValueError(f"`nperseg` must be at least 2, got {nperseg}.")
        freqs, psd = _welch_psd(signal, good, stretches, sample_rate, nperseg)
        curves.append((freqs, psd, f"Welch, nperseg = {nperseg}", None))

    fig = plt.figure(figsize=(6.4 * 1.3, 4.8 * 1.3))

    with plt.style.context(STYLE_FILE):
        ax = fig.add_subplot()
        for freqs, psd, label, lw in curves:
            if psd.shape[1] == 1:
                ax.loglog(freqs, psd[:, 0], lw=lw, label=label)
            else:
                # the mean of the channels; those with no estimate (NaN) are left out
                mean = np.nanmean(psd, axis=1)
                ax.loglog(freqs, mean, lw=lw, label=f"{label}")

        if signal.shape[1] == 1:
            if isinstance(data, Scan):
                channel_id = data.channel_ids[columns][0]
                ax.set_title(f"Channel {columns.start} (ID {channel_id})")
            else:
                ax.set_title(f"Channel {columns.start}")

        if unit is None:
            ax.set_ylabel("PSD [arbitrary units]")
        else:
            ax.set_ylabel(f"PSD [{unit}$^2$/Hz]")
        ax.set_xlabel("Frequency [Hz]")
        ax.legend(frameon=False, fontsize=12)

    return fig
