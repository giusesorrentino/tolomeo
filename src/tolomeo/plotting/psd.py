# Adapted from:
# https://github.com/thomaswmorris/maria/blob/master/maria/plotting/tod.py
# Original implementation by Thomas W. Morris (Maria).

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.signal import periodogram

from .._lib.spectral import _prepare_for_spectrum
from ..tod import Scan, TimeOrderedData

__all__ = ["plot_psd"]

STYLE_FILE = Path(__file__).resolve().parent / "resources" / "default.mplstyle"


# def _mean_psd(signal, flags, segments, fs, nperseg, background=None, statistic="median"):
#     """
#     PSD (Welch) di ogni rivelatore in ogni subscan, poi combinate in un solo spettro.

#     In ogni subscan i campioni flaggati o fuori da `background` (False = sulla sorgente) sono sostituiti da una
#     retta tra i campioni buoni (`_prepare_for_spectrum`), cosi' Welch lavora su dati contigui. I subscan piu'
#     corti di `nperseg` sono saltati; un rivelatore con meno di 2 campioni buoni in un subscan e' saltato li'.

#     statistic = "median": mediana sui rivelatori in ogni subscan, poi mediana sui subscan (robusta a rivelatori
#     e subscan cattivi, ma un po' sotto la media). "mean": media sui rivelatori, poi media sui subscan pesata
#     con la loro lunghezza.

#     Ritorna (freqs, psd) con psd di forma (n_freqs,).
#     """
#     n_samples, n_det = signal.shape
#     if background is None:
#         background = np.ones((n_samples, n_det), dtype=bool)
#     elif background.ndim == 1:
#         background = np.broadcast_to(background[:, None], (n_samples, n_det))
#     combine = np.nanmedian if statistic == "median" else np.nanmean   # sui rivelatori, dentro ogni subscan

#     freqs, per_subscan = None, []
#     for start, stop in pairwise(segments):
#         if stop - start < nperseg:                                  # troppo corto per un segmento
#             continue
#         good = background[start:stop] & flags[start:stop, None]     # (n_t, n_det)
#         piece = signal[start:stop].astype(float)                    # copia del subscan
#         for j in np.flatnonzero(~good.all(axis=0)):                 # solo i rivelatori con campioni da riempire
#             piece[:, j] = (_prepare_for_spectrum(piece[:, j], good[:, j]) if good[:, j].sum() >= 2
#                            else np.nan)
#         freqs, pxx = welch(piece, fs=fs, window="hann", nperseg=nperseg, axis=0)   # (n_freqs, n_det): una PSD per rivelatore
#         per_subscan.append(combine(pxx, axis=1))                    # (n_freqs,): una PSD per subscan

#     if not per_subscan:
#         raise ValueError("No subscan has at least `nperseg` samples.")
#     return freqs, np.nanmedian(np.array(per_subscan), axis=0)       # (n_subscans, n_freqs) -> mediana sui subscan


def _mean_psd(
    signal, flags, segments, fs, nfft=None, background=None, statistic="mean"
):
    """
    Periodogramma di ogni rivelatore su tutto il subscan, poi media (o mediana) sui rivelatori dentro ogni
    subscan e mediana sui subscan.

    I subscan hanno lunghezze diverse: con nfft uguale per tutti (zero padding) le frequenze sono le stesse e
    gli spettri si possono combinare. nfft = None: la potenza di 2 >= del subscan piu' lungo. La risoluzione
    vera di ogni subscan resta fs / (sua lunghezza): lo zero padding interpola, non aggiunge informazione.

    Attenzione alla mediana sui rivelatori: un periodogramma (senza la media sui segmenti di Welch) ha in ogni
    bin una distribuzione esponenziale, la cui mediana e' ln 2 = 0.69 volte la media. Con statistic="median" lo
    spettro esce circa il 30% piu' basso; per il livello usa "mean".
    """
    n_samples, n_det = signal.shape
    if background is None:
        background = np.ones((n_samples, n_det), dtype=bool)
    elif background.ndim == 1:
        background = np.broadcast_to(background[:, None], (n_samples, n_det))
    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(np.diff(segments).max())))
    combine = (
        np.nanmedian if statistic == "median" else np.nanmean
    )  # sui rivelatori, dentro ogni subscan

    freqs, per_subscan = None, []
    for start, stop in pairwise(segments):
        if stop - start < 2:
            continue
        good = background[start:stop] & flags[start:stop, None]  # (n_t, n_det)
        piece = signal[start:stop].astype(float)  # copia del subscan
        for j in np.flatnonzero(
            ~good.all(axis=0)
        ):  # solo i rivelatori con campioni da riempire
            piece[:, j] = (
                _prepare_for_spectrum(piece[:, j], good[:, j])
                if good[:, j].sum() >= 2
                else np.nan
            )
        # finestra di Hann su tutto il subscan, FFT lunga nfft (zero padding), densita' spettrale a un lato
        freqs, pxx = periodogram(
            piece, fs=fs, window="hann", nfft=nfft, detrend="constant", axis=0
        )  # (n_freqs, n_det)
        per_subscan.append(combine(pxx, axis=1))  # (n_freqs,): una PSD per subscan

    if not per_subscan:
        raise ValueError("No subscan has at least 2 samples.")
    return freqs, np.nanmedian(
        np.array(per_subscan), axis=0
    )  # (n_subscans, n_freqs) -> mediana sui subscan


def plot_psd(
    data: Scan | TimeOrderedData,
    nperseg: int = 512,
    unit: str = "rad",
    background: NDArray[np.bool_] | None = None,
    statistic: str = "median",
) -> Figure:
    tod = data.tod if isinstance(data, Scan) else data

    fig = plt.figure(figsize=(6.4 * 1.3, 4.8 * 1.3))

    with plt.style.context(STYLE_FILE):
        f, psd = _mean_psd(
            tod.signal,
            tod.flags,
            tod.segments,
            tod.sample_rate,
            nperseg,
            background,
            statistic,
        )

        ax = fig.add_subplot()
        ax.loglog(f[1:], psd[1:], lw=0.8)
        ax.set_ylabel(f"PSD [{unit}$^2$/Hz]")
        ax.set_xlabel("Frequency [Hz]")

    return fig
