"""
The normal equations ``A m = b`` of the GLS map, and the operators they are made of.

The pointing operator ``P`` / ``P^T``, the noise filter ``N^-1`` on blocks, the
right-hand side ``b = P^T N^-1 d``, the diagonal preconditioner and the normal operator
``A = P^T N^-1 P``. ``_normal_equations`` builds them for ``gls_map``.

The one loop that matters
-------------------------
The right-hand side and the product ``A x`` are the same loop. For every block:

1. fill a zeroed buffer ``(n_det, block_len)`` with samples;
2. apply ``N^-1`` to it (``InverseNoiseFilter``);
3. add the core of the result to a map through ``P^T``.

They differ only in step 1: the right-hand side reads the measured TOD, the normal
operator reads ``P x`` (the current map, read at the pointing pixels). That loop is
``_filter_and_bin``, parametrised by "how to fill a block".

Maps here have ``n_unknowns`` elements: the sky pixels followed by the virtual pixel.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, rfft

from ._blocks import BlockLayout, BlockWindow
from ._noise import InverseNoise

logger = logging.getLogger(__name__)

# rows (time samples) copied at once when PointingOperator transposes the pixels
_TILE = 4096


class PointingOperator:
    """``P`` (map -> blocks of TOD) and ``P^T`` (blocks of TOD -> map), one block at a time."""

    def __init__(self, pixels: NDArray[np.integer], layout: BlockLayout) -> None:
        """
        ``pixels`` has shape ``(n_samples, n_det)``, with the flagged samples already
        on the virtual pixel (see ``gls_map`` in ``_maps.py``).

        Builds ``pixels_by_det``, a copy of the pixels with shape ``(n_det, n_samples)``
        and dtype ``intp`` (the type NumPy wants for indexing), one contiguous row per
        detector.
        """
        self.layout = layout

        n_samples, n_det = pixels.shape

        # The blocks are buffers of shape (n_det, block_len), so the pixels of a block
        # are needed in the same order: detector-major. The transposition is done in
        # tiles of _TILE rows. Each tile is a contiguous piece of pixels, read
        # sequentially, and it is written as _TILE consecutive values per detector row.
        # Transposing the whole array in one go instead walks through the 1.5 GB of memory
        # with a large stride, which is very slow when the machine is short of RAM.
        out = np.empty((n_det, n_samples), dtype=np.intp)
        for r0 in range(0, n_samples, _TILE):
            # the last tile is shorter: the slices stop at the end of the array
            out[:, r0 : r0 + _TILE] = pixels[r0 : r0 + _TILE].T

        self.pixels_by_det = out

    def map_to_block(
        self, in_map: NDArray[np.float64], window: BlockWindow
    ) -> NDArray[np.float64]:
        """
        ``P x`` for one block: shape ``(n_det, block_len)``.

        A new zeroed buffer where the positions ``left_pad .. left_pad + n`` (with
        ``n = read_end - read_start``) hold ``in_map[pixel]`` for the samples
        ``read_start .. read_end`` of every detector. The rest of the buffer stays zero
        (the part of the block that sticks out of the TOD).
        """
        time_buf = np.zeros(shape=(self.layout.n_det, self.layout.block_len))
        n = window.read_end - window.read_start

        # pixels of the samples of this block, (n_det, n): contiguous rows
        pixels = self.pixels_by_det[:, window.read_start : window.read_end]

        # gather: the map value at every one of those pixels, same shape (n_det, n)
        values = in_map[pixels]

        # place them after the left_pad zeros
        time_buf[:, window.left_pad : window.left_pad + n] = values

        return time_buf

    def block_to_map(
        self,
        time_buf: NDArray[np.float64],
        window: BlockWindow,
        out_map: NDArray[np.float64],
    ) -> None:
        """
        ``P^T`` for one block: add the core of a filtered block to ``out_map``, in place.

        The core of the block is at the positions ``context_len .. context_len + n``,
        with ``n = core_end - core_start``, and its samples belong to the pixels of the
        TOD samples ``core_start .. core_end`` of every detector. Several samples (and
        several detectors) hit the same pixel, so a plain ``out_map[pixels] += values``
        would be **wrong** (with repeated indices only one of the additions is kept).
        ``np.bincount`` with ``weights=`` sums the values that fall in the same pixel;
        ``np.add.at`` would also be correct but much slower.
        """
        n = window.core_end - window.core_start
        context_len = self.layout.context_len

        # the core of the filtered block, (n_det, n): it starts `context_len` samples
        # into the block
        values = time_buf[:, context_len : context_len + n]

        # the pixels of the same samples, with the same shape (n_det, n)
        pixels = self.pixels_by_det[:, window.core_start : window.core_end]

        # values and pixels are flattened in the same order, so that every value stays
        # paired with its pixel. minlength gives a result as long as the map
        summed = np.bincount(
            pixels.ravel(), weights=values.ravel(), minlength=out_map.size
        )
        out_map += summed


class InverseNoiseFilter:
    """
    ``N^-1`` as an operator on blocks.

    A block holds ``nperseg`` samples of every detector, shape ``(n_det, nperseg)``;
    the same length is used by the Welch estimate of the spectra:

    1. real FFT of every detector -> ``n_freqs = nperseg // 2 + 1`` complex bins;
    2. multiplication of every detector's spectrum by its own inverse spectrum;
    3. inverse real FFT back to ``nperseg`` samples per detector.

    With cross-spectra step 2 would mix the detectors: for every column ``i`` (the pair
    ``(f, r)``) and every frequency ``k``,
    ``out[f, k] += spectrum[r, k] * inv_noiseps[k, i]`` and, if ``f != r``,
    ``out[r, k] += spectrum[f, k] * inv_noiseps[k, i]``.
    """

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
            Inverse noise spectrum of every detector (see :meth:`from_inv_noiseps`).
            Shape ``(n_det, n_freqs)``.
        """
        self.layout = layout
        self.weights = weights  # (n_det, n_freqs)

    @classmethod
    def from_inv_noiseps(
        cls,
        inv_noiseps: NDArray[np.float64],
        det_pairs: NDArray[np.integer],
        layout: BlockLayout,
    ) -> InverseNoiseFilter:
        """
        Build the filter from the arrays that ``gls_map`` passes to the kernel.

        ``inv_noiseps`` is ``(n_freqs, n_cols)`` and ``det_pairs`` is
        ``(n_cols, 2)``; only auto spectra are supported.
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

        # np.empty left the rows uninitialised: every detector must have been written
        # once, otherwise a row would hold garbage (missing) or the last column wins
        # (repeated)
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

        * **Normalisation.** ``scipy.fft.irfft`` already divides by ``n``, so the
          weights are used as they are. With all the weights equal to 1 the filter
          returns its input.
        * **The input buffer is overwritten** (``overwrite_x=True``, to save memory):
          give a buffer that you own and do not use it afterwards.
        * **One FFT worker.** The parallelism is across blocks (threads in
          ``_filter_and_bin``); if every FFT also asked for all the cores, the threads
          would compete for them.
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


def _filter_and_bin(
    fill_block: Callable[[BlockWindow], NDArray[np.float64]],
    noise_filter: InverseNoiseFilter,
    pointing: PointingOperator,
    layout: BlockLayout,
    n_unknowns: int,
    workers: int = 1,
) -> NDArray[np.float64]:
    """
    ``P^T N^-1`` applied to the blocks that ``fill_block`` produces; returns a map.

    For every block: ``fill_block`` gives the samples, ``noise_filter`` applies ``N^-1``
    and ``pointing`` adds the core of the result to a map of ``n_unknowns`` elements.

    ``workers`` is the number of threads (an integer >= 1: ``None`` and negative values
    are resolved before, in ``_lib.parallel._resolve_workers``). The blocks are split in
    ``workers`` contiguous ranges, one per thread. Each thread accumulates into its own
    map, so no lock is needed, and at the end the maps are summed. With ``workers=1`` no
    thread is created. Threads help because the heavy calls (FFT, gather, copies,
    ``bincount``) release the GIL; the Python glue between them does not run in parallel.

    The floating-point sum is not associative, so the result changes at rounding level
    with ``workers``: compare maps with a tolerance, never with ``==``.

    ``fill_block`` is called from several threads: it must return a *new*
    ``(n_det, block_len)`` array for every call and only read shared data.
    """
    out_map = np.zeros(n_unknowns)

    # never more threads than blocks: a thread without blocks would do nothing
    workers = min(workers, layout.n_blocks)

    # split the blocks 0 .. n_blocks in `workers` contiguous ranges: range k is
    # block_edges[k] .. block_edges[k + 1]. dtype=int floors the inner edges, while the
    # first (0) and the last (n_blocks) are exact, so every block is in exactly one range
    block_edges = np.linspace(0, layout.n_blocks, workers + 1, dtype=int)

    # the job of one thread: filter the blocks i0 .. i1 - 1 and bin them into a map of
    # its own (local_map), that nobody else touches
    def run_range(i0: int, i1: int) -> NDArray[np.float64]:
        """Filter and bin the blocks ``i0 .. i1 - 1``; return a map of ``n_unknowns``."""
        local_map = np.zeros(n_unknowns)

        for i in range(i0, i1):
            window = layout.window(i)  # where block i lives in the TOD

            time_buf = fill_block(window)  # (n_det, block_len) samples of the block
            time_buf = noise_filter.filter_block(time_buf=time_buf)  # apply N^-1
            # add the core of the filtered block to the local map (P^T)
            pointing.block_to_map(time_buf=time_buf, window=window, out_map=local_map)

        return local_map

    # a single worker: no pool, everything runs in the calling thread
    if workers == 1:
        return run_range(0, layout.n_blocks)

    # one task per range. submit returns a Future at once, so all the tasks run at the
    # same time; the `with` block waits for all of them before it ends
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start, stop in pairwise(block_edges):
            futures.append(executor.submit(run_range, start, stop))

    # sum the local maps. result() returns the map of a task, or raises here again any
    # exception that happened inside its thread
    for future in futures:
        out_map += future.result()

    return out_map


def _build_rhs(
    tod: NDArray[np.float64],
    pointing: PointingOperator,
    noise_filter: InverseNoiseFilter,
    layout: BlockLayout,
    n_unknowns: int,
    workers: int = 1,
) -> NDArray[np.float64]:
    """
    Right-hand side ``b = P^T N^-1 d``.

    A call to ``_filter_and_bin`` where ``fill_block`` copies the measured TOD instead of
    applying ``P``: a zeroed ``(n_det, block_len)`` buffer with the samples
    ``read_start .. read_end`` of ``tod``, transposed, starting at the position
    ``left_pad``.

    ``tod`` is ``(n_samples, n_det)``, so the transpose of a time window is a strided
    copy; one block is only a few MB, so this is cheap.
    """
    n_det = layout.n_det
    block_len = layout.block_len

    def fill_from_tod(window: BlockWindow) -> NDArray[np.float64]:
        """Block of the measured TOD, shape ``(n_det, block_len)``, zero-padded."""
        # 1. new buffer of zeros, shape (n_det, block_len)
        # 2. n = number of samples of this window
        # 3. copy the TOD samples of the window into the buffer
        # 4. return the buffer
        buffer = np.zeros(shape=(n_det, block_len))
        n = window.read_end - window.read_start

        # the window of tod is (n, n_det); its transpose (n_det, n) goes after the
        # left_pad zeros of the buffer
        buffer[:, window.left_pad : window.left_pad + n] = tod[
            window.read_start : window.read_end
        ].T

        return buffer

    return _filter_and_bin(
        fill_block=fill_from_tod,
        noise_filter=noise_filter,
        pointing=pointing,
        layout=layout,
        n_unknowns=n_unknowns,
        workers=workers,
    )


def _build_diagonal_preconditioner(
    pixels_by_det: NDArray[np.integer],
    lag_zero: NDArray[np.float64],
    n_unknowns: int,
    workers: int = 1,  # idk
) -> NDArray[np.float64]:
    """
    ``diag`` of length ``n_unknowns`` (map pixels + the virtual pixel).

    The preconditioner ``M`` is the diagonal of ``P^T N^-1 P`` approximated with the
    zero-lag noise weight of each detector: for every sample (flagged ones included,
    they sit in the virtual pixel) add ``lag_zero[det]`` to the pixel it points to. So
    ``diag[p]`` is the weighted number of samples that fall in pixel ``p``. Applying
    it, ``M^-1 r``, is the ``precondition`` method of ``TemperatureNormalOperator``.

    ``lag_zero`` has one value per detector; it is computed by
    ``estimate_inverse_noise``.

    ``pixels_by_det`` has shape ``(n_det, n_samples)`` and dtype ``intp``: the pixel of
    every sample, one row per detector (``PointingOperator.pixels_by_det``). Detector
    major, so that the pixels of one detector are contiguous in memory and the loop
    below reads them sequentially.
    """
    diag = np.zeros(n_unknowns)
    n_det = pixels_by_det.shape[0]

    def run_det(pixels_by_det: NDArray[np.integer], idet: int) -> NDArray[np.floating]:
        """Contribution of detector ``idet`` to the diagonal, shape ``(n_unknowns,)``."""
        local_diag = np.zeros(n_unknowns)
        ids = pixels_by_det[idet]
        counts = np.bincount(ids, minlength=n_unknowns)
        local_diag += lag_zero[idet] * counts

        return local_diag

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for idet in range(n_det):
            futures.append(executor.submit(run_det, pixels_by_det, idet))

    for future in futures:
        diag += future.result()

    # for idet in range(n_det):
    #     # the pixels hit by this detector, one per sample: a contiguous row
    #     ids = pixels_by_det[idet]

    #     # number of samples of this detector in each pixel. minlength makes the result
    #     # always have n_unknowns elements, also when the last pixels (the virtual one
    #     # among them) have no hit, so that it can be added to diag
    #     counts = np.bincount(ids, minlength=n_unknowns)

    #     # every sample of this detector weighs lag_zero[idet]
    #     diag += lag_zero[idet] * counts

    return diag


class TemperatureNormalOperator:
    """
    ``A = P^T N^-1 P`` and its diagonal preconditioner.

    This is the object the CG talks to; see ``_solver.CGOperator``.
    """

    def __init__(
        self,
        diag: NDArray[np.float64],
        layout: BlockLayout,
        pixels: NDArray[np.integer],
        inv_noiseps: NDArray[np.float64],
        det_pairs: NDArray[np.integer],
        workers: int = 1,
    ) -> None:
        """
        Build the ``PointingOperator`` and the ``InverseNoiseFilter`` once, here.

        ``diag`` is needed only by ``precondition``. Since ``1 / diag`` does not change
        during the CG, it is computed once here (``0`` where ``diag == 0``, no ``inf``,
        no warning) and kept as ``inv_diag``, instead of dividing at every iteration.
        ``workers`` is the number of threads of every product ``A x``.
        """
        self.layout = layout

        # the operator has its own PointingOperator (and so its own pixels_by_det)
        self.pointing = PointingOperator(pixels=pixels, layout=layout)
        self.noise_filter = InverseNoiseFilter.from_inv_noiseps(
            inv_noiseps=inv_noiseps,
            det_pairs=det_pairs,
            layout=layout,
        )

        # 1 / diag where diag > 0, 0 elsewhere (pixels with no hits): np.divide only
        # computes the positions selected by `where`, the others keep the zeros of `out`
        self.inv_diag = np.divide(1.0, diag, out=np.zeros_like(diag), where=diag > 0)
        self.workers = workers

    def project(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """``A x`` for a vector ``x`` of ``n_unknowns`` elements: ``P^T N^-1 (P x)``."""
        # the lambda is safe with threads: every task calls it with its own window and
        # only reads x
        return _filter_and_bin(
            fill_block=lambda window: self.pointing.map_to_block(x, window),
            noise_filter=self.noise_filter,
            pointing=self.pointing,
            layout=self.layout,
            n_unknowns=x.size,
            workers=self.workers,
        )

    def precondition(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        ``M^-1 r``: ``r`` divided pixel by pixel by ``diag`` (0 where ``diag == 0``).

        The CG calls it as ``precondition(g)`` to get the preconditioned residual ``h``.
        """
        # one vectorised product by the precomputed 1 / diag, no loop
        return r * self.inv_diag


def _normal_equations(
    tod: NDArray[np.float64],
    pixels: NDArray[np.integer],
    inv_noise: InverseNoise,
    n_unknowns: int,
    workers: int,
) -> tuple[TemperatureNormalOperator, NDArray[np.float64]]:
    """
    The system ``A m = b`` of the GLS map.

    Returns the operator ``A = P^T N^-1 P``, with its diagonal preconditioner (what the
    CG needs, see ``_solver.CGOperator``), and the right-hand side
    ``b = P^T N^-1 d``.

    ``tod`` and ``pixels`` are ``(n_samples, n_det)``. ``pixels`` already sends the
    flagged samples to the virtual pixel, so the maps here have ``n_unknowns`` elements:
    the pixels of the sky followed by the virtual pixel.
    """
    n_samples, n_det = tod.shape
    layout = BlockLayout(n_samples=n_samples, n_det=n_det, nperseg=inv_noise.nperseg)

    t0 = time.perf_counter()
    pointing = PointingOperator(pixels=pixels, layout=layout)
    noise_filter = InverseNoiseFilter.from_inv_noiseps(
        inv_noiseps=inv_noise.inv_noiseps, det_pairs=inv_noise.det_pairs, layout=layout
    )
    b = _build_rhs(
        tod=tod,
        pointing=pointing,
        noise_filter=noise_filter,
        layout=layout,
        n_unknowns=n_unknowns,
        workers=workers,
    )
    logger.info(f"[MAPMAKER] RHS vector constructed in {time.perf_counter() - t0:.2f}s")
    del noise_filter  # the operator below builds its own

    t1 = time.perf_counter()
    diag = _build_diagonal_preconditioner(
        pixels_by_det=pointing.pixels_by_det,
        lag_zero=inv_noise.lag_zero,
        n_unknowns=n_unknowns,
        workers=workers,
    )
    logger.info(
        f"[MAPMAKER] Diagonal preconditioner built in {time.perf_counter() - t1:.2f}s"
    )
    del pointing

    operator = TemperatureNormalOperator(
        diag=diag,
        layout=layout,
        pixels=pixels,
        inv_noiseps=inv_noise.inv_noiseps,
        det_pairs=inv_noise.det_pairs,
        workers=workers,
    )
    return operator, b
