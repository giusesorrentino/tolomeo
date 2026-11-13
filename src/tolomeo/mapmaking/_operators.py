"""
Pointing operator ``P`` / ``P^T``, the right-hand side ``b = P^T N^-1 d`` and the
normal operator ``A = P^T N^-1 P``.

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

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from ._blocks import BlockLayout, BlockWindow
from ._noise import InverseNoiseFilter

# rows (time samples) copied at once when PointingOperator transposes the pixels
_TILE = 4096


class PointingOperator:
    """``P`` (map -> blocks of TOD) and ``P^T`` (blocks of TOD -> map), one block at a time."""

    def __init__(self, pixels: NDArray[np.integer], layout: BlockLayout) -> None:
        """
        ``pixels`` has shape ``(n_samples, n_det)``, with the flagged samples already
        on the virtual pixel (``_flags._redirect_flagged_samples``).

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

        The core of the block is at the positions ``padding .. padding + n`` with
        ``n = core_end - core_start``, and its samples belong to the pixels of the TOD
        samples ``core_start .. core_end`` of every detector. Several samples (and
        several detectors) hit the same pixel, so a plain ``out_map[pixels] += values``
        would be **wrong** (with repeated indices only one of the additions is kept).
        ``np.bincount`` with ``weights=`` sums the values that fall in the same pixel;
        ``np.add.at`` would also be correct but much slower.
        """
        n = window.core_end - window.core_start
        padding = self.layout.padding

        # the core of the filtered block, (n_det, n): it starts `padding` samples into
        # the block
        values = time_buf[:, padding : padding + n]

        # the pixels of the same samples, with the same shape (n_det, n)
        pixels = self.pixels_by_det[:, window.core_start : window.core_end]

        # values and pixels are flattened in the same order, so that every value stays
        # paired with its pixel. minlength gives a result as long as the map
        summed = np.bincount(
            pixels.ravel(), weights=values.ravel(), minlength=out_map.size
        )
        out_map += summed


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


class TemperatureNormalOperator:
    """
    ``A = P^T N^-1 P`` and its diagonal preconditioner.

    This is the object the CG talks to; see ``_solver.LinearOperator``.
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


# Tests worth writing for this file:
#   * adjointness: <P x, y> == <x, P^T y> for random x, y (use a tiny layout);
#   * A is symmetric and positive: <u, A v> == <A u, v> and <v, A v> > 0;
#   * with all noise weights equal to 1, N^-1 is the identity, so A x is the
#     hit-weighted map and the final GLS map must equal ``binned_map``.
