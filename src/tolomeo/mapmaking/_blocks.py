"""
Block geometry of the overlap-save noise filter.

Why blocks
----------
``N^-1`` is a convolution in time with the kernel of the noise weights, i.e. a product
in frequency. The weights are known only at the ``nperseg // 2 + 1`` frequencies of a
block of ``nperseg`` samples, the length of the blocks of the noise estimate, so the
filter works on blocks of ``block_len = nperseg`` samples: ``rfft`` of a block, product
by the weights, ``irfft``. One FFT of the whole TOD would need the weights at many more
frequencies, and its circular convolution would mix the end of the scan with the start.
The blocks also keep the memory small and are shared among threads.

Core and context
----------------
Block ``b`` is responsible for the TOD samples ``core_start .. core_end``, its core:
``core_len = nperseg // 2`` samples from ``b * core_len`` (the last core may be
shorter). The filtered value of a sample needs its neighbours too, so the block also
holds ``context_len`` samples before the core and the rest of the block after it
(``block_len - context_len - core_len`` samples, equal to ``context_len`` when
``nperseg`` is a multiple of 4)::

    block b:   |<-- context_len -->|<--- core (core_len) --->|<-- right context -->|
               |<--------------------- block_len = 2 * core_len ------------------>|
               ^ read_start_raw = core_start - context_len

The context is made of real samples of the TOD, not zeros. Near the edges of the block
the circular convolution of the FFT is wrong, but there are only context samples there:
the filtered core is exact if the kernel ``h(tau)`` is zero for ``|tau| > context_len``.
Of every block only the core is kept, and the cores of all the blocks tile
``0 .. n_samples`` with no gap and no overlap. Away from the edges of the TOD a sample
is in the core of one block and in the context of a neighbouring one.

Edges of the TOD
----------------
The context of the first block starts before sample 0, and the last blocks go past the
last sample: there the block holds zeros, the TOD extended with zeros that ``N^-1``
assumes at the edges. For one block, :class:`BlockWindow` gives the samples really
read, ``read_start .. read_end``, and ``left_pad``, the zeros before them.

Example with ``nperseg = 8`` (``core_len = 4``, ``context_len = 2``) and a TOD of 10
samples. Under every position of the block, the TOD sample it holds (``.`` is a zero);
the ranges are Python slices, with the end excluded::

    position:   0  1  2  3  4  5  6  7
    block 0:    .  .  0  1  2  3  4  5    core 0..4, read 0..6, left_pad 2
    block 1:    2  3  4  5  6  7  8  9    core 4..8, read 2..10, left_pad 0
    block 2:    6  7  8  9  .  .  .  .    core 8..10, read 6..10, left_pad 0

The core is always at the positions ``context_len .. context_len + core_len`` of the
block (2 .. 6 here), whatever the block.
"""

import math
from typing import Any

from pydantic import BaseModel, ConfigDict


class BlockWindow(BaseModel):
    """
    Position of one block in the TOD.

    The block holds the TOD samples ``read_start .. read_end`` at the positions
    ``left_pad .. left_pad + (read_end - read_start)`` and zeros everywhere else.
    Its core is the TOD samples ``core_start .. core_end``, found in the block at the
    positions ``context_len .. context_len + (core_end - core_start)``
    (:attr:`BlockLayout.context_len`).
    """

    model_config = ConfigDict(frozen=True)

    core_start: int
    core_end: int
    read_start: int
    read_end: int
    left_pad: int


class BlockLayout(BaseModel):
    """
    Sizes of the TOD and of its blocks.

    ``n_pixels`` is deliberately *not* stored here: the number of unknowns (the map
    pixels plus the virtual pixel) is passed explicitly to the functions that need it,
    so it can never get out of sync with the layout.

    Parameters
    ----------
    n_samples : int
        Samples per detector.
    n_det : int
        Number of detectors.
    nperseg : int
        Samples in one block, even: the ``nperseg`` of the noise model
        (:attr:`~tolomeo.mapmaking.InverseNoise.nperseg`).
    """

    model_config = ConfigDict(frozen=True)

    n_samples: int
    n_det: int
    nperseg: int

    def model_post_init(self, context: Any, /) -> None:
        """Raise ``ValueError`` if a size cannot describe a TOD."""
        # reject sizes that cannot describe a TOD; nperseg even and >= 2 so that
        # core_len, which is nperseg // 2, is at least 1 and exactly half a block
        if self.nperseg < 2 or self.nperseg % 2:
            raise ValueError(
                f"`nperseg` must be even and at least 2, got {self.nperseg}"
            )
        if self.n_samples < 1:
            raise ValueError(f"`n_samples` must be at least 1, got {self.n_samples}")
        if self.n_det < 1:
            raise ValueError(f"`n_det` must be at least 1, got {self.n_det}")

    @property
    def block_len(self) -> int:
        """Length of each FFT block: ``nperseg``."""
        return self.nperseg

    @property
    def core_len(self) -> int:
        """Samples kept from each block (the core): ``nperseg // 2``."""
        return self.nperseg // 2

    @property
    def context_len(self) -> int:
        """Context samples before the core: ``core_len // 2`` (integer division)."""
        return self.core_len // 2

    @property
    def n_blocks(self) -> int:
        """Blocks needed to cover the TOD: ``ceil(n_samples / core_len)``."""
        # the last block may be only partly filled, so round up
        return math.ceil(self.n_samples / self.core_len)

    def window(self, iblock: int) -> BlockWindow:
        """
        Where block ``iblock`` lives in the TOD, as a ``BlockWindow``.

        ``iblock`` must be in ``0 .. n_blocks``; it is not checked.
        """
        # the TOD samples this block is responsible for (the last core is shorter)
        core_start = iblock * self.core_len
        core_end = min(core_start + self.core_len, self.n_samples)

        # the block starts `context_len` samples before its core: negative for the
        # first block, where the block sticks out of the TOD on the left
        read_start_raw = core_start - self.context_len

        # zeros needed before the first real sample (non-zero only for the first block)
        left_pad = max(-read_start_raw, 0)

        # the TOD samples that are really read, clipped to the TOD itself (the block
        # also sticks out on the right for the last blocks: zeros there)
        read_start = max(read_start_raw, 0)
        read_end = min(read_start_raw + self.block_len, self.n_samples)

        return BlockWindow(
            core_start=core_start,
            core_end=core_end,
            read_start=read_start,
            read_end=read_end,
            left_pad=left_pad,
        )
