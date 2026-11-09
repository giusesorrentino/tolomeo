"""
Block geometry of the overlap-save noise filter.

Why blocks
----------
``N^-1`` is a convolution in time, i.e. a product in frequency. A whole TOD is far too
long for one FFT, so it is cut in blocks of ``block_len = 2 * (n_freqs - 1)`` samples.
Consecutive blocks start ``core_len = n_freqs - 1`` samples apart, so every sample is
inside two blocks, but it is *kept* from one only: the "core" of that block.

::

    block b:   |<-- padding -->|<------ core (core_len) ------>|<-- right context -->|
               |<------------------- block_len = 2 * core_len ---------------------->|
               ^ read_start_raw = core_start - padding

The samples before and after the core are real neighbouring samples of the TOD (context,
not zeros): they take the edge effects of the circular convolution, so the core is
exact as long as the filter response is shorter than the context. Zeros appear only
where the block sticks out of the TOD (before sample 0, after the last sample).

The cores of all the blocks tile ``0 .. n_samples`` with no gap and no overlap.
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
    positions ``padding .. padding + (core_end - core_start)``.
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
    n_freqs : int
        Frequency bins of the inverse noise spectra (rows of ``inv_noiseps``).
    """

    model_config = ConfigDict(frozen=True)

    n_samples: int
    n_det: int
    n_freqs: int

    def model_post_init(self, context: Any, /) -> None:
        """Raise ``ValueError`` if a size cannot describe a TOD."""
        # reject sizes that cannot describe a TOD; n_freqs >= 2 so that core_len,
        # which is n_freqs - 1, is at least 1
        if self.n_freqs < 2:
            raise ValueError(f"`n_freqs` must be at least 2, got {self.n_freqs}")
        if self.n_samples < 1:
            raise ValueError(f"`n_samples` must be at least 1, got {self.n_samples}")
        if self.n_det < 1:
            raise ValueError(f"`n_det` must be at least 1, got {self.n_det}")

    @property
    def block_len(self) -> int:
        """Length of each FFT block: ``2 * (n_freqs - 1)``."""
        return 2 * (self.n_freqs - 1)

    @property
    def core_len(self) -> int:
        """
        Samples kept from each block (the core): ``n_freqs - 1``.

        Called ``valid_len`` in Rust; here ``valid`` is the mask of the good samples.
        """
        return self.n_freqs - 1

    @property
    def padding(self) -> int:
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

        # the block starts `padding` samples before its core: negative for the first
        # block, where the block sticks out of the TOD on the left
        read_start_raw = core_start - self.padding

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


# Tests worth writing for this file (no Rust needed):
#   * the cores of all blocks tile 0..n_samples exactly (concatenate them and compare);
#   * read_end - read_start + left_pad <= block_len for every block;
#   * a TOD shorter than one block, and one whose length is an exact multiple of core_len;
#   * for an interior block, left_pad == 0 and read_end - read_start == block_len.
