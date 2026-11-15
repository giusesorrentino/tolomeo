import numpy as np
import pytest
from pydantic import ValidationError

from tolomeo.mapmaking._blocks import BlockLayout, BlockWindow

LAYOUTS = [(10, 8), (16, 8), (1, 8), (1000, 64), (97, 12), (50, 6), (5, 2)]
LAYOUT_IDS = [f"n{n}_nperseg{nperseg}" for n, nperseg in LAYOUTS]


def _windows(layout):
    return [layout.window(i) for i in range(layout.n_blocks)]


# -----------------------------------------------------------------------------
# BlockLayout
# -----------------------------------------------------------------------------


def test_block_layout_sizes():
    layout = BlockLayout(n_samples=10, n_det=3, nperseg=8)

    assert layout.block_len == 8
    assert layout.core_len == 4
    assert layout.context_len == 2
    assert layout.n_blocks == 3  # ceil(10 / 4)


def test_block_layout_example_of_the_docstring():
    # Check the example of the module docstring: nperseg = 8 and 10 samples.
    layout = BlockLayout(n_samples=10, n_det=1, nperseg=8)

    assert _windows(layout) == [
        BlockWindow(core_start=0, core_end=4, read_start=0, read_end=6, left_pad=2),
        BlockWindow(core_start=4, core_end=8, read_start=2, read_end=10, left_pad=0),
        BlockWindow(core_start=8, core_end=10, read_start=6, read_end=10, left_pad=0),
    ]


@pytest.mark.parametrize("n_samples, nperseg", LAYOUTS, ids=LAYOUT_IDS)
def test_block_cores_tile_the_tod(n_samples, nperseg):
    # Check that the cores of the blocks cover every sample once, in order.
    layout = BlockLayout(n_samples=n_samples, n_det=1, nperseg=nperseg)

    cores = [np.arange(w.core_start, w.core_end) for w in _windows(layout)]

    np.testing.assert_array_equal(np.concatenate(cores), np.arange(n_samples))
    assert all(core.size > 0 for core in cores)


@pytest.mark.parametrize("n_samples, nperseg", LAYOUTS, ids=LAYOUT_IDS)
def test_block_window_bounds(n_samples, nperseg):
    # Check that every block reads samples of the TOD, that they fit in the block,
    # that they include the core and that the core is at the position context_len.
    layout = BlockLayout(n_samples=n_samples, n_det=1, nperseg=nperseg)

    for w in _windows(layout):
        assert 0 <= w.read_start <= w.core_start < w.core_end <= w.read_end
        assert w.read_end <= n_samples
        assert w.left_pad + (w.read_end - w.read_start) <= layout.block_len
        assert w.left_pad + (w.core_start - w.read_start) == layout.context_len


def test_block_window_interior_block_is_full():
    # Check that a block away from the edges is all real samples, with no zero.
    layout = BlockLayout(n_samples=1000, n_det=1, nperseg=64)

    w = layout.window(5)

    assert w.left_pad == 0
    assert w.read_end - w.read_start == layout.block_len


@pytest.mark.parametrize("nperseg", [4, 8, 12, 8192])
def test_block_context_symmetric_when_nperseg_multiple_of_4(nperseg):
    layout = BlockLayout(n_samples=10, n_det=1, nperseg=nperseg)

    right_context = layout.block_len - layout.context_len - layout.core_len

    assert right_context == layout.context_len


def test_block_context_asymmetric_otherwise():
    # nperseg = 6: core of 3 samples, 1 before it and 2 after it
    layout = BlockLayout(n_samples=10, n_det=1, nperseg=6)

    assert (layout.context_len, layout.core_len) == (1, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nperseg": 7},
        {"nperseg": 0},
        {"nperseg": -2},
        {"n_samples": 0},
        {"n_det": 0},
    ],
    ids=["odd", "zero", "negative", "no_samples", "no_detectors"],
)
def test_block_layout_invalid_size_raises(kwargs):
    with pytest.raises(ValueError, match=r"must be"):
        BlockLayout(**({"n_samples": 10, "n_det": 1, "nperseg": 8} | kwargs))


def test_block_layout_is_frozen():
    layout = BlockLayout(n_samples=10, n_det=1, nperseg=8)

    with pytest.raises(ValidationError):
        layout.nperseg = 16
