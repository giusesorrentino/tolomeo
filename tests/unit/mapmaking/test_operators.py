import numpy as np
import pytest

from tolomeo.mapmaking import InverseNoise
from tolomeo.mapmaking._blocks import BlockLayout
from tolomeo.mapmaking._operators import (
    InverseNoiseFilter,
    NormalOperator,
    PointingOperator,
    _build_diagonal_preconditioner,
    _build_rhs,
    _filter_and_bin,
    _normal_equations,
)

rng = np.random.default_rng(17)

N_SAMPLES, N_DET, NPERSEG = 50, 3, 16  # core of 8 samples, context of 4
N_PIXELS = 12  # the unknowns of the tests, virtual pixel included
N_FREQS = NPERSEG // 2 + 1
LAYOUT = BlockLayout(n_samples=N_SAMPLES, n_det=N_DET, nperseg=NPERSEG)
AUTO_PAIRS = np.array([(j, j) for j in range(N_DET)], dtype=np.uintp)


def _random_pixels(n_samples=N_SAMPLES, n_det=N_DET):
    return rng.integers(0, N_PIXELS, (n_samples, n_det))


def _short_kernel(n_det=N_DET, context_len=NPERSEG // 4):
    """
    Symmetric kernels of half length ``context_len``, one per detector, and the weights
    that give them.

    The weights are the real FFT of the kernels: with them the overlap-save filter is
    exactly the linear convolution with the kernel. Kernel ``j`` is
    ``[h_K .. h_1, h_0, h_1 .. h_K]``, with ``h_0`` large enough that every weight is
    positive.
    """
    taps = rng.uniform(-1, 1, (n_det, context_len + 1))
    taps[:, 0] = 2 * np.abs(taps[:, 1:]).sum(axis=1) + 1
    kernels = np.concatenate([taps[:, :0:-1], taps], axis=1)  # (n_det, 2K + 1)

    # the same kernel, circular, on a block of NPERSEG samples
    circular = np.zeros((n_det, NPERSEG))
    circular[:, : context_len + 1] = taps
    circular[:, -context_len:] = taps[:, :0:-1]
    weights = np.fft.rfft(circular, axis=1).real  # (n_det, n_freqs)

    return kernels, weights


def _operator(pixels, weights, diag=None, workers=1):
    return NormalOperator(
        diag=np.ones(N_PIXELS) if diag is None else diag,
        layout=BlockLayout(
            n_samples=pixels.shape[0], n_det=pixels.shape[1], nperseg=NPERSEG
        ),
        pixels=pixels,
        inv_noiseps=weights.T,
        det_pairs=np.array([(j, j) for j in range(pixels.shape[1])], dtype=np.uintp),
        workers=workers,
    )


# -----------------------------------------------------------------------------
# PointingOperator
# -----------------------------------------------------------------------------


def test_pointing_operator_pixels_by_detector():
    # More samples than a tile of the transposition, so that it is done in pieces.
    pixels = rng.integers(0, 100, (10_000, 2))
    layout = BlockLayout(n_samples=10_000, n_det=2, nperseg=NPERSEG)

    pointing = PointingOperator(pixels, layout)

    np.testing.assert_array_equal(pointing.pixels_by_det, pixels.T)
    assert pointing.pixels_by_det.dtype == np.intp
    assert pointing.pixels_by_det.flags.c_contiguous


def test_pointing_operator_map_to_block():
    # Check the gather on the example of the docstring of _blocks: block 0 has two
    # zeros, then the map read at the pixels of the samples 0 .. 6.
    layout = BlockLayout(n_samples=10, n_det=1, nperseg=8)
    pointing = PointingOperator(np.arange(10)[:, None], layout)
    in_map = 10.0 + np.arange(10)

    blocks = [pointing.map_to_block(in_map, layout.window(i)) for i in range(3)]

    np.testing.assert_array_equal(blocks[0], [[0, 0, 10, 11, 12, 13, 14, 15]])
    np.testing.assert_array_equal(blocks[1], [[12, 13, 14, 15, 16, 17, 18, 19]])
    np.testing.assert_array_equal(blocks[2], [[16, 17, 18, 19, 0, 0, 0, 0]])


def test_pointing_operator_block_to_map_adds_the_core():
    # Check that only the core of the block is added, at the pixels of the core
    # samples, and that the values of samples in the same pixel are summed.
    layout = BlockLayout(n_samples=10, n_det=2, nperseg=8)
    pixels = np.column_stack([np.arange(10), np.full(10, 3)])
    pointing = PointingOperator(pixels, layout)
    block = np.arange(16.0).reshape(2, 8)
    out_map = np.ones(10)

    pointing.block_to_map(block, layout.window(1), out_map)  # core 4 .. 8

    expected = np.ones(10)
    expected[4:8] += block[0, 2:6]
    expected[3] += block[1, 2:6].sum()
    np.testing.assert_array_equal(out_map, expected)


# -----------------------------------------------------------------------------
# InverseNoiseFilter
# -----------------------------------------------------------------------------


def test_inverse_noise_filter_unit_weights_is_identity():
    noise_filter = InverseNoiseFilter(LAYOUT, np.ones((N_DET, N_FREQS)))
    block = rng.standard_normal((N_DET, NPERSEG))

    filtered = noise_filter.filter_block(block.copy())

    np.testing.assert_allclose(filtered, block, atol=1e-14)


def test_inverse_noise_filter_scales_every_detector():
    scale = np.array([0.5, 2.0, 7.0])
    noise_filter = InverseNoiseFilter(
        LAYOUT, np.repeat(scale[:, None], N_FREQS, axis=1)
    )
    block = rng.standard_normal((N_DET, NPERSEG))

    filtered = noise_filter.filter_block(block.copy())

    np.testing.assert_allclose(filtered, scale[:, None] * block, atol=1e-13)


def test_inverse_noise_filter_from_inv_noiseps():
    # Check that column i goes to the detector of det_pairs[i], in any order.
    inv_noiseps = rng.random((N_FREQS, N_DET))
    det_pairs = np.array([(2, 2), (0, 0), (1, 1)], dtype=np.uintp)

    noise_filter = InverseNoiseFilter.from_inv_noiseps(inv_noiseps, det_pairs, LAYOUT)

    np.testing.assert_array_equal(noise_filter.weights, inv_noiseps[:, [1, 2, 0]].T)


def test_inverse_noise_filter_cross_spectrum_raises():
    det_pairs = np.array([(0, 0), (1, 1), (0, 2)], dtype=np.uintp)

    with pytest.raises(NotImplementedError, match=r"cross spectrum"):
        InverseNoiseFilter.from_inv_noiseps(np.ones((N_FREQS, 3)), det_pairs, LAYOUT)


@pytest.mark.parametrize(
    "det_pairs, match",
    [
        ([(0, 0), (1, 1), (5, 5)], r"refers to detector 5"),
        ([(0, 0), (1, 1)], r"missing = \[2\]"),
        ([(0, 0), (1, 1), (1, 1)], r"repeated = \[1\]"),
    ],
    ids=["out_of_range", "missing", "repeated"],
)
def test_inverse_noise_filter_invalid_pairs_raise(det_pairs, match):
    det_pairs = np.array(det_pairs, dtype=np.uintp)
    inv_noiseps = np.ones((N_FREQS, len(det_pairs)))

    with pytest.raises(ValueError, match=match):
        InverseNoiseFilter.from_inv_noiseps(inv_noiseps, det_pairs, LAYOUT)


# -----------------------------------------------------------------------------
# normal operator and right-hand side
# -----------------------------------------------------------------------------


def test_overlap_save_is_a_linear_convolution():
    # Check that, with every sample in its own pixel, A x is the linear convolution of
    # x with a kernel shorter than the context, across the edges of the blocks and
    # with zeros outside the TOD.
    kernels, weights = _short_kernel(n_det=1)
    operator = _operator(
        np.arange(N_SAMPLES)[:, None], weights, diag=np.ones(N_SAMPLES)
    )
    x = rng.standard_normal(N_SAMPLES)

    np.testing.assert_allclose(
        operator.project(x), np.convolve(x, kernels[0], mode="same"), atol=1e-12
    )


def test_normal_operator_with_unit_weights_is_the_hit_count():
    # With N^-1 = 1, A = P^T P is diagonal: A x = hits * x.
    pixels = _random_pixels()
    operator = _operator(pixels, np.ones((N_DET, N_FREQS)))
    x = rng.standard_normal(N_PIXELS)

    hits = np.bincount(pixels.ravel(), minlength=N_PIXELS)
    np.testing.assert_allclose(operator.project(x), hits * x, atol=1e-12)


def test_normal_operator_symmetric_positive():
    # Check that <y, A x> = <A y, x> and <x, A x> > 0, the two properties the CG needs,
    # with a kernel shorter than the context.
    _, weights = _short_kernel()
    operator = _operator(_random_pixels(), weights)
    x, y = rng.standard_normal((2, N_PIXELS))

    assert np.isclose(y @ operator.project(x), operator.project(y) @ x, rtol=1e-12)
    assert x @ operator.project(x) > 0


def test_normal_operator_does_not_depend_on_workers():
    _, weights = _short_kernel()
    pixels = _random_pixels()
    x = rng.standard_normal(N_PIXELS)

    one = _operator(pixels, weights, workers=1).project(x)
    many = _operator(pixels, weights, workers=3).project(x)

    np.testing.assert_allclose(one, many, rtol=1e-12, atol=1e-12)


def test_normal_operator_precondition():
    # Check that M^-1 r divides by diag, with 0 where diag is 0.
    diag = np.linspace(0, 5, N_PIXELS)
    operator = _operator(_random_pixels(), np.ones((N_DET, N_FREQS)), diag=diag)
    r = rng.standard_normal(N_PIXELS)

    expected = np.zeros(N_PIXELS)
    expected[1:] = r[1:] / diag[1:]
    np.testing.assert_allclose(operator.precondition(r), expected)


def test_build_rhs_with_unit_weights_bins_the_tod():
    # With N^-1 = 1, b = P^T d: the sum of the samples of every pixel.
    pixels = _random_pixels()
    tod = rng.standard_normal((N_SAMPLES, N_DET))
    pointing = PointingOperator(pixels, LAYOUT)
    noise_filter = InverseNoiseFilter(LAYOUT, np.ones((N_DET, N_FREQS)))

    b = _build_rhs(tod, pointing, noise_filter, LAYOUT, N_PIXELS)

    expected = np.bincount(pixels.ravel(), weights=tod.ravel(), minlength=N_PIXELS)
    np.testing.assert_allclose(b, expected, atol=1e-12)


def test_filter_and_bin_is_the_transpose_of_the_gather():
    # Check that <P x, d> = <x, P^T d>: the gather and the scatter of the blocks are
    # the transpose of each other.
    pixels = _random_pixels()
    pointing = PointingOperator(pixels, LAYOUT)
    noise_filter = InverseNoiseFilter(LAYOUT, np.ones((N_DET, N_FREQS)))
    x = rng.standard_normal(N_PIXELS)
    d = rng.standard_normal((N_SAMPLES, N_DET))

    p_transpose_d = _build_rhs(d, pointing, noise_filter, LAYOUT, N_PIXELS)

    assert np.isclose((x[pixels] * d).sum(), x @ p_transpose_d, rtol=1e-12)


def test_filter_and_bin_does_not_depend_on_workers():
    pixels = _random_pixels()
    pointing = PointingOperator(pixels, LAYOUT)
    _, weights = _short_kernel()
    noise_filter = InverseNoiseFilter(LAYOUT, weights)
    x = rng.standard_normal(N_PIXELS)

    def fill(window):
        return pointing.map_to_block(x, window)

    one = _filter_and_bin(fill, noise_filter, pointing, LAYOUT, N_PIXELS, workers=1)
    many = _filter_and_bin(fill, noise_filter, pointing, LAYOUT, N_PIXELS, workers=4)

    np.testing.assert_allclose(one, many, rtol=1e-12, atol=1e-12)


def test_build_diagonal_preconditioner():
    # Every sample adds the zero-lag weight of its detector to its pixel.
    pixels = _random_pixels()
    lag_zero = np.array([1.0, 0.5, 3.0])

    diag = _build_diagonal_preconditioner(
        pixels.T.copy(), lag_zero, N_PIXELS, workers=2
    )

    expected = sum(
        lag_zero[j] * np.bincount(pixels[:, j], minlength=N_PIXELS)
        for j in range(N_DET)
    )
    np.testing.assert_allclose(diag, expected)


def test_normal_equations():
    pixels = _random_pixels()
    tod = rng.standard_normal((N_SAMPLES, N_DET))
    lag_zero = np.array([1.0, 0.5, 3.0])
    inv_noise = InverseNoise(
        inv_noiseps=np.ones((N_FREQS, N_DET)),
        det_pairs=AUTO_PAIRS,
        lag_zero=lag_zero,
        sample_rate=244.0,
        channel_ids=[3, 7, 12],
    )

    operator, b = _normal_equations(tod, pixels, inv_noise, N_PIXELS, workers=1)

    hits = np.bincount(pixels.ravel(), minlength=N_PIXELS)
    np.testing.assert_allclose(
        b,
        np.bincount(pixels.ravel(), weights=tod.ravel(), minlength=N_PIXELS),
        atol=1e-12,
    )
    x = rng.standard_normal(N_PIXELS)
    np.testing.assert_allclose(operator.project(x), hits * x, atol=1e-12)
    # the preconditioner is built from lag_zero
    r = np.ones(N_PIXELS)
    diag = sum(
        lag_zero[j] * np.bincount(pixels[:, j], minlength=N_PIXELS)
        for j in range(N_DET)
    )
    np.testing.assert_allclose(operator.precondition(r), 1 / diag)
