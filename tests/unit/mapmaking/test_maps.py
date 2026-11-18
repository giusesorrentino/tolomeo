import numpy as np
import pytest

from tests.common import DEC0, PIXEL_SIZE, RA0, make_scan, with_signal
from tolomeo.map import MapGeometry
from tolomeo.mapmaking import InverseNoise, binned_map, estimate_inverse_noise, gls_map
from tolomeo.mapmaking._pointing import _pointing_matrix

rng = np.random.default_rng(29)

NPERSEG = 64
GEOMETRY = MapGeometry(nx=24, ny=18, pixel_size=PIXEL_SIZE, center=(RA0, DEC0))


def _white_inverse_noise(scan, level=None):
    """Noise model with flat inverse spectra: ``N^-1`` is ``level`` times one."""
    n_det = len(scan.channel_ids)
    level = np.ones(n_det) if level is None else np.asarray(level)
    return InverseNoise(
        inv_noiseps=np.tile(level, (NPERSEG // 2 + 1, 1)),
        det_pairs=np.array([(j, j) for j in range(n_det)], dtype=np.uintp),
        lag_zero=level.copy(),
        sample_rate=scan.tod.sample_rate,
        channel_ids=scan.channel_ids,
    )


def _colored_inverse_noise(scan):
    """
    Noise model with the same smooth inverse spectrum for every detector: ``1/f`` noise
    below a knee at a tenth of the Nyquist frequency, white above, and a weight above
    zero at ``f = 0``. With it ``A`` is (nearly) symmetric and positive definite, so the
    CG and MINRES converge to the same map.
    """
    n_det = len(scan.channel_ids)
    freqs = np.fft.rfftfreq(NPERSEG, d=1 / scan.tod.sample_rate)
    f_knee = 0.1 * freqs[-1]
    level = 1 / (1 + (f_knee / np.maximum(freqs, freqs[1] / 2)) ** 2)
    return InverseNoise(
        inv_noiseps=np.tile(level[:, None], (1, n_det)),
        det_pairs=np.array([(j, j) for j in range(n_det)], dtype=np.uintp),
        lag_zero=np.full(n_det, np.fft.irfft(level, n=NPERSEG)[0]),
        sample_rate=scan.tod.sample_rate,
        channel_ids=scan.channel_ids,
    )


def _expected_binned(scan, geometry=GEOMETRY):
    """Hits, mean and sample standard deviation of every pixel, one pixel at a time."""
    pixels, valid = _pointing_matrix(scan.coordinates, scan.offsets, geometry)
    used = valid & scan.tod.flags[:, None] & np.isfinite(scan.tod.signal)

    hits = np.zeros(geometry.n_pixels, dtype=int)
    mean = np.full(geometry.n_pixels, np.nan)
    std = np.full(geometry.n_pixels, np.nan)
    for p in np.unique(pixels[used]):
        values = scan.tod.signal[used & (pixels == p)]
        hits[p] = values.size
        mean[p] = values.mean()
        if values.size > 1:
            std[p] = values.std(ddof=1)

    shape = (geometry.ny, geometry.nx)
    return hits.reshape(shape), mean.reshape(shape), std.reshape(shape)


# -----------------------------------------------------------------------------
# binned_map
# -----------------------------------------------------------------------------


def test_binned_map_mean_std_hits():
    scan = make_scan()

    result = binned_map(scan, GEOMETRY)

    hits, mean, std = _expected_binned(scan)
    np.testing.assert_array_equal(result.hits, hits)
    np.testing.assert_allclose(result.image, mean, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(result.std, std, rtol=1e-9, equal_nan=True)
    assert result.geometry == GEOMETRY


def test_binned_map_skips_flagged_and_nan_samples():
    scan = make_scan()
    scan.tod.flags[:300] = False
    scan.tod.signal[400:420, 1] = np.nan

    result = binned_map(scan, GEOMETRY)

    hits, mean, _ = _expected_binned(scan)
    np.testing.assert_array_equal(result.hits, hits)
    np.testing.assert_allclose(result.image, mean, rtol=1e-12, equal_nan=True)
    assert np.isfinite(result.image[hits > 0]).all()


def test_binned_map_constant_sky():
    # A sky of 5 everywhere gives 5 in every observed pixel, with no dispersion.
    scan = with_signal(make_scan(), np.full(make_scan().tod.signal.shape, 5.0))

    result = binned_map(scan, GEOMETRY)

    observed = result.hits > 0
    np.testing.assert_allclose(result.image[observed], 5.0)
    np.testing.assert_allclose(result.std[result.hits > 1], 0.0, atol=1e-12)
    assert np.isnan(result.image[~observed]).all()


# -----------------------------------------------------------------------------
# gls_map
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["cg", "minres"])
def test_gls_map_with_white_weights_is_the_binned_map(solver):
    # With N^-1 proportional to the identity, A = P^T P is diagonal and the GLS map is
    # the mean of the samples of every pixel. Starting from zero, the solver must find
    # it.
    scan = make_scan()
    inv_noise = _white_inverse_noise(scan)

    result = gls_map(
        scan,
        GEOMETRY,
        inv_noise,
        initial_map=np.zeros(GEOMETRY.n_pixels),
        solver=solver,
        tol=1e-12,
    )

    binned = binned_map(scan, GEOMETRY)
    observed = binned.hits > 0
    assert result.solver == solver
    assert result.converged
    assert result.n_iter >= 1
    np.testing.assert_array_equal(result.hits, binned.hits)
    np.testing.assert_allclose(
        result.image[observed], binned.image[observed], atol=1e-10
    )
    np.testing.assert_array_equal(result.image[~observed], 0.0)


def test_gls_map_starts_from_the_binned_map():
    # With white weights the binned map already solves the system: no iteration.
    scan = make_scan()

    result = gls_map(scan, GEOMETRY, _white_inverse_noise(scan))

    assert result.n_iter == 0
    assert result.converged
    assert result.solver == "cg"  # the default


@pytest.mark.parametrize("residual_norm", ["plain", "preconditioned", "m_inverse"])
def test_gls_map_cg_and_minres_give_the_same_map(residual_norm):
    # Check that the two solvers, from zero and with a tight tolerance, converge to the
    # same map with colored noise. The tolerance of the comparison, 1e-6 of the peak, is
    # 100 times the relative residual: room for the conditioning of A.
    scan = make_scan()
    inv_noise = _colored_inverse_noise(scan)

    maps = {
        solver: gls_map(
            scan,
            GEOMETRY,
            inv_noise,
            initial_map=np.zeros(GEOMETRY.n_pixels),
            solver=solver,
            residual_norm=residual_norm,
            tol=1e-8,
            workers=1,
        )
        for solver in ["cg", "minres"]
    }

    cg, minres = maps["cg"], maps["minres"]
    assert cg.converged and minres.converged
    assert cg.residual <= 1e-8 and minres.residual <= 1e-8
    np.testing.assert_allclose(
        minres.image, cg.image, rtol=0, atol=1e-6 * np.abs(cg.image).max()
    )


def test_gls_map_weights_and_unit():
    # The weight of a pixel is the sum of the zero-lag weights of its samples.
    scan = make_scan()
    scan = scan.model_copy(update={"tod": scan.tod.model_copy(update={"unit": "K"})})
    level = np.array([1.0, 2.0, 0.5, 4.0])

    result = gls_map(scan, GEOMETRY, _white_inverse_noise(scan, level))

    pixels, valid = _pointing_matrix(scan.coordinates, scan.offsets, GEOMETRY)
    expected = sum(
        level[j] * np.bincount(pixels[valid[:, j], j], minlength=GEOMETRY.n_pixels)
        for j in range(4)
    )
    np.testing.assert_allclose(result.weights.ravel(), expected)
    assert result.unit == "K"


def test_gls_map_flagged_samples_do_not_touch_the_map():
    # Flagged samples go to the virtual pixel: a huge value there changes nothing.
    scan = make_scan()
    scan.tod.flags[100:200] = False
    reference = gls_map(scan, GEOMETRY, _white_inverse_noise(scan), tol=1e-12)

    scan.tod.signal[100:200] = 1e6
    result = gls_map(scan, GEOMETRY, _white_inverse_noise(scan), tol=1e-12)

    np.testing.assert_allclose(result.image, reference.image, atol=1e-8)
    np.testing.assert_array_equal(result.hits, reference.hits)


@pytest.mark.parametrize("solver", ["cg", "minres"])
def test_gls_map_not_converged_warns(solver):
    scan = make_scan()
    inv_noise = estimate_inverse_noise(scan, nperseg=NPERSEG, workers=1)

    with pytest.warns(RuntimeWarning, match=r"did not converge.*Increase `maxiter`"):
        result = gls_map(
            scan,
            GEOMETRY,
            inv_noise,
            initial_map=np.zeros(GEOMETRY.n_pixels),
            solver=solver,
            tol=1e-14,
            maxiter=1,
        )

    assert not result.converged
    assert result.n_iter == 1


def test_gls_map_min_hits_frac():
    # Check that the pixels with fewer hits than the fraction of the maximum are left
    # empty, and the others keep their hits.
    scan = make_scan()
    all_hits = binned_map(scan, GEOMETRY).hits

    result = gls_map(scan, GEOMETRY, _white_inverse_noise(scan), min_hits_frac=0.5)

    keep = all_hits >= 0.5 * all_hits.max()
    np.testing.assert_array_equal(result.hits[keep], all_hits[keep])
    np.testing.assert_array_equal(result.hits[~keep], 0)
    np.testing.assert_array_equal(result.image[~keep], 0.0)


def test_gls_map_does_not_depend_on_workers():
    scan = make_scan()
    inv_noise = estimate_inverse_noise(scan, nperseg=NPERSEG, workers=1)

    one = gls_map(scan, GEOMETRY, inv_noise, workers=1)
    many = gls_map(scan, GEOMETRY, inv_noise, workers=3)

    assert one.n_iter == many.n_iter
    np.testing.assert_allclose(one.image, many.image, atol=1e-10)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"solver": "gc"}, r"`solver` must be 'cg' or 'minres'"),
        ({"residual_norm": "max"}, r"`residual_norm` must be"),
        ({"min_hits_frac": 1.0}, r"`min_hits_frac` must be in \[0, 1\)"),
        ({"initial_map": np.zeros((18, 24))}, r"`initial_map` has shape"),
        ({"workers": 0}, r"workers"),
    ],
    ids=["solver", "residual_norm", "min_hits_frac", "initial_map", "workers"],
)
def test_gls_map_invalid_option_raises(kwargs, match):
    scan = make_scan()

    with pytest.raises(ValueError, match=match):
        gls_map(scan, GEOMETRY, _white_inverse_noise(scan), **kwargs)


def test_gls_map_noise_of_another_scan_raises():
    scan = make_scan()
    other = make_scan(n_det=3)

    with pytest.raises(ValueError, match=r"estimate it on this scan"):
        gls_map(scan, GEOMETRY, _white_inverse_noise(other))


def test_gls_map_no_sample_in_the_map_raises():
    scan = make_scan()
    far = GEOMETRY.model_copy(update={"center": (RA0 + 0.1, DEC0)})

    with pytest.raises(ValueError, match=r"No valid sample"):
        gls_map(scan, far, _white_inverse_noise(scan))
