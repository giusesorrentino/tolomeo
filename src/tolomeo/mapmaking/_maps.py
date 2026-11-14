"""
The maps of a scan: ``binned_map`` and ``gls_map``.

``binned_map`` is the mean of the samples of every pixel, with no noise weighting, for
quick checks. ``gls_map`` is the generalized least squares (GLS) map: it builds the
normal equations ``A m = b``, with ``A = P^T N^-1 P`` and ``b = P^T N^-1 d``
(``_operators._normal_equations``), and solves them by preconditioned conjugate
gradient (``_solver._conjugate_gradient``). Both sum the samples of every pixel with
``_bin``.

The other private modules of ``mapmaking``:

==================  ==============================================================
``_pointing.py``    the pixel of every sample of every detector
``_blocks.py``      sizes and positions of the blocks of the noise filter
``_noise.py``       the noise model, the inverse noise spectra
``_operators.py``   ``P`` and ``P^T``, the filter ``N^-1``, the right-hand side,
                    ``A`` and its diagonal preconditioner
``_solver.py``      the conjugate gradient
==================  ==============================================================

Dependencies go one way: ``_blocks`` and ``_noise`` <- ``_operators`` <- this
module; ``_pointing`` and ``_solver`` depend on nothing else in ``mapmaking``.
"""

import time
import warnings

import numpy as np
from numpy.typing import NDArray

from .._lib.parallel import _resolve_workers
from .._typing import ResidualNorm
from ..map import BinnedMapResult, MapGeometry, MapResult
from ..tod import Scan
from ._noise import InverseNoise
from ._operators import _normal_equations
from ._pointing import _pointing_matrix
from ._solver import _conjugate_gradient

__all__ = ["binned_map", "gls_map"]


def binned_map(scan: Scan, geometry: MapGeometry) -> BinnedMapResult:
    """
    Mean and standard deviation of the samples in every pixel, for quick checks.

    A sample is used if it falls inside the map, is not flagged in ``scan.tod.flags``
    and is finite. No noise weighting: every used sample counts the same.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        ``scan.tod.signal`` is ``(n_samples, n_det)`` float64; NaN samples are skipped.
    geometry : :class:`~tolomeo.map.MapGeometry`
        Map grid; the maps have shape ``(ny, nx)``.

    Returns
    -------
    :class:`~tolomeo.map.BinnedMapResult`
        ``image`` (mean, NaN where ``hits == 0``), ``hits`` and ``std`` (sample standard
        deviation, NaN where ``hits <= 1``), all ``(ny, nx)``.
    """
    # (n_samples, n_det) signal, one column per detector
    tod = scan.tod.signal
    n_pixels = geometry.n_pixels

    # pixel of every sample and whether it falls inside the map, both (n_samples, n_det)
    pixels, valid = _pointing_matrix(scan.coordinates, scan.offsets, geometry)

    # a sample is used only if it is inside the map, not flagged in the TOD and finite.
    # scan.tod.flags is (n_samples,): [:, None] repeats it for every detector
    used = valid & scan.tod.flags[:, None] & np.isfinite(tod)

    # for every pixel: the number of samples (hits), the sum of their values and the sum
    # of their squared values
    hits = np.bincount(pixels[used], minlength=n_pixels)
    total = _bin(pixels, used, n_pixels, values=tod)
    sumsq = _bin(pixels, used, n_pixels, values=tod, power=2)

    # the mean needs at least one sample in the pixel, the standard deviation at least two
    observed = hits > 0
    observed_std = hits > 1

    mean = np.full(n_pixels, np.nan)
    var = np.full(n_pixels, np.nan)

    mean[observed] = total[observed] / hits[observed]
    # unbiased sample variance from the sums: (sum v^2 - (sum v)^2 / n) / (n - 1)
    var[observed_std] = (
        sumsq[observed_std] - total[observed_std] ** 2 / hits[observed_std]
    ) / (hits[observed_std] - 1)
    # rounding can make a tiny variance slightly negative: clip it before the square root
    std = np.sqrt(np.maximum(var, 0.0))

    # flat maps back to 2-D images, (ny, nx)
    mean = mean.reshape((geometry.ny, geometry.nx))
    hits = hits.reshape((geometry.ny, geometry.nx))
    std = std.reshape((geometry.ny, geometry.nx))

    return BinnedMapResult(image=mean, hits=hits, std=std, geometry=geometry)


def gls_map(
    scan: Scan,
    geometry: MapGeometry,
    inv_noise: InverseNoise,
    *,
    initial_map: NDArray[np.float64] | None = None,
    residual_norm: ResidualNorm = "plain",
    tol: float = 1e-3,
    maxiter: int = 300,
    min_hits_frac: float | None = None,
    workers: int | None = None,
) -> MapResult:
    """
    Generalized least squares (GLS) map of a cleaned scan.

    Solves ``A m = b`` with ``A = P^T N^-1 P`` and ``b = P^T N^-1 d`` by preconditioned
    conjugate gradient (see ``FUNZIONAMENTO.md``). The samples outside the map or flagged
    in ``scan.tod.flags`` are sent to a virtual pixel and never touch the map.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        Cleaned scan: ``scan.tod.signal`` is ``(n_samples, n_det)`` float64 without NaN,
        ``scan.tod.flags`` is ``(n_samples,)`` bool with ``True`` for a good sample.
    geometry : :class:`~tolomeo.map.MapGeometry`
        Map grid; the map has shape ``(ny, nx)``.
    inv_noise : :class:`~tolomeo.mapmaking.InverseNoise`
        Noise model of ``scan``, from :func:`~tolomeo.mapmaking.estimate_inverse_noise`:
        same detectors, in the same order, and same sampling frequency.
    initial_map : :class:`~numpy.ndarray` of float64, optional
        Starting point of the CG, flat (``iy * nx + ix``). None means the binned map:
        the mean of the used samples of every pixel, 0 where there is none.
        Shape ``(n_pixels,)``.
    residual_norm : :class:`~tolomeo._typing.ResidualNorm`
        Norm of the relative residual that stops the CG (ROMA's ``itol`` 1 and 2).
        Default ``"plain"``.
    tol : float
        Relative residual at which the CG stops.
        Default ``1e-3``.
    maxiter : int
        Maximum number of CG iterations.
        Default ``300``.
    min_hits_frac : float, optional
        Use only the samples that fall in the pixels with at least ``min_hits_frac`` times
        the hits of the most observed pixel: the others are flagged like the samples
        outside the map, and their pixels stay empty (hits 0). Between 0 (included) and 1
        (excluded). None means every pixel.
        Default None.
    workers : int, optional
        Threads of the pointing and the CG, with SciPy's convention:
        None is one thread, a positive number that many, ``-1`` all the cores, ``-2``
        all but one.

    Returns
    -------
    :class:`~tolomeo.map.MapResult`
        ``image``, ``hits`` and ``weights`` with shape ``(ny, nx)``, the number of CG
        iterations, the final relative residual and whether the CG converged. A
        ``RuntimeWarning`` is issued if it did not.

    Raises
    ------
    ValueError
        If an option is out of range, if ``initial_map`` does not have one value per
        pixel, if ``inv_noise`` does not belong to ``scan`` or if no sample is used.
    """
    n_pixels = geometry.n_pixels

    # check the options first: a wrong one must fail now and not after the (slow)
    # pointing computation
    if residual_norm not in ("plain", "preconditioned"):
        raise ValueError(
            f"`residual_norm` must be 'plain' or 'preconditioned', got {residual_norm!r}."
        )
    if min_hits_frac is not None and not 0.0 <= min_hits_frac < 1.0:
        raise ValueError(f"`min_hits_frac` must be in [0, 1), got {min_hits_frac}.")
    if initial_map is not None and np.shape(initial_map) != (n_pixels,):
        raise ValueError(
            f"`initial_map` has shape {np.shape(initial_map)}, expected ({n_pixels},): "
            "one value per pixel of the map."
        )

    # the noise model must belong to this scan: one of another scan would give a wrong
    # map without any error
    inv_noise.check_scan(scan)

    # turn None / negative values into a number of threads, and reject 0: the same
    # number is used by the pointing and the CG
    workers = _resolve_workers(workers)

    # the time-ordered data, (n_samples, n_det)
    tod = scan.tod.signal
    n_samples, n_det = tod.shape

    print(
        f"[POINTING] Projecting {n_samples} samples x {n_det} detectors on a "
        f"{geometry.nx}x{geometry.ny} "
        f"map ({geometry.projection}, pixel = {geometry.pixel_size:.2f} arcsec)"
    )

    # pointing: the pixel of every sample of every detector, and whether it falls inside
    # the map
    t0 = time.perf_counter()
    pixels, valid = _pointing_matrix(
        scan.coordinates, scan.offsets, geometry, workers=workers
    )
    dt = time.perf_counter() - t0
    n_out = int((~valid).sum())

    print(
        f"[POINTING] {n_out} of {valid.size} samples fall outside the map "
        f"({100 * n_out / valid.size:.1f}%), "
        f"done in {dt:.2f}s"
    )

    # a sample is used only if it is inside the map and not flagged in the TOD
    # (scan.tod.flags is (n_samples,): the [:, None] repeats it for every detector)
    valid &= scan.tod.flags[:, None]

    if not valid.any():
        raise ValueError(
            "No valid sample: all the samples fall outside the map or are flagged. "
            "Check geometry.center (radians) and the pixel size."
        )

    # hits per pixel: the samples that are used and fall in the pixel
    hits = np.bincount(pixels[valid], minlength=n_pixels)

    # pixels with too few hits: their samples are flagged too, and the pixels stay empty
    if min_hits_frac is not None:
        keep = hits >= min_hits_frac * hits.max()
        n_pix_drop = int(((hits > 0) & ~keep).sum())
        n_samp_drop = int(hits[~keep].sum())
        # pixels is 0 where valid is False, so keep[pixels] is always a valid index
        valid &= keep[pixels]
        hits[~keep] = 0
        print(
            f"[POINTING] min_hits_frac = {min_hits_frac}: {n_samp_drop} samples in "
            f"{n_pix_drop} pixels with fewer than {min_hits_frac * hits.max():.0f} hits "
            "are not used"
        )

    # starting point of the CG: the binned map (mean of the used samples of every pixel,
    # 0 in the pixels without samples)
    if initial_map is None:
        total = _bin(pixels, valid, n_pixels, values=tod)
        initial_map = np.divide(total, hits, out=np.zeros(n_pixels), where=hits > 0)
        print("[MAP] Starting the CG from the binned map")

    # the flagged samples are not dropped: they still go through the noise filter, so
    # that it sees a continuous time stream, but they point to one more unknown, the
    # virtual pixel (index n_pixels), which is thrown away at the end
    n_flagged = valid.size - np.count_nonzero(valid)
    print(
        f"[MAPMAKER] {n_flagged} of {valid.size} samples are flagged "
        f"({100.0 * n_flagged / valid.size:.1f}%): "
        "they are assigned to a virtual pixel"
    )
    # np.where makes a copy (`pixels` is still needed for the weight map); the virtual
    # pixel is a scalar of the dtype of `pixels`, so that the copy keeps that dtype
    solver_pixels = np.where(valid, pixels, pixels.dtype.type(n_pixels))
    n_unknowns = n_pixels + 1

    print(
        f"[MAPMAKER] Config: TOD size = {n_samples}, pixels = {n_pixels}, "
        f"detectors = {n_det}, workers = {workers}"
    )
    start_total = time.perf_counter()

    operator, b = _normal_equations(tod, solver_pixels, inv_noise, n_unknowns, workers)
    del solver_pixels  # the operator keeps its own copy, by detector

    # the virtual pixel starts from 0
    x = np.zeros(n_unknowns)
    x[:n_pixels] = initial_map

    print(f"[MAPMAKER] Launching CG solver (tol = {tol:.1e}, maxiter = {maxiter})")
    n_iter, residual = _conjugate_gradient(b, x, operator, residual_norm, tol, maxiter)

    print(
        f"[MAPMAKER] Map-making completed in {time.perf_counter() - start_total:.2f}s "
        f"(steps: {n_iter}, final residual: {residual:.3e})"
    )

    # the CG stopped at maxiter before reaching tol: the map is not the GLS solution yet
    # (the large scales are the last to converge)
    converged = bool(residual <= tol)
    if not converged:
        warnings.warn(
            f"The conjugate gradient did not converge: residual {residual:.2e} > tol "
            f"{tol:.1e} after {n_iter} iterations. Increase `maxiter`.",
            RuntimeWarning,
            stacklevel=2,
        )

    # weight map: for every pixel, the sum of the zero-lag weight of the detector of
    # each sample used in it (the diagonal of the preconditioner, without the virtual
    # pixel)
    weights = _bin(pixels, valid, n_pixels, det_weights=inv_noise.lag_zero)

    n_hit = int((hits > 0).sum())
    print(
        f"[MAP] {n_hit} of {n_pixels} pixels have at least one hit "
        f"({100 * n_hit / n_pixels:.1f}%), max hits = {hits.max()}"
    )

    # the flat maps go back to 2-D, (ny, nx); the virtual pixel is dropped
    shape = (geometry.ny, geometry.nx)
    return MapResult(
        image=x[:n_pixels].reshape(shape),
        hits=hits.reshape(shape),
        n_iter=n_iter,
        residual=residual,
        geometry=geometry,
        weights=weights.reshape(shape),
        converged=converged,
        unit=scan.tod.unit,
    )


def _bin(
    pixels: NDArray[np.integer],
    used: NDArray[np.bool_],
    n_pixels: int,
    *,
    values: NDArray[np.float64] | None = None,
    power: int = 1,
    det_weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Sum over the used samples of every pixel.

    One detector at a time, so that no array with all the used samples is created.

    Parameters
    ----------
    pixels, used : :class:`~numpy.ndarray`
        Pixel of every sample, and whether it is used.
        Shape ``(n_samples, n_det)``.
    n_pixels : int
        Pixels of the map.
    values : :class:`~numpy.ndarray` of float64, optional
        What is summed, raised to ``power``. None means 1 for every sample: the sum is
        the number of samples.
        Shape ``(n_samples, n_det)``.
    power : int
        Power of ``values``.
        Default ``1``.
    det_weights : :class:`~numpy.ndarray` of float64, optional
        One factor per detector, applied to the sums of its samples (for example
        ``lag_zero``).
        Shape ``(n_det,)``.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        Shape ``(n_pixels,)``.
    """
    total = np.zeros(n_pixels)
    for idet in range(pixels.shape[1]):
        rows = used[:, idet]
        weights = None if values is None else values[rows, idet] ** power
        sums = np.bincount(pixels[rows, idet], weights=weights, minlength=n_pixels)
        total += sums if det_weights is None else det_weights[idet] * sums
    return total
