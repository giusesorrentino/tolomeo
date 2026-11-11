"""
GLS map of a cleaned scan: ``gls_map`` and, below it, ``_tod2map``.

``gls_map`` is the entry point. It takes a ``Scan``, a ``MapGeometry`` and the noise
model of the scan (an ``InverseNoise``, from ``estimate_inverse_noise``), projects the
samples on the map and returns a ``MapResult``. The heavy part is
``_tod2map``, which knows nothing about ``Scan``: from arrays to the flat map, by
preconditioned conjugate gradient.

The other private modules of ``mapmaking`` each mirror one file of the Rust reference
implementation, which lives in ``rust/src``:

======================  ==========================================================
``_blocks.py``          ``roma/config.rs``  (block sizes and positions)
``_flags.py``           ``roma/temperature/flags.rs``
``_noise.py``           ``roma/operators/noise_filter.rs`` and ``roma/fft.rs``
                        (the estimation of the spectra has no Rust counterpart)
``_operators.py``       ``roma/operators/pointing.rs``, ``temperature/rhs.rs`` and
                        ``temperature/normal_operator.rs``
``_preconditioner.py``  ``roma/temperature/preconditioner.rs``
``_solver.py``          ``roma/solver/*``
``_gls.py``             ``roma/temperature/mapmaker.rs`` and ``python.rs``
======================  ==========================================================

Dependencies go one way: ``_blocks`` <- ``_noise`` <- ``_operators`` <- this
module; ``_flags``, ``_preconditioner`` and ``_solver`` depend on nothing in the package.

``_tod2map`` keeps the arguments and the three returned values of the Rust
``tod2map_ffi`` (plus the optional ``workers``), so the two implementations can be
compared on identical inputs.

Array conventions of ``_tod2map`` (as they arrive from ``gls_map``):

* ``tod``         ``(n_samples, n_det)`` float64
* ``pixels``   ``(n_samples, n_det)`` uintp: flat pixel ``iy * nx + ix`` of every sample
* ``valid``       ``(n_samples, n_det)`` bool or ``None``; ``False`` = flagged sample
* ``inv_noiseps`` ``(n_freqs, n_cols)`` float64
* ``det_pairs``   ``(n_cols, 2)`` uintp
* ``lag_zero``    ``(n_det,)`` float64
* ``n_pixels``    pixels of the sky map, virtual pixel *not* included
"""

import time
import warnings

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .._lib.parallel import _resolve_workers
from .._typing import ResidualNorm
from ..map import MapGeometry, MapResult
from ..tod import Scan
from ._blocks import BlockLayout
from ._flags import _count_flagged, _first_out_of_range_pixel, _redirect_flagged_samples
from ._noise import InverseNoise, InverseNoiseFilter
from ._operators import PointingOperator, TemperatureNormalOperator, _build_rhs
from ._pointing import _pointing_matrix
from ._preconditioner import _build_diagonal_preconditioner
from ._solver import _conjugate_gradient

__all__ = ["gls_map"]


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
    """
    # check the options first: a wrong one must fail now and not after the (slow)
    # pointing computation
    if residual_norm not in ("plain", "preconditioned"):
        raise ValueError(
            f"`residual_norm` must be 'plain' or 'preconditioned', got {residual_norm!r}."
        )
    if min_hits_frac is not None and not 0.0 <= min_hits_frac < 1.0:
        raise ValueError(f"`min_hits_frac` must be in [0, 1), got {min_hits_frac}.")

    # the noise model must belong to this scan: one of another scan would give a wrong
    # map without any error
    _check_inv_noise(scan, inv_noise)

    # turn None / negative values into a number of threads, and reject 0: the same
    # number is used by the pointing and the CG
    workers = _resolve_workers(workers)

    # the time-ordered data, (n_samples, n_det), in the form that the kernels read
    tod = _as_kernel_array(scan.tod.signal, np.float64, "tod")
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
    hits = np.bincount(pixels[valid], minlength=geometry.n_pixels)

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

    n_pixels = geometry.n_pixels

    # starting point of the CG: the binned map (mean of the used samples of every pixel,
    # 0 in the pixels without samples). One detector at a time, so that no
    # (n_samples, n_det) array is created
    if initial_map is None:
        total = np.zeros(n_pixels)
        for idet in range(n_det):
            used = valid[:, idet]
            total += np.bincount(
                pixels[used, idet], weights=tod[used, idet], minlength=n_pixels
            )
        initial_map = np.divide(total, hits, out=np.zeros(n_pixels), where=hits > 0)
        print("[MAP] Starting the CG from the binned map")

    image_flat, n_iter, residual = _tod2map(
        tod,
        pixels,
        valid,
        inv_noise.inv_noiseps,
        inv_noise.det_pairs,
        inv_noise.lag_zero,
        n_pixels,
        initial_map,
        residual_norm,
        tol,
        maxiter,
        workers=workers,
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
    # pixel). One detector at a time, so that no (n_samples, n_det) array is created
    weights = np.zeros(n_pixels)
    for idet, weight in enumerate(inv_noise.lag_zero):
        used = valid[:, idet]
        weights += weight * np.bincount(pixels[used, idet], minlength=n_pixels)

    # hit map: computed above, before the CG
    n_hit = int((hits > 0).sum())
    print(
        f"[MAP] {n_hit} of {geometry.n_pixels} pixels have at least one hit "
        f"({100 * n_hit / geometry.n_pixels:.1f}%), max hits = {hits.max()}"
    )

    # the flat map goes back to 2-D, (ny, nx)
    return MapResult(
        image=image_flat.reshape((geometry.ny, geometry.nx)),
        hits=hits.reshape((geometry.ny, geometry.nx)),
        n_iter=n_iter,
        residual=residual,
        geometry=geometry,
        weights=weights.reshape((geometry.ny, geometry.nx)),
        converged=converged,
        unit=scan.tod.unit,
    )


def _check_inv_noise(scan: Scan, inv_noise: InverseNoise) -> None:
    """
    Refuse a noise model that does not belong to ``scan``. TODO: to implement.

    A model of another scan would give a wrong map without any error.
    """
    # - inv_noise.detector_ids must be equal to scan.detector_ids, in the same order:
    #   column j of inv_noiseps is used for column j of the TOD.
    # - inv_noise.sample_rate must be equal to scan.tod.sample_rate within a small
    #   relative tolerance (both come from the median time step): the bins of the
    #   spectra are at k * sample_rate / block_len.
    # - raise a ValueError that says what differs (for the detectors: how many, or the
    #   first position where they differ).


def _as_kernel_array(array: ArrayLike, dtype: DTypeLike, name: str) -> NDArray:
    """
    Return ``array`` as the C-contiguous ``dtype`` array that the Rust kernel needs.

    The kernel reads every array as one flat block of memory, in row-major order, and
    does not convert types: an array of the wrong type raises a TypeError, and an array
    in column-major (Fortran) order is accepted but read in the wrong order, which would
    give a wrong map without any error.

    If ``array`` is already right it is returned as it is, without a copy. Otherwise a
    converted copy is made and a ``RuntimeWarning`` says which array it was and why. It is
    a warning and not a log because the logger of the package has a NullHandler, so a log
    would stay invisible unless the user enables logging.

    Parameters
    ----------
    array : array_like
        The array to check.
    dtype : numpy dtype
        The type the kernel needs for this array.
    name : str
        Name of the array, only used in the warning.
    """
    array = np.asarray(array)

    reasons = []
    if array.dtype != np.dtype(dtype):
        reasons.append(f"it is {array.dtype} and the kernel needs {np.dtype(dtype)}")
    if not array.flags.c_contiguous:
        reasons.append("it is not C-contiguous")

    if reasons:
        reason = " and ".join(reasons)
        warnings.warn(
            f"[INPUT] `{name}`: {reason}, making a converted copy",
            RuntimeWarning,
            stacklevel=2,
        )

    return np.ascontiguousarray(array, dtype=dtype)


def _check_inputs(
    tod: NDArray[np.float64],
    pixels: NDArray[np.integer],
    valid: NDArray[np.bool_] | None,
    inv_noiseps: NDArray[np.float64],
    det_pairs: NDArray[np.integer],
    lag_zero: NDArray[np.float64],
    n_pixels: int,
    initial_map: NDArray[np.float64] | None,
    residual_norm: str,
) -> None:
    """
    Raise ``ValueError`` (or ``TypeError``) with a clear message if an input is wrong.

    The checks, in order:

    1. ``residual_norm`` is ``"plain"`` or ``"preconditioned"``;
    2. ``tod`` is 2-D and ``pixels`` has its shape;
    3. ``valid``, if given, has the shape of ``tod``;
    4. ``inv_noiseps`` is 2-D with at least 2 rows (``n_freqs``);
    5. ``lag_zero`` has ``n_det`` elements;
    6. ``det_pairs`` has shape ``(n_cols, 2)``;
    7. ``inv_noiseps`` has ``n_cols`` columns;
    8. ``initial_map``, if given, has ``n_pixels`` elements;
    9. the dtypes of ``pixels``, ``valid`` and ``det_pairs``;
    10. no *valid* sample has a pixel outside ``0 .. n_pixels``
        (``_flags._first_out_of_range_pixel``; the message names the sample and detector).
    """
    if residual_norm not in ("plain", "preconditioned"):
        raise ValueError(
            f"`residual_norm` must be 'plain' or 'preconditioned', got {residual_norm!r}"
        )

    # the number of dimensions is checked first, so that the shape messages below can
    # index the shapes safely
    if tod.ndim != 2:
        raise ValueError(f"`tod` must be 2-D (n_samples, n_det), got shape {tod.shape}")

    if pixels.shape != tod.shape:
        raise ValueError(
            f"`tod` shape {tod.shape} does not match `pixels` shape {pixels.shape}"
        )

    if valid is not None and valid.shape != tod.shape:
        raise ValueError(
            f"`valid` shape {valid.shape} does not match `tod` shape {tod.shape}"
        )

    n_det = tod.shape[1]

    if inv_noiseps.ndim != 2:
        raise ValueError(
            f"`inv_noiseps` must be 2-D (n_freqs, n_cols), got shape {inv_noiseps.shape}"
        )

    # the filter blocks have 2 * (n_freqs - 1) samples: n_freqs = 1 would give empty ones
    if inv_noiseps.shape[0] < 2:
        raise ValueError("`inv_noiseps` must contain at least two frequency bins")

    if lag_zero.shape != (n_det,):
        raise ValueError(
            f"`lag_zero` must have n_det = {n_det} elements, got shape {lag_zero.shape}"
        )

    if det_pairs.ndim != 2 or det_pairs.shape[1] != 2:
        raise ValueError(f"`det_pairs` shape {det_pairs.shape} must be (n_cols, 2)")

    # one column of inv_noiseps per detector pair
    n_cols = det_pairs.shape[0]
    if inv_noiseps.shape[1] != n_cols:
        raise ValueError(
            f"`inv_noiseps` has {inv_noiseps.shape[1]} columns, but `det_pairs` "
            f"has {n_cols} rows"
        )

    if initial_map is not None and initial_map.shape != (n_pixels,):
        raise ValueError(
            f"`initial_map` shape {initial_map.shape} must be ({n_pixels},), "
            "one value per map pixel"
        )

    # NumPy does not fix the dtypes in the signature. A float `pixels` would fail much
    # later with a cryptic IndexError, and an integer `valid` would not behave as a mask
    if not np.issubdtype(pixels.dtype, np.integer):
        raise TypeError(f"`pixels` must have an integer dtype, got {pixels.dtype}")

    if valid is not None and valid.dtype != np.bool_:
        raise TypeError(f"`valid` must have dtype bool, got {valid.dtype}")

    if not np.issubdtype(det_pairs.dtype, np.integer):
        raise TypeError(
            f"`det_pairs` must have an integer dtype, got {det_pairs.dtype}"
        )

    # only the samples that are really used must point inside the map: the flagged
    # ones are sent to the virtual pixel, whatever their pixel is
    found = _first_out_of_range_pixel(pixels, valid, n_pixels)
    if found is not None:
        sample, det = found
        raise ValueError(
            f"`pixels[{sample}, {det}]` = {pixels[sample, det]} is outside the "
            f"map (n_pixels = {n_pixels}). Flag the sample (`flags`) or fix the "
            "pixelization."
        )


def _tod2map(
    tod: NDArray[np.float64],
    pixels: NDArray[np.integer],
    valid: NDArray[np.bool_] | None,
    inv_noiseps: NDArray[np.float64],
    det_pairs: NDArray[np.integer],
    lag_zero: NDArray[np.float64],
    n_pixels: int,
    initial_map: NDArray[np.float64] | None,
    residual_norm: ResidualNorm,
    tol: float,
    maxiter: int,
    workers: int | None = None,
) -> tuple[NDArray[np.float64], int, float]:
    """
    GLS map by preconditioned CG. Returns ``(map_flat, iterations, residual)``.

    The first eleven arguments, and the three returned values, are those of the Rust
    ``tod2map_ffi`` (array conventions in the docstring of the module). The map is flat:
    ``n_pixels`` elements, the virtual pixel is not included.

    ``workers`` is the number of threads that process the blocks of the TOD, with the
    convention of SciPy: ``None`` is 1, a positive integer is that number, a negative one
    counts back from the number of cores (``-1`` all the cores, ``-2`` all but one) and
    ``0`` is an error (see ``_resolve_workers``). The FFT inside each thread always runs
    with one worker. For a fair comparison with the Rust backend, set ``RAYON_NUM_THREADS``
    to the same number of threads.
    """

    # 1. refuse inconsistent inputs, before any heavy work
    _check_inputs(
        tod=tod,
        pixels=pixels,
        valid=valid,
        inv_noiseps=inv_noiseps,
        det_pairs=det_pairs,
        lag_zero=lag_zero,
        n_pixels=n_pixels,
        initial_map=initial_map,
        residual_norm=residual_norm,
    )

    # decided once, here, and passed to _build_rhs and to the operator: the same number
    # in the RHS and in every A x, and the one printed in the log. (From gls_map it is
    # already a positive integer, and resolving it again changes nothing)
    workers = _resolve_workers(workers)

    start_total = time.perf_counter()

    # 2. flagged samples: count them and send them to the virtual pixel, which is one
    # more unknown after the map pixels (index n_pixels)
    if valid is not None:
        n_flagged = _count_flagged(valid)
        print(
            f"[MAPMAKER] {n_flagged} of {valid.size} samples are flagged "
            f"({100.0 * n_flagged / max(valid.size, 1):.1f}%): "
            "they are assigned to a virtual pixel"
        )
        pixels = _redirect_flagged_samples(
            pixels=pixels, valid=valid, virtual_pixel=n_pixels
        )
        n_unknowns = n_pixels + 1
    else:
        n_unknowns = n_pixels

    # 3. sizes of the TOD and of the filter blocks
    n_samples, n_det = tod.shape
    n_freqs = inv_noiseps.shape[0]

    layout = BlockLayout(
        n_samples=n_samples,
        n_det=n_det,
        n_freqs=n_freqs,
    )

    print("[MAPMAKER] Starting Temperature Mapmaking pipeline")
    print(
        f"[MAPMAKER] Config: TOD size = {n_samples}, pixels = {n_pixels}, "
        f"detectors = {n_det}, workers = {workers}"
    )

    # 4. right-hand side b = P^T N^-1 d. The time includes building the pointing
    # operator, which transposes the pixels
    t0 = time.perf_counter()
    pointing = PointingOperator(pixels=pixels, layout=layout)
    noise_filter = InverseNoiseFilter.from_inv_noiseps(
        inv_noiseps=inv_noiseps, det_pairs=det_pairs, layout=layout
    )

    b = _build_rhs(
        tod=tod,
        pointing=pointing,
        noise_filter=noise_filter,
        layout=layout,
        n_unknowns=n_unknowns,
        workers=workers,
    )
    print(f"[MAPMAKER] RHS vector constructed in {time.perf_counter() - t0:.2f}s")

    del noise_filter  # free some memory on my poor Mac

    # 5. diagonal of the preconditioner, from the pixels already transposed by `pointing`
    t1 = time.perf_counter()
    diag = _build_diagonal_preconditioner(
        pixels_by_det=pointing.pixels_by_det,
        lag_zero=lag_zero,
        n_unknowns=n_unknowns,
        workers=workers,
    )
    print(
        f"[MAPMAKER] Diagonal preconditioner built in {time.perf_counter() - t1:.2f}s"
    )

    # the operator below builds its own pointing operator: free this one (1.5 GB)
    del pointing

    # 6. the unknowns x of A x = b, map pixels and virtual pixel, at their starting
    # value: zeros, or initial_map in the first n_pixels elements (the virtual pixel
    # starts at zero)
    x = np.zeros(n_unknowns)

    if initial_map is not None:
        x[:n_pixels] = initial_map

    # 7. normal operator A = P^T N^-1 P with its preconditioner, and the CG that solves
    # A x = b
    operator = TemperatureNormalOperator(
        diag=diag,
        layout=layout,
        pixels=pixels,
        inv_noiseps=inv_noiseps,
        det_pairs=det_pairs,
        workers=workers,
    )

    print(f"[MAPMAKER] Launching CG solver (tol = {tol:.1e}, maxiter = {maxiter})")

    n_iter, residual = _conjugate_gradient(b, x, operator, residual_norm, tol, maxiter)

    total_time = time.perf_counter() - start_total
    print(
        f"[MAPMAKER] Pipeline completed in {total_time:.2f}s "
        f"(steps: {n_iter}, final residual: {residual:.3e})"
    )

    # drop the virtual pixel (if any); a copy, so that the n_unknowns array can be freed
    return x[:n_pixels].copy(), n_iter, residual
