import numpy as np

from ..map import BinnedMapResult, MapGeometry
from ..tod import Scan
from ._pointing import _pointing_matrix

__all__ = ["binned_map"]


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

    # pixel of every sample and whether it falls inside the map, both (n_samples, n_det)
    pixels, valid = _pointing_matrix(scan.coordinates, scan.offsets, geometry)

    # a sample is used only if it is inside the map, not flagged in the TOD and finite.
    # scan.tod.flags is (n_samples,): [:, None] repeats it for every detector
    used = valid & scan.tod.flags[:, None] & np.isfinite(tod)

    # boolean indexing flattens: both are 1-D, one entry per used sample
    used_pixels = pixels[used]
    used_values = tod[used]

    # bincount adds up, for every pixel, what falls into it: the number of samples (hits),
    # the sum of their values and the sum of their squared values
    hits = np.bincount(used_pixels, minlength=geometry.n_pixels)
    total = np.bincount(used_pixels, minlength=geometry.n_pixels, weights=used_values)
    sumsq = np.bincount(
        used_pixels, minlength=geometry.n_pixels, weights=used_values**2
    )

    # the mean needs at least one sample in the pixel, the standard deviation at least two
    observed = hits > 0
    observed_std = hits > 1

    mean = np.full(geometry.n_pixels, np.nan)
    var = np.full(geometry.n_pixels, np.nan)

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
