from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from math import ceil
from os import cpu_count

import numpy as np
from astropy.stats import sigma_clip
from numpy.typing import NDArray

from ..coords.transforms import radec2latlon
from ..map import MapGeometry
from ..tod import Scan
from .config import EdgeMask, Mask, RadiusMask, SigmaClipMask

ARCMIN2RAD = np.pi / (180 * 60)
BLOCK = 16384


def _radius_mask(
    scan: Scan, mask: RadiusMask, geometry: MapGeometry
) -> NDArray[np.bool_]:
    """
    Samples that are far from the source, for every detector.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        Scan to clean. Its boresight (``scan.coordinates``) and the detector offsets
        (``scan.offsets``) give the position of every detector at every sample.
    mask : :class:`~tolomeo.processing.RadiusMask`
        ``mask.radius`` is in arcmin. ``mask.center`` is ``"auto"``, that is
        :attr:`~tolomeo.tod.Scan.center`, or ``(ra, dec)`` in degrees.

    Returns
    -------
    :class:`~numpy.ndarray` of bool
        True where the detector is at least ``mask.radius`` away from the center: the
        samples that can be used to estimate the background.
        Shape ``(n_samples, n_det)``.
    """
    if mask.center == "auto":
        ra0, dec0 = scan.center
    else:
        ra0, dec0 = np.deg2rad(mask.center)

    radius = mask.radius * ARCMIN2RAD
    coords, offsets = scan.coordinates, scan.offsets
    n_samples, n_det = coords.ra.size, offsets["lon"].size
    background = np.zeros((n_samples, n_det), dtype=bool)

    def mask_block(start: int, stop: int) -> None:
        lat, lon = radec2latlon(
            coords.ra[start:stop],
            coords.dec[start:stop],
            ra0,
            dec0,
            geometry.projection,
            coords.par_angle[start:stop],
            offsets["lon"],
            offsets["lat"],
            frame=geometry.frame,
        )

        # a boresight with NaN coordinates gives a NaN distance, and NaN >= radius is
        # False: those samples come out as not usable, which is what we want
        background[start:stop] = (np.hypot(lat, lon) >= radius).T

    workers = cpu_count() or 1
    futures = []

    block_len = min(BLOCK, ceil(n_samples / (4 * workers)))
    chunk_edges = [*range(0, n_samples, block_len), n_samples]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start, stop in pairwise(chunk_edges):
            futures.append(executor.submit(mask_block, start, stop))

    for future in futures:
        future.result()  # nothing to collect: it only re-raises the errors of the blocks

    return background


def _sigma_clip_mask(scan: Scan, mask: SigmaClipMask) -> NDArray[np.bool_]:
    """
    Samples that do not stand out of the noise, for every detector.

    The source is not looked for by position but by amplitude: in each subscan, the
    samples of a detector that are more than ``mask.sigma`` standard deviations away
    from its median are taken to be the source, and left out.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        Scan to clean. Only ``scan.tod`` is used: the signal, the flags and the
        subscan boundaries (``segments``).
    mask : :class:`~tolomeo.processing.SigmaClipMask`
        ``mask.sigma`` and ``mask.maxiters`` go to :func:`astropy.stats.sigma_clip`.

    Returns
    -------
    :class:`~numpy.ndarray` of bool
        True where the sample can be used to estimate the background: it was not
        clipped and it is not flagged. Shape ``(n_samples, n_det)``, one mask per
        detector.
    """
    tod = scan.tod
    background = np.zeros_like(tod.signal, dtype=bool)

    workers = cpu_count() or 1
    futures = []

    def mask_block(start: int, stop: int) -> None:
        signal = np.where(tod.flags[start:stop, None], tod.signal[start:stop], np.nan)
        clipped = sigma_clip(
            signal, sigma=mask.sigma, maxiters=mask.maxiters, axis=0, stdfunc="mad_std"
        )
        background[start:stop] = ~clipped.mask

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start, stop in pairwise(tod.segments):
            if stop <= start:
                continue
            futures.append(executor.submit(mask_block, start, stop))

    for future in futures:
        future.result()

    return background


def _edge_mask(scan: Scan, mask: EdgeMask) -> NDArray[np.bool_]:
    """
    Samples at the two ends of every subscan.

    It assumes that the telescope scans across the source, so that the source is
    in the middle of each subscan and the two ends are clean.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        Scan to clean. Only ``scan.tod`` is used: the flags and the subscan
        boundaries (``segments``).
    mask : :class:`~tolomeo.processing.EdgeMask`
        ``mask.fraction`` is the part of each subscan, at each end, that is used.

    Returns
    -------
    :class:`~numpy.ndarray` of bool
        True for the first and the last ``mask.fraction`` of the samples of each
        subscan, and not flagged. Shape ``(n_samples,)``: the same samples for every
        detector, which is accepted by the steps and takes far less memory than one
        mask per detector.
    """
    tod = scan.tod
    background = np.zeros(tod.signal.shape[0], dtype=bool)

    for start, stop in pairwise(tod.segments):
        if stop <= start:
            continue
        n_edge = int(mask.fraction * (stop - start))
        background[start : start + n_edge] = True
        background[stop - n_edge : stop] = True
        background[start:stop] &= scan.tod.flags[start:stop]

    return background


def _build_mask(
    scan: Scan, mask: Mask, geometry: MapGeometry | None = None
) -> NDArray[np.bool_]:
    """
    Build the ``background`` array of the mask of the config.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        Scan to clean.
    mask : :class:`~tolomeo.processing.RadiusMask`, ``SigmaClipMask`` or ``EdgeMask``
        The mask of :class:`~tolomeo.processing.FilteringConfig`. The kind of mask
        is its ``type``.
    geometry : :class:`~tolomeo.map.MapGeometry`, optional
        Needed only by :class:`~tolomeo.processing.RadiusMask`.

    Returns
    -------
    :class:`~numpy.ndarray` of bool
        True for the samples that can be used to estimate the background. Shape
        ``(n_samples,)`` or ``(n_samples, n_det)``, depending on the mask.
    """
    match mask:
        case RadiusMask():
            if geometry is None:
                raise ValueError(
                    "A radius mask needs `geometry`: the distance from the source is "
                    "measured on the plane of the projection of the map."
                )
            return _radius_mask(scan, mask, geometry)
        case SigmaClipMask():
            return _sigma_clip_mask(scan, mask)
        case EdgeMask():
            return _edge_mask(scan, mask)
        case _:
            raise TypeError(f"Unknown mask: {mask!r}")
