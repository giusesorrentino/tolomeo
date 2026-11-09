"""Pointing matrix: the map pixel seen by every detector at every time sample."""

from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from math import ceil

import numpy as np
from numpy.typing import NDArray

from ..coords import Coordinates
from ..coords.transforms import radec2xy, rot
from ..map import MapGeometry

_BLOCK = 16384


def _pointing_matrix(
    coords: Coordinates,
    offsets: dict[str, NDArray[np.float64]],
    geometry: MapGeometry,
    workers: int = 1,
) -> tuple[NDArray[np.uintp], NDArray[np.bool_]]:
    """
    Pixel of every sample of every detector, and whether it falls inside the map.

    Parameters
    ----------
    coords : :class:`~tolomeo.coords.Coordinates`
        Boresight ``ra``, ``dec`` and ``par_angle``, each ``(n_samples,)``, in radians.
    offsets : dict
        ``"lon"`` and ``"lat"``: offset of every detector from the boresight, each
        ``(n_det,)``, in radians.
    geometry : :class:`~tolomeo.map.MapGeometry`
        Map grid and projection, centred on ``geometry.center``.

    Returns
    -------
    pixels : :class:`~numpy.ndarray` of uintp
        Flat pixel ``iy * nx + ix``. It is 0 where ``valid`` is False, so it only means
        something together with ``valid``.
        Shape ``(n_samples, n_det)``.
    valid : :class:`~numpy.ndarray` of bool
        True where the sample falls inside the map.
        Shape ``(n_samples, n_det)``.
    """
    if geometry.frame == "azel":
        raise NotImplementedError(
            "Frame 'azel' is not implemented in the mapmaker yet."
        )

    # centre of the map, which is also the tangent point of the projection (radians)
    ra0, dec0 = geometry.center
    ra, dec = coords.ra, coords.dec

    pixel_size = np.deg2rad(geometry.pixel_size / 3600.0)  # arcsec -> radians
    nx, ny = geometry.nx, geometry.ny
    # position of the lower-left corner of the map on the tangent plane: the map is
    # centred on the tangent point
    xmin = -nx * pixel_size / 2
    ymin = -ny * pixel_size / 2

    n_samples = coords.ra.size
    n_det = offsets["lon"].size

    # outputs, both (n_samples, n_det): the flat pixel index iy * nx + ix of every sample,
    # and whether the sample falls inside the map. `pixels` is 0 where `valid` is False,
    # so it only means something together with `valid`
    pixels = np.zeros((n_samples, n_det), dtype=np.uintp)
    valid = np.zeros((n_samples, n_det), dtype=bool)

    # 1. boresight on the tangent plane, shape (n_samples,)
    x, y = radec2xy(ra, dec, ra0, dec0, projection=geometry.projection)

    block_len = min(_BLOCK, ceil(n_samples / (4 * workers)))
    chunk_edges = [*range(0, n_samples, block_len), n_samples]

    def project_block(start: int, stop: int) -> None:
        """
        Fill ``pixels[start:stop]`` and ``valid[start:stop]`` for every detector.

        Each task writes only its own rows of the two shared arrays, so no lock is
        needed.
        """
        x_blk, y_blk, p_blk = x[start:stop], y[start:stop], coords.par_angle[start:stop]
        for j in range(n_det):
            # offset of detector j from the boresight (radians)
            x_off, y_off = offsets["lon"][j], offsets["lat"][j]

            # the offset rotates with the parallactic angle, so it changes along the scan
            x_rot, y_rot = rot(x_off, y_off, p_blk)

            # position of the detector on the tangent plane. x_det grows toward the west,
            # so that the east is on the left of the image, as in the standard orientation
            x_det = -x_blk + x_rot
            y_det = y_blk + y_rot

            # pixel indices: distance from the lower-left corner in pixel units, rounded down
            # (still floats here, and NaN where the coordinates are NaN)
            ix = np.floor((x_det - xmin) / pixel_size)
            iy = np.floor((y_det - ymin) / pixel_size)

            # inside the map if both indices are in range. Comparing a NaN would warn: it
            # is silenced, and the comparison is simply False
            with np.errstate(invalid="ignore"):
                inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

            # flat row-major index, as in an image of shape (ny, nx); the samples outside get
            # pixel 0 so that the index is always valid
            pixels[start:stop, j] = np.where(inside, iy * nx + ix, 0).astype(np.uintp)
            valid[start:stop, j] = inside

    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start, stop in pairwise(chunk_edges):
            futures.append(executor.submit(project_block, start, stop))

    for future in futures:
        future.result()

    # # 2. one detector at a time: its position is the boresight plus its offset
    # for j in range(n_det):
    #     # offset of detector j from the boresight (radians)
    #     x_off, y_off = offsets["lon"][j], offsets["lat"][j]
    #     par_angle = coords.par_angle

    #     # the offset rotates with the parallactic angle, so it changes along the scan
    #     x_rot, y_rot = rot(x_off, y_off, par_angle)

    #     # position of the detector on the tangent plane. x_det grows toward the west,
    #     # so that the east is on the left of the image, as in the standard orientation
    #     x_det = -x + x_rot
    #     y_det = y + y_rot

    #     # pixel indices: distance from the lower-left corner in pixel units, rounded down
    #     # (still floats here, and NaN where the coordinates are NaN)
    #     ix = np.floor((x_det - xmin) / pixel_size)
    #     iy = np.floor((y_det - ymin) / pixel_size)

    #     # inside the map if both indices are in range. Comparing a NaN would warn: it
    #     # is silenced, and the comparison is simply False
    #     with np.errstate(invalid="ignore"):
    #         inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    #     # flat row-major index, as in an image of shape (ny, nx); the samples outside get
    #     # pixel 0 so that the index is always valid
    #     pixels[:, j] = np.where(inside, iy * nx + ix, 0).astype(np.uintp)
    #     valid[:, j] = inside

    return pixels, valid
