"""Write a map to FITS, and read it back."""

from pathlib import Path

import numpy as np
from astropy.io import fits

from ..map import MapGeometry, MapResult

__all__ = ["read_map", "write_map"]

_SUFFIXES = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz")


def _check_suffix(path: Path) -> None:
    """Raise ``ValueError`` if ``path`` does not end with a FITS suffix (``_SUFFIXES``)."""
    if not path.name.lower().endswith(_SUFFIXES):
        raise ValueError(f"`path` must end with one of {_SUFFIXES}, got {path.name!r}.")


def write_map(
    result: MapResult,
    path: str | Path,
    *,
    extra_header: dict | None = None,
    overwrite: bool = False,
) -> None:
    """
    Write a map and its hit count to a FITS file.

    The primary HDU (``EXTNAME = "SCI"``) holds ``result.image`` with the WCS of
    ``result.geometry`` and the keywords ``NITER`` and ``RESID``; the extension
    ``"HITS"`` holds ``result.hits`` with the same header.

    Parameters
    ----------
    result : :class:`~tolomeo.map.MapResult`
        The map: ``image`` and ``hits`` of shape ``(ny, nx)``.
    path : str or :class:`~pathlib.Path`
        Output file; it must end with a FITS suffix (``.fits``, ``.fit``, ``.fts``,
        optionally ``.gz``).
    extra_header : dict, optional
        More header keywords (for example ``{"OBJECT": "Jupiter"}``), added to both
        HDUs.
    overwrite : bool
        Replace ``path`` if it exists.
        Default ``False``.
    """

    _check_suffix(Path(path))

    wcs = result.geometry.to_wcs()

    hdr = wcs.to_header()
    hdr.update(extra_header)

    hdr["NITER"] = (result.n_iter, "Conjugate gradient iterations")
    hdr["RESID"] = (result.residual, "Final CG residual")
    hdr["EXTNAME"] = "SCI"

    primary = fits.PrimaryHDU(result.image, header=hdr)
    hits_hdu = fits.ImageHDU(result.hits, header=hdr, name="HITS")

    fits.HDUList([primary, hits_hdu]).writeto(path, overwrite=overwrite)


def read_map(path: str | Path) -> MapResult:
    """
    Read a map written by :func:`write_map`.

    Parameters
    ----------
    path : str or :class:`~pathlib.Path`
        FITS file with the map in the primary HDU and the ``"HITS"`` extension.

    Returns
    -------
    :class:`~tolomeo.map.MapResult`
        ``image`` (float64) and ``hits`` (int64) of shape ``(ny, nx)``, ``n_iter`` and
        ``residual`` from the header, and the geometry rebuilt from ``CDELT2``,
        ``CRVAL1``, ``CRVAL2`` and ``CTYPE1``.

    Notes
    -----
    ``CRPIX`` is not read: the geometry assumes the projection center in the middle
    of the grid, as :meth:`~tolomeo.map.MapGeometry.to_wcs` writes it. A
    FITS file from another program, with a different ``CRPIX``, would get a wrong
    geometry. A file that is not a tolomeo map raises ``ValueError``.
    """
    path = Path(path).resolve()

    _check_suffix(path)

    with fits.open(path) as hdul:
        data = hdul[0].data

        if data is None:
            raise ValueError(
                f"{path} is not a tolomeo map: the primary HDU has no data."
            )
        try:
            image = data.astype(np.float64)
            hits = hdul["HITS"].data.astype(np.int64)
            header = hdul[0].header
            n_iter = header["NITER"]
            residual = header["RESID"]

            ny, nx = image.shape
            geometry = MapGeometry(
                nx=nx,
                ny=ny,
                pixel_size=abs(header["CDELT2"]) * 3600,
                center=(np.deg2rad(header["CRVAL1"]), np.deg2rad(header["CRVAL2"])),
                projection=header["CTYPE1"][-3:].lower(),  # "RA---SIN" -> "sin"
            )
        except KeyError as exc:
            raise ValueError(f"{path} is not a tolomeo map: missing {exc}") from exc

        return MapResult(
            image=image, hits=hits, n_iter=n_iter, residual=residual, geometry=geometry
        )
