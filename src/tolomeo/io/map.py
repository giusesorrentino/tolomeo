"""Write a map to FITS, and read it back."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from astropy.io import fits

from ..map import MapGeometry, MapResult

__all__ = ["check_fits_suffix", "read_map", "write_map"]

_SUFFIXES = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz")


def check_fits_suffix(path: Path) -> None:
    """Raise ``ValueError`` if ``path`` does not end with a FITS suffix (``_SUFFIXES``)."""
    if not path.name.lower().endswith(_SUFFIXES):
        raise ValueError(f"`path` must end with one of {_SUFFIXES}, got {path.name!r}.")


def write_map(
    result: MapResult,
    path: str | Path,
    *,
    extra_header: dict | None = None,
    config: str | None = None,
    overwrite: bool = False,
) -> None:
    """
    Write a map, its hit count and its weights to a FITS file.

    * Primary HDU (``EXTNAME = "SCI"``): ``result.image`` with the WCS of
      ``result.geometry`` and the keywords ``BUNIT`` (if ``result.unit`` is known),
      ``NITER``, ``RESID``, ``CONVERGD``, ``SOLVER``, ``TOLOMEO`` (version of the
      package), those of ``extra_header``, and ``FRAME``, ``OBSRA``, ``OBSDEC`` and
      ``PROJ`` (frame, centre in degrees and projection of the geometry, which
      :func:`read_map` needs for an ``"azel"`` map, whose WCS has neither the centre
      nor the projection).
    * ``"HITS"``: ``result.hits``, with the WCS and ``BUNIT = "count"``.
    * ``"WEIGHTS"``: ``result.weights``, with the WCS; only if they are not None.
    * ``"CONFIG"``: a table with one column, ``LINE``, that holds the lines of
      ``config``; only if it is given.

    Parameters
    ----------
    result : :class:`~tolomeo.map.MapResult`
        The map: ``image`` and ``hits`` of shape ``(ny, nx)``.
    path : str or :class:`~pathlib.Path`
        Output file; it must end with a FITS suffix (``.fits``, ``.fit``, ``.fts``,
        optionally ``.gz``).
    extra_header : dict, optional
        More header keywords (for example ``{"OBJECT": "Jupiter"}``), added to the
        image HDUs (``SCI``, ``HITS`` and ``WEIGHTS``).
    config : str, optional
        Text that says how the map was made, for example the YAML file of the run. It
        is stored line by line, so that the FITS file is enough to know it. Characters
        that are not ASCII are written as escape sequences (``\\xe8``).
    overwrite : bool
        Replace ``path`` if it exists. Defaults to ``False``.
    """

    check_fits_suffix(Path(path))
    geometry = result.geometry

    # header shared by the image HDUs: the WCS and the keywords of the caller
    base = geometry.to_wcs().to_header()
    base.update(extra_header or {})

    # what read_map needs to rebuild the geometry: the WCS of an azel map has neither
    # the centre nor the projection
    base["FRAME"] = (geometry.frame, "Frame of the map: radec or azel")
    base["OBSRA"] = (np.rad2deg(geometry.center[0]), "[deg] RA of the map centre")
    base["OBSDEC"] = (np.rad2deg(geometry.center[1]), "[deg] Dec of the map centre")
    base["PROJ"] = (geometry.projection, "Projection of the map")

    try:
        base["TOLOMEO"] = (version("tolomeo"), "Version of tolomeo")
    except PackageNotFoundError:
        pass

    # the keywords of the solver describe the map, not the hits or the weights
    hdr = base.copy()
    if result.unit is not None:
        hdr["BUNIT"] = (result.unit, "Unit of the map")
    hdr["NITER"] = (result.n_iter, "Iterations of the solver")
    hdr["RESID"] = (result.residual, "Final relative residual")
    hdr["CONVERGD"] = (result.converged, "The solver reached the tolerance")
    hdr["SOLVER"] = (result.solver, "Solver of the map: cg or minres")
    hdr["EXTNAME"] = "SCI"

    hdus = [fits.PrimaryHDU(result.image, header=hdr)]

    hits_hdr = base.copy()
    hits_hdr["BUNIT"] = ("count", "Samples used in the pixel")
    hdus.append(fits.ImageHDU(result.hits, header=hits_hdr, name="HITS"))

    if result.weights is not None:
        weights_hdr = base.copy()
        weights_hdr["COMMENT"] = "Sum of the zero-lag noise weights of the samples."
        hdus.append(fits.ImageHDU(result.weights, header=weights_hdr, name="WEIGHTS"))

    if config is not None:
        # FITS strings are ASCII: anything else becomes an escape sequence
        lines = [
            line.encode("ascii", "backslashreplace").decode("ascii")
            for line in config.splitlines()
        ] or [""]
        width = max(1, max(len(line) for line in lines))
        column = fits.Column(name="LINE", format=f"{width}A", array=np.array(lines))
        hdus.append(fits.BinTableHDU.from_columns([column], name="CONFIG"))

    fits.HDUList(hdus).writeto(path, overwrite=overwrite)


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
        ``residual`` from the header, and the geometry rebuilt from the header: the
        pixel size from ``CDELT2``, the frame from ``FRAME``, and the centre and the
        projection from ``CRVAL1``, ``CRVAL2`` and ``CTYPE1`` for ``"radec"``, from
        ``OBSRA``, ``OBSDEC`` and ``PROJ`` for ``"azel"``. A file without ``FRAME``,
        written before it existed, is read as ``"radec"``. ``weights``, ``converged``,
        ``solver`` and ``unit`` come from the ``"WEIGHTS"`` extension and the keywords
        ``CONVERGD``, ``SOLVER`` and ``BUNIT``; a file written before they existed
        gives None, True, ``"cg"`` (the only solver there was) and None.

    Notes
    -----
    ``CRPIX`` is not read: the geometry assumes the projection center in the middle
    of the grid, as :meth:`~tolomeo.map.MapGeometry.to_wcs` writes it. A
    FITS file from another program, with a different ``CRPIX``, would get a wrong
    geometry. A file that is not a tolomeo map raises ``ValueError``.
    """
    path = Path(path).resolve()

    check_fits_suffix(path)

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

            frame = header.get("FRAME", "radec")  # files written before: radec
            if frame == "azel":
                center = (np.deg2rad(header["OBSRA"]), np.deg2rad(header["OBSDEC"]))
                projection = header["PROJ"]
            else:
                center = (np.deg2rad(header["CRVAL1"]), np.deg2rad(header["CRVAL2"]))
                projection = header["CTYPE1"][-3:].lower()

            ny, nx = image.shape
            geometry = MapGeometry(
                nx=nx,
                ny=ny,
                pixel_size=abs(header["CDELT2"]) * 3600,
                center=center,
                projection=projection,
                frame=frame,
            )
        except KeyError as exc:
            raise ValueError(f"{path} is not a tolomeo map: missing {exc}") from exc

        weights = None
        if "WEIGHTS" in hdul:
            weights = hdul["WEIGHTS"].data.astype(np.float64)

        return MapResult(
            image=image,
            hits=hits,
            n_iter=n_iter,
            residual=residual,
            geometry=geometry,
            weights=weights,
            converged=bool(header.get("CONVERGD", True)),
            solver=header.get("SOLVER", "cg"),
            unit=header.get("BUNIT"),
        )
