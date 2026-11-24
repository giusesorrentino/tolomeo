from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from .parallel import _resolve_workers

if TYPE_CHECKING:
    from ..coords import Coordinates
    from ..map import MapGeometry
    from ..mapmaking import InverseNoise
    from ..tod import Scan

logger = logging.getLogger(__name__)


def write_roma_fits(
    scan: Scan,
    geometry: MapGeometry | None,
    inv_noise: InverseNoise,
    directory: str | Path,
    *,
    pointing: Literal["pixels", "radec"] = "pixels",
    workers: int | None = -1,
) -> None:
    """
    Write the TOD, the pointing, the flags and the inverse noise spectra for ROMA.

    One file per detector and per kind of data, numbered from 1 in the order of
    ``scan.channel_ids``, as ROMA wants:

    * ``tod_N.fits``: the signal, float64;
    * with ``pointing="pixels"``, ``pix_N.fits``: the pixel of every sample in
      ``geometry``, int64, counted from 0 and flat row-major (``iy * nx + ix``) as in
      tolomeo;
    * with ``pointing="radec"``, ``radec_N.fits``: the position of the detector on the
      sky at every sample, two float64 columns ``RA`` and ``DEC`` in degrees (FK5,
      J2000), so that ROMA chooses its own pixels;
    * ``flags_N.fits``: 0 for a good sample, 1 for a flagged one, for one outside the
      map (``pointing="pixels"``) or for one without a finite position
      (``pointing="radec"``), int32 (the opposite of ``scan.tod.flags``, where True
      is good);
    * ``invps_N_N.fits``: the inverse noise spectrum of the detector, float64.

    Every file has an empty primary HDU and a binary table, with one column (two in
    ``radec_N.fits``). The header of the table gives the channel (``CHANNEL``, the
    ``XXX`` of ``chp_XXX``) and the sampling frequency (``FSAMP``, Hz). That of
    ``tod_N.fits`` also gives the unit of the TOD (``BUNIT``), if known; that of
    ``pix_N.fits`` the WCS of ``geometry`` and its size (``MAPNX``, ``MAPNY``); that
    of ``radec_N.fits`` the frame (``RADESYS``, ``EQUINOX``). Existing files are
    replaced.

    The position of a detector is its offset from the boresight (``scan.offsets``),
    turned by the parallactic angle as in the pointing of tolomeo, on the plane
    tangent to the sky at the boresight (see :func:`_detector_radec`). The pixels of
    :func:`~tolomeo.mapmaking.gls_map` add the same offsets on the plane of the map
    instead, tangent at its centre: the two positions differ by about the product of
    the offset and of the distance from the centre, both in radians (0.5 arcsec for an
    offset of 3 arcmin at 10 arcmin from the centre).

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        Scan to write, raw or cleaned. ``scan.tod.signal`` must be finite.
    geometry : :class:`~tolomeo.map.MapGeometry` or None
        Grid of the pixels. Not used with ``pointing="radec"``, where it can be None.
    inv_noise : :class:`~tolomeo.mapmaking.InverseNoise`
        Noise model of ``scan``: only the auto-spectra are written. ROMA needs an odd
        number of bins, so ``nperseg`` must be a multiple of 4.
    directory : str or :class:`~pathlib.Path`
        Folder of the files, created if missing.
    pointing : {"pixels", "radec"}, optional
        ``"pixels"`` writes the pixel of every sample in ``geometry``; ``"radec"``
        writes the RA and Dec of every sample, and no pixels. Defaults to
        ``"pixels"``.
    workers : int or None, optional
        Threads of the pointing in pixels, with the convention of ``gls_map``: None is
        one thread, a positive number that many, ``-1`` all the cores. Defaults to
        ``-1``.

    Raises
    ------
    ValueError
        If ``pointing`` is not ``"pixels"`` or ``"radec"``, if ``geometry`` is None
        with ``pointing="pixels"``, if the signal is not finite, if ``inv_noise`` does
        not belong to ``scan``, or if its spectra have an even number of bins.

    Examples
    --------
    From the FITS subscans of MISTRAL to the files of ROMA. The paths are only
    examples. The cleaning, the calibration and the grid are the same as for a map of
    tolomeo, so that ROMA gets the same data.

    .. code-block:: python

        from tolomeo._lib.roma import write_roma_fits
        from tolomeo.io import read_scan, read_skydip
        from tolomeo.map import MapGeometry
        from tolomeo.mapmaking import estimate_inverse_noise
        from tolomeo.calibration import calibrate, skydip_responsivity
        from tolomeo.processing import clean_subscans

        # one folder per scan, all joined in one Scan; only the good channels of the
        # beammap are read
        scan = read_scan(
            ["path/to/scan_1/", "path/to/scan_2/"],
            beammap_path="path/to/beammap.dat",
        )

        # baseline and common mode of every subscan; for a bright source, give
        # `background` too, so that the source does not enter the fits
        scan = clean_subscans(scan, overwrite_scan=True)

        # TOD in kelvin, with the skydip of the same night. Without a usable skydip
        # read_skydip raises FileNotFoundError: skip these three lines, and the TOD
        # stays in the units of the instrument (rad)
        skydips = read_skydip("path/to/gain_cal/", scan.channel_ids)
        responsivity = skydip_responsivity(skydips)
        scan = calibrate(scan, responsivity, tau_atm=0.16)

        # grid of the pixels: centre, pixel size, projection and frame can be chosen,
        # see MapGeometry.from_scan
        geometry = MapGeometry.from_scan(scan, pixel_size="4 arcsec")

        # nperseg a multiple of 4: ROMA wants spectra with an odd number of bins
        inv_noise = estimate_inverse_noise(scan, nperseg=8192)

        write_roma_fits(scan, geometry, inv_noise, "path/to/roma_input/")

    Or the RA and Dec of every detector, for ROMA to pixelize: no grid is needed.

    .. code-block:: python

        write_roma_fits(scan, None, inv_noise, "path/to/roma_input/", pointing="radec")
    """
    # imported here: the modules of _lib import nothing from tolomeo when they load
    from ..mapmaking._pointing import _pointing_matrix

    # the arguments are checked first, before the slow part
    if pointing not in ("pixels", "radec"):
        raise ValueError(f"`pointing` must be 'pixels' or 'radec', got {pointing!r}.")
    if pointing == "pixels" and geometry is None:
        raise ValueError(
            "`pointing='pixels'` needs `geometry`: give it, or use pointing='radec'."
        )

    signal = scan.tod.signal
    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "The signal has NaN or inf: ROMA filters the TOD with FFTs, which do not "
            "accept them."
        )

    inv_noise.check_scan(scan)
    n_ps = inv_noise.inv_noiseps.shape[0]
    if n_ps % 2 == 0:
        raise ValueError(
            f"The spectra have {n_ps} bins: ROMA needs an odd number, so `nperseg` "
            "must be a multiple of 4."
        )

    if pointing == "pixels":
        # (n_samples, n_det): the pixel of every sample, and whether it is in the map
        pixels, inside = _pointing_matrix(
            scan.coordinates, scan.offsets, geometry, workers=_resolve_workers(workers)
        )

        # the WCS of the grid and its size, for the header of the pointing files
        grid = geometry.to_wcs().to_header()
        grid["MAPNX"] = (geometry.nx, "pixels of the map along x")
        grid["MAPNY"] = (geometry.ny, "pixels of the map along y")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for j, channel in enumerate(scan.channel_ids):
        n = j + 1  # ROMA counts the detectors from 1

        common = fits.Header()
        common["CHANNEL"] = (int(channel), "MISTRAL channel, the XXX of chp_XXX")
        common["FSAMP"] = (scan.tod.sample_rate, "[Hz] sampling frequency")

        tod_header = common.copy()
        if scan.tod.unit is not None:
            tod_header["BUNIT"] = (scan.tod.unit, "unit of the TOD")

        if pointing == "pixels":
            pix_header = common.copy()
            pix_header.update(grid)
            _write_column(
                directory / f"pix_{n}.fits",
                "PIXEL",
                "K",
                pixels[:, j].astype(np.int64),
                pix_header,
            )
            # a sample outside the map has no pixel
            good = scan.tod.flags & inside[:, j]
        else:
            ra, dec = _detector_radec(
                scan.coordinates, scan.offsets["lon"][j], scan.offsets["lat"][j]
            )
            radec_header = common.copy()
            radec_header["RADESYS"] = ("FK5", "frame of RA and DEC")
            radec_header["EQUINOX"] = (2000.0, "equinox of RA and DEC")
            columns = [
                fits.Column(name="RA", format="D", unit="deg", array=ra),
                fits.Column(name="DEC", format="D", unit="deg", array=dec),
            ]
            _write_table(directory / f"radec_{n}.fits", columns, radec_header)
            # a sample without a finite position (NaN in the raw pointing) has none
            good = scan.tod.flags & np.isfinite(ra) & np.isfinite(dec)

        # ROMA wants 0 for a good sample
        flags = (~good).astype(np.int32)

        _write_column(directory / f"tod_{n}.fits", "TOD", "D", signal[:, j], tod_header)
        _write_column(directory / f"flags_{n}.fits", "FLAG", "J", flags, common)
        _write_column(
            directory / f"invps_{n}_{n}.fits",
            "INVPS",
            "D",
            inv_noise.inv_noiseps[:, j],
            common,
        )

    logger.info(
        f"[ROMA] Files of {len(scan.channel_ids)} detectors written in {directory} "
        f"(pointing = {pointing})"
    )


def _detector_radec(
    coords: Coordinates, lon_offset: float, lat_offset: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""
    RA and Dec of one detector at every sample, in degrees.

    The offset is turned by the parallactic angle :math:`p` as in
    :func:`~tolomeo.mapmaking._pointing._pointing_matrix`, which gives
    :math:`(x_r, y_r)`. The pointing adds :math:`x_r` to a coordinate that grows
    toward the west, so on the sky the detector is :math:`\xi = -x_r` to the east
    and :math:`\eta = y_r` to the north of the boresight
    :math:`(\alpha_0, \delta_0)`. These are taken on the plane tangent to the sky at
    the boresight, and the inverse gnomonic projection gives

    .. math::

        \alpha = \alpha_0 + \operatorname{atan2}\left(\xi,\;
            \cos\delta_0 - \eta \sin\delta_0\right), \qquad
        \delta = \operatorname{atan2}\left(\sin\delta_0 + \eta \cos\delta_0,\;
            \sqrt{\xi^2 + (\cos\delta_0 - \eta \sin\delta_0)^2}\right).

    Parameters
    ----------
    coords : :class:`~tolomeo.coords.Coordinates`
        Boresight ``ra``, ``dec`` and ``par_angle``, in radians, each of shape
        ``(n_samples,)``.
    lon_offset, lat_offset : float
        Offset of the detector from the boresight, in radians.

    Returns
    -------
    ra, dec : :class:`~numpy.ndarray` of float64
        In degrees, the RA in ``[0, 360)``; NaN where the boresight is NaN.
        Shape ``(n_samples,)``.
    """
    cos_p, sin_p = np.cos(coords.par_angle), np.sin(coords.par_angle)
    x_rot = lon_offset * cos_p - lat_offset * sin_p
    y_rot = lon_offset * sin_p + lat_offset * cos_p
    xi, eta = -x_rot, y_rot

    sin_dec0, cos_dec0 = np.sin(coords.dec), np.cos(coords.dec)
    denominator = cos_dec0 - eta * sin_dec0
    ra = coords.ra + np.arctan2(xi, denominator)
    dec = np.arctan2(sin_dec0 + eta * cos_dec0, np.hypot(xi, denominator))
    return np.rad2deg(ra % (2 * np.pi)), np.rad2deg(dec)


def _write_column(
    path: Path, name: str, fmt: str, values: NDArray, header: fits.Header
) -> None:
    """
    Write one FITS file for ROMA: an empty primary HDU and a table with one column.

    Parameters
    ----------
    path : :class:`~pathlib.Path`
        File to write; an existing one is replaced.
    name : str
        Name of the column.
    fmt : str
        FITS format of the column: ``"D"`` float64, ``"K"`` int64, ``"J"`` int32.
    values : :class:`~numpy.ndarray`
        The column. Shape ``(n_rows,)``.
    header : :class:`~astropy.io.fits.Header`
        Keywords added to the header of the table.
    """
    _write_table(path, [fits.Column(name=name, format=fmt, array=values)], header)


def _write_table(path: Path, columns: list[fits.Column], header: fits.Header) -> None:
    """
    Write one FITS file for ROMA: an empty primary HDU and a table with ``columns``.

    Parameters
    ----------
    path : :class:`~pathlib.Path`
        File to write; an existing one is replaced.
    columns : list of :class:`~astropy.io.fits.Column`
        The columns of the table, all of the same length.
    header : :class:`~astropy.io.fits.Header`
        Keywords added to the header of the table.
    """
    table = fits.BinTableHDU.from_columns(columns)
    table.header.update(header)
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=True)
