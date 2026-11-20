import logging
import warnings
from datetime import datetime
from enum import IntFlag
from pathlib import Path

import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.spatial.distance import pdist, squareform

from .._lib.units import _to_unit
from ..coords.transforms import radec2xy, rot
from ..tod import Scan
from ._beammap import BeamMap

__all__ = ["reconstruct_focal_plane"]

logger = logging.getLogger(__name__)

# one radian in arcsec, and the sigma of a Gaussian in units of its FWHM
_RAD2ARCSEC = 180 / np.pi * 3600
_FWHM2SIGMA = 1 / np.sqrt(8 * np.log(2))

# columns of a beammap file: id, lon, lat, their sigmas, flag, as in the MISTRAL ones
_BEAMMAP_FMT = ["%2d", "%12.6f", "%12.6f", "%12.3e", "%12.3e", "%2d"]


class ChannelFlag(IntFlag):
    """
    Reasons why a channel of a beammap is flagged.

    The flag of a channel is the sum of its reasons, so that one number holds all
    of them: 0 for a good channel, 10 for ``LOW_SNR | AT_EDGE``. Whether a channel
    has a given reason is tested with ``flags & reason``, nonzero if it has it
    (for example ``flags & ChannelFlag.LOW_SNR``); iterating over the class gives
    the reasons one by one.

    Attributes
    ----------
    FIT_FAILED : int
        1: the Gaussian fit did not converge, or its covariance could not be
        estimated. The channel has no position.
    LOW_SNR : int
        2: the fitted amplitude over its error is below ``min_snr``.
    DUPLICATE : int
        4: another channel is less than 1 arcsec away, usually because the same
        tone is written in two channels (identical data).
    AT_EDGE : int
        8: the centre of the fit is closer than ``fit_radius`` to the edge of the
        maps, where its cutout is cut.
    """

    FIT_FAILED = 1
    LOW_SNR = 2
    DUPLICATE = 4
    AT_EDGE = 8


def _gaussian_2d(
    xy: NDArray[np.float64],
    amplitude: float,
    x0: float,
    y0: float,
    sigma: float,
    offset: float,
) -> NDArray[np.float64]:
    """Circular 2-D Gaussian plus a constant; ``xy`` is ``(2, n_pixels)``, arcsec."""
    x, y = xy
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return amplitude * np.exp(-0.5 * r2 / sigma**2) + offset


def reconstruct_focal_plane(
    scan: Scan,
    ref_channel: int,
    *,
    pixel_size: float | str = 4.0,
    fit_radius: float | str = 30.0,
    fwhm: float | str = 12.0,
    min_snr: float = 10.0,
    output: str | Path | None = None,
) -> BeamMap:
    r"""
    Offsets, beams and gains of the detectors from a scan on a bright source.

    Every detector gets its own binned map, made with the boresight pointing only
    (all the offsets zero), in the horizontal frame and centered on the source.
    Detector :math:`j` looks at :math:`\mathbf b(t) + \boldsymbol\delta_j`, with
    :math:`\mathbf b(t)` the boresight and :math:`\boldsymbol\delta_j` its offset, so
    it sees the source when the boresight is at :math:`-\boldsymbol\delta_j`: in its
    map the source appears at :math:`\mathbf x_{0,j} = -\boldsymbol\delta_j`. A 2-D
    Gaussian fit of every map gives :math:`\mathbf x_{0,j}`, hence the offset.

    Parameters
    ----------
    scan : :class:`~tolomeo.tod.Scan`
        Scan on the source, already cleaned with the source masked (see Notes), with
        the position of the source of every sample (read by
        :func:`~tolomeo.io.read_scan` from the FITS headers). Its ``offsets`` are
        ignored: the maps use the boresight only.
    ref_channel : int
        Channel whose offset is set to zero: the offsets are relative to it.
    pixel_size : float or str, optional
        Side of the pixels of the maps. A number is in arcsec. About FWHM / 3 or
        smaller. Defaults to 4.
    fit_radius : float or str, optional
        Radius of the cutout used in the fit, around the brightest pixel of the
        smoothed map. A number is in arcsec. Two or three FWHM. Defaults to 30.
    fwhm : float or str, optional
        Expected FWHM of the beam, for the smoothing and the starting point of the
        fit. A number is in arcsec. Defaults to 13.
    min_snr : float, optional
        A detector whose fitted amplitude has a lower signal-to-noise ratio is
        flagged. Defaults to 10.
    output : str, :class:`~pathlib.Path` or None, optional
        Beammap text file to write, in the format of the MISTRAL ones (see
        :func:`_write_beammap`). Defaults to ``None``: nothing is written.

    Returns
    -------
    BeamMap
        For every channel of ``scan``: ``channel_ids``; ``lon`` and ``lat``, the
        offsets in radians relative to ``ref_channel``; ``flags``, 0 for a good
        channel, otherwise the sum of its :class:`ChannelFlag`, as in the column 6
        of the file.

    Raises
    ------
    ValueError
        If the scan has no position of the source, if it has no valid sample, or if
        the fit of ``ref_channel`` fails.
    """
    # everything in arcsec from here on
    pix = _to_unit(pixel_size, u.arcsec)
    radius = _to_unit(fit_radius, u.arcsec)
    fwhm = _to_unit(fwhm, u.arcsec)

    # boresight relative to the source, in the horizontal orientation (the "azel"
    # branch of _pointing_matrix). The source is the one of every sample: a planet
    # moves from one subscan to the next
    coords = scan.coordinates
    _ = scan.source  # raises if the scan has no position of the source
    x, y = radec2xy(
        coords.ra, coords.dec, coords.source_ra, coords.source_dec, projection="sin"
    )
    u_bs, v_bs = rot(-x, y, -coords.par_angle)
    u_bs, v_bs = u_bs * _RAD2ARCSEC, v_bs * _RAD2ARCSEC

    # the maps cover the whole scan (Notes, point 8): an nx x ny grid centred on the
    # source, whose half sides are the largest distances of the valid samples of the
    # boresight from it; floor + 1 and not ceil, so that the farthest sample is inside
    valid = scan.tod.flags & np.isfinite(u_bs) & np.isfinite(v_bs)
    if not valid.any():
        raise ValueError("No valid sample in the scan.")
    nx = 2 * (int(np.abs(u_bs[valid]).max() // pix) + 1)
    ny = 2 * (int(np.abs(v_bs[valid]).max() // pix) + 1)
    logger.info(f"Maps of {nx} x {ny} pixels of {pix:g} arcsec.")

    # a fit centred closer than fit_radius to the edge of the maps is flagged
    edge_x = nx * pix / 2 - radius
    edge_y = ny * pix / 2 - radius

    # pixel of every sample, the same for all the detectors (offsets ignored):
    # pixel (iy, ix) is iy * nx + ix. Every valid sample falls inside by
    # construction: the check only guards against the rounding
    ix = np.floor((u_bs + nx * pix / 2) / pix)
    iy = np.floor((v_bs + ny * pix / 2) / pix)
    good = valid & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    index = (iy[good] * nx + ix[good]).astype(np.intp)

    def to_map(weights: NDArray[np.float64] | None) -> NDArray[np.float64]:
        """Sum over the samples of every pixel, as an ny x nx map."""
        return np.bincount(index, weights=weights, minlength=nx * ny).reshape(ny, nx)

    hits = to_map(None)
    seen = hits > 0

    # position of every pixel: the mean position of its samples, not its centre (the
    # rows of the raster are not centred in the pixels); the centres are only used for
    # the starting point of the fit
    x_pix = np.where(seen, to_map(u_bs[good]) / np.maximum(hits, 1), np.nan)
    y_pix = np.where(seen, to_map(v_bs[good]) / np.maximum(hits, 1), np.nan)
    x_centres = (np.arange(nx) + 0.5) * pix - nx * pix / 2
    y_centres = (np.arange(ny) + 0.5) * pix - ny * pix / 2
    x_centre, y_centre = np.meshgrid(x_centres, y_centres)

    n_det = scan.tod.signal.shape[1]
    smooth_sigma = fwhm * _FWHM2SIGMA / pix  # in pixels

    # map of every detector, mean of its samples in every pixel, and its noise,
    # robust to the few samples on the source (MAD)
    maps = np.zeros((n_det, ny, nx))
    noise = np.empty(n_det)
    for j in range(n_det):
        d = scan.tod.signal[good, j]
        level = to_map(d)[seen] / hits[seen]
        # the median of the map is removed: the pixels with no hit stay at zero,
        # and must not stand out of the rest of the smoothed map
        maps[j][seen] = level - np.median(level)
        noise[j] = 1.4826 * np.median(np.abs(d - np.median(d)))

    # sign of the source in every detector (Notes, point 7): +1 if the largest
    # excursion of its smoothed map is up, -1 if down, 0 if the map is not finite
    signs = np.zeros(n_det)
    for j in range(n_det):
        s = gaussian_filter(maps[j], smooth_sigma)[seen]
        med = np.median(s)
        if np.isfinite(med):
            signs[j] = 1 if s.max() - med >= med - s.min() else -1
    majority = 1 if signs.sum() >= 0 else -1
    logger.info(
        f"Sign of the source: {majority:+d} in {np.sum(signs == majority):.0f} "
        f"detectors, {-majority:+d} in {np.sum(signs == -majority):.0f}."
    )
    maps *= signs[:, None, None]

    x0 = np.full(n_det, np.nan)
    y0 = np.full(n_det, np.nan)
    x0_err = np.full(n_det, np.nan)
    y0_err = np.full(n_det, np.nan)
    snr = np.full(n_det, np.nan)
    reasons = np.zeros(n_det, dtype=np.int64)

    for j in range(n_det):
        m = maps[j]
        err = noise[j] / np.sqrt(np.maximum(hits, 1))  # error of every pixel

        # starting point: brightest pixel of the map smoothed over one beam
        smooth = gaussian_filter(m, smooth_sigma)
        k_y, k_x = np.unravel_index(np.argmax(smooth), smooth.shape)
        x_start, y_start = x_centre[k_y, k_x], y_centre[k_y, k_x]

        # fit on the observed pixels within fit_radius of the starting point
        cut = seen & (np.hypot(x_centre - x_start, y_centre - y_start) < radius)
        p0 = [smooth[k_y, k_x], x_start, y_start, fwhm * _FWHM2SIGMA, 0.0]
        try:
            with warnings.catch_warnings():
                # a covariance that cannot be estimated is a failed fit
                warnings.simplefilter("error", OptimizeWarning)
                p, cov = curve_fit(
                    _gaussian_2d,
                    np.vstack([x_pix[cut], y_pix[cut]]),
                    m[cut],
                    p0=p0,
                    sigma=err[cut],
                    absolute_sigma=True,
                )
        except (RuntimeError, ValueError, OptimizeWarning):
            reasons[j] |= ChannelFlag.FIT_FAILED
            continue  # no fit: the detector stays flagged

        errors = np.sqrt(np.diag(cov))
        x0[j], y0[j] = p[1], p[2]
        x0_err[j], y0_err[j] = errors[1], errors[2]
        snr[j] = p[0] / errors[0]

        if np.abs(x0[j]) > edge_x or np.abs(y0[j]) > edge_y:
            reasons[j] |= ChannelFlag.AT_EDGE

    reasons[snr < min_snr] |= ChannelFlag.LOW_SNR

    # offsets: minus the position of the source, in radians, relative to ref_channel
    lon = np.deg2rad(-x0 / 3600)
    lat = np.deg2rad(-y0 / 3600)
    k = scan.channel_ids.index(ref_channel)
    if not np.isfinite(lon[k]):
        raise ValueError(f"The reference channel {ref_channel} has no fit.")

    ref_lon, ref_lat = lon[k], lat[k]
    lon -= ref_lon
    lat -= ref_lat

    offsets = np.column_stack((lon, lat))
    pair_distances = pdist(offsets, metric="euclidean")
    duplicate = squareform(pair_distances * _RAD2ARCSEC < 1).any(axis=0)
    reasons[duplicate] |= ChannelFlag.DUPLICATE

    if output is not None:
        _write_beammap(
            output,
            channel_ids=np.array(scan.channel_ids),
            lon=np.rad2deg(lon),
            lat=np.rad2deg(lat),
            lon_sigma=x0_err / 3600,
            lat_sigma=y0_err / 3600,
            reasons=reasons,
            min_snr=min_snr,
            ref_channel=ref_channel,
            ref_lon=np.rad2deg(ref_lon),
            ref_lat=np.rad2deg(ref_lat),
        )

    return BeamMap(
        channel_ids=np.array(scan.channel_ids), flags=reasons, lat=lat, lon=lon
    )


def _write_beammap(
    path: str | Path,
    *,
    channel_ids: NDArray[np.integer],
    lon: NDArray[np.float64],
    lat: NDArray[np.float64],
    lon_sigma: NDArray[np.float64],
    lat_sigma: NDArray[np.float64],
    reasons: NDArray[np.integer],
    min_snr: float,
    ref_channel: int,
    ref_lon: float,
    ref_lat: float,
) -> None:
    """
    Write a beammap text file in the format of the MISTRAL ones.

    One row per channel: id, lon and lat offsets (deg), their sigmas (deg) in the two
    columns where the old files have the Tcal (always zero there), flag: 0 for a good
    channel, otherwise the sum of its :class:`ChannelFlag`, listed in the header. The
    header gives the date, the number of rows, the columns and the offsets of
    ``ref_channel`` from the boresight, which were subtracted from all the others.
    :func:`~tolomeo.io.read_scan` reads the columns 1, 2, 3 and 6 only, and keeps
    the channels with flag 0, as for the old files.

    Parameters
    ----------
    path : str or :class:`~pathlib.Path`
        File to write; an existing one is replaced.
    channel_ids : :class:`~numpy.ndarray` of int
        Ids of the channels.
        Shape ``(n_det,)``.
    lon, lat : :class:`~numpy.ndarray` of float64
        Offsets relative to ``ref_channel``, in degrees; NaN for a channel with no
        fit, written as 0 (it is flagged).
        Shape ``(n_det,)``.
    lon_sigma, lat_sigma : :class:`~numpy.ndarray` of float64
        Sigmas of the offsets from the fit of every channel, in degrees (the error of
        the reference channel is not included); NaN written as 0.
        Shape ``(n_det,)``.
    reasons : :class:`~numpy.ndarray` of int
        Flag of every channel: 0 for a good one, otherwise the sum of its
        :class:`ChannelFlag`.
        Shape ``(n_det,)``.
    min_snr : float
        Threshold of :attr:`ChannelFlag.LOW_SNR`, written in the header.
    ref_channel : int
        Channel whose offsets are zero.
    ref_lon, ref_lat : float
        Offsets of ``ref_channel`` from the boresight, in degrees.
    """
    # one line per reason of the flag, from ChannelFlag: they cannot diverge
    flag_lines = [
        f"  {flag.value} {flag.name}"
        + (f" (SNR < {min_snr:g})" if flag is ChannelFlag.LOW_SNR else "")
        for flag in ChannelFlag
    ]
    header = "\n".join(
        [
            f"Feed list created: {datetime.now().astimezone():%b %d %H:%M:%S %Y}",
            f"nfeeds: {channel_ids.size}",
            "Col.1: id",
            "Col.2: lon-offset (deg)",
            "Col.3: lat-offset (deg)",
            "Col.4: lon-offset sigma (deg)",
            "Col.5: lat-offset sigma (deg)",
            "Col.6: flag (0=>Ok, otherwise the sum of the reasons below)",
            *flag_lines,
            f"chp{ref_channel} az_off (deg): {ref_lon:.12g}",
            f"chp{ref_channel} el_off (deg): {ref_lat:.12g}",
        ]
    )
    rows = np.column_stack(
        [
            channel_ids,
            np.nan_to_num(lon),
            np.nan_to_num(lat),
            np.nan_to_num(lon_sigma),
            np.nan_to_num(lat_sigma),
            reasons,
        ]
    )
    np.savetxt(path, rows, fmt=_BEAMMAP_FMT, header=header, comments="# ")
