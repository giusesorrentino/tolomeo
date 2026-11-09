"""
Small coordinate transforms: rotations, boresight shifts and map projections.

All the angles are in radians.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._typing import Frame, Projection


def rot(
    x: ArrayLike, y: ArrayLike, theta: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Rotate (x, y) by theta radians.

    Parameters
    ----------
    x, y : array_like
        Coordinates to rotate; any shapes that broadcast with ``theta``.
    theta : array_like
        Rotation angle in radians, counter-clockwise.

    Returns
    -------
    x_rot, y_rot : :class:`~numpy.ndarray` of float64
        The rotated coordinates, with the broadcast shape of the inputs.
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return x * c - y * s, x * s + y * c


def shift_boresight(
    ra: NDArray[np.float64], dec: NDArray[np.float64], dra: float, ddec: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Shift the boresight by an offset measured on the sky.

    Parameters
    ----------
    ra, dec : :class:`~numpy.ndarray` of float64
        Boresight coordinates, in radians.
        Shape ``(n_samples,)``.
    dra : float
        Offset along the RA direction, as an angle on the sky, in radians: the RA
        coordinate changes by ``dra / cos(dec)``.
    ddec : float
        Offset in Dec, in radians.

    Returns
    -------
    ra, dec : :class:`~numpy.ndarray` of float64
        The shifted coordinates, in radians (new arrays).
        Shape ``(n_samples,)``.
    """
    return ra + dra / np.cos(dec), dec + ddec


def radec2xy(
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    ra0: float,
    dec0: float,
    *,
    projection: Projection,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Project celestial coordinates onto a tangent plane.

    All angles are in radians. Returned x and y are angular coordinates
    in radians relative to the projection center.

    Parameters
    ----------
    ra, dec : :class:`~numpy.ndarray` of float64
        Coordinates to project, in radians (any broadcastable shapes work).
        Shape ``(n,)``.
    ra0, dec0 : float
        Projection center, in radians.
    projection : :class:`~tolomeo._typing.Projection`
        * sin: orthographic (slant) projection; only the hemisphere facing the
          center is defined.
        * tan: also known as gnomonic; only the hemisphere facing the center is
          defined.
        * zea: also known as Lambert azimuthal equal-area projection; the whole
          sphere except the antipode of the center is defined.

    Returns
    -------
    x, y : :class:`~numpy.ndarray` of float64
        Coordinates on the plane, in radians, with x growing toward the east
        (increasing RA) and y toward the north. NaN where the projection is not
        defined.
        Shape ``(n,)``.
    """

    dra = np.asarray(ra) - ra0
    dec = np.asarray(dec)

    sin_dec0 = np.sin(dec0)
    cos_dec0 = np.cos(dec0)
    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)
    cos_dra = np.cos(dra)

    x = cos_dec * np.sin(dra)
    y = cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_dra

    projection = projection.lower()

    if projection not in ["sin", "tan", "zea"]:
        raise ValueError(f"Unknown projection '{projection}'. ")

    if projection == "sin":
        # Only the visible hemisphere projects onto the SIN plane.
        c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra
        c = np.clip(c, -1.0, 1.0)
        invalid = c < 0

        x = np.where(invalid, np.nan, x)
        y = np.where(invalid, np.nan, y)

    elif projection == "tan":
        c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra
        c = np.clip(c, -1.0, 1.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            x = x / c
            y = y / c

        # The gnomonic projection is only defined in the forward hemisphere.
        invalid = c <= 0
        x = np.where(invalid, np.nan, x)
        y = np.where(invalid, np.nan, y)

    elif projection == "zea":
        c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra
        c = np.clip(c, -1.0, 1.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.sqrt(2 / (1 + c))
            x *= k
            y *= k

        invalid = c <= -1

        x = np.where(invalid, np.nan, x)
        y = np.where(invalid, np.nan, y)

    return x, y


def xy2latlon(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    par_angle: NDArray[np.float64],
    x_offset: NDArray[np.float64],
    y_offset: NDArray[np.float64],
    center_ra: float,
    center_dec: float,
    frame: Frame,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Apply detector offsets to projected boresight coordinates.

    Returns (lat, lon), in radians, with shape
    (n_detectors, n_samples).

    Parameters
    ----------
    x, y : :class:`~numpy.ndarray` of float64
        Boresight on the projection plane, in radians.
        Shape ``(n_samples,)``.
    par_angle : :class:`~numpy.ndarray` of float64
        Parallactic angle, in radians.
        Shape ``(n_samples,)``.
    x_offset, y_offset : :class:`~numpy.ndarray` of float64
        Detector offsets from the boresight, in radians.
        Shape ``(n_detectors,)``.
    center_ra, center_dec : float
        Map center, in radians; used only with ``frame="azel"``.
    frame : {"radec", "azel"}
        ``"radec"``: the offsets are rotated by the parallactic angle and added on
        the plane. ``"azel"``: the boresight is rotated relative to the center
        and the offsets are added without rotation.

    Returns
    -------
    lat, lon : :class:`~numpy.ndarray` of float64
        Detector coordinates on the plane, in radians.
        Shape ``(n_detectors, n_samples)``.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    par_angle = np.asarray(par_angle)
    x_offset = np.asarray(x_offset)
    y_offset = np.asarray(y_offset)

    xO = x_offset[:, None]
    yO = y_offset[:, None]
    theta = par_angle[None, :]

    if frame == "radec":
        xoff_rot, yoff_rot = rot(xO, yO, theta)
        lat = y[None, :] + yoff_rot
        lon = x[None, :] - xoff_rot
        return lat, lon

    if frame == "azel":
        x_rot, y_rot = rot(x, y, par_angle)
        lat = y_rot[None, :] + yO
        lon = x_rot[None, :] - xO / np.cos(lat)
        return lat, lon

    raise NotImplementedError(f"Frame '{frame!r}' is not available.")


def radec2latlon(
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    center_ra: float,
    center_dec: float,
    projection: Projection,
    par_angle: NDArray[np.float64],
    x_offset: NDArray[np.float64],
    y_offset: NDArray[np.float64],
    frame: Frame,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Convert boresight RA/Dec plus detector offsets to map coordinates.

    :func:`radec2xy` followed by :func:`xy2latlon`.

    Parameters
    ----------
    ra, dec : :class:`~numpy.ndarray` of float64
        Boresight coordinates, in radians.
        Shape ``(n_samples,)``.
    center_ra, center_dec : float
        Projection center, in radians.
    projection : :class:`~tolomeo._typing.Projection`
        Projection of the map.
    par_angle : :class:`~numpy.ndarray` of float64
        Parallactic angle, in radians.
        Shape ``(n_samples,)``.
    x_offset, y_offset : :class:`~numpy.ndarray` of float64
        Detector offsets from the boresight, in radians.
        Shape ``(n_detectors,)``.
    frame : {"radec", "azel"}
        See :func:`xy2latlon`.

    Returns
    -------
    lat, lon : :class:`~numpy.ndarray` of float64
        Detector coordinates on the plane, in radians.
        Shape ``(n_detectors, n_samples)``.
    """
    x, y = radec2xy(
        ra,
        dec,
        center_ra,
        center_dec,
        projection=projection,
    )

    return xy2latlon(
        x,
        y,
        par_angle,
        x_offset,
        y_offset,
        center_ra,
        center_dec,
        frame,
    )
