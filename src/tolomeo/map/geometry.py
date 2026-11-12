from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.wcs import WCS
from pydantic import BaseModel, ConfigDict, Field

from .._lib.units import Arcsec, Radians, _to_unit
from .._typing import Frame, Projection
from ..coords.transforms import radec2xy
from ..tod import Scan

__all__ = ["MapGeometry"]


class MapGeometry(BaseModel):
    """
    Grid of a map on the sky: size, pixel size, center and projection.

    The map has shape ``(ny, nx)`` and the flat pixel of ``(iy, ix)`` is
    ``iy * nx + ix``. The grid is centred on the projection center.

    Attributes
    ----------
    nx, ny : int
        Number of pixels along x and y, both > 0.
        Default ``256``.
    pixel_size : float
        Side of a pixel, > 0. It is stored in arcsec: a number is in arcsec, and
        ``"0.1 arcmin"`` or ``0.1 * astropy.units.arcmin`` is converted to arcsec.
        Default ``3.0``.
    center : tuple of float
        ``(ra, dec)`` of the projection center. It is stored in radians: a number is
        in radians, and ``"83.6 deg"`` or an :class:`~astropy.units.Quantity` is
        converted to radians.
    projection : :class:`~tolomeo._typing.Projection`
        Projection of the sphere on the map plane.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nx: int = Field(default=256, gt=0)
    ny: int = Field(default=256, gt=0)
    pixel_size: Arcsec = Field(default=3.0, gt=0)  # implicit unit: arcsec
    center: tuple[Radians, Radians]  # (ra, dec); implicit unit: radians
    projection: Projection = "sin"
    frame: Frame = "radec"

    @property
    def n_pixels(self) -> int:
        """Number of pixels of the map, ``nx * ny``."""
        return self.nx * self.ny

    def to_wcs(self) -> WCS:
        """
        Return the celestial WCS of the map grid.

        The WCS puts every sky position in the same pixel as
        ``tolomeo.mapmaking._pointing._pointing_matrix``. Its pixel axes are ``(x, y)``,
        so it matches an image stored as ``image[iy, ix]`` with shape ``(ny, nx)``.

        Returns
        -------
        :class:`astropy.wcs.WCS`
            Two axes, RA then Dec, in degrees. ``CDELT1`` is negative: x grows toward
            the west, so the east is on the left of the image, as in the pixelization.
        """
        step = self.pixel_size / 3600.0  # deg
        ra0, dec0 = np.rad2deg(self.center)
        proj = self.projection.upper()

        wcs = WCS(naxis=2)
        wcs.wcs.ctype = [f"RA---{proj}", f"DEC--{proj}"]
        wcs.wcs.cunit = ["deg", "deg"]
        wcs.wcs.cdelt = [-step, step]

        # CRPIX is 1-based and counts from pixel centers, while the pixelization counts
        # from the grid edge (xmin = -n * step / 2); the projection center is at x = y = 0,
        # i.e. at n / 2 pixels from the edge, which is n / 2 + 0.5 in FITS units.
        # It is the corner shared by the central pixels if n is even, the center of
        # the middle pixel if n is odd.
        crpix = [self.nx / 2 + 0.5, self.ny / 2 + 0.5]

        wcs.wcs.crval = [ra0, dec0]
        wcs.wcs.crpix = crpix

        return wcs

    @classmethod
    def from_beam(
        cls,
        beam_fwhm: float | str | u.Quantity,
        center: tuple[float, float],
        projection: Projection = "sin",
        *,
        pixels_per_beam: float = 3.0,
        nx: int = 256,
        ny: int = 256,
    ) -> MapGeometry:
        """
        Geometry with ``pixels_per_beam`` pixels across the beam FWHM.

        Parameters
        ----------
        beam_fwhm : float, str or :class:`~astropy.units.Quantity`
            Full width at half maximum of the beam. A number is in arcsec; a string
            such as ``"12 arcsec"`` or a quantity is converted to arcsec.
        center : tuple of float
            ``(ra, dec)`` of the projection center, in radians (see ``center`` of
            :class:`MapGeometry`).
        projection : :class:`~tolomeo._typing.Projection`
            Projection of the map.
        pixels_per_beam : float
            The pixel size is ``beam_fwhm / pixels_per_beam``.
            Default ``3.0``.
        nx, ny : int
            Number of pixels along x and y.
            Default ``384``.

        Returns
        -------
        :class:`~tolomeo.map.MapGeometry`
        """
        return cls(
            nx=nx,
            ny=ny,
            pixel_size=_to_unit(beam_fwhm, u.arcsec) / pixels_per_beam,
            center=center,
            projection=projection,
        )

    @classmethod
    def from_scan(
        cls,
        scan: Scan,
        *,
        pixel_size: float | str = 3.0,
        projection: Projection = "sin",
        frame: Frame = "radec",
        center: tuple[Radians, Radians] | None = None,
        margin: int = 10,
    ) -> MapGeometry:
        """
        Geometry that covers every detector of ``scan``, centred on ``center``.

        The valid samples of the boresight (``scan.tod.flags`` True and finite
        coordinates) are projected on the plane of ``projection`` around ``center``.
        The half-size of the map along each axis is the largest distance of the
        boresight from the centre plus the largest offset of a detector from the
        boresight, so the whole array fits at every sample. The offsets rotate with
        the parallactic angle, so their largest radius is used in both directions.
        ``margin`` pixels are then added on each side.

        Parameters
        ----------
        scan : :class:`~tolomeo.tod.Scan`
            Scan to cover.
        pixel_size : float, str or :class:`~astropy.units.Quantity`
            Side of a pixel. A number is in arcsec; a string with a unit
            (``"4 arcsec"``) or a quantity is converted.
            Default ``3.0``.
        projection : {"sin", "tan", "zea"}
            Projection of the map. Default ``"sin"``.
        frame : {"radec", "azel"}
            Coordinates of the map. ``"azel"`` is not supported by the mapmaker
            yet. Default ``"radec"``.
        center : tuple of float, optional
            Projection centre, in radians: ``(ra, dec)`` for ``"radec"``,
            ``(az, el)`` for ``"azel"``. None means :attr:`~tolomeo.tod.Scan.center`
            for ``"radec"``, and the circular mean of the azimuth and the mean of
            the elevation of the valid samples for ``"azel"``.
        margin : int
            Pixels added on each side of the map. Default ``10``.

        Returns
        -------
        :class:`MapGeometry`
            ``nx`` and ``ny`` are even: the grid goes from ``-nx * pixel_size / 2``
            to ``nx * pixel_size / 2``, so the centre falls on the corner shared by
            the four central pixels.

        Raises
        ------
        ValueError
            If ``frame`` is not ``"radec"`` or ``"azel"``.
        """
        pix = np.deg2rad(_to_unit(pixel_size, u.arcsec) / 3600)
        coords = scan.coordinates

        # the frame only chooses the spherical coordinates; the projection is the same
        if frame == "radec":
            lon, lat = coords.ra, coords.dec
        elif frame == "azel":
            lon, lat = coords.az, coords.el
        else:
            raise ValueError(f"Unknown frame: {frame!r}")

        ok = scan.tod.flags & np.isfinite(lon) & np.isfinite(lat)

        if center is None:
            if frame == "radec":
                center = scan.center
            else:
                # circular mean: the azimuth wraps at 0/360 deg
                center = (
                    np.angle(np.mean(np.exp(1j * lon[ok]))) % (2 * np.pi),
                    np.mean(lat[ok]),
                )

        lon0, lat0 = center
        x, y = radec2xy(lon[ok], lat[ok], lon0, lat0, projection=projection)
        r_array = np.hypot(scan.offsets["lon"], scan.offsets["lat"]).max()
        half_x = np.abs(x).max() + r_array
        half_y = np.abs(y).max() + r_array

        nx = 2 * (int(np.ceil(half_x / pix)) + margin)
        ny = 2 * (int(np.ceil(half_y / pix)) + margin)

        return cls(
            nx=nx,
            ny=ny,
            pixel_size=pixel_size,
            center=center,
            projection=projection,
            frame=frame,
        )
