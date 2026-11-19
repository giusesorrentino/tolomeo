from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy.wcs import WCS
from pydantic import BaseModel, ConfigDict, Field

from .._lib.units import Arcsec, Radians, _to_unit
from .._typing import Frame, Projection
from ..coords.transforms import radec2xy, rot
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
    frame : {"radec", "azel"}
        Orientation of the map. ``"radec"``: north up and east on the left.
        ``"azel"``: still centred on ``center`` (the source, in RA/Dec), but turned
        with the parallactic angle, so that the elevation is up and the azimuth grows
        to the right. Default ``"radec"``.
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
        Return the WCS of the map grid.

        The WCS puts every sky position in the same pixel as
        ``tolomeo.mapmaking._pointing._pointing_matrix``. Its pixel axes are ``(x, y)``,
        so it matches an image stored as ``image[iy, ix]`` with shape ``(ny, nx)``.

        Returns
        -------
        :class:`astropy.wcs.WCS`
            ``"radec"``: two celestial axes, RA then Dec, in degrees, in the FK5 frame
            with the J2000 equinox (``RADESYS = 'FK5'``, ``EQUINOX = 2000``), the frame
            of the ``raj2000`` and ``decj2000`` columns of the MISTRAL files. ``CDELT1``
            is negative: x grows toward the west, so the east is on the left of the
            image, as in the pixelization.
            ``"azel"``: two linear axes, ``AZOFFSET`` and ``ELOFFSET``, in degrees from
            the centre. x grows with the azimuth, so ``CDELT1`` is positive. The
            orientation of the map on the sky changes along the scan, so this WCS
            cannot give RA and Dec.
        """
        step = self.pixel_size / 3600.0  # deg
        ra0, dec0 = np.rad2deg(self.center)
        proj = self.projection.upper()

        wcs = WCS(naxis=2)

        # CRPIX is 1-based and counts from pixel centers, while the pixelization counts
        # from the grid edge (xmin = -n * step / 2); the projection center is at x = y = 0,
        # i.e. at n / 2 pixels from the edge, which is n / 2 + 0.5 in FITS units.
        # It is the corner shared by the central pixels if n is even, the center of
        # the middle pixel if n is odd.
        crpix = [self.nx / 2 + 0.5, self.ny / 2 + 0.5]

        if self.frame == "radec":
            wcs.wcs.radesys = "FK5"
            wcs.wcs.equinox = 2000.0

            wcs.wcs.ctype = [f"RA---{proj}", f"DEC--{proj}"]
            wcs.wcs.cunit = ["deg", "deg"]
            wcs.wcs.cdelt = [-step, step]

            wcs.wcs.crval = [ra0, dec0]
            wcs.wcs.crpix = crpix
        else:
            # offsets from the centre: not celestial types, so astropy treats them as
            # linear axes
            wcs.wcs.ctype = ["AZOFFSET", "ELOFFSET"]
            wcs.wcs.cunit = ["deg", "deg"]
            wcs.wcs.cdelt = [step, step]  # positive: x grows with azimuth
            wcs.wcs.crval = [0.0, 0.0]  # the source is at offset 0
            wcs.wcs.crpix = crpix  # same as for radec

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
        margin: float = 0.1,
    ) -> MapGeometry:
        """
        Geometry that covers every detector of ``scan``, centred on ``center``.

        The valid samples of the boresight (``scan.tod.flags`` True and finite RA, Dec
        and parallactic angle) are projected on the plane of ``projection`` around
        ``center``; for ``"azel"`` they are then turned with the parallactic angle, as
        in ``tolomeo.mapmaking._pointing._pointing_matrix``. The half-size of the map
        along each axis is the largest distance of the boresight from the centre plus
        the largest offset of a detector from the boresight, so the whole array fits
        at every sample. The offsets rotate with the parallactic angle in ``"radec"``,
        so their largest radius is used in both directions. A fraction ``margin`` of
        the half-size, rounded up to whole pixels, is then added on each side, so that
        ``nx`` and ``ny`` grow by about ``100 * margin`` percent.

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
            Orientation of the map, see :class:`MapGeometry`. Default ``"radec"``.
        center : tuple of float, optional
            Projection centre ``(ra, dec)``, in radians, for both frames. None means
            :attr:`~tolomeo.tod.Scan.center`.
        margin : float
            Fraction of the half-size of the map added on each side, from 0 to 1.
            Default ``0.1``: ``nx`` and ``ny`` about 10% larger.

        Returns
        -------
        :class:`MapGeometry`
            ``nx`` and ``ny`` are even: the grid goes from ``-nx * pixel_size / 2``
            to ``nx * pixel_size / 2``, so the centre falls on the corner shared by
            the four central pixels.

        Raises
        ------
        ValueError
            If ``frame`` is not ``"radec"`` or ``"azel"``, or if ``margin`` is not
            between 0 and 1.
        """
        if frame not in ("radec", "azel"):
            raise ValueError(f"Unknown frame: {frame!r}")

        if not 0 <= margin <= 1:
            raise ValueError(f"`margin` must be between 0 and 1, got {margin}.")

        pix = np.deg2rad(_to_unit(pixel_size, u.arcsec) / 3600)
        coords = scan.coordinates

        # the boresight is always projected around the center in RA/Dec; in azel it is
        # then turned into the horizontal orientation, as in _pointing_matrix
        ra, dec, par = coords.ra, coords.dec, coords.par_angle
        ok = scan.tod.flags & np.isfinite(ra) & np.isfinite(dec) & np.isfinite(par)

        if center is None:
            center = scan.center

        ra0, dec0 = center
        x, y = radec2xy(ra[ok], dec[ok], ra0, dec0, projection=projection)
        r_array = np.hypot(scan.offsets["lon"], scan.offsets["lat"]).max()

        if frame == "azel":
            x, y = rot(-x, y, -par[ok])

        half_x = np.abs(x).max() + r_array
        half_y = np.abs(y).max() + r_array

        half_px_x = int(np.ceil(half_x / pix))
        half_px_y = int(np.ceil(half_y / pix))
        nx = 2 * (half_px_x + int(np.ceil(margin * half_px_x)))
        ny = 2 * (half_px_y + int(np.ceil(margin * half_px_y)))

        return cls(
            nx=nx,
            ny=ny,
            pixel_size=pixel_size,
            center=center,
            projection=projection,
            frame=frame,
        )
