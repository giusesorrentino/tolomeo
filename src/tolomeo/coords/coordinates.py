from __future__ import annotations

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class Coordinates(BaseModel):
    """
    Boresight coordinates and detector pointing transformations.

    The class stores the celestial pointing of the telescope boresight
    as a function of time and provides methods to calculate the pointing
    of individual detectors and project their positions onto a map.

    Attributes
    ----------
    ra : :class:`~numpy.ndarray` of float64
        Right ascension of the telescope boresight at each time sample,
        in radians, in the ICRS reference frame. Shape: ``(n_samples,)``.

    dec : :class:`~numpy.ndarray` of float64
        Declination of the telescope boresight at each time sample,
        in radians, in the ICRS reference frame. Shape: ``(n_samples,)``.

    par_angle : :class:`~numpy.ndarray` of float64
        Parallactic angle at each time sample, in radians. It specifies
        the rotation between the detector-coordinate axes and the
        celestial-coordinate axes. Shape: ``(n_samples,)``.

    az : :class:`~numpy.ndarray` of float64
        Azimuth of the boresight at each time sample, in radians.
        Shape: ``(n_samples,)``.

    el : :class:`~numpy.ndarray` of float64
        Elevation of the boresight at each time sample, in radians.
        Shape: ``(n_samples,)``.

    Notes
    -----
    Detector offsets are assumed to be angular offsets in radians,
    expressed in the instrument's focal-plane coordinate system.
    The x and y offsets must correspond to the same detector ordering.

    The boresight arrays ``ra``, ``dec`` and ``par_angle`` must contain
    the same number of samples.
    """

    # numpy arrays are not pydantic types: they are only checked with isinstance
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ra: NDArray[np.float64]
    dec: NDArray[np.float64]
    par_angle: NDArray[np.float64]
    az: NDArray[np.float64]
    el: NDArray[np.float64]

    def detector_pointing(
        self,
        x_offset: float,
        y_offset: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Calculate the celestial pointing of **one** detector at every time sample.

        The detector offset is rotated according to the parallactic angle and
        applied to the boresight position using spherical offsets on the
        celestial sphere.

        Parameters
        ----------
        x_offset : float
            Horizontal angular offset of the detector from the boresight, in
            radians.
        y_offset : float
            Vertical angular offset of the detector from the boresight, in
            radians.

        Returns
        -------
        ra_det : :class:`~numpy.ndarray` of float64
            Right ascension of the detector at every time sample, in radians, in
            the ICRS reference frame.
            Shape ``(n_samples,)``.
        dec_det : :class:`~numpy.ndarray` of float64
            Declination of the detector at every time sample, in radians, in the
            ICRS reference frame.
            Shape ``(n_samples,)``.

        Notes
        -----
        The returned coordinates account for the spherical geometry
        of the celestial sphere; the detector offsets are not simply
        added directly to RA and Dec.
        """

        # ra = np.asarray(self.ra)[None, :]
        # dec = np.asarray(self.dec)[None, :]
        # pa = np.asarray(self.par_angle)[None, :]

        # x = np.asarray(x_offsets)[:, None]
        # y = np.asarray(y_offsets)[:, None]

        # # rotate detector offsets by the parallactic angle.
        # x_rot = x * np.cos(pa) - y * np.sin(pa)
        # y_rot = x * np.sin(pa) + y * np.cos(pa)

        # boresight = SkyCoord(
        #     ra=ra * units.rad,
        #     dec=dec * units.rad,
        #     frame="icrs",
        # )

        # pointing = boresight.spherical_offsets_by(
        #     d_lon=-x_rot * units.rad,
        #     d_lat=y_rot * units.rad,
        # )

        # return pointing.ra.rad, pointing.dec.rad
        pa = self.par_angle

        x_rot = x_offset * np.cos(pa) - y_offset * np.sin(pa)
        y_rot = x_offset * np.sin(pa) + y_offset * np.cos(pa)

        boresight = SkyCoord(
            ra=self.ra * units.rad,
            dec=self.dec * units.rad,
            frame="icrs",
        )

        pointing = boresight.spherical_offsets_by(
            d_lon=-x_rot * units.rad,
            d_lat=y_rot * units.rad,
        )

        return pointing.ra.rad, pointing.dec.rad

    def select(self, mask: NDArray[np.bool_]) -> Coordinates:
        """
        Keep only the samples where ``mask`` is True.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` of bool
            True for the samples to keep.
            Shape ``(n_samples,)``.

        Returns
        -------
        :class:`~tolomeo.coords.Coordinates`
            New object whose five arrays have shape ``(mask.sum(),)`` (copies).
        """
        if mask.dtype != bool:
            raise TypeError("")
        if mask.shape != (self.ra.shape[0],):
            raise ValueError("")

        return self.model_copy(
            update={
                "ra": self.ra[mask],
                "dec": self.dec[mask],
                "par_angle": self.par_angle[mask],
                "az": self.az[mask],
                "el": self.el[mask],
            }
        )

    @property
    def center(self) -> tuple[float, float]:
        """
        Mean boresight position (ra, dec) in radians.

        The samples with a NaN or inf pointing are ignored: the raw scans have some, and
        a single NaN would make the whole center NaN. The mean of the RA is a circular
        mean (mean of sin and cos), so it is right also across RA = 0.
        """
        good = np.isfinite(self.ra) & np.isfinite(self.dec)
        if not good.any():
            raise ValueError(
                "No finite boresight coordinates: cannot compute the center."
            )

        ra0 = np.arctan2(np.mean(np.sin(self.ra[good])), np.mean(np.cos(self.ra[good])))
        dec0 = np.mean(self.dec[good])
        return float(ra0), float(dec0)

    # def to_pixels(
    #     self,
    #     ra_det: NDArray[np.float64],
    #     dec_det: NDArray[np.float64],
    #     *,
    #     center: tuple[float, float],
    #     pixel_size: float,
    #     npix: tuple[int, int],
    #     projection: Literal["sin", "gnom"],
    # ) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.bool_]]:
    #     """
    #     Project detector coordinates onto a map and find their pixels.

    #     Detector coordinates are projected onto a plane centered on
    #     the specified map center. The resulting plane coordinates are
    #     converted into integer pixel indices.

    #     Parameters
    #     ----------
    #     ra_det : :class:`numpy.ndarray`
    #         Right ascension of each detector at each time sample,
    #         in radians. Typically returned by :meth:`detector_pointing`.
    #         Shape: ``(n_detectors, n_samples)``.

    #     dec_det : :class:`numpy.ndarray`
    #         Declination of each detector at each time sample,
    #         in radians. Typically returned by :meth:`detector_pointing`.
    #         Shape: ``(n_detectors, n_samples)``.

    #     center : tuple[float, float]
    #         Map projection center as ``(center_ra, center_dec)``,
    #         in radians. Both values are celestial coordinates in the
    #         ICRS reference frame.

    #     pixel_size : float
    #         Angular pixel size in degrees, measured in the projected
    #         map plane. Must be strictly positive.

    #     npix : tuple[int, int]
    #         Map dimensions as ``(nx, ny)``, where ``nx`` is the number
    #         of pixels along the x-axis and ``ny`` is the number of
    #         pixels along the y-axis.

    #     projection : {"sin", "gnom"}
    #         Projection used to transform celestial coordinates into
    #         map-plane coordinates. ``"sin"`` selects the orthographic
    #         (SIN) projection; ``"gnom"`` selects the gnomonic projection.

    #     Returns
    #     -------
    #     ix : NDArray[np.intp]
    #         Integer x-pixel indices, with shape matching ``ra_det``.

    #     iy : NDArray[np.intp]
    #         Integer y-pixel indices, with shape matching ``dec_det``.

    #     valid : NDArray[np.bool_]
    #         Boolean mask identifying coordinates that project to finite
    #         positions inside the map boundaries. Has the same shape as
    #         the detector-coordinate arrays.

    #     Notes
    #     -----
    #     Pixel indices are zero-based. Coordinates outside the map
    #     boundaries, or outside the valid domain of the selected
    #     projection, are marked ``False`` in ``valid``.

    #     The pixel origin is defined relative to the geometric center
    #     of the map. The returned indices do not by themselves specify
    #     how the map should be displayed or whether either axis should
    #     be reversed.
    #     """

    #     # # sta cosa va fatta for each detector nb!
    #     # x, y = radec2xy(
    #     #     ra_det,
    #     #     dec_det,
    #     #     center[0],
    #     #     center[1],
    #     #     projection=projection,
    #     # )

    #     # # convert projected angular coordinates from radians to degrees.
    #     # x = np.rad2deg(x)
    #     # y = np.rad2deg(y)

    #     # nx, ny = npix
    #     # half_width_x = nx * pixel_size / 2
    #     # half_width_y = ny * pixel_size / 2

    #     # ix = np.floor((x + half_width_x) / pixel_size).astype(np.intp)
    #     # iy = np.floor((y + half_width_y) / pixel_size).astype(np.intp)

    #     # valid = (
    #     #     np.isfinite(x)
    #     #     & np.isfinite(y)
    #     #     & (ix >= 0)
    #     #     & (ix < nx)
    #     #     & (iy >= 0)
    #     #     & (iy < ny)
    #     # )

    #     # pix = np.full(ra_det.shape, -1, dtype=np.int64)
    #     # pix[valid] = iy[valid] * nx + ix[valid]

    #     nx, ny = npix
    #     n_detectors, n_samples = ra_det.shape

    #     pix = np.full((n_detectors, n_samples), -1, dtype=np.int64)
    #     valid = np.zeros((n_detectors, n_samples), dtype=bool)

    #     half_width_x = nx * pixel_size / 2
    #     half_width_y = ny * pixel_size / 2

    #     for k in range(n_detectors):
    #         x, y = radec2xy(
    #             ra_det[k],
    #             dec_det[k],
    #             center[0],
    #             center[1],
    #             projection=projection,
    #         )

    #         x = np.rad2deg(x)
    #         y = np.rad2deg(y)

    #         finite = np.isfinite(x) & np.isfinite(y)

    #         ix = np.full(x.shape, -1, dtype=np.int64)
    #         iy = np.full(y.shape, -1, dtype=np.int64)

    #         ix[finite] = np.floor(
    #             (x[finite] + half_width_x) / pixel_size
    #         ).astype(np.int64)

    #         iy[finite] = np.floor(
    #             (y[finite] + half_width_y) / pixel_size
    #         ).astype(np.int64)

    #         inside = (
    #             finite
    #             & (ix >= 0)
    #             & (ix < nx)
    #             & (iy >= 0)
    #             & (iy < ny)
    #         )

    #         pix[k, inside] = iy[inside] * nx + ix[inside]
    #         valid[k] = inside

    #     return pix, valid

    #     # e poi devo salvare la pixellizzazione in un array del tipo
    #     # fits.BinTableHDU.from_columns([
    #     #     fits.Column(name="pix", format="K", array=pix_safe)
    #     # ]).writeto(output_dir / f"pix_{k}.fits", overwrite=True)

    #     # return ix, iy, valid
