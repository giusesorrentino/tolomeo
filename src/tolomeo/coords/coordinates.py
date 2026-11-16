from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class Coordinates(BaseModel):
    """
    Boresight coordinates of a scan.

    The class stores the celestial pointing of the telescope boresight
    as a function of time.

    Attributes
    ----------
    ra : :class:`~numpy.ndarray` of float64
        Right ascension of the telescope boresight at each time sample, in radians,
        in the FK5 reference frame (J2000 equinox). Shape: ``(n_samples,)``.

    dec : :class:`~numpy.ndarray` of float64
        Declination of the telescope boresight at each time sample, in radians,
        in the FK5 reference frame (J2000 equinox). Shape: ``(n_samples,)``.

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

        ra0 = np.arctan2(
            np.mean(np.sin(self.ra[good])), np.mean(np.cos(self.ra[good]))
        ) % (2 * np.pi)
        dec0 = np.mean(self.dec[good])
        return float(ra0), float(dec0)
