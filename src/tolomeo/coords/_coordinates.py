from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class Coordinates(BaseModel):
    """
    Boresight coordinates of a scan, and the name and position of its source.

    The class stores the celestial pointing of the telescope boresight
    as a function of time, and which source the scan observes and where it is.

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
    source_ra, source_dec : :class:`~numpy.ndarray` of float64 or None
        Position of the source (the target of the scan) at each time sample, in
        radians, J2000, as written in the header of every FITS subscan: the same
        at every sample for a fixed source, different from one subscan to the
        next for a planet. None if it is not known (a scan built by hand).
        Shape: ``(n_samples,)``.
    source_name : str or None
        Name of the source, as written in the ``SOURCE`` keyword of the header of
        the FITS subscans (for example ``"MARS"``). None if it is not known.

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
    source_ra: NDArray[np.float64] | None = None
    source_dec: NDArray[np.float64] | None = None
    source_name: str | None = None

    def select(self, mask: NDArray[np.bool_]) -> Coordinates:
        """
        Keep only the samples where ``mask`` is True.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` of bool
            True for the samples to keep. Shape ``(n_samples,)``.

        Returns
        -------
        :class:`~tolomeo.coords.Coordinates`
            New object whose arrays have shape ``(mask.sum(),)`` (copies).
        """
        if mask.dtype != bool:
            raise TypeError("")
        if mask.shape != (self.ra.shape[0],):
            raise ValueError("")

        update = {
            "ra": self.ra[mask],
            "dec": self.dec[mask],
            "par_angle": self.par_angle[mask],
            "az": self.az[mask],
            "el": self.el[mask],
        }

        # the position of the source is optional: a scan built by hand may not
        # have it
        for name in ("source_ra", "source_dec"):
            values = getattr(self, name)
            if values is not None:
                update[name] = values[mask]

        return self.model_copy(update=update)
