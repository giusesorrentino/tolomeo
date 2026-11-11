from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class TimeOrderedData(BaseModel):
    """
    Time-ordered data (TOD) of all the detectors of one observation.

    Attributes
    ----------
    signal : :class:`~numpy.ndarray` of float64
        One row per time sample, one column per detector, in the units of the
        instrument.
        Shape ``(n_samples, n_det)``.
    time : :class:`~numpy.ndarray` of float64
        Time of every sample, modified Julian date (days).
        Shape ``(n_samples,)``.
    flags : :class:`~numpy.ndarray` of bool
        True for a good sample, the same for all the detectors.
        Shape ``(n_samples,)``.
    segments : :class:`~numpy.ndarray` of int64
        First row of every subscan, followed by ``n_samples``: subscan ``k`` is
        ``signal[segments[k]:segments[k + 1]]``.
        Shape ``(n_subscans + 1,)``.
    passband : float or 2-tuple of floats, optional
        ...
    unit : str, optional
        Unit of ``signal``, as a FITS unit string (``"K"`` after
        :func:`~tolomeo.processing.calibrate`). None means not known: the raw units of
        the instrument.
    """

    # numpy arrays are not pydantic types: they are only checked with isinstance
    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: NDArray[np.float64]  # shape (n_samples, n_det)
    time: NDArray[np.float64]  # shape (n_samples,)
    flags: NDArray[np.bool_]  # shape (n_samples,)
    segments: NDArray[np.int64]
    passband: tuple[float, float] | None = None
    unit: str | None = None

    def select(
        self, mask: NDArray[np.bool_]
    ) -> TimeOrderedData:  # mask has shape (n_samples,)
        """
        Keep only the samples where ``mask`` is True.

        Parameters
        ----------
        mask : :class:`~numpy.ndarray` of bool
            True for the samples to keep.
            Shape ``(n_samples,)``.

        Returns
        -------
        :class:`~tolomeo.tod.TimeOrderedData`
            A new object with ``n_samples = mask.sum()``; ``segments`` is moved to the
            positions of the kept samples. The arrays are copies.
        """
        if mask.dtype != bool:
            raise TypeError("")
        if mask.shape != (self.signal.shape[0],):
            raise ValueError("")

        return self.model_copy(
            update={
                "signal": self.signal[mask, :],
                "time": self.time[mask],
                "flags": self.flags[mask],
                "segments": np.concatenate(([0], np.cumsum(mask)))[self.segments],
            }
        )

    @property
    def sample_rate(self) -> float:
        """Sampling frequency in Hz, from the median time step between samples."""
        dt = np.diff(
            self.time * 86400, axis=0
        )  # remember that time is in MJD, so we convert to seconds
        return 1 / np.median(dt)
