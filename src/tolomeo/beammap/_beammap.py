from enum import IntFlag
from typing import Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, model_validator

__all__ = ["BeamMap", "ChannelFlag"]


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
        2: the fitted amplitude over its error is below the threshold, the
        ``snr_threshold`` of the :class:`BeamMap`.
    DUPLICATE : int
        4: another channel is less than 1 arcsec away, usually because the same
        tone is written in two channels (identical data).
    AT_EDGE : int
        8: the centre of the fit is closer than ``fit_radius`` to the edge of the
        maps, where its cutout is cut.
    UNKNOWN : int
        16: flagged in a file that does not say why: in the older MISTRAL files the
        flag is only 0 or 1.
    """

    FIT_FAILED = 1
    LOW_SNR = 2
    DUPLICATE = 4
    AT_EDGE = 8
    UNKNOWN = 16


class BeamMap(BaseModel):
    r"""
    Positions of the channels on the focal plane, and their flags.

    One entry per channel, in the same order in every array. The offsets are those
    of a beammap file, which :func:`~tolomeo.io.read_scan` reads: the position of
    every channel relative to a reference channel, fixed in the horizontal frame
    (on the plane of the projection: about :math:`\Delta\mathrm{az}\cos\mathrm{el}`
    and :math:`\Delta\mathrm{el}`). What a file may not say (the errors, the
    reference channel, the threshold of the signal-to-noise ratio, the source and
    the time of the scan) is None.

    Attributes
    ----------
    channel_ids : :class:`~numpy.ndarray` of int
        Id of every channel, the number in the name of its column ``chp_XXX`` in
        the FITS files. Every id appears once. Shape ``(n_det,)``.
    flags : :class:`~numpy.ndarray` of int
        0 for a good channel, otherwise the sum of the reasons why it is flagged,
        the :class:`ChannelFlag`: ``flags & reason`` is nonzero for the channels
        flagged for that reason. It is the column 6 of a beammap file.
        Shape ``(n_det,)``.
    lat, lon : :class:`~numpy.ndarray` of float
        Offsets of every channel in elevation and in azimuth, in radians, relative
        to the reference channel. NaN for a channel whose fit failed.
        Shape ``(n_det,)``.
    lat_error, lon_error : :class:`~numpy.ndarray` of float or None
        Errors of the offsets from the fit of every channel, in radians (the error
        of the reference channel is not included); NaN for a channel whose fit
        failed. None if not known: the old files have the Tcal in those columns.
        Shape ``(n_det,)``. Defaults to ``None``.
    ref_channel : int or None
        Channel whose offsets are zero, the reference of all the others.
        Defaults to ``None``.
    ref_lat, ref_lon : float or None
        Offsets of ``ref_channel`` from the boresight, in radians, subtracted from
        all the channels. Defaults to ``None``.
    snr_threshold : float or None
        Signal-to-noise ratio below which a channel is flagged
        :attr:`ChannelFlag.LOW_SNR`. Defaults to ``None``.
    source_name : str or None
        Name of the source of the scan, as in the FITS headers. Defaults to ``None``.
    time_range : tuple of float or None
        Start and end of the scan, in MJD (UTC). Defaults to ``None``.

    Raises
    ------
    ValueError
        If an array does not have the shape of ``channel_ids``, or an id appears
        twice.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    channel_ids: NDArray[np.integer]
    flags: NDArray[np.integer]  # 0 good, otherwise the sum of its ChannelFlag
    lat: NDArray[np.floating]
    lon: NDArray[np.floating]
    lat_error: NDArray[np.floating] | None = None
    lon_error: NDArray[np.floating] | None = None
    ref_channel: int | None = None
    ref_lat: float | None = None
    ref_lon: float | None = None
    snr_threshold: float | None = None
    source_name: str | None = None
    time_range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        """One entry per channel in every array, and every id once."""
        shape = self.channel_ids.shape
        for name in ("flags", "lat", "lon", "lat_error", "lon_error"):
            value = getattr(self, name)
            if value is not None and value.shape != shape:
                raise ValueError(
                    f"`{name}` has shape {value.shape}, `channel_ids` {shape}."
                )

        if np.unique(self.channel_ids).size != self.channel_ids.size:
            raise ValueError("duplicate detector ids in the beammap.")

        return self
