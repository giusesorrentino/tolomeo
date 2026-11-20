import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

__all__ = ["BeamMap"]


class BeamMap(BaseModel):
    r"""
    Positions of the channels on the focal plane, and their flags.

    One entry per channel, in the same order in every array. The offsets are those
    of a beammap file, which :func:`~tolomeo.io.read_scan` reads: the position of
    every channel relative to a reference channel, fixed in the horizontal frame
    (on the plane of the projection: about :math:`\Delta\mathrm{az}\cos\mathrm{el}`
    and :math:`\Delta\mathrm{el}`).

    Attributes
    ----------
    channel_ids : :class:`~numpy.ndarray` of int
        Id of every channel, the number in the name of its column ``chp_XXX`` in
        the FITS files. Shape ``(n_det,)``.
    flags : :class:`~numpy.ndarray` of int
        0 for a good channel, otherwise the sum of the reasons why it is flagged,
        the :class:`~tolomeo.beammap.ChannelFlag`: ``flags & reason`` is nonzero for
        the channels flagged for that reason. It is the column 6 of a beammap file.
        Shape ``(n_det,)``.
    lon, lat : :class:`~numpy.ndarray` of float
        Offsets of every channel in azimuth and in elevation, in radians,
        relative to the reference channel. NaN for a channel whose fit failed
        (0 in a beammap file). Shape ``(n_det,)``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    channel_ids: NDArray[np.uint]
    flags: NDArray[np.integer]  # 0 good, otherwise the sum of its ChannelFlag
    lat: NDArray[np.floating]
    lon: NDArray[np.floating]
