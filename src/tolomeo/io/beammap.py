import re
from datetime import datetime
from pathlib import Path

import numpy as np

from ..beammap._beammap import BeamMap, ChannelFlag

__all__ = ["read_beammap", "write_beammap"]

# columns of a beammap file: id, lon, lat, their errors, flag, as in the MISTRAL ones
_FMT = ["%2d", "%12.6f", "%12.6f", "%12.3e", "%12.3e", "%2d"]

# lines of the header: the flag of the files of write_beammap (in the old ones 0 is
# good and 1 bad, and the columns 4 and 5 have the Tcal), the threshold of LOW_SNR,
# and the offsets of the reference channel from the boresight
_REASONS = "otherwise the sum of the reasons below"
_SNR = re.compile(r"LOW_SNR \(SNR < (\S+)\)")
_REF = re.compile(r"chp(\d+) (az|el)_off \(deg\): (\S+)")


def read_beammap(path: str | Path) -> BeamMap:
    """
    Read a beammap text file.

    The columns are id, lon and lat offsets (deg), two more columns and a flag;
    the header says how to read them:

    - written by :func:`write_beammap`: the two columns are the errors of the
      offsets, and the flag is the sum of the :class:`~tolomeo.beammap.ChannelFlag`
      of the channel. A channel whose fit failed gets back NaN offsets and errors,
      written as 0;
    - the older MISTRAL files: the two columns are the Tcal, not read, and the flag
      is 0 (good) or 1 (bad), with no reason: a bad channel gets
      :attr:`~tolomeo.beammap.ChannelFlag.UNKNOWN`, so that the reason stays
      unknown if the beammap is written again.

    The offsets of the reference channel and the threshold of the signal-to-noise
    ratio are read from the header when it has them, otherwise they are None.

    Parameters
    ----------
    path : str or :class:`~pathlib.Path`
        Beammap file.

    Returns
    -------
    :class:`~tolomeo.beammap.BeamMap`
        Every channel of the file, in its order, with the offsets in radians.

    Raises
    ------
    ValueError
        If the file has fewer than six columns, or an id appears twice.
    """
    with open(path) as f:
        header = [line[1:].strip() for line in f if line.startswith("#")]

    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    if data.shape[1] < 6:
        raise ValueError("the beammap must have at least six columns")

    channel_ids = data[:, 0].astype(np.int64)
    lon = np.deg2rad(data[:, 1])
    lat = np.deg2rad(data[:, 2])
    flags = data[:, 5].astype(np.int64)

    lon_error = lat_error = snr_threshold = None
    if not any(_REASONS in line for line in header):
        # an old file: 1 is bad, for a reason that it does not say
        flags = np.where(flags != 0, int(ChannelFlag.UNKNOWN), 0)
    else:
        lon_error = np.deg2rad(data[:, 3])
        lat_error = np.deg2rad(data[:, 4])
        # a failed fit has no position, and was written as 0
        failed = (flags & ChannelFlag.FIT_FAILED) != 0
        for values in (lon, lat, lon_error, lat_error):
            values[failed] = np.nan
        for line in header:
            if match := _SNR.search(line):
                snr_threshold = float(match.group(1))

    ref_channel, ref = None, {}
    for line in header:
        if match := _REF.search(line):
            ref_channel = int(match.group(1))
            ref[match.group(2)] = float(np.deg2rad(float(match.group(3))))

    return BeamMap(
        channel_ids=channel_ids,
        flags=flags,
        lat=lat,
        lon=lon,
        lat_error=lat_error,
        lon_error=lon_error,
        ref_channel=ref_channel,
        ref_lat=ref.get("el"),
        ref_lon=ref.get("az"),
        snr_threshold=snr_threshold,
    )


def write_beammap(beammap: BeamMap, path: str | Path) -> None:
    """
    Write a beammap text file, in the format of the MISTRAL ones.

    One row per channel: id, lon and lat offsets (deg), their errors (deg) in the
    two columns where the old files have the Tcal, and the flag: 0 for a good
    channel, otherwise the sum of its :class:`~tolomeo.beammap.ChannelFlag`, listed
    in the header. The header also gives the date, the number of rows, the columns
    and, when they are known, the offsets of the reference channel from the
    boresight, subtracted from all the others. NaN (a failed fit) and unknown
    errors are written as 0. :func:`read_beammap` reads the file back;
    :func:`~tolomeo.io.read_scan` reads the columns 1, 2, 3 and 6 and keeps the
    channels with flag 0, as with the old files.

    Parameters
    ----------
    beammap : :class:`~tolomeo.beammap.BeamMap`
        Beammap to write.
    path : str or :class:`~pathlib.Path`
        File to write; an existing one is replaced.
    """
    threshold = beammap.snr_threshold
    # one line per reason of the flag, from ChannelFlag: they cannot diverge
    flag_lines = [
        f"  {flag.value} {flag.name}"
        + (
            f" (SNR < {threshold:g})"
            if flag is ChannelFlag.LOW_SNR and threshold is not None
            else ""
        )
        for flag in ChannelFlag
    ]

    ref_lines = []
    ref = (beammap.ref_channel, beammap.ref_lon, beammap.ref_lat)
    if None not in ref:
        channel, ref_lon, ref_lat = ref
        ref_lines = [
            f"chp{channel} az_off (deg): {np.rad2deg(ref_lon):.12g}",
            f"chp{channel} el_off (deg): {np.rad2deg(ref_lat):.12g}",
        ]

    header = "\n".join(
        [
            f"Feed list created: {datetime.now().astimezone():%b %d %H:%M:%S %Y}",
            f"nfeeds: {beammap.channel_ids.size}",
            "Col.1: id",
            "Col.2: lon-offset (deg)",
            "Col.3: lat-offset (deg)",
            "Col.4: lon-offset error (deg)",
            "Col.5: lat-offset error (deg)",
            f"Col.6: flag (0=>Ok, {_REASONS})",
            *flag_lines,
            *ref_lines,
        ]
    )

    def degrees(values: np.ndarray | None) -> np.ndarray:
        """Radians to degrees, NaN and unknown to 0."""
        if values is None:
            return np.zeros(beammap.channel_ids.size)
        return np.nan_to_num(np.rad2deg(values))

    rows = np.column_stack(
        [
            beammap.channel_ids,
            degrees(beammap.lon),
            degrees(beammap.lat),
            degrees(beammap.lon_error),
            degrees(beammap.lat_error),
            beammap.flags,
        ]
    )
    np.savetxt(path, rows, fmt=_FMT, header=header, comments="# ")
