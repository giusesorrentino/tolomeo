import os
import re
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from ..beammap._beammap import BeamMap
from ..coords import Coordinates
from ..coords._transforms import shift_boresight
from ..tod import Scan, ScanSpec, TimeOrderedData
from ._beammap import read_beammap

__all__ = ["read_scan"]

# unit strings accepted for the angular columns of the DATA TABLE
_RADIAN_UNITS = {"rad", "radian", "radians"}
_ARCSEC_TO_RAD = np.pi / (180 * 3600)

# name of a detector column of the PH TABLE: chp_ and the id of the detector
_CHANNEL_COLUMN = re.compile(r"chp_\d+")


class SubScan(NamedTuple):
    """
    Arrays of one FITS subscan, as :func:`_read_subscan` returns them.

    All the arrays have ``n_samples`` rows; the angles are in radians.
    ``source_ra``, ``source_dec`` and ``source_name`` are the position and the name
    of the source written in the primary header of the subscan (no name: None).
    """

    n_samples: int
    flags: NDArray[np.bool_]  # (n_samples,), True = good sample
    signal: NDArray[np.float64]  # (n_samples, n_det)
    time: NDArray[np.float64]  # (n_samples,), modified julian date
    ra: NDArray[np.float64]  # (n_samples,), radians
    dec: NDArray[np.float64]
    par_angle: NDArray[np.float64]
    az: NDArray[np.float64]
    el: NDArray[np.float64]
    source_ra: float  # radians, J2000
    source_dec: float
    source_name: str | None


def read_scan(
    tod_dirs: str | Path | ScanSpec | Sequence[str | Path | ScanSpec],
    beammap_path: str | Path | BeamMap | None = None,
    channels: int | Sequence[int] | None = None,
) -> Scan:
    """
    Read MISTRAL subscans and a beammap into a :class:`~tolomeo.tod.Scan`.

    Parameters
    ----------
    tod_dirs : str, :class:`~pathlib.Path`, :class:`~tolomeo.tod.ScanSpec` or a sequence of them
        Directory or list of directories with the FITS subscans of each scan.
    beammap_path : str, :class:`~pathlib.Path`, :class:`~tolomeo.beammap.BeamMap` or None, optional
        Beammap: a text file (see :func:`~tolomeo.io.read_beammap`) or a
        :class:`~tolomeo.beammap.BeamMap`. Only its good channels (flag 0) are
        read. None (default) reads every ``chp_XXX`` column of the FITS files,
        with zero offsets: for a scan made to measure the beammap.
    channels : int, sequence of int or None, optional
        Detectors to read, among the good ones of the beammap (all the columns of
        the FITS files without a beammap): None (default) reads all of them, an
        int ``n`` the first ``n`` by id, a sequence the detectors with those ids
        (the ``XXX`` of the ``chp_XXX`` columns).

    Returns
    -------
    :class:`~tolomeo.tod.Scan`
        TOD of the detectors, boresight coordinates, position and name of the
        source from the headers of the subscans and detector offsets (all angles in
        radians).

    Raises
    ------
    NotADirectoryError
        If an entry of `tod_dirs` is not a directory.
    FileNotFoundError
        If no FITS file is found.
    ValueError
        If the beammap has no good detector, `channels` is not valid, or a
        subscan is inconsistent or has no position of the source in its header.
    """
    dirs = [tod_dirs] if isinstance(tod_dirs, (str, Path, ScanSpec)) else tod_dirs
    dirs = [ScanSpec(path=d) if not isinstance(d, ScanSpec) else d for d in dirs]

    fits_files = []

    for spec in dirs:
        if not spec.path.is_dir():
            raise NotADirectoryError(f"{spec.path} is not a directory.")

        # sorting by name keeps the subscans in time order
        files = sorted(spec.path.glob("*.fit*", case_sensitive=False))

        for f in files:
            fits_files.append((f, spec))

    if not fits_files:
        raise FileNotFoundError("No FITS subscans found in the given directories.")

    if beammap_path is None:
        # a scan made to measure the beammap: every detector, at the boresight
        channel_ids = _read_channel_ids(fits_files[0][0])
        lon_offset = np.zeros(channel_ids.size)
        lat_offset = np.zeros(channel_ids.size)
    else:
        beammap = (
            beammap_path
            if isinstance(beammap_path, BeamMap)
            else read_beammap(beammap_path)
        )
        channel_ids, lon_offset, lat_offset = _good_channels(beammap)
    keep = _select_detectors(channel_ids, channels)

    channel_ids = channel_ids[keep]
    lon_offset = lon_offset[keep]
    lat_offset = lat_offset[keep]

    columns = [f"chp_{i:03d}" for i in channel_ids]

    workers = os.cpu_count() or 1
    futures = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_read_subscan, file, spec, columns)
            for file, spec in fits_files
        ]
        subscans = [future.result() for future in futures]
    lengths = [s.n_samples for s in subscans]

    # start index of each subscan, the last value is the total number of samples
    segments = np.concatenate(([0], np.cumsum(lengths)))

    coords = Coordinates(
        ra=np.concatenate([s.ra for s in subscans]),
        dec=np.concatenate([s.dec for s in subscans]),
        par_angle=np.concatenate([s.par_angle for s in subscans]),
        az=np.concatenate([s.az for s in subscans]),
        el=np.concatenate([s.el for s in subscans]),
        # one position per subscan, from its header, repeated over its samples
        source_ra=np.concatenate([np.full(s.n_samples, s.source_ra) for s in subscans]),
        source_dec=np.concatenate(
            [np.full(s.n_samples, s.source_dec) for s in subscans]
        ),
        source_name=_source_name(subscans),
    )
    tod = TimeOrderedData(
        signal=np.concatenate([s.signal for s in subscans], axis=0),
        time=np.concatenate([s.time for s in subscans]),
        flags=np.concatenate([s.flags for s in subscans]),
        segments=segments,
    )
    offsets = {"lon": lon_offset, "lat": lat_offset}  # radians, one value per detector

    return Scan(
        tod=tod,
        coordinates=coords,
        offsets=offsets,
        channel_ids=channel_ids.tolist(),
    )


def _read_angle(hdu: fits.BinTableHDU, name: str) -> NDArray[np.float64]:
    """
    Read an angular column of a FITS table as float64, checking it is in radians.

    Parameters
    ----------
    hdu : :class:`~astropy.io.fits.BinTableHDU`
        The table, for example the ``"DATA TABLE"`` of a subscan.
    name : str
        Column name.

    Returns
    -------
    :class:`~numpy.ndarray` of float64
        The column, in radians (a copy). ``ValueError`` if its unit is not radians.
        Shape ``(n_samples,)``.
    """
    unit = hdu.columns[name].unit

    if str(unit).lower() not in _RADIAN_UNITS:
        raise ValueError(f"column {name!r} is in {unit!r}, expected radians")

    return np.array(hdu.data[name], dtype=np.float64)


def _good_channels(
    beammap: BeamMap,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Ids and offsets (radians) of the good channels of a beammap, sorted by id.

    Raises
    ------
    ValueError
        If the beammap has no good channel.
    """
    good = beammap.flags == 0
    if not good.any():
        raise ValueError("the beammap contains no good detectors.")

    channel_ids = np.asarray(beammap.channel_ids[good], dtype=np.int64)
    order = np.argsort(channel_ids)  # same ordering as mistral2roma

    return (
        channel_ids[order],
        np.asarray(beammap.lon[good][order], dtype=np.float64),
        np.asarray(beammap.lat[good][order], dtype=np.float64),
    )


def _source_name(subscans: list[SubScan]) -> str | None:
    """Names of the sources in the headers, joined by commas; None if none."""
    names = sorted({s.source_name for s in subscans if s.source_name})
    return ", ".join(names) if names else None


def _read_channel_ids(file: Path) -> NDArray[np.int64]:
    """
    Ids of all the detectors of a FITS subscan, sorted.

    Parameters
    ----------
    file : :class:`~pathlib.Path`
        A FITS subscan, whose ``"PH TABLE"`` has one ``chp_XXX`` column per
        detector.

    Returns
    -------
    :class:`~numpy.ndarray` of int64
        The ``XXX`` of the ``chp_XXX`` columns, sorted. Shape ``(n_det,)``.

    Raises
    ------
    ValueError
        If the table has no ``chp_XXX`` column.
    """
    with fits.open(file, memmap=False) as hdul:
        names = hdul["PH TABLE"].columns.names

    # chp_ has four characters: the rest of the name is the id
    ids = [int(name[4:]) for name in names if _CHANNEL_COLUMN.fullmatch(name)]
    if not ids:
        raise ValueError(f"No chp_XXX column in the PH TABLE of {file.name}.")

    return np.sort(np.array(ids, dtype=np.int64))


def _read_subscan(file: Path, spec: ScanSpec, columns: list[str]) -> SubScan:
    """
    Read one FITS subscan: the detector columns and the boresight pointing.

    Runs in a worker process of :func:`read_scan`.

    Parameters
    ----------
    file : :class:`~pathlib.Path`
        The FITS file, with a ``"DATA TABLE"`` (time, flags, pointing) and a
        ``"PH TABLE"`` (one column per detector) of the same length.
    spec : :class:`~tolomeo.tod.ScanSpec`
        The scan the file belongs to; its ``ra_offset`` and ``dec_offset`` (arcsec)
        are added to the boresight.
    columns : list of str
        Detector columns to read, for example ``"chp_012"``, in the output order.
        Length ``n_det``.

    Returns
    -------
    :class:`~tolomeo.io._scan.SubScan`
        ``signal`` of shape ``(n_samples, n_det)`` and the other arrays of shape
        ``(n_samples,)``; ``source_ra``, ``source_dec`` and ``source_name`` from the
        primary header. ``ValueError`` if the two tables have different lengths, a
        detector column is missing or the header has no position of the source.
    """
    with fits.open(file, memmap=False) as hdul:
        data_hdu = hdul["DATA TABLE"]
        dt = data_hdu.data
        pt = hdul["PH TABLE"].data

        if len(pt) != len(dt):
            raise ValueError(f"Pointing/TOD length mismatch in {file.name}.")

        missing = [c for c in columns if c not in pt.names]
        if missing:
            raise ValueError(f"Detector column {missing[0]!r} missing in {file.name}.")

        # position of the source: radians and J2000, as the comments of the two
        # keywords say (a header has no unit to check, unlike the columns)
        header = hdul[0].header
        if "RightAscension" not in header or "Declination" not in header:
            raise ValueError(
                "No position of the source (RightAscension, Declination) in the "
                f"header of {file.name}."
            )
        source_ra = float(header["RightAscension"])
        source_dec = float(header["Declination"])
        source_name = str(header.get("SOURCE", "")).strip() or None

        r = _read_angle(data_hdu, "raj2000")
        d = _read_angle(data_hdu, "decj2000")

        ra, dec = shift_boresight(
            r, d, spec.ra_offset * _ARCSEC_TO_RAD, spec.dec_offset * _ARCSEC_TO_RAD
        )

        n_samples = len(dt)
        flags = dt["flag_track"].astype(bool)
        signal = np.column_stack([pt[c] for c in columns]).astype(
            np.float64, copy=False
        )
        time = np.array(dt["time"], dtype=np.float64)
        par_angle = _read_angle(data_hdu, "par_angle")
        az = _read_angle(data_hdu, "az")
        el = _read_angle(data_hdu, "el")

        return SubScan(
            n_samples=n_samples,
            flags=flags,
            signal=signal,
            time=time,
            ra=ra,
            dec=dec,
            par_angle=par_angle,
            az=az,
            el=el,
            source_ra=source_ra,
            source_dec=source_dec,
            source_name=source_name,
        )


def _select_detectors(
    channel_ids: NDArray[np.integer],
    channels: int | Sequence[int] | None,
) -> NDArray[np.bool_]:
    """
    Which detectors to read, among those that can be read.

    Parameters
    ----------
    channel_ids : :class:`~numpy.ndarray` of int
        Ids of the detectors that can be read, sorted: the good ones of the beammap
        (:func:`_good_channels`) or all the columns (:func:`_read_channel_ids`).
        Shape ``(n_det,)``.
    channels : int, sequence of int or None
        None keeps all of them; an int ``n`` keeps the first ``n`` by id; a
        sequence keeps the detectors with those ids.

    Returns
    -------
    :class:`~numpy.ndarray` of bool
        True for the detectors to read. Shape ``(n_det,)``.

    Raises
    ------
    ValueError
        If ``n`` is smaller than 1, or an id is not in ``channel_ids``.
    """
    if channels is None:
        return np.ones(channel_ids.size, dtype=bool)
    if isinstance(channels, int):  # the first n good detectors, by id
        if channels < 1:
            raise ValueError(f"`channels` must be at least 1, got {channels}.")
        return np.arange(channel_ids.size) < channels
    wanted = np.asarray(channels, dtype=int)
    missing = np.setdiff1d(wanted, channel_ids)
    if missing.size:  # a typo must not silently drop a detector
        raise ValueError(f"not detectors that can be read: {missing.tolist()}")
    return np.isin(channel_ids, wanted)
