import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from ..coords import Coordinates
from ..coords.transforms import shift_boresight
from ..tod import Scan, ScanSpec, TimeOrderedData

__all__ = ["read_scan"]

# unit strings accepted for the angular columns of the DATA TABLE
_RADIAN_UNITS = {"rad", "radian", "radians"}
_ARCSEC_TO_RAD = np.pi / (180 * 3600)


class SubScan(NamedTuple):
    """
    Arrays of one FITS subscan, as :func:`_read_subscan` returns them.

    All the arrays have ``n_samples`` rows; the angles are in radians.
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


def read_scan(
    tod_dirs: str | Path | ScanSpec | Sequence[str | Path | ScanSpec],
    beammap_path: str | Path,
) -> Scan:
    """
    Read MISTRAL subscans and a beammap into a :class:`MISTRALScan`.

    Parameters
    ----------
    tod_dirs : str, :class:`~pathlib.Path`, :class:`~tolomeo.tod.ScanSpec` or a sequence of them
        Directory or list of directories with the FITS subscans of each scan.
    beammap_path : str or :class:`~pathlib.Path`
        Beammap text file, with columns id, lon and lat offsets (deg), two
        columns not used here and a flag (0 = good detector).

    Returns
    -------
    :class:`~tolomeo.tod.Scan`
        TOD of the good detectors, boresight coordinates and detector offsets
        (all angles in radians).

    Raises
    ------
    NotADirectoryError
        If an entry of `tod_dirs` is not a directory.
    FileNotFoundError
        If no FITS file is found.
    ValueError
        If the beammap has no good detector or a subscan is inconsistent.
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

    detector_ids, lon_offset, lat_offset = _read_beammap(beammap_path)
    columns = [f"chp_{i:03d}" for i in detector_ids]

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
        detector_ids=detector_ids.tolist(),
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


def _read_beammap(
    beammap_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ids and offsets (radians) of the good detectors, sorted by id."""
    beammap = np.atleast_2d(np.loadtxt(beammap_path, comments="#"))

    if beammap.shape[1] < 6:
        raise ValueError("the beammap must have at least six columns")

    # columns: id, lon (deg), lat (deg), two unused ones, flag (0 = good)
    ids, lon_offset, lat_offset, kid_flag = beammap[:, [0, 1, 2, 5]].T
    good = kid_flag == 0

    detector_ids = ids[good].astype(int)
    if detector_ids.size == 0:
        raise ValueError("the beammap contains no good detectors.")
    if np.unique(detector_ids).size != detector_ids.size:
        raise ValueError("duplicate detector ids in the beammap.")

    order = np.argsort(detector_ids)  # same ordering as mistral2roma

    return (
        detector_ids[order],
        np.deg2rad(lon_offset[good][order]),
        np.deg2rad(lat_offset[good][order]),
    )


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
    :class:`~tolomeo.io.scan.SubScan`
        ``signal`` of shape ``(n_samples, n_det)`` and the other arrays of shape
        ``(n_samples,)``. ``ValueError`` if the two tables have different lengths or
        a detector column is missing.
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
        )
