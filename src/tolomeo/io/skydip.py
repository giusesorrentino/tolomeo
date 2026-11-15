from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count
from pathlib import Path
from typing import NamedTuple

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

from .._lib.messages import _warn
from .scan import _read_angle

__all__ = ["SkyDip", "read_skydip"]

# a file with fewer valid samples is skipped (as in the skydip of mistral2roma)
_MIN_SAMPLES = 20


class SkyDip(NamedTuple):
    """
    Valid samples of one skydip FITS file.

    ``el`` and ``signal`` have the same number of rows.
    """

    el: NDArray[np.float64]  # (n_samples,), radians
    signal: NDArray[np.float64]  # (n_samples, n_det), raw units


def read_skydip(gain_dir: str | Path, detector_ids: Sequence[int]) -> list[SkyDip]:
    """
    Read the skydip FITS files of a directory, or a single one.

    Parameters
    ----------
    gain_dir : str or :class:`~pathlib.Path`
        Directory with the ``GAIN_CAL`` FITS files, or the path of a single FITS file.
    detector_ids : sequence of int
        Detectors to read, for example ``scan.detector_ids``. Their columns are
        ``chp_000``, ``chp_001``, ... in the ``"PH TABLE"``.

    Returns
    -------
    list of :class:`SkyDip`
        One per usable file, in the order of the file names. Only the samples with
        a valid elevation and ``flag_track`` set are kept. A file is skipped, with a
        warning that lists its name, if it is empty or truncated, if it lacks one of
        the two tables, if they have a different number of rows, or if it has fewer
        than 20 valid samples. The columns of ``signal`` are in the order of
        ``detector_ids``.

    Raises
    ------
    NotADirectoryError
        If ``gain_dir`` is neither a directory nor a file.
    FileNotFoundError
        If there is no usable file.
    """
    gain_dir = Path(gain_dir).resolve()

    if gain_dir.is_file():
        # a single skydip, named explicitly: its extension is not checked
        files = [gain_dir]
    elif gain_dir.is_dir():
        # sorting by name keeps the files in time order
        files = sorted(gain_dir.glob("*.fit*", case_sensitive=False))
    else:
        raise NotADirectoryError(f"{gain_dir} is neither a directory nor a file.")

    workers = cpu_count() or 1

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_read_skydip_file, file, detector_ids) for file in files
        ]
        results = [future.result() for future in futures]

    # a file that cannot be used comes back as None
    skipped = [file.name for file, result in zip(files, results) if result is None]
    used = [result for result in results if result is not None]

    if skipped:
        _warn(
            f"{len(skipped)} of {len(files)} skydip files skipped: "
            f"{', '.join(skipped)}.",
            RuntimeWarning,
        )

    if not used:
        raise FileNotFoundError(f"No usable skydip file in {gain_dir}.")

    return used


def _read_skydip_file(file: Path, detector_ids: Sequence[int]) -> SkyDip | None:
    """
    Read one skydip FITS file, or return None if it has to be skipped.

    It runs in a worker process of :func:`read_skydip`, so it is defined at the top
    level of the module: a function defined inside another cannot be sent to a process.

    Parameters
    ----------
    file : :class:`~pathlib.Path`
        The FITS file, with a ``"DATA TABLE"`` (elevation, flags) and a
        ``"PH TABLE"`` (one column per detector) of the same length.
    detector_ids : sequence of int
        Detectors to read, in the output order. Their columns are ``chp_000``,
        ``chp_001``, ... in the ``"PH TABLE"``.

    Returns
    -------
    :class:`SkyDip` or None
        None if the file has to be skipped (see :func:`read_skydip`).
    """
    columns = [f"chp_{i:03d}" for i in detector_ids]

    try:
        hdul = fits.open(file, memmap=False)
    except OSError:  # an empty file
        return None

    with hdul:
        try:
            if "DATA TABLE" not in hdul or "PH TABLE" not in hdul:
                return None

            pt = hdul["PH TABLE"].data
            dt = hdul["DATA TABLE"].data
        except (OSError, ValueError):  # a truncated file
            return None

        if len(dt) != len(pt):
            return None

        el = _read_angle(hdul["DATA TABLE"], "el")

        ok = (
            dt["flag_track"].astype(bool)
            & np.isfinite(el)
            & (el > 0)
            & (el <= np.pi / 2)
        )

        if np.count_nonzero(ok) < _MIN_SAMPLES:
            return None

        signal = np.column_stack([pt[c] for c in columns]).astype(
            np.float64, copy=False
        )

    return SkyDip(el=el[ok], signal=signal[ok])
