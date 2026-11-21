from itertools import pairwise

import numpy as np
from astropy.io import fits

from tolomeo.coords import Coordinates
from tolomeo.coords.transforms import radec2latlon
from tolomeo.mapmaking._pointing import _pointing_matrix
from tolomeo.tod import Scan, TimeOrderedData

# --- BASE GLOBAL CONSTANTS ---
ARCSEC = np.deg2rad(1 / 3600)  # one arcsec in radians
ARCMIN = 60 * ARCSEC

SAMPLE_RATE = 244.0  # Hz
MJD_START = 60600.0  # modified Julian date of the first sample

# centre of the synthetic field (radians), near 3C84
RA0 = np.deg2rad(52.0)
DEC0 = np.deg2rad(41.5)
SOURCE_NAME = "FAKE"  # the SOURCE keyword of the headers

# detectors: beammap ids and offsets from the boresight (radians)
DETECTOR_IDS = [3, 7, 12, 20]
OFFSETS_LON = np.array([-20.0, 10.0, 0.0, 25.0]) * ARCSEC
OFFSETS_LAT = np.array([15.0, -10.0, 0.0, 5.0]) * ARCSEC

# raster: subscans along RA, one row of Dec after the other
N_SUBSCANS = 8
SUBSCAN_LEN = 256
SCAN_HALF_WIDTH = 90 * ARCSEC  # half length of a subscan on the sky
ROW_STEP = 15 * ARCSEC  # distance in Dec between two subscans

# default map grid of the tests
PIXEL_SIZE = 10.0  # arcsec

N_DET = len(DETECTOR_IDS)


def raster(n_subscans=N_SUBSCANS, subscan_len=SUBSCAN_LEN, gap=0.0):
    """
    Boresight of a raster around (RA0, DEC0), and the time of every sample.

    Return ``ra``, ``dec`` (radians), ``time`` (MJD) and ``segments``. The subscans go
    back and forth along RA, one Dec row each; ``gap`` seconds of nothing separate two
    subscans in time.
    """
    n_samples = n_subscans * subscan_len
    along = np.linspace(-SCAN_HALF_WIDTH, SCAN_HALF_WIDTH, subscan_len)

    ra = np.empty(n_samples)
    dec = np.empty(n_samples)
    seconds = np.empty(n_samples)

    for k in range(n_subscans):
        rows = slice(k * subscan_len, (k + 1) * subscan_len)
        sweep = along if k % 2 == 0 else along[::-1]
        ra[rows] = RA0 + sweep / np.cos(DEC0)
        dec[rows] = DEC0 + (k - (n_subscans - 1) / 2) * ROW_STEP
        seconds[rows] = (np.arange(subscan_len) + k * subscan_len) / SAMPLE_RATE
        seconds[rows] += k * gap

    segments = np.arange(n_subscans + 1, dtype=np.int64) * subscan_len
    return ra, dec, MJD_START + seconds / 86400, segments


def gaussian_source(scan, amplitude, fwhm=30 * ARCSEC, center=(RA0, DEC0)):
    """Signal of a Gaussian source seen by every detector, ``(n_samples, n_det)``."""
    coords = scan.coordinates
    lat, lon = radec2latlon(
        coords.ra,
        coords.dec,
        center[0],
        center[1],
        "sin",
        coords.par_angle,
        scan.offsets["lon"],
        scan.offsets["lat"],
        frame="radec",
    )
    sigma = fwhm / np.sqrt(8 * np.log(2))
    return (amplitude * np.exp(-(lat**2 + lon**2) / (2 * sigma**2))).T


def pixelized_source(scan, geometry, amplitude, fwhm=30 * ARCSEC):
    """
    Gaussian source at the centre of ``geometry``, sampled at the centres of its pixels.

    Return the sky ``(ny, nx)`` and the signal it gives to every detector of ``scan``,
    ``(n_samples, n_det)``: the value of the pixel the sample falls in, 0 outside the
    map. A sky constant inside every pixel has no pixelization error, so a map made
    from this signal must give the sky back.
    """
    pixel_size = geometry.pixel_size * ARCSEC
    y, x = np.mgrid[: geometry.ny, : geometry.nx]
    r2 = ((x - geometry.nx / 2 + 0.5) ** 2 + (y - geometry.ny / 2 + 0.5) ** 2) * (
        pixel_size**2
    )
    sigma = fwhm / np.sqrt(8 * np.log(2))
    sky = amplitude * np.exp(-r2 / (2 * sigma**2))

    pixels, valid = _pointing_matrix(scan.coordinates, scan.offsets, geometry)
    return sky, np.where(valid, sky.ravel()[pixels], 0.0)


def make_scan(
    *,
    n_subscans=N_SUBSCANS,
    subscan_len=SUBSCAN_LEN,
    n_det=N_DET,
    noise_rms=1.0,
    par_angle=0.0,
    gap=0.0,
    seed=0,
):
    """
    Synthetic ``Scan``: a raster with white noise of rms ``noise_rms`` in every
    detector.

    All the samples are good. ``par_angle`` (radians) is the same at every sample.
    """
    rng = np.random.default_rng(seed)
    ra, dec, time, segments = raster(n_subscans, subscan_len, gap)
    n_samples = ra.size

    coords = Coordinates(
        ra=ra,
        dec=dec,
        par_angle=np.full(n_samples, par_angle),
        az=np.linspace(1.0, 1.1, n_samples),
        el=np.full(n_samples, 0.8),
        source_ra=np.full(n_samples, RA0),
        source_dec=np.full(n_samples, DEC0),
        source_name=SOURCE_NAME,
    )
    tod = TimeOrderedData(
        signal=noise_rms * rng.standard_normal((n_samples, n_det)),
        time=time,
        flags=np.ones(n_samples, dtype=bool),
        segments=segments,
    )
    return Scan(
        tod=tod,
        coordinates=coords,
        offsets={"lon": OFFSETS_LON[:n_det].copy(), "lat": OFFSETS_LAT[:n_det].copy()},
        channel_ids=DETECTOR_IDS[:n_det],
    )


def make_two_passes(**kwargs):
    """
    Two passes of the raster of ``make_scan`` over the same field, one after the
    other, as two scans read together: the second starts when the first ends.

    ``kwargs`` go to both ``make_scan``; the noise of the second pass has another
    seed, so the two are independent.
    """
    first = make_scan(**kwargs)
    second = make_scan(**(kwargs | {"seed": kwargs.get("seed", 0) + 1}))
    n_first = first.tod.signal.shape[0]
    shift = n_first / SAMPLE_RATE / 86400  # days

    def join(a, b):
        return np.concatenate([a, b])

    tod = first.tod.model_copy(
        update={
            "signal": join(first.tod.signal, second.tod.signal),
            "time": join(first.tod.time, second.tod.time + shift),
            "flags": join(first.tod.flags, second.tod.flags),
            "segments": join(first.tod.segments, second.tod.segments[1:] + n_first),
        }
    )
    names = ["ra", "dec", "par_angle", "az", "el", "source_ra", "source_dec"]
    first_coords, second_coords = first.coordinates, second.coordinates
    coords = first_coords.model_copy(
        update={
            name: join(getattr(first_coords, name), getattr(second_coords, name))
            for name in names
        }
    )
    return first.model_copy(update={"tod": tod, "coordinates": coords})


def with_signal(scan, signal):
    """Copy of ``scan`` with ``signal`` instead of its TOD."""
    return scan.model_copy(
        update={"tod": scan.tod.model_copy(update={"signal": np.asarray(signal)})}
    )


# -----------------------------------------------------------------------------
# MISTRAL FITS files
# -----------------------------------------------------------------------------


def _angle_column(name, values, unit="rad"):
    return fits.Column(name=name, format="D", unit=unit, array=values)


def write_subscan(path, scan, start, stop, angle_unit="rad"):
    """Write the samples ``start .. stop`` of ``scan`` as a MISTRAL FITS subscan."""
    coords = scan.coordinates
    rows = slice(start, stop)

    data_table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="time", format="D", array=scan.tod.time[rows]),
            fits.Column(
                name="flag_track",
                format="I",
                array=scan.tod.flags[rows].astype(np.int16),
            ),
            _angle_column("raj2000", coords.ra[rows], angle_unit),
            _angle_column("decj2000", coords.dec[rows], angle_unit),
            _angle_column("par_angle", coords.par_angle[rows], angle_unit),
            _angle_column("az", coords.az[rows], angle_unit),
            _angle_column("el", coords.el[rows], angle_unit),
        ],
        name="DATA TABLE",
    )
    ph_table = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name=f"chp_{det:03d}", format="D", array=scan.tod.signal[rows, j]
            )
            for j, det in enumerate(scan.channel_ids)
        ],
        name="PH TABLE",
    )
    # the position of the source goes in the primary header, as MISTRAL writes it
    primary = fits.PrimaryHDU()
    if coords.source_ra is not None:
        primary.header["HIERARCH RightAscension"] = float(coords.source_ra[start])
        primary.header["HIERARCH Declination"] = float(coords.source_dec[start])
    if coords.source_name is not None:
        primary.header["SOURCE"] = coords.source_name
    fits.HDUList([primary, data_table, ph_table]).writeto(path)


def write_beammap(path, ids, lon, lat, flags=None):
    """Write a beammap: id, lon and lat offsets in degrees, two unused columns, flag."""
    flags = np.zeros(len(ids)) if flags is None else np.asarray(flags)
    rows = np.column_stack(
        [
            ids,
            np.rad2deg(lon),
            np.rad2deg(lat),
            np.ones(len(ids)),
            np.ones(len(ids)),
            flags,
        ]
    )
    np.savetxt(path, rows, header="id lon(deg) lat(deg) tcal1 tcal2 flag")


def write_scan_dir(folder, scan):
    """
    Write ``scan`` as MISTRAL files: one FITS file per subscan in ``folder/tod`` and a
    beammap ``folder/beammap.dat``. Return the two paths.
    """
    tod_dir = folder / "tod"
    tod_dir.mkdir(parents=True)

    for k, (start, stop) in enumerate(pairwise(scan.tod.segments)):
        write_subscan(tod_dir / f"subscan_{k:03d}.fits", scan, start, stop)

    beammap = folder / "beammap.dat"
    write_beammap(beammap, scan.channel_ids, scan.offsets["lon"], scan.offsets["lat"])
    return tod_dir, beammap


def write_skydip(path, el, signal, channel_ids, flags=None):
    """Write a skydip FITS file: elevation (radians) and one column per detector."""
    flags = np.ones(el.size, dtype=np.int16) if flags is None else flags
    data_table = fits.BinTableHDU.from_columns(
        [
            _angle_column("el", el),
            fits.Column(name="flag_track", format="I", array=flags),
        ],
        name="DATA TABLE",
    )
    ph_table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name=f"chp_{det:03d}", format="D", array=signal[:, j])
            for j, det in enumerate(channel_ids)
        ],
        name="PH TABLE",
    )
    fits.HDUList([fits.PrimaryHDU(), data_table, ph_table]).writeto(path)


# -----------------------------------------------------------------------------
# YAML of the command line
# -----------------------------------------------------------------------------

# smallest working configuration of `tolomeo`, for the files written by `write_run`.
# On the short scans of make_scan() the CG does not get much below a residual of 1e-2,
# so the tolerance is loose: these tests check the command line, not the solution
RUN_YAML = """\
scan:
  tod_dirs: tod
  beammap: beammap.dat
noise:
  nperseg: 128
gls:
  tol: 5.0e-2
output:
  path: out/map.fits
"""


def write_run(folder, text=RUN_YAML, scan=None):
    """
    Write ``scan`` as MISTRAL files in ``folder`` (see :func:`write_scan_dir`) and
    ``text`` as ``folder/run.yaml``; return the path of the YAML file.

    ``scan`` defaults to ``make_scan()``. The paths in ``text`` are relative to
    ``folder``, as the command line reads them.
    """
    write_scan_dir(folder, make_scan() if scan is None else scan)
    path = folder / "run.yaml"
    path.write_text(text)
    return path
