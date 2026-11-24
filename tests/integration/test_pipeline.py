# Integration tests for the map-making pipeline.
#
# These tests verify that the stages of tolomeo work correctly together: synthetic
# MISTRAL files are read with read_scan, cleaned with process_tod, mapped with
# estimate_inverse_noise and gls_map, and written and read back with write_map and
# read_map. The sky is a Gaussian source sampled at the centres of the pixels, so that
# the TOD is exactly P m plus white noise and the map must give m back: a sky that
# changes inside a pixel would add the pixelization error, which the GLS does not
# remove.

import numpy as np
import pytest

from tests.common import (
    DEC0,
    DETECTOR_IDS,
    RA0,
    make_scan,
    pixelized_source,
    with_signal,
    write_scan_dir,
    write_skydip,
)
from tolomeo.calibration import calibrate, skydip_responsivity
from tolomeo.io import read_map, read_scan, read_skydip, write_map
from tolomeo.map import MapGeometry
from tolomeo.mapmaking import binned_map, estimate_inverse_noise, gls_map
from tolomeo.processing import (
    Baseline,
    CommonMode,
    FilteringConfig,
    RadiusMask,
    process_tod,
)

AMPLITUDE = 5.0  # peak of the source, in the units of the TOD
NOISE_RMS = 0.05
NX, NY = 36, 30
PIXEL_SIZE = 10.0  # arcsec
GEOMETRY = MapGeometry(nx=NX, ny=NY, pixel_size=PIXEL_SIZE, center=(RA0, DEC0))
MASK = RadiusMask(radius=1.0)

# the source sampled at the centres of the pixels, (ny, nx)
SKY, _ = pixelized_source(make_scan(), GEOMETRY, AMPLITUDE)

# the source is on the corner of the four central pixels
CENTRAL = (slice(NY // 2 - 1, NY // 2 + 1), slice(NX // 2 - 1, NX // 2 + 1))


def _sky_scan(geometry=GEOMETRY, gain=1.0, par_angle=0.0):
    """Raster over the pixelized source, plus white noise, times ``gain``."""
    scan = make_scan(noise_rms=NOISE_RMS, par_angle=par_angle)
    _, signal = pixelized_source(scan, geometry, AMPLITUDE)
    return with_signal(scan, gain * (signal + scan.tod.signal))


def _clean_and_map(scan, geometry=GEOMETRY):
    """process_tod, estimate_inverse_noise and gls_map, with the source masked."""
    config = FilteringConfig(steps=[Baseline(), CommonMode()], mask=MASK)
    clean = process_tod(scan, config, geometry=geometry)
    inv_noise = estimate_inverse_noise(
        clean, nperseg=128, mask=MASK, geometry=geometry, workers=1
    )
    return clean, gls_map(clean, geometry, inv_noise, workers=1)


def _assert_sky(result):
    # The map is the sky, within 2% of the amplitude in every pixel with some hits,
    # and its peak is in the four central pixels.
    observed = result.hits >= 5
    error = result.image[observed] - SKY[observed]

    assert result.converged
    assert np.abs(error).max() < 0.02 * AMPLITUDE
    assert np.max(result.image[observed]) == np.max(result.image[CENTRAL])


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


def test_pipeline_recovers_the_sky(tmp_path):
    # From the FITS files to the FITS map, and the map read back is the one written.
    tod_dir, beammap = write_scan_dir(tmp_path, _sky_scan())

    scan = read_scan(tod_dir, beammap)
    _, result = _clean_and_map(scan)
    write_map(result, tmp_path / "map.fits")
    read = read_map(tmp_path / "map.fits")

    _assert_sky(result)
    np.testing.assert_array_equal(read.image, result.image)
    np.testing.assert_array_equal(read.hits, result.hits)


def test_gls_close_to_binned_with_white_noise():
    # With white noise the GLS map and the binned map of the same cleaned scan are
    # the same sky.
    clean, result = _clean_and_map(_sky_scan())

    binned = binned_map(clean, GEOMETRY)
    observed = binned.hits >= 5

    difference = result.image[observed] - binned.image[observed]
    assert np.abs(difference).max() < 0.02 * AMPLITUDE


def test_calibrated_pipeline_in_kelvin(tmp_path):
    # The raw TOD is the sky temperature times a responsivity of -2: the skydips find
    # it, and the calibrated map is the sky in kelvin.
    t_atm, tau_atm, responsivity = 267.0, 0.16, -2.0
    el = np.linspace(np.deg2rad(20), np.deg2rad(80), 200)
    t_sky = t_atm * (1 - np.exp(-tau_atm / np.sin(el)))
    skydip = 3.0 + responsivity * t_sky[:, None] * np.ones(len(DETECTOR_IDS))
    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()
    write_skydip(cal_dir / "skydip.fits", el, skydip, DETECTOR_IDS)

    scan = _sky_scan(gain=responsivity)
    skydips = read_skydip(cal_dir, scan.channel_ids)
    calibrated = calibrate(
        scan, skydip_responsivity(skydips, T_atm=t_atm, tau_atm=tau_atm)
    )
    _, result = _clean_and_map(calibrated)

    assert result.unit == "K"
    _assert_sky(result)


@pytest.mark.parametrize("frame", ["radec", "azel"])
def test_pipeline_with_parallactic_angle(frame):
    # The detectors turn around the boresight: the map is still the sky, in both
    # frames (each with the sky sampled on its own grid).
    geometry = GEOMETRY.model_copy(update={"frame": frame})

    _, result = _clean_and_map(_sky_scan(geometry, par_angle=0.6), geometry)

    _assert_sky(result)
