# End-to-end tests for the command line.
#
# These tests simulate a realistic use case: synthetic MISTRAL files (subscans,
# beammap and skydips) and a YAML configuration are written to a folder, the command
# `tolomeo` is run on them, and the test verifies that the FITS map is written, with
# the sky in kelvin and the keywords of the run, and the image if it is asked for.

import shutil

import numpy as np
import pytest
from astropy.io import fits

from tests.common import (
    DEC0,
    DETECTOR_IDS,
    RA0,
    make_scan,
    pixelized_source,
    with_signal,
    write_run,
    write_skydip,
)
from tolomeo._cli import main
from tolomeo.map import MapGeometry

AMPLITUDE = 5.0  # K
RESPONSIVITY = -2.0  # raw units per kelvin
T_ATM, TAU_ATM = 267.0, 0.16
GEOMETRY = MapGeometry(nx=36, ny=30, pixel_size=10.0, center=(RA0, DEC0))

RUN_YAML = f"""\
scan:
  tod_dirs:
    - path: tod
  beammap: beammap.dat
geometry:
  nx: {GEOMETRY.nx}
  ny: {GEOMETRY.ny}
  pixel_size: {GEOMETRY.pixel_size} arcsec
  center: [{np.rad2deg(RA0)}, {np.rad2deg(DEC0)}]
calibration:
  skydip_dir: cal
  T_atm: {T_ATM}
  tau_atm: {TAU_ATM}
  correct_opacity: false
filtering:
  mask:
    type: radius
    radius: 1.0
  steps:
    - type: baseline
    - type: common_mode
noise:
  nperseg: 128
  leave_one_out: false  # a single scan
mapmaking:
  workers: 1
output:
  path: out/map.fits
  header:
    OBJECT: synthetic
"""


def _write_observation(folder, text):
    """
    Write the subscans of a raster over a pixelized source, in raw units, the
    skydips and the YAML ``text``; return the sky in kelvin and the YAML path.
    """
    scan = make_scan(noise_rms=0.05)
    sky, signal = pixelized_source(scan, GEOMETRY, AMPLITUDE)
    scan = with_signal(scan, RESPONSIVITY * (signal + scan.tod.signal))

    cal_dir = folder / "cal"
    cal_dir.mkdir(parents=True)
    el = np.linspace(np.deg2rad(20), np.deg2rad(80), 200)
    t_sky = T_ATM * (1 - np.exp(-TAU_ATM / np.sin(el)))
    skydip = 3.0 + RESPONSIVITY * t_sky[:, None] * np.ones(len(DETECTOR_IDS))
    write_skydip(cal_dir / "skydip.fits", el, skydip, DETECTOR_IDS)

    return sky, write_run(folder, text, scan)


def test_full_run(tmp_path):
    # Verify that the command makes the map of the sky, in kelvin, and writes it with
    # the keywords of the run and the YAML file.
    sky, config = _write_observation(tmp_path, RUN_YAML)

    code = main([str(config)])

    assert code == 0
    with fits.open(tmp_path / "out" / "map.fits") as hdul:
        assert [hdu.name for hdu in hdul] == ["SCI", "HITS", "WEIGHTS", "CONFIG"]

        header = hdul["SCI"].header
        assert header["BUNIT"] == "K"
        assert header["CONVERGD"]
        assert header["SOLVER"] == "cg"  # the default
        assert header["CALIB"]
        assert header["NPERSEG"] == 128
        assert header["GLSTOL"] == 1e-3
        assert header["NDET"] == len(DETECTOR_IDS)
        assert header["OBJECT"] == "synthetic"
        assert header["TELESCOP"] == "SRT"
        assert list(hdul["CONFIG"].data["LINE"]) == RUN_YAML.splitlines()

        image, hits = hdul["SCI"].data, hdul["HITS"].data
        observed = hits >= 5
        assert np.abs(image[observed] - sky[observed]).max() < 0.02 * AMPLITUDE


@pytest.mark.skipif(shutil.which("latex") is None, reason="the plot style needs LaTeX")
def test_full_run_with_plot(tmp_path):
    # Verify that the image of the map is written when `output.plot` is set.
    text = RUN_YAML.replace(
        "  path: out/map.fits\n", "  path: out/map.fits\n  plot: out/map.png\n"
    )
    _, config = _write_observation(tmp_path, text)

    code = main([str(config)])

    assert code == 0
    assert (tmp_path / "out" / "map.png").stat().st_size > 0
