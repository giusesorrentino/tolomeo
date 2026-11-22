import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from tests.common import (
    DEC0,
    DETECTOR_IDS,
    MJD_START,
    PIXEL_SIZE,
    RA0,
    make_scan,
    make_two_passes,
    write_skydip,
)
from tolomeo._cli.config import (
    CalibrationConfig,
    GLSConfig,
    NoiseConfig,
    OutputConfig,
    RunConfig,
)
from tolomeo._cli.runner import RunMapmaker
from tolomeo.map import MapGeometry, MapResult
from tolomeo.mapmaking import estimate_inverse_noise
from tolomeo.processing import FilteringConfig, RadiusMask

GEOMETRY = MapGeometry(nx=36, ny=30, pixel_size=PIXEL_SIZE, center=(RA0, DEC0))


def _config(tmp_path, **kwargs):
    """A RunConfig of a synthetic scan, with the given fields replaced."""
    fields = {
        "scan": make_scan(),
        "output": OutputConfig(path=tmp_path / "out" / "map.fits"),
        "geometry": GEOMETRY,
        "noise": NoiseConfig(nperseg=128, leave_one_out=False),  # a single scan
        # loose: on this short scan the CG does not get much below 1e-2
        "gls": GLSConfig(tol=5e-2, workers=1),
    }
    return RunConfig(**(fields | kwargs))


def _skydip_dir(tmp_path):
    """A folder with two skydip files of the detectors of make_scan()."""
    folder = tmp_path / "cal"
    folder.mkdir()
    el = np.linspace(np.deg2rad(20), np.deg2rad(80), 100)
    t_sky = 267.0 * (1 - np.exp(-0.16 / np.sin(el)))
    signal = 5.0 - 2.0 * t_sky[:, None] * np.ones(len(DETECTOR_IDS))
    for name in ["a.fits", "b.fits"]:
        write_skydip(folder / name, el, signal, DETECTOR_IDS)
    return folder


# -----------------------------------------------------------------------------
# RunMapmaker
# -----------------------------------------------------------------------------


def test_run_writes_the_map(tmp_path):
    config = _config(tmp_path, yaml_text="name: test\n")

    result = RunMapmaker(config).run()

    assert isinstance(result, MapResult)
    with fits.open(config.output.path) as hdul:
        assert [hdu.name for hdu in hdul] == ["SCI", "HITS", "WEIGHTS", "CONFIG"]
        np.testing.assert_array_equal(hdul["SCI"].data, result.image)


def test_run_uses_the_solver_of_the_config(tmp_path):
    # Check that gls.solver and gls.residual_norm reach gls_map, and that the solver is
    # written in the FITS map.
    gls = GLSConfig(tol=5e-2, workers=1, solver="minres", residual_norm="m_inverse")
    config = _config(tmp_path, gls=gls)

    result = RunMapmaker(config).run()

    assert result.solver == "minres"
    with fits.open(config.output.path) as hdul:
        assert hdul["SCI"].header["SOLVER"] == "minres"


def test_run_refuses_an_existing_output(tmp_path):
    # Check that it stops before the cleaning, which works in place on the scan.
    config = _config(tmp_path)
    config.output.path.parent.mkdir()
    config.output.path.touch()
    original = config.scan.tod.signal.copy()

    with pytest.raises(FileExistsError, match=r"already exists"):
        RunMapmaker(config).run()

    np.testing.assert_array_equal(config.scan.tod.signal, original)


def test_run_overwrites_when_asked(tmp_path):
    output = OutputConfig(path=tmp_path / "map.fits", overwrite=True)
    output.path.touch()

    RunMapmaker(_config(tmp_path, output=output)).run()

    assert output.path.stat().st_size > 0


def test_header(tmp_path):
    config = _config(
        tmp_path,
        gls=GLSConfig(tol=1e-4, maxiter=50, min_hits_frac=0.2),
        output=OutputConfig(path=tmp_path / "map.fits", header={"NDET": 99}),
    )
    runner = RunMapmaker(config)
    inv_noise = estimate_inverse_noise(config.scan, nperseg=128, workers=1)

    header = runner._header(config.scan, inv_noise)

    assert header["TELESCOP"][0] == "SRT"
    assert header["INSTRUME"][0] == "MISTRAL"
    assert header["DATE-OBS"][0] == Time(MJD_START, format="mjd").isot
    assert header["MJD-OBS"][0] == MJD_START
    assert header["CALIB"][0] is False
    assert header["NPERSEG"][0] == 128  # the value used, not the one of the config
    assert header["GLSTOL"][0] == 1e-4
    assert header["MAXITER"][0] == 50
    assert header["MINHITS"][0] == 0.2
    assert header["NDET"] == 99  # the keywords of output.header come last
    assert "TATM" not in header


def test_header_with_calibration(tmp_path):
    calibration = CalibrationConfig(skydip_dir=tmp_path, T_atm=270, tau_atm=0.1)
    config = _config(tmp_path, calibration=calibration)
    inv_noise = estimate_inverse_noise(config.scan, nperseg=128, workers=1)

    header = RunMapmaker(config)._header(config.scan, inv_noise)

    assert header["CALIB"][0] is True
    assert header["TATM"][0] == 270.0
    assert header["TAUATM"][0] == 0.1
    assert header["OPACORR"][0] is True
    assert "MINHITS" not in header


def test_calibrate_without_calibration_returns_the_scan(tmp_path):
    config = _config(tmp_path)

    assert RunMapmaker(config)._calibrate(config.scan) is config.scan


def test_calibrate_with_skydips(tmp_path):
    # The skydips give a responsivity of -2 per kelvin for every detector.
    calibration = CalibrationConfig(
        skydip_dir=_skydip_dir(tmp_path), correct_opacity=False
    )
    config = _config(tmp_path, calibration=calibration)

    scan = RunMapmaker(config)._calibrate(config.scan)

    np.testing.assert_allclose(scan.tod.signal, config.scan.tod.signal / -2.0)
    assert scan.tod.unit == "K"


def test_estimate_noise_uses_the_mask_of_filtering(tmp_path):
    # Without noise.mask, the mask of the filtering keeps the source out of the
    # spectra too.
    mask = RadiusMask(radius=0.5)
    config = _config(tmp_path, filtering=FilteringConfig(mask=mask))

    inv_noise = RunMapmaker(config)._estimate_noise(config.scan)

    expected = estimate_inverse_noise(
        config.scan, nperseg=128, mask=mask, geometry=GEOMETRY, workers=1
    )
    np.testing.assert_allclose(inv_noise.inv_noiseps, expected.inv_noiseps)


def test_estimate_noise_leave_one_out_without_the_mask_of_filtering(tmp_path):
    # Check that with noise.leave_one_out the mask of the filtering is not passed:
    # estimate_inverse_noise would refuse it.
    scan = make_two_passes()
    calibrated = scan.model_copy(
        update={"tod": scan.tod.model_copy(update={"unit": "K"})}
    )
    config = _config(
        tmp_path,
        scan=calibrated,
        filtering=FilteringConfig(mask=RadiusMask(radius=0.5)),
        noise=NoiseConfig(nperseg=128, leave_one_out=True),
    )

    inv_noise = RunMapmaker(config)._estimate_noise(config.scan)

    expected = estimate_inverse_noise(
        config.scan, nperseg=128, geometry=GEOMETRY, leave_one_out=True, workers=1
    )
    np.testing.assert_allclose(inv_noise.inv_noiseps, expected.inv_noiseps)


def test_estimate_noise_single_scan_without_a_mask(tmp_path):
    # Check that with one scan the leave-one-out gives way, with a warning, to the
    # spectra of all the good samples: neither it nor the mask of the filtering.
    config = _config(
        tmp_path,
        filtering=FilteringConfig(mask=RadiusMask(radius=0.5)),
        noise=NoiseConfig(nperseg=128, leave_one_out=True),
    )

    with pytest.warns(UserWarning, match="estimated without a mask"):
        inv_noise = RunMapmaker(config)._estimate_noise(config.scan)

    expected = estimate_inverse_noise(config.scan, nperseg=128, workers=1)
    np.testing.assert_allclose(inv_noise.inv_noiseps, expected.inv_noiseps)
