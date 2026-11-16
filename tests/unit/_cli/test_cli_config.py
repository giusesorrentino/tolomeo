import numpy as np
import pytest
from pydantic import ValidationError

from tests.common import (
    ARCSEC,
    DEC0,
    RA0,
    RUN_YAML,
    make_scan,
    write_run,
    write_scan_dir,
)
from tolomeo._cli.config import (
    CalibrationConfig,
    GLSConfig,
    NoiseConfig,
    OutputConfig,
    RunConfig,
)
from tolomeo.map import MapGeometry
from tolomeo.processing import Bandpass, Lowpass
from tolomeo.processing.masks import RadiusMask

# -----------------------------------------------------------------------------
# NoiseConfig
# -----------------------------------------------------------------------------


def test_noise_config_defaults():
    config = NoiseConfig()

    assert config.nperseg == 8192
    assert config.filters == []
    assert config.mask is None


@pytest.mark.parametrize("nperseg", ["auto", "AUTO", " Auto "])
def test_noise_config_auto_nperseg(nperseg):
    assert NoiseConfig(nperseg=nperseg).nperseg is None


@pytest.mark.parametrize("nperseg", [255, 0, "many"])
def test_noise_config_invalid_nperseg_raises(nperseg):
    with pytest.raises(ValidationError):
        NoiseConfig(nperseg=nperseg)


def test_noise_config_filters_and_mask():
    config = NoiseConfig(
        filters=[
            {"type": "lowpass", "fc": "30 Hz"},
            {"type": "bandpass", "f_lower": 0.1, "f_upper": 20},
        ],
        mask={"type": "radius", "radius": 2},
    )

    assert [type(f) for f in config.filters] == [Lowpass, Bandpass]
    assert isinstance(config.mask, RadiusMask)


def test_noise_config_unknown_key_raises():
    with pytest.raises(ValidationError):
        NoiseConfig(n_freqs=4097)


# -----------------------------------------------------------------------------
# GLSConfig, CalibrationConfig, OutputConfig
# -----------------------------------------------------------------------------


def test_gls_config_defaults():
    config = GLSConfig()

    assert config.solver == "cg"
    assert config.residual_norm == "plain"
    assert config.tol == 1e-3
    assert config.maxiter == 300
    assert config.min_hits_frac is None
    assert config.workers == -1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tol": 0},
        {"maxiter": 0},
        {"min_hits_frac": 1.0},
        {"residual_norm": "max"},
        {"solver": "gc"},
    ],
    ids=["tol", "maxiter", "min_hits_frac", "residual_norm", "solver"],
)
def test_gls_config_invalid_value_raises(kwargs):
    with pytest.raises(ValidationError):
        GLSConfig(**kwargs)


def test_calibration_config_temperature_in_kelvin(tmp_path):
    config = CalibrationConfig(skydip_dir=tmp_path, T_atm="300 K")

    assert config.T_atm == 300.0
    assert config.tau_atm == 0.16
    assert config.correct_opacity


def test_output_config_invalid_suffix_raises(tmp_path):
    with pytest.raises(ValidationError, match=r"must end with one of"):
        OutputConfig(path=tmp_path / "map.png")


# -----------------------------------------------------------------------------
# RunConfig.from_yaml
# -----------------------------------------------------------------------------


def test_from_yaml_minimal(tmp_path, monkeypatch):
    # Check that the relative paths are relative to the folder of the YAML file, not
    # to the working directory, and that the geometry is fitted to the scan.
    path = write_run(tmp_path)
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)

    config = RunConfig.from_yaml(path)

    scan = make_scan()
    np.testing.assert_array_equal(config.scan.tod.signal, scan.tod.signal)
    assert config.output.path == (tmp_path / "out" / "map.fits").resolve()
    assert config.geometry == MapGeometry.from_scan(config.scan)
    assert config.calibration is None
    assert config.noise.nperseg == 128
    assert config.yaml_text == RUN_YAML


def test_from_yaml_scan_offsets(tmp_path):
    # Check that the offsets of an entry of tod_dirs are applied to the boresight, and
    # that an empty one is zero.
    text = RUN_YAML.replace(
        "  tod_dirs: tod\n",
        "  tod_dirs:\n    - path: tod\n      ra_offset:\n      dec_offset: 0.1 arcmin\n",
    )

    config = RunConfig.from_yaml(write_run(tmp_path, text))

    scan = make_scan()
    np.testing.assert_allclose(config.scan.coordinates.ra, scan.coordinates.ra)
    np.testing.assert_allclose(
        config.scan.coordinates.dec, scan.coordinates.dec + 6 * ARCSEC, atol=1e-10
    )


def test_from_yaml_several_scans(tmp_path):
    text = RUN_YAML.replace("  tod_dirs: tod\n", "  tod_dirs: [tod, second/tod]\n")
    path = write_run(tmp_path, text)
    write_scan_dir(tmp_path / "second", make_scan(seed=1))

    config = RunConfig.from_yaml(path)

    assert config.scan.tod.signal.shape[0] == 2 * make_scan().tod.signal.shape[0]


def test_from_yaml_explicit_geometry(tmp_path):
    # Numbers in the implicit unit and strings with a unit give the same geometry.
    text = RUN_YAML + (
        "geometry:\n"
        "  nx: 40\n"
        "  ny: 30\n"
        "  pixel_size: 0.2 arcmin\n"
        "  projection: tan\n"
        "  center: [52.0, 41.5 deg]\n"
    )

    config = RunConfig.from_yaml(write_run(tmp_path, text))

    geometry = config.geometry
    assert (geometry.nx, geometry.ny) == (40, 30)
    assert np.isclose(geometry.pixel_size, 12.0)
    assert geometry.projection == "tan"
    np.testing.assert_allclose(geometry.center, (RA0, DEC0))


def test_from_yaml_automatic_size_with_center(tmp_path):
    text = (
        RUN_YAML + "geometry:\n  pixel_size: 5\n  center: [52.0, 41.5]\n  margin: 2\n"
    )

    config = RunConfig.from_yaml(write_run(tmp_path, text))

    expected = MapGeometry.from_scan(
        config.scan, pixel_size=5, center=(RA0, DEC0), margin=2
    )
    assert config.geometry == expected


def test_from_yaml_sections(tmp_path):
    text = (
        "scan:\n"
        "  tod_dirs: tod\n"
        "  beammap: beammap.dat\n"
        "calibration:\n"
        "  skydip_dir: cal\n"
        "  T_atm: 270\n"
        "filtering:\n"
        "  steps:\n"
        "    - type: baseline\n"
        "      deg: 2\n"
        "gls:\n"
        "  tol: 1.0e-5\n"
        "  min_hits_frac: 0.2\n"
        "output:\n"
        "  path: map.fits\n"
        "  overwrite: true\n"
        "  header:\n"
        "    OBJECT: 3C84\n"
        "  plot: plots/map.png\n"
    )

    config = RunConfig.from_yaml(write_run(tmp_path, text))

    assert config.calibration.skydip_dir == (tmp_path / "cal").resolve()
    assert config.calibration.T_atm == 270.0
    assert config.filtering.steps[0].deg == 2
    assert config.gls.tol == 1e-5
    assert config.gls.min_hits_frac == 0.2
    assert config.output.overwrite
    assert config.output.header == {"OBJECT": "3C84"}
    assert config.output.plot == (tmp_path / "plots" / "map.png").resolve()


@pytest.mark.parametrize(
    "text, match",
    [
        ("", r"is empty"),
        ("output:\n  path: map.fits\n", r"`scan` is missing or empty"),
        (
            "scan:\n  beammap: beammap.dat\noutput:\n  path: map.fits\n",
            r"`scan.tod_dirs` is missing or empty",
        ),
        (
            "scan:\n  tod_dirs: tod\noutput:\n  path: map.fits\n",
            r"`scan.beammap` is missing or empty",
        ),
        (
            "scan:\n  tod_dirs: [{ra_offset: 1}]\n  beammap: beammap.dat\n",
            r"`scan.tod_dirs\[\].path` is missing or empty",
        ),
        (
            RUN_YAML.replace("  path: out/map.fits\n", "  overwrite: true\n"),
            "output.path",
        ),
        (RUN_YAML + "cleaning:\n  steps: []\n", r"now called `filtering`"),
        (RUN_YAML.replace("nperseg: 128", "n_freqs: 4097"), r"is now `noise.nperseg`"),
        (RUN_YAML.replace("nperseg: 128", "nperseg: 127"), r"must be even"),
        (RUN_YAML + "geometry:\n  nx: 40\n", r"both numbers or both `auto`"),
        (
            RUN_YAML + "geometry:\n  center: [1, 2, 3]\n",
            r"must be `auto` or \[ra, dec\]",
        ),
    ],
    ids=[
        "empty",
        "no_scan",
        "no_tod_dirs",
        "no_beammap",
        "no_tod_path",
        "no_output_path",
        "cleaning",
        "n_freqs",
        "odd_nperseg",
        "only_nx",
        "bad_center",
    ],
)
def test_from_yaml_invalid_file_raises(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        RunConfig.from_yaml(write_run(tmp_path, text))


@pytest.mark.xfail(reason="from_yaml ignores the keys it does not read")
@pytest.mark.parametrize(
    "text",
    [
        RUN_YAML + "filterng:\n  steps: []\n",
        RUN_YAML.replace(
            "  beammap: beammap.dat\n", "  beammap: beammap.dat\n  typo: 1\n"
        ),
    ],
    ids=["section", "scan_key"],
)
def test_from_yaml_unknown_key_raises(tmp_path, text):
    # The docstring of RunConfig: unknown keys are an error, so that a typo does not
    # go unnoticed (here the misspelled section would leave the default filtering).
    with pytest.raises(ValueError):
        RunConfig.from_yaml(write_run(tmp_path, text))


def test_from_yaml_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunConfig.from_yaml(tmp_path / "missing.yaml")


def test_from_yaml_missing_data_raises(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(RUN_YAML)

    with pytest.raises(NotADirectoryError):
        RunConfig.from_yaml(path)
