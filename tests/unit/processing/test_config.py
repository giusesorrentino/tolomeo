from importlib.resources import files

import astropy.units as u
import numpy as np
import pytest
from pydantic import ValidationError

from tolomeo.processing import (
    Bandpass,
    Baseline,
    CommonMode,
    FilteringConfig,
    Highpass,
    Lowpass,
    RemoveModes,
    RemoveSpline,
    Window,
)
from tolomeo.processing.masks import EdgeMask, RadiusMask, SigmaClipMask


def _valid_config_dict():
    """Return a valid filtering configuration with every kind of step."""
    return {
        "mask": {"type": "radius", "radius": 3.5, "center": "auto"},
        "steps": [
            {"type": "baseline", "deg": 2},
            {"type": "common_mode", "method": "median"},
            {"type": "remove_spline", "knot_spacing": "30 s"},
            {"type": "window", "name": "tukey", "kwargs": {"alpha": 0.1}},
            {"type": "highpass", "fc": "50 mHz"},
            {"type": "lowpass", "fc": 30},
            {"type": "bandpass", "f_lower": 0.1, "f_upper": 20, "order": 2},
            {"type": "remove_modes", "modes_to_remove": 1},
        ],
    }


STEP_CLASSES = [
    Baseline,
    CommonMode,
    RemoveSpline,
    Window,
    Highpass,
    Lowpass,
    Bandpass,
    RemoveModes,
]


# -----------------------------------------------------------------------------
# steps
# -----------------------------------------------------------------------------


def test_step_defaults():
    assert Baseline().deg == 1
    assert CommonMode().method == "corr"
    assert Lowpass(fc=1.0).order == 4


@pytest.mark.parametrize(
    "fc, expected", [(0.05, 0.05), ("50 mHz", 0.05), (50 * u.mHz, 0.05)]
)
def test_filter_cutoff_in_hz(fc, expected):
    assert np.isclose(Highpass(fc=fc).fc, expected)


def test_spline_knot_spacing_in_days():
    assert np.isclose(RemoveSpline(knot_spacing="43200 s").knot_spacing, 0.5)


@pytest.mark.parametrize("f_lower, f_upper", [(10.0, 10.0), (20.0, 5.0)])
def test_bandpass_empty_band_raises(f_lower, f_upper):
    with pytest.raises(ValidationError):
        Bandpass(f_lower=f_lower, f_upper=f_upper)


@pytest.mark.parametrize(
    "step, kwargs",
    [
        (Baseline, {"deg": -1}),
        (Lowpass, {"fc": 0.0}),
        (Lowpass, {"fc": 1.0, "order": 0}),
        (RemoveModes, {"modes_to_remove": -1}),
        (CommonMode, {"method": "mode"}),
    ],
)
def test_step_invalid_value_raises(step, kwargs):
    with pytest.raises(ValidationError):
        step(**kwargs)


def test_step_unknown_key_raises():
    with pytest.raises(ValidationError):
        Baseline(degree=2)


def test_step_is_frozen():
    with pytest.raises(ValidationError):
        Baseline().deg = 3


# -----------------------------------------------------------------------------
# FilteringConfig
# -----------------------------------------------------------------------------


def test_filtering_config_default_steps():
    config = FilteringConfig()

    assert [type(step) for step in config.steps] == [Baseline, CommonMode]
    assert config.mask is None


def test_filtering_config_from_dict():
    # Check that every entry of `steps` becomes the class named by its `type`.
    config = FilteringConfig.from_dict(_valid_config_dict())

    assert [type(step) for step in config.steps] == STEP_CLASSES
    assert isinstance(config.mask, RadiusMask)
    assert np.isclose(config.steps[4].fc, 0.05)


@pytest.mark.parametrize(
    "mask, cls",
    [
        ({"type": "radius", "radius": 2}, RadiusMask),
        ({"type": "sigma_clip", "sigma": 5}, SigmaClipMask),
        ({"type": "edges", "fraction": 0.2}, EdgeMask),
    ],
)
def test_filtering_config_mask_types(mask, cls):
    config = FilteringConfig.from_dict({"mask": mask})

    assert isinstance(config.mask, cls)


def test_filtering_config_unknown_step_raises():
    with pytest.raises(ValidationError):
        FilteringConfig.from_dict({"steps": [{"type": "deglitch"}]})


def test_filtering_config_from_yaml(tmp_path):
    path = tmp_path / "filtering.yaml"
    path.write_text(
        "steps:\n  - type: baseline\n    deg: 3\n  - type: highpass\n    fc: 1\n"
    )

    config = FilteringConfig.from_yaml(path)

    assert config.steps == [Baseline(deg=3), Highpass(fc=1.0)]


def test_filtering_config_empty_yaml_is_the_default(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")

    assert FilteringConfig.from_yaml(path) == FilteringConfig()


def test_filtering_config_packaged_yaml():
    # Check that the YAML shipped with the package is a valid configuration.
    path = files("tolomeo.processing") / "resources" / "filtering.yaml"

    config = FilteringConfig.from_yaml(path)

    assert config.steps == [CommonMode(method="median"), Baseline(deg=1)]
