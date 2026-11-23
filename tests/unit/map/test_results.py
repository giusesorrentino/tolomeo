import numpy as np
import pytest
from pydantic import ValidationError

from tests.common import DEC0, RA0
from tolomeo.map import BinnedMapResult, MapGeometry, MapResult

GEOMETRY = MapGeometry(nx=4, ny=3, pixel_size=10.0, center=(RA0, DEC0))
SHAPE = (3, 4)  # (ny, nx)


def _map_result(**kwargs):
    """A valid MapResult, with the given fields replaced."""
    fields = {
        "image": np.zeros(SHAPE),
        "hits": np.ones(SHAPE, dtype=np.int64),
        "n_iter": 3,
        "residual": 1e-4,
        "geometry": GEOMETRY,
        "solver": "cg",
        "weights": np.ones(SHAPE),
    }
    return MapResult(**(fields | kwargs))


# -----------------------------------------------------------------------------
# MapResult
# -----------------------------------------------------------------------------


def test_map_result_defaults():
    result = _map_result(weights=None)

    assert result.weights is None
    assert result.converged
    assert result.unit is None
    assert result.source_name is None


@pytest.mark.parametrize("name", ["image", "hits", "weights"])
def test_map_result_wrong_shape_raises(name):
    # Check that the maps must have the shape (ny, nx) of the geometry, not (nx, ny).
    wrong = np.zeros(SHAPE[::-1], dtype=np.int64 if name == "hits" else float)

    with pytest.raises(ValueError, match=rf"`{name}` must have shape"):
        _map_result(**{name: wrong})


def test_map_result_without_solver_raises():
    # The solver has no default: a map must say which solver made it.
    with pytest.raises(ValidationError, match=r"solver"):
        MapResult(
            image=np.zeros(SHAPE),
            hits=np.ones(SHAPE, dtype=np.int64),
            n_iter=3,
            residual=1e-4,
            geometry=GEOMETRY,
        )


@pytest.mark.parametrize("solver", ["CG", "gc"])
def test_map_result_invalid_solver_raises(solver):
    with pytest.raises(ValidationError, match=r"solver"):
        _map_result(solver=solver)


def test_map_result_is_frozen():
    with pytest.raises(ValidationError):
        _map_result().n_iter = 5


# -----------------------------------------------------------------------------
# MapResult.to
# -----------------------------------------------------------------------------


def test_map_result_to_mk():
    # Check that the image is multiplied by the factor and the weights, an inverse
    # variance, divided by its square; the original map does not change.
    result = _map_result(
        image=np.full(SHAPE, 2.0), weights=np.full(SHAPE, 4.0), unit="K"
    )

    converted = result.to("mK")

    np.testing.assert_allclose(converted.image, 2000.0)
    np.testing.assert_allclose(converted.weights, 4e-6)
    assert converted.unit == "mK"
    np.testing.assert_array_equal(converted.hits, result.hits)
    np.testing.assert_allclose(result.image, 2.0)
    np.testing.assert_allclose(result.weights, 4.0)
    assert result.unit == "K"


def test_map_result_to_round_trip():
    image = np.arange(12.0).reshape(SHAPE)
    result = _map_result(image=image, unit="K")

    back = result.to("uK").to("K")

    np.testing.assert_allclose(back.image, image)
    np.testing.assert_allclose(back.weights, result.weights)
    assert back.unit == "K"


def test_map_result_to_without_weights():
    assert _map_result(weights=None, unit="K").to("mK").weights is None


@pytest.mark.parametrize(
    ("frequency", "beam_fwhm", "nu", "theta"),
    [(90.0, 12.0, 90e9, 12.0), ("150 GHz", "0.3 arcmin", 150e9, 18.0)],
)
def test_map_result_to_jy_per_beam_rayleigh_jeans(frequency, beam_fwhm, nu, theta):
    # Check the factor against the Rayleigh-Jeans law written by hand,
    # S = 2 k_B nu^2 T Omega / c^2, with the solid angle of a Gaussian beam.
    k_b, c = 1.380649e-23, 299792458.0  # SI, exact
    omega = np.pi * np.deg2rad(theta / 3600) ** 2 / (4 * np.log(2))
    jy_per_kelvin = 2 * k_b * nu**2 * omega / c**2 / 1e-26
    result = _map_result(image=np.ones(SHAPE), unit="K")

    converted = result.to("Jy/beam", frequency=frequency, beam_fwhm=beam_fwhm)

    np.testing.assert_allclose(converted.image, jy_per_kelvin, rtol=1e-12)
    np.testing.assert_allclose(converted.weights, jy_per_kelvin**-2, rtol=1e-12)
    assert converted.unit == "Jy beam-1"


def test_map_result_to_default_beam():
    # 1 K is 0.954 Jy/beam at 90 GHz with a FWHM of 12 arcsec, as the docstring says.
    converted = _map_result(image=np.ones(SHAPE), unit="K").to("Jy/beam")

    np.testing.assert_allclose(converted.image, 0.954, rtol=1e-3)


def test_map_result_from_jy_per_beam():
    # Check that the conversion works in the other direction too.
    result = _map_result(image=np.ones(SHAPE), unit="mJy/beam")

    back = result.to("mK").to("mJy/beam")

    np.testing.assert_allclose(back.image, 1.0)
    assert back.unit == "mJy beam-1"


@pytest.mark.parametrize(
    ("unit", "target", "match"),
    [
        (None, "mK", "not known"),
        ("K", "blah", "did not parse"),
        ("K", "Jy", "not convertible"),  # a flux density of a map is per beam
        ("K", "s", "not convertible"),
        ("rad", "K", "calibrate"),  # a map of a TOD that was not calibrated
    ],
)
def test_map_result_to_invalid_raises(unit, target, match):
    with pytest.raises(ValueError, match=match):
        _map_result(unit=unit).to(target)


@pytest.mark.parametrize("kwargs", [{"frequency": 0.0}, {"beam_fwhm": "-12 arcsec"}])
def test_map_result_to_non_positive_beam_raises(kwargs):
    with pytest.raises(ValueError, match="must be positive"):
        _map_result(unit="K").to("mJy/beam", **kwargs)


# -----------------------------------------------------------------------------
# BinnedMapResult
# -----------------------------------------------------------------------------


def test_binned_map_result_is_frozen():
    result = BinnedMapResult(
        image=np.zeros(SHAPE),
        hits=np.zeros(SHAPE, dtype=np.int64),
        std=np.zeros(SHAPE),
        geometry=GEOMETRY,
    )

    with pytest.raises(ValidationError):
        result.image = np.ones(SHAPE)
