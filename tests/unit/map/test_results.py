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


@pytest.mark.parametrize("name", ["image", "hits", "weights"])
def test_map_result_wrong_shape_raises(name):
    # Check that the maps must have the shape (ny, nx) of the geometry, not (nx, ny).
    wrong = np.zeros(SHAPE[::-1], dtype=np.int64 if name == "hits" else float)

    with pytest.raises(ValueError, match=rf"`{name}` must have shape"):
        _map_result(**{name: wrong})


def test_map_result_is_frozen():
    with pytest.raises(ValidationError):
        _map_result().n_iter = 5


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
