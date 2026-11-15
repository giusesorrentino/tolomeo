import numpy as np
import pytest

from tolomeo._signal.splines import bspline_basis_from_knots, bspline_knots

T = np.linspace(0.0, 1.0, 500)


# -----------------------------------------------------------------------------
# bspline_knots
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3])
def test_bspline_knots_spacing_and_coverage(order):
    # Check that the knots are equally spaced and that the interior ones cover the
    # data, with order + 1 extra knots on each side.
    knots = bspline_knots(T, spacing=0.1, order=order)

    np.testing.assert_allclose(np.diff(knots), 0.1)
    assert knots[order] <= T.min() + 1e-12
    assert knots[-order - 1] >= T.max() - 1e-12


# -----------------------------------------------------------------------------
# bspline_basis_from_knots
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2, 3])
def test_bspline_basis_shape(order):
    knots = bspline_knots(T, spacing=0.1, order=order)

    basis = bspline_basis_from_knots(T, knots, order)

    assert basis.shape == (len(knots) - order - 1, T.size)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_bspline_basis_partition_of_unity(order):
    # Check that the basis functions sum to one at every sample, as B-splines do.
    knots = bspline_knots(T, spacing=0.1, order=order)

    basis = bspline_basis_from_knots(T, knots, order)

    np.testing.assert_allclose(basis.sum(axis=0), 1.0, atol=1e-12)
    assert (basis >= 0).all()
