import numpy as np
import pytest

from tolomeo.mapmaking._solver import _conjugate_gradient

rng = np.random.default_rng(19)

N = 20


class _MatrixOperator:
    """``A`` given as a dense matrix, with the Jacobi preconditioner."""

    def __init__(self, matrix, precondition=True):
        self.matrix = matrix
        diag = np.diag(matrix)
        self.inv_diag = 1 / diag if precondition else np.ones_like(diag)

    def project(self, x):
        return self.matrix @ x

    def precondition(self, r):
        return r * self.inv_diag


def _spd_matrix(n=N, condition=100.0):
    """Random symmetric positive definite matrix with the given condition number."""
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q @ np.diag(np.geomspace(1, condition, n)) @ q.T


# -----------------------------------------------------------------------------
# _conjugate_gradient
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("residual_norm", ["plain", "preconditioned"])
@pytest.mark.parametrize("precondition", [True, False], ids=["jacobi", "none"])
def test_conjugate_gradient_solves_spd_system(residual_norm, precondition):
    matrix = _spd_matrix()
    b = rng.standard_normal(N)
    x = np.zeros(N)

    n_iter, residual = _conjugate_gradient(
        b,
        x,
        _MatrixOperator(matrix, precondition),
        residual_norm,
        tol=1e-10,
        maxiter=200,
    )

    assert 0 < n_iter <= 200
    assert residual <= 1e-10
    np.testing.assert_allclose(x, np.linalg.solve(matrix, b), rtol=1e-6, atol=1e-8)


def test_conjugate_gradient_diagonal_system_in_one_iteration():
    # With a diagonal A the Jacobi preconditioner is its inverse: one iteration.
    matrix = np.diag(rng.uniform(1, 10, N))
    b = rng.standard_normal(N)
    x = np.zeros(N)

    n_iter, _ = _conjugate_gradient(
        b, x, _MatrixOperator(matrix), "plain", tol=1e-12, maxiter=50
    )

    assert n_iter == 1
    np.testing.assert_allclose(x, b / np.diag(matrix))


def test_conjugate_gradient_zero_rhs():
    x = np.full(N, 3.0)

    result = _conjugate_gradient(
        np.zeros(N), x, _MatrixOperator(_spd_matrix()), "plain", tol=1e-6, maxiter=10
    )

    assert result == (0, 0.0)
    np.testing.assert_array_equal(x, 3.0)  # x is not touched


def test_conjugate_gradient_initial_guess_satisfies_tol():
    matrix = _spd_matrix()
    solution = rng.standard_normal(N)
    x = solution.copy()

    n_iter, residual = _conjugate_gradient(
        matrix @ solution, x, _MatrixOperator(matrix), "plain", tol=1e-6, maxiter=10
    )

    assert n_iter == 0
    assert residual <= 1e-6


def test_conjugate_gradient_stops_at_maxiter():
    matrix = _spd_matrix(condition=1e6)
    b = rng.standard_normal(N)

    n_iter, residual = _conjugate_gradient(
        b, np.zeros(N), _MatrixOperator(matrix, False), "plain", tol=1e-12, maxiter=3
    )

    assert n_iter == 3
    assert residual > 1e-12


def test_conjugate_gradient_residual_is_relative():
    # Check the returned residual against |A x - b| / |b| (plain norm).
    matrix = _spd_matrix()
    b = rng.standard_normal(N)
    x = np.zeros(N)

    _, residual = _conjugate_gradient(
        b, x, _MatrixOperator(matrix), "plain", tol=1e-30, maxiter=5
    )

    expected = np.linalg.norm(matrix @ x - b) / np.linalg.norm(b)
    assert np.isclose(residual, expected, rtol=1e-6)
