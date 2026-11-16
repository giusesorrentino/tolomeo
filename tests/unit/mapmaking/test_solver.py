import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator, minres

from tolomeo.mapmaking._solver import _conjugate_gradient, _minimal_residual

rng = np.random.default_rng(19)

N = 20
NORMS = ["plain", "preconditioned", "m_inverse"]

# the tests of both solvers take the solver as the argument `solve`
both_solvers = pytest.mark.parametrize(
    "solve", [_conjugate_gradient, _minimal_residual], ids=["cg", "minres"]
)


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


def _relative_residual(operator, b, x, residual_norm):
    """The relative residual of ``x`` in ``residual_norm``, computed from scratch."""
    r = b - operator.project(x)
    match residual_norm:
        case "plain":
            return np.linalg.norm(r) / np.linalg.norm(b)
        case "preconditioned":
            precond_r, precond_b = operator.precondition(r), operator.precondition(b)
            return np.linalg.norm(precond_r) / np.linalg.norm(precond_b)
        case "m_inverse":
            return np.sqrt(
                (r @ operator.precondition(r)) / (b @ operator.precondition(b))
            )


# -----------------------------------------------------------------------------
# _conjugate_gradient and _minimal_residual
# -----------------------------------------------------------------------------


@both_solvers
@pytest.mark.parametrize("residual_norm", NORMS)
@pytest.mark.parametrize("precondition", [True, False], ids=["jacobi", "none"])
def test_solver_solves_spd_system(solve, residual_norm, precondition):
    matrix = _spd_matrix()
    b = rng.standard_normal(N)
    x = np.zeros(N)

    n_iter, residual = solve(
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


@both_solvers
def test_solver_diagonal_system_in_one_iteration(solve):
    # With a diagonal A the Jacobi preconditioner is its inverse: one iteration.
    matrix = np.diag(rng.uniform(1, 10, N))
    b = rng.standard_normal(N)
    x = np.zeros(N)

    n_iter, _ = solve(b, x, _MatrixOperator(matrix), "plain", tol=1e-12, maxiter=50)

    assert n_iter == 1
    np.testing.assert_allclose(x, b / np.diag(matrix))


@both_solvers
@pytest.mark.parametrize("residual_norm", NORMS)
def test_solver_zero_rhs(solve, residual_norm):
    x = np.full(N, 3.0)

    result = solve(
        np.zeros(N),
        x,
        _MatrixOperator(_spd_matrix()),
        residual_norm,
        tol=1e-6,
        maxiter=10,
    )

    assert result == (0, 0.0)
    np.testing.assert_array_equal(x, 3.0)  # x is not touched


@both_solvers
def test_solver_initial_guess_satisfies_tol(solve):
    matrix = _spd_matrix()
    solution = rng.standard_normal(N)
    x = solution.copy()

    n_iter, residual = solve(
        matrix @ solution, x, _MatrixOperator(matrix), "plain", tol=1e-6, maxiter=10
    )

    assert n_iter == 0
    assert residual <= 1e-6
    np.testing.assert_array_equal(x, solution)


@both_solvers
def test_solver_stops_at_maxiter(solve):
    matrix = _spd_matrix(condition=1e6)
    b = rng.standard_normal(N)

    n_iter, residual = solve(
        b, np.zeros(N), _MatrixOperator(matrix, False), "plain", tol=1e-12, maxiter=3
    )

    assert n_iter == 3
    assert residual > 1e-12


@both_solvers
@pytest.mark.parametrize("residual_norm", NORMS)
def test_solver_residual_is_relative(solve, residual_norm):
    # Check the returned residual against the one of the returned x, computed from
    # scratch. MINRES carries the plain and the preconditioned residual by recurrence
    # and gives the m_inverse one from its rotations: after 5 steps of a well
    # conditioned problem both agree with the true residual to rounding.
    matrix = _spd_matrix()
    operator = _MatrixOperator(matrix)
    b = rng.standard_normal(N)
    x = np.zeros(N)

    _, residual = solve(b, x, operator, residual_norm, tol=1e-30, maxiter=5)

    expected = _relative_residual(operator, b, x, residual_norm)
    assert np.isclose(residual, expected, rtol=1e-6)


@both_solvers
def test_solver_unknown_without_weight(solve):
    # An unknown that no sample touches (an empty pixel) has a zero row and column in
    # A, 0 in b and 0 in M^-1: the solvers must ignore it and leave it at its start.
    matrix = _spd_matrix()
    matrix[0, :] = matrix[:, 0] = 0.0
    operator = _MatrixOperator(matrix, False)
    # 1 / diag, and 0 where diag is 0, as NormalOperator does
    diag = np.diag(matrix)
    operator.inv_diag = np.divide(1.0, diag, out=np.zeros(N), where=diag > 0)
    b = rng.standard_normal(N)
    b[0] = 0.0
    x = np.zeros(N)
    x[0] = 7.0

    _, residual = solve(b, x, operator, "plain", tol=1e-10, maxiter=200)

    assert residual <= 1e-10
    assert x[0] == 7.0
    np.testing.assert_allclose(
        x[1:], np.linalg.solve(matrix[1:, 1:], b[1:]), rtol=1e-6, atol=1e-8
    )


@both_solvers
def test_solver_unknown_residual_norm_raises(solve):
    # "minres" was the old name of "m_inverse"
    operator = _MatrixOperator(_spd_matrix())
    b = rng.standard_normal(N)

    with pytest.raises(ValueError, match=r"`residual_norm` must be"):
        solve(b, np.zeros(N), operator, "minres", tol=1e-6, maxiter=10)


# -----------------------------------------------------------------------------
# _minimal_residual
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("n_steps", [1, 2, 5, 10])
def test_minimal_residual_matches_scipy(n_steps):
    # Check the iterate after n_steps against scipy.sparse.linalg.minres, with the same
    # Jacobi preconditioner and the same random start. Both are MINRES, so they must
    # agree to rounding: over 300 random systems like this one the relative distance
    # was at most 6e-12 up to 10 steps. After many more steps both lose the
    # orthogonality of the Lanczos vectors and drift apart.
    matrix = _spd_matrix()
    operator = _MatrixOperator(matrix)
    b = rng.standard_normal(N)
    x0 = rng.standard_normal(N)
    x = x0.copy()

    _minimal_residual(b, x, operator, "m_inverse", tol=0.0, maxiter=n_steps)

    preconditioner = LinearOperator((N, N), matvec=operator.precondition)
    expected, _ = minres(matrix, b, x0=x0, M=preconditioner, rtol=0.0, maxiter=n_steps)
    assert np.linalg.norm(x - expected) <= 1e-9 * np.linalg.norm(expected)


def test_minimal_residual_m_inverse_residual_decreases():
    # MINRES minimises ||r||_M^-1 over a Krylov space that grows at every step, so the
    # m_inverse residual can only go down; it must also be the true one.
    matrix = _spd_matrix(condition=1e4)
    operator = _MatrixOperator(matrix)
    b = rng.standard_normal(N)

    residuals = []
    for n_steps in range(1, N + 1):
        x = np.zeros(N)
        _, residual = _minimal_residual(
            b, x, operator, "m_inverse", tol=0.0, maxiter=n_steps
        )
        residuals.append(residual)
        expected = _relative_residual(operator, b, x, "m_inverse")
        assert np.isclose(residual, expected, rtol=1e-6)

    assert np.all(np.diff(residuals) <= 1e-12 * residuals[0])


def test_minimal_residual_solves_indefinite_system():
    # Unlike the CG, MINRES needs A symmetric but not positive definite. The
    # preconditioner must stay positive definite: here it is the identity.
    q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    eigenvalues = np.geomspace(1, 100, N) * np.where(np.arange(N) % 2, -1.0, 1.0)
    matrix = q @ np.diag(eigenvalues) @ q.T
    b = rng.standard_normal(N)
    x = np.zeros(N)

    _, residual = _minimal_residual(
        b, x, _MatrixOperator(matrix, False), "plain", tol=1e-10, maxiter=200
    )

    assert residual <= 1e-10
    np.testing.assert_allclose(x, np.linalg.solve(matrix, b), rtol=1e-6, atol=1e-8)


def test_minimal_residual_singular_system_stops_early():
    # A = diag(1, 2, 3, 0) and b = (1, 1, 1, 1): the last component of b cannot be
    # reached, so the best relative residual is 1 / ||b|| = 0.5, found in 3 steps. At
    # step 4 T_k is singular and MINRES stops (breakdown or condition number, whichever
    # comes first at this rounding) and returns x and the residual of step 3.
    matrix = np.diag([1.0, 2.0, 3.0, 0.0])
    operator = _MatrixOperator(matrix, False)
    b = np.ones(4)
    x = np.zeros(4)

    n_iter, residual = _minimal_residual(
        b, x, operator, "m_inverse", tol=1e-10, maxiter=50
    )

    x_step3 = np.zeros(4)
    _minimal_residual(b, x_step3, operator, "m_inverse", tol=1e-10, maxiter=3)
    assert n_iter == 3
    assert np.isclose(residual, 0.5)
    np.testing.assert_array_equal(x, x_step3)
    np.testing.assert_allclose(x[:3], [1.0, 1 / 2, 1 / 3])
