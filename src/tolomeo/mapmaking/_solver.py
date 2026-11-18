import logging
import time
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .._typing import ResidualNorm

logger = logging.getLogger(__name__)


class CGOperator(Protocol):
    """What the solvers (CG and MINRES) need from the problem."""

    def project(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """``A x``, for a vector ``x`` of shape ``(n_unknowns,)``."""
        ...

    def precondition(self, r: NDArray[np.float64]) -> NDArray[np.float64]:
        """``M^-1 r``, the preconditioner applied to ``r`` of shape ``(n_unknowns,)``."""
        ...


def _conjugate_gradient(
    b: NDArray[np.float64],
    x: NDArray[np.float64],
    operator: CGOperator,
    residual_norm: ResidualNorm,
    tol: float,
    maxiter: int,
) -> tuple[int, float]:
    """
    Solve ``A x = b``. ``x`` is the starting point and is updated **in place**.

    ``operator`` gives ``A x`` and ``M^-1 r`` (see ``CGOperator``). The loop stops
    when the relative residual (norm chosen with ``residual_norm``) is ``<= tol``, or
    after ``maxiter`` iterations. The norms are those of ``_minimal_residual``;
    ``"m_inverse"`` is ``sqrt(g . h)``, which the loop computes anyway for ``beta``.

    Return ``(iterations, residual)``: the number of completed iterations and the
    relative residual reached. It is ``0`` iterations if ``b`` is zero or if the
    starting point already satisfies ``tol``.

    Names: ``g`` is the residual, ``h`` the preconditioned residual (inside the loop it
    also holds ``A d``), ``d`` the search direction, ``d0`` and ``d1`` the scalar
    ``g . h`` of the previous and of the current iteration.
    """
    start_time = time.perf_counter()

    def log_total_time() -> None:
        """Log the time since the start of the CG, as the Rust backend does."""
        logger.info(f"Total time was {time.perf_counter() - start_time:.5e} seconds.")

    # g = A x - b is the residual; h = M^-1 g is the preconditioned residual
    g = operator.project(x) - b
    h = operator.precondition(g)

    # g . h = r . M^-1 r (g = -r): the squared M^-1 norm of the residual, and the d0 of
    # the first step
    d0 = g @ h

    # the residual is measured relative to the size of b, in the norm chosen with
    # residual_norm. `@` on two 1-D arrays is the dot product; max() because rounding
    # can make r . M^-1 r slightly negative when it is about 0
    match residual_norm:
        case "plain":
            ref_norm, this_norm = np.sqrt(b @ b), np.sqrt(g @ g)
        case "preconditioned":
            precond_b = operator.precondition(b)
            ref_norm, this_norm = np.sqrt(precond_b @ precond_b), np.sqrt(h @ h)
        case "m_inverse":
            ref_norm = np.sqrt(max(b @ operator.precondition(b), 0.0))
            this_norm = np.sqrt(max(d0, 0.0))
        case _:
            raise ValueError(
                "`residual_norm` must be 'plain', 'preconditioned' or 'm_inverse', "
                f"got {residual_norm!r}."
            )

    # b = 0 means x = 0 is the solution, and the relative residual would divide by zero
    if ref_norm == 0.0:
        logger.info(
            "Conjugate Gradient: RHS vector `b` is zero. Solved in 0 iterations."
        )
        log_total_time()
        return 0, 0.0

    # last completed iteration and its residual, returned if the loop ends early: with
    # no step done (maxiter = 0, or zero curvature at the first step) they are those
    # of the starting point
    n_iter, res = 0, float(this_norm / ref_norm)

    # step 0: the residual of the starting point, written like the steps of the loop
    logger.info(f"[CG {0:3.0f}%] step {0:4d}/{maxiter} ▸ residual = {res:.3e}")

    # the starting point may already be good enough
    if res <= tol:
        logger.info(
            f"Conjugate Gradient: Initial guess satisfies tolerance "
            f"({res:.3e} <= {tol:.3e})."
        )
        log_total_time()
        return 0, res

    # first search direction: steepest descent on the preconditioned residual
    d = -h

    for it in range(1, maxiter + 1):
        # h = A d, and d2 = d . A d is the curvature of the problem along d
        h = operator.project(d)
        d2 = d @ h
        # NOTE: after this `break` the code below prints the "Reached max iterations"
        # message, which is not exactly what happened; kept as in the Rust backend
        if d2 == 0.0:
            logger.info(
                f"Conjugate Gradient: Zero curvature encountered at iteration {it}. "
                "Stopping."
            )
            break

        # step length along d; then move the solution and update the residual
        tau = d0 / d2

        x += tau * d
        g += tau * h

        # preconditioned residual, and the relative residual that is compared with tol
        h = operator.precondition(g)

        # g . h is needed for the next direction (beta) and is the squared M^-1 norm of
        # the residual
        d1 = g @ h

        match residual_norm:
            case "plain":
                this_norm = np.sqrt(g @ g)
            case "preconditioned":
                this_norm = np.sqrt(h @ h)
            case _:  # "m_inverse"
                this_norm = np.sqrt(max(d1, 0.0))
        n_iter = it
        res = this_norm / ref_norm

        # one line per iteration, with the same text as the Rust backend so that the
        # two logs can be compared
        logger.info(
            f"[CG {100 * it / maxiter:3.0f}%] step {it:4d}/{maxiter} "
            f"▸ residual = {res:.3e}"
        )

        if res <= tol:
            logger.info(
                f"Conjugate Gradient: Converged in {n_iter} iterations "
                f"(res = {res:.3e} <= {tol:.3e})."
            )
            log_total_time()
            return n_iter, float(res)

        # new search direction: the new preconditioned residual plus a fraction beta of
        # the previous direction (Fletcher-Reeves)
        beta = d1 / d0
        d0 = d1

        d = beta * d - h

    logger.info(
        f"Conjugate Gradient: Reached max iterations ({maxiter}) without full "
        f"convergence. Final residual: {res:.3e}"
    )
    log_total_time()
    return n_iter, float(res)


def _minimal_residual(
    b: NDArray[np.float64],
    x: NDArray[np.float64],
    operator: CGOperator,
    residual_norm: ResidualNorm,
    tol: float,
    maxiter: int,
) -> tuple[int, float]:
    """
    Solve ``A x = b`` by preconditioned MINRES. ``x`` is the starting point and is
    updated **in place**.

    Same arguments and return value as ``_conjugate_gradient``. MINRES (Paige and
    Saunders, 1975) builds the Lanczos basis of the Krylov space in the ``M^-1`` inner
    product and, at every iteration, takes the ``x`` of that space with the smallest
    ``||r||_M^-1 = sqrt(r . M^-1 r)``, where ``r = b - A x``. Translation of ROMA's
    ``Solvers/linCG_MINRES_T.F90`` without the MPI reductions.

    ``residual_norm`` chooses the relative residual compared with ``tol``:
    ``"plain"`` is ``||r||_2 / ||b||_2``, ``"preconditioned"`` is
    ``||M^-1 r||_2 / ||M^-1 b||_2`` and ``"m_inverse"`` is ``||r||_M^-1 / ||b||_M^-1``
    (ROMA's ``itol`` 1, 2 and 3). ``"m_inverse"`` is free and decreases at every
    iteration. For the other two MINRES does not give the residual: it is carried by
    the recurrence ``r_k = r_{k-1} - phi_k A d_k``, with no extra product by ``A``.

    Return ``(iterations, residual)``: the number of completed iterations and the
    relative residual of the returned ``x``. It is ``0`` iterations if ``b`` is zero or
    if the starting point already satisfies ``tol``. The loop also stops before
    ``maxiter``, with ``residual > tol``, if ``T_k`` is singular at double precision
    (breakdown, or estimated condition number ``>= 0.1 / eps``): ``x`` and the residual
    are then those of the previous iteration.

    Names follow the Fortran: ``v`` are the Lanczos vectors and ``z = M^-1 v``,
    ``alpha`` and ``beta`` the diagonal and the off-diagonal of the tridiagonal
    ``T_k``, ``c`` and ``s`` the Givens rotations of its QR factorization, ``eps1``,
    ``eps2`` and ``gamma_k`` the entries of column ``k`` of ``R``, ``d`` the direction
    vectors, ``phi_bar`` the rotated right-hand side (``|phi_bar| = ||r||_M^-1``). The
    suffixes ``_cur``, ``_old`` and ``_oldest`` are the iterations ``k``, ``k - 1`` and
    ``k - 2``.
    """
    start_time = time.perf_counter()
    eps = np.finfo(np.float64).eps

    def log_total_time() -> None:
        """Log the time since the start of MINRES."""
        logger.info(f"Total time was {time.perf_counter() - start_time:.5e} seconds.")

    # initial residual r = b - A x, z = M^-1 r and beta_1 = ||r||_M^-1. max() because
    # rounding can make r . z slightly negative when it is about 0
    r = b - operator.project(x)
    z = operator.precondition(r)
    beta = np.sqrt(max(r @ z, 0.0))

    # norm of b that makes the residual relative, and the residual of the starting point
    match residual_norm:
        case "plain":
            ref_norm, this_norm = np.sqrt(b @ b), np.sqrt(r @ r)
        case "preconditioned":
            precond_b = operator.precondition(b)
            ref_norm, this_norm = np.sqrt(precond_b @ precond_b), np.sqrt(z @ z)
        case "m_inverse":
            ref_norm = np.sqrt(max(b @ operator.precondition(b), 0.0))
            this_norm = beta
        case _:
            raise ValueError(
                "`residual_norm` must be 'plain', 'preconditioned' or 'm_inverse', "
                f"got {residual_norm!r}."
            )

    # b = 0 means x = 0 is the solution, and the relative residual would divide by zero
    if ref_norm == 0.0:
        logger.info("MINRES: RHS vector `b` is zero. Solved in 0 iterations.")
        log_total_time()
        return 0, 0.0

    # last completed iteration and its residual, returned if the loop ends early
    n_iter, res = 0, float(this_norm / ref_norm)

    # step 0: the residual of the starting point, written like the steps of the loop
    logger.info(f"[MINRES {0:3.0f}%] step {0:4d}/{maxiter} ▸ residual = {res:.3e}")

    # the starting point may already be good enough; with beta_1 = 0 (r != 0 is
    # possible only if M^-1 is singular) there is no Krylov space to search
    if res <= tol or beta == 0.0:
        logger.info(
            f"MINRES: Initial guess satisfies tolerance ({res:.3e} <= {tol:.3e})."
            if res <= tol
            else "MINRES: r . M^-1 r = 0 at the starting point. Stopping."
        )
        log_total_time()
        return 0, res

    # with "plain" and "preconditioned" the residual r is updated with A d_k, which
    # follows the same recurrence as d_k: A d_k and A d_{k-1}, 0 at the start
    track_residual = residual_norm != "m_inverse"
    Ad_cur, Ad_old = np.zeros_like(b), np.zeros_like(b)

    # first Lanczos vector, of unit M^-1 norm: v_1 = r / beta_1, z_1 = M^-1 v_1; v_0 = 0
    v = r / beta
    z = z / beta
    v_old = np.zeros_like(b)

    # rotations of the two previous iterations: Q_0 = Q_-1 = identity
    c_old, s_old = 1.0, 0.0
    c_oldest, s_oldest = 1.0, 0.0

    # rotated right-hand side, and the direction vectors d_k, d_{k-1} (d_0 = d_-1 = 0)
    phi_bar = beta
    d_cur, d_old = np.zeros_like(b), np.zeros_like(b)

    # squared Frobenius norm of T_k and the extreme gamma_k, for the two stop tests
    Anorm2 = 0.0
    gmax, gmin = 0.0, np.inf

    for it in range(1, maxiter + 1):
        # Lanczos step: w = A z_k - alpha_k v_k - beta_k v_{k-1} (at it = 1 the last
        # term is 0, since v_0 = 0), then beta_{k+1} = ||w||_M^-1
        Av = operator.project(z)
        alpha = z @ Av
        w = Av - alpha * v - beta * v_old
        z_new = operator.precondition(w)
        beta_new = np.sqrt(max(w @ z_new, 0.0))

        # squared entries of column k of T_k: beta_k above the diagonal (from k = 2:
        # at k = 1 beta is beta_1, the norm of r, not an entry of T_k), alpha_k on the
        # diagonal and beta_{k+1} below it
        if it > 1:
            Anorm2 += beta**2
        Anorm2 += alpha**2 + beta_new**2
        Anorm = np.sqrt(Anorm2)

        # the previous rotations applied to column k: Q_{k-2} gives eps1 (0 at k = 2,
        # where Q_0 is the identity), Q_{k-1} gives eps2 and the diagonal delta_bar.
        # Column 1 has nothing above the diagonal
        if it == 1:
            eps1, eps2, delta_bar = 0.0, 0.0, alpha
        else:
            eps1 = s_oldest * beta
            eps2 = c_old * (c_oldest * beta) + s_old * alpha
            delta_bar = -s_old * (c_oldest * beta) + c_old * alpha

        # the new rotation Q_k zeroes beta_{k+1}; gamma_k is the diagonal entry of R
        gamma_k = np.sqrt(delta_bar**2 + beta_new**2)

        # breakdown: T_k is singular at double precision. gamma_k has the scale of
        # M^-1 A, so it is compared with eps * ||T_k|| and not with eps alone
        if gamma_k <= eps * Anorm:
            logger.info(
                f"MINRES: breakdown at iteration {it}, gamma_k <= eps*||T_k||. "
                "Stopping."
            )
            break

        # gmax / gmin estimates cond(T_k): at 0.1 / eps more iterations cannot improve
        # the solution
        gmax = max(gmax, gamma_k)
        gmin = min(gmin, gamma_k)
        if gmax / gmin >= 0.1 / eps:
            logger.info(
                f"MINRES: estimated cond(T_k) = {gmax / gmin:.3e} exceeds 0.1/eps at "
                f"iteration {it}. Stopping."
            )
            break

        c_cur = delta_bar / gamma_k
        s_cur = beta_new / gamma_k

        # phi_k is the step along d_k; the new |phi_bar| is ||r_k||_M^-1
        phi_k = c_cur * phi_bar
        phi_bar = -s_cur * phi_bar

        # d_k = (z_k - eps1 d_{k-2} - eps2 d_{k-1}) / gamma_k;
        # x_k = x_{k-1} + phi_k d_k
        d_oldest, d_old = d_old, d_cur
        d_cur = (z - eps1 * d_oldest - eps2 * d_old) / gamma_k
        x += phi_k * d_cur

        if track_residual:
            Ad_oldest, Ad_old = Ad_old, Ad_cur
            Ad_cur = (Av - eps1 * Ad_oldest - eps2 * Ad_old) / gamma_k
            r -= phi_k * Ad_cur

        # shift the rotations: k becomes k - 1, and k - 1 becomes k - 2
        c_oldest, s_oldest = c_old, s_old
        c_old, s_old = c_cur, s_cur

        # next Lanczos vectors (with beta_{k+1} = 0 the loop stops below)
        v_old = v
        if beta_new > 0.0:
            v = w / beta_new
            z = z_new / beta_new
        beta = beta_new

        match residual_norm:
            case "plain":
                this_norm = np.sqrt(r @ r)
            case "preconditioned":
                precond_r = operator.precondition(r)
                this_norm = np.sqrt(precond_r @ precond_r)
            case _:  # "m_inverse"
                this_norm = abs(phi_bar)
        n_iter, res = it, float(this_norm / ref_norm)

        # one line per iteration, in the format of the CG
        logger.info(
            f"[MINRES {100 * it / maxiter:3.0f}%] step {it:4d}/{maxiter} "
            f"▸ residual = {res:.3e}"
        )

        if res <= tol:
            logger.info(
                f"MINRES: Converged in {n_iter} iterations "
                f"(res = {res:.3e} <= {tol:.3e})."
            )
            break

        # beta_{k+1} = 0: the Krylov space is invariant and cannot grow
        if beta_new <= 0.0:
            logger.info(f"MINRES: beta = 0 at iteration {it}. Stopping.")
            break
    else:
        logger.info(
            f"MINRES: Reached max iterations ({maxiter}) without full convergence. "
            f"Final residual: {res:.3e}"
        )

    # free diagnostic: how ill-conditioned the preconditioned system is
    if gmax > 0.0:
        logger.info(f"MINRES: estimated cond(T_k) = {gmax / gmin:.3e}")
    log_total_time()

    return n_iter, res
