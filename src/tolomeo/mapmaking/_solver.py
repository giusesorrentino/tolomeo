"""
Preconditioned conjugate gradient (CG).

It is Fletcher-Reeves CG on ``A x = b`` with a preconditioner ``M^-1``. The iterations are
the same as in the Rust backend, so that a difference between the two means a bug and not
a different formulation.

The residual that stops the loop depends on ``residual_norm``:

* ``"plain"``          ``||g|| / ||b||``        (ROMA's ``itol = 1``)
* ``"preconditioned"`` ``||M^-1 g|| / ||M^-1 b||`` (ROMA's ``itol = 2``)
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .._typing import ResidualNorm

# The true residual ``A x - b`` is recomputed from scratch every 30 iterations: the
# recurrence ``g += tau * h`` used in between slowly drifts away from it, because the
# rounding errors accumulate.
_REPLACE_RESIDUAL_EVERY = 30


class CGOperator(Protocol):
    """What the CG needs from the problem."""

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
    after ``maxiter`` iterations.

    Return ``(iterations, residual)``: the number of completed iterations and the
    relative residual reached. It is ``0`` iterations if ``b`` is zero or if the
    starting point already satisfies ``tol``.

    Names: ``g`` is the residual, ``h`` the preconditioned residual (inside the loop it
    also holds ``A d``), ``d`` the search direction, ``d0`` and ``d1`` the scalar
    ``g . h`` of the previous and of the current iteration.
    """
    start_time = time.perf_counter()
    plain = residual_norm == "plain"

    def print_total_time() -> None:
        """Print the time since the start of the CG, as the Rust backend does."""
        print(f"Total time was {time.perf_counter() - start_time:.5e} seconds.")

    # last completed iteration and its residual, returned if the loop ends early
    n_iter, res = 0, 0.0

    # g = A x - b is the residual; h = M^-1 g is the preconditioned residual
    g = operator.project(x) - b
    h = operator.precondition(g)

    # the residual is measured relative to the size of b (preconditioned or not,
    # depending on residual_norm). `@` on two 1-D arrays is the dot product
    if plain:
        ref_norm, this_norm = np.sqrt(b @ b), np.sqrt(g @ g)
    else:
        precond_b = operator.precondition(b)
        ref_norm, this_norm = np.sqrt(precond_b @ precond_b), np.sqrt(h @ h)

    # b = 0 means x = 0 is the solution, and the relative residual would divide by zero
    if ref_norm == 0.0:
        print("Conjugate Gradient: RHS vector `b` is zero. Solved in 0 iterations.")
        print_total_time()
        return 0, 0.0

    # the starting point may already be good enough
    err = this_norm / ref_norm
    if err <= tol:
        print(
            f"Conjugate Gradient: Initial guess satisfies tolerance "
            f"({err:.3e} <= {tol:.3e})."
        )
        print_total_time()
        return 0, float(err)

    # first search direction: steepest descent on the preconditioned residual
    d = -h
    d0 = g @ h

    for it in range(1, maxiter + 1):
        # h = A d, and d2 = d . A d is the curvature of the problem along d
        h = operator.project(d)
        d2 = d @ h
        # NOTE: after this `break` the code below prints the "Reached max iterations"
        # message, which is not exactly what happened; kept as in the Rust backend
        if d2 == 0.0:
            print(
                f"Conjugate Gradient: Zero curvature encountered at iteration {it}. "
                "Stopping."
            )
            break

        # step length along d; then move the solution and update the residual
        tau = d0 / d2

        x += tau * d
        g += tau * h

        # the recurrence for g drifts away from the true residual: recompute it
        if it % _REPLACE_RESIDUAL_EVERY == 0:
            g = operator.project(x) - b

        # preconditioned residual, and the relative residual that is compared with tol
        h = operator.precondition(g)

        this_norm = np.sqrt(g @ g) if plain else np.sqrt(h @ h)
        n_iter = it
        res = this_norm / ref_norm

        # one line per iteration, with the same text as the Rust backend so that the
        # two logs can be compared
        print(
            f"[CG {100 * it / maxiter:3.0f}%] step {it:4d}/{maxiter} "
            f"▸ residual = {res:.3e}"
        )

        if res <= tol:
            print(
                f"Conjugate Gradient: Converged in {n_iter} iterations "
                f"(res = {res:.3e} <= {tol:.3e})."
            )
            print_total_time()
            return n_iter, float(res)

        # new search direction: the new preconditioned residual plus a fraction beta of
        # the previous direction (Fletcher-Reeves)
        d1 = g @ h
        beta = d1 / d0
        d0 = d1

        d = beta * d - h

    print(
        f"Conjugate Gradient: Reached max iterations ({maxiter}) without full "
        f"convergence. Final residual: {res:.3e}"
    )
    print_total_time()
    return n_iter, float(res)
