use super::linalg::dot;
use crate::pyprint;
use crate::roma::solver::LinearOperator;
use rayon::prelude::*;
use std::time::Instant;

/// Norm of the residual used by the stopping test (ROMA's `itol`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualNorm {
    /// ||r|| / ||b|| (ROMA's `itol = 1`)
    Plain,
    /// ||M^-1 r|| / ||M^-1 b|| (ROMA's `itol = 2`)
    Preconditioned,
}

pub fn conjugate_gradient<O: LinearOperator>(
    b: &[f64],
    x: &mut [f64],
    operator: &mut O,
    residual_norm: ResidualNorm,
    tol: f64,
    maxiter: usize,
) -> (usize, f64) {
    let start_time = Instant::now();

    let mut iter: usize = 0;
    let mut res: f64 = 0.0;

    let mut g = vec![0.0; b.len()];
    let mut h = vec![0.0; b.len()];
    let mut d = vec![0.0; b.len()];

    operator.project(x, &mut d);

    g.par_iter_mut()
        .zip(d.par_iter())
        .zip(b.par_iter())
        .for_each(|((ig, id), ib)| *ig = *id - *ib);

    operator.precondition(&g, &mut h);

    let (ref_norm, this_norm) = match residual_norm {
        ResidualNorm::Plain => (dot(b, b).sqrt(), dot(&g, &g).sqrt()),
        ResidualNorm::Preconditioned => {
            operator.precondition(b, &mut d);
            (dot(&d, &d).sqrt(), dot(&h, &h).sqrt())
        }
    };

    if ref_norm == 0.0 {
        let elapsed = start_time.elapsed().as_secs_f64();
        pyprint!("Conjugate Gradient: RHS vector `b` is zero. Solved in 0 iterations."); // o [CG Info] ?
        pyprint!("Total time was {elapsed:.5e} seconds.");
        return (0, 0.0);
    }

    let err = this_norm / ref_norm;

    if err <= tol {
        let elapsed = start_time.elapsed().as_secs_f64();
        pyprint!("Conjugate Gradient: Initial guess satisfies tolerance ({err:.3e} <= {tol:.3e}).");
        pyprint!("Total time was {elapsed:.5e} seconds.");
        return (0, err);
    }

    d.par_iter_mut()
        .zip(h.par_iter())
        .for_each(|(id, ih)| *id = -*ih);

    let mut d0 = dot(&g, &h);

    for it in 1..=maxiter {
        operator.project(&d, &mut h);

        let d2 = dot(&d, &h);
        if d2 == 0.0 {
            pyprint!("Conjugate Gradient: Zero curvature encountered at iteration {it}. Stopping.");
            break;
        }

        let tau = d0 / d2;

        x.par_iter_mut()
            .zip(d.par_iter())
            .zip(g.par_iter_mut())
            .zip(h.par_iter())
            .for_each(|(((xi, &di), gi), &hi)| {
                *xi += tau * di;
                *gi += tau * hi;
            });

        if it % 30 == 0 {
            operator.project(x, &mut h); // h = A * x
            g.par_iter_mut().zip(h.par_iter()).zip(b.par_iter())
                .for_each(|((gi, &hi), &bi)| *gi = hi - bi); // g = Ax - b
        }

        operator.precondition(&g, &mut h);

        let this_norm = match residual_norm {
            ResidualNorm::Plain => dot(&g, &g).sqrt(),
            ResidualNorm::Preconditioned => dot(&h, &h).sqrt(),
        };

        iter = it;
        res = this_norm / ref_norm;

        let pct = (iter as f64 / maxiter as f64) * 100.0;
        pyprint!(
            "[CG {:3.0}%] step {:4}/{} ▸ residual = {:.3e}",
            pct,
            it,
            maxiter,
            res
        );

        if res <= tol {
            let elapsed = start_time.elapsed().as_secs_f64();
            pyprint!("Conjugate Gradient: Converged in {iter} iterations (res = {res:.3e} <= {tol:.3e}).");
            pyprint!("Total time was {elapsed:.5e} seconds.");
            return (iter, res);
        }

        let d1 = dot(&g, &h);
        let beta = d1 / d0;
        d0 = d1;

        d.par_iter_mut()
            .zip(h.par_iter())
            .for_each(|(id, &ih)| *id = beta * *id - ih);
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    pyprint!("Conjugate Gradient: Reached max iterations ({maxiter}) without full convergence. Final residual: {res:.3e}");
    pyprint!("Total time was {elapsed:.5e} seconds.");
    (iter, res)
}
