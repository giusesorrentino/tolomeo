use super::flags::{count_flagged, redirect_flagged_samples};
use super::normal_operator::TemperatureNormalOperator;
use super::preconditioner::build_diagonal_preconditioner;
use super::rhs::build_rhs;
use crate::pyprint;
use crate::roma::config::MapmakingConfig;
use crate::roma::fft::FFTPlans;
use crate::roma::solver::conjugate_gradient::{conjugate_gradient, ResidualNorm};
use std::borrow::Cow;
use std::time::Instant;

pub struct TemperatureMapmakingResult {
    pub map: Vec<f64>,
    pub iterations: usize,
    pub residual: f64,
}

/// Temperature mapmaker (ROMA's iGLS solved with a preconditioned CG).
///
/// `config.n_pixels` is the number of pixels of the *sky map*. If `valid` is given
/// (same `[sample * n_det + det]` layout as `tod`; `true` = use the sample,
/// `false` = flagged), every flagged sample is moved to one extra virtual pixel
/// appended after the map pixels, exactly as ROMA does (`pixel_virt`). The virtual
/// pixel is solved for together with the map, but it is not part of the returned map,
/// which always has `config.n_pixels` elements.
#[allow(clippy::too_many_arguments)]
pub fn tod2map(
    tod: &[f64],
    pixel_ids: &[usize],
    valid: Option<&[bool]>,
    inv_noiseps: &[f64],
    det_pairs: &[(usize, usize)],
    lag_zero: &[f64],
    config: &MapmakingConfig,
    initial_map: Option<&[f64]>,
    residual_norm: ResidualNorm,
    tol: f64,
    maxiter: usize,
) -> TemperatureMapmakingResult {
    let start_total = Instant::now();

    let n_map_pixels = config.n_pixels;

    // flagged samples -> virtual pixel (index `n_map_pixels`); the solver then works on
    // `n_map_pixels + 1` unknowns, the last one being the virtual pixel.
    let (pixel_ids, solver_config): (Cow<[usize]>, MapmakingConfig) = match valid {
        Some(valid) => {
            let n_flagged = count_flagged(valid);
            pyprint!(
                "[MAPMAKER] {} of {} samples are flagged ({:.1}%): they are assigned to a virtual pixel",
                n_flagged,
                valid.len(),
                100.0 * n_flagged as f64 / valid.len().max(1) as f64
            );

            (
                Cow::Owned(redirect_flagged_samples(pixel_ids, valid, n_map_pixels)), // devo capire bene Cow
                MapmakingConfig {
                    n_pixels: n_map_pixels + 1,
                    ..config.clone()
                },
            )
        }
        None => (Cow::Borrowed(pixel_ids), config.clone()),
    };
    let pixel_ids: &[usize] = &pixel_ids;
    let config = &solver_config;

    pyprint!("[MAPMAKER] Starting Temperature Mapmaking pipeline");
    pyprint!(
        "[MAPMAKER] Config: TOD size = {}, pixels = {}, detectors = {}",
        config.n_samples,
        n_map_pixels,
        config.n_det
    );

    let t0 = Instant::now();
    let plans = FFTPlans::new(config.block_len);
    let rhs = build_rhs(tod, pixel_ids, inv_noiseps, det_pairs, config, &plans);
    pyprint!(
        "[MAPMAKER] RHS vector constructed in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    let t1 = Instant::now();
    let diag = build_diagonal_preconditioner(pixel_ids, lag_zero, config);
    pyprint!(
        "[MAPMAKER] Diagonal preconditioner built in {:.2}s",
        t1.elapsed().as_secs_f64()
    );

    let mut map = vec![0.0; config.n_pixels];
    if let Some(initial_map) = initial_map {
        assert_eq!(
            initial_map.len(),
            n_map_pixels,
            "`initial_map` must have one value per map pixel"
        );
        map[..n_map_pixels].copy_from_slice(initial_map);
    }

    let mut normal_operator =
        TemperatureNormalOperator::new(&diag, config, pixel_ids, inv_noiseps, det_pairs);

    pyprint!(
        "[MAPMAKER] Launching CG solver (tol = {:.1e}, maxiter = {})",
        tol,
        maxiter
    );

    let (iterations, residual) =
        conjugate_gradient(&rhs, &mut map, &mut normal_operator, residual_norm, tol, maxiter);

    // the virtual pixel is an unknown of the system, not part of the sky map
    map.truncate(n_map_pixels);

    let total_time = start_total.elapsed().as_secs_f64();
    pyprint!(
        "[MAPMAKER] Pipeline completed in {:.2}s (steps: {}, final residual: {:.3e})",
        total_time,
        iterations,
        residual
    );

    TemperatureMapmakingResult {
        map,
        iterations,
        residual,
    }
}
