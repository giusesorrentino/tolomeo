use crate::roma::config::MapmakingConfig;

pub fn build_diagonal_preconditioner(
    pixel_ids: &[usize],
    lag_zero: &[f64],
    config: &MapmakingConfig,
) -> Vec<f64> {
    // Build the diagonal approximation used by ROMA's make_diag:
    // for each valid sample and detector, add lag_zero[detector] to its pointed pixel.
    // top uses the same flattened [sample, detector] layout as TOD, with zero-based
    // pixel indices into a map of length config.n_pixels.
    // Return the diagonal in the same pixel ordering/length as the solver map.

    let mut diag = vec![0.0; config.n_pixels];

    // sample-major loop: `pixel_ids` is read sequentially (one row of `n_det` pixels per
    // sample). A detector-major loop would jump `n_det` elements at every step and touch
    // the whole array once per detector, which is very slow for large TODs.
    for row in pixel_ids.chunks_exact(config.n_det) {
        for (&pixel, &weight) in row.iter().zip(lag_zero) {
            diag[pixel] += weight;
        }
    }

    diag
}
