//! Flagged samples, handled the same way ROMA does it.
//!
//! In ROMA (`read_pf` in `readers.F90`) every sample whose flag differs from
//! `data_ok` has its pixel replaced by one extra *virtual pixel*,
//! `pixel_virt = n_pixels_map + 1` (`initialisers.F90`). That pixel is a normal
//! unknown of the linear system: flagged samples still enter the noise
//! filter, the right-hand side, the hit count and the diagonal
//! preconditioner, but they can no longer contaminate any pixel of the sky
//! map. The virtual pixel is never written to the output maps.
//!
//! Here the convention is zero-based: map pixels are `0..n_map_pixels` and the
//! virtual pixel is `n_map_pixels`.

use rayon::prelude::*;

/// Returns a copy of `pixel_ids` where every flagged sample (`valid == false`)
/// points to `virtual_pixel`. Both slices use the `[sample * n_det + det]` layout.
pub fn redirect_flagged_samples(
    pixel_ids: &[usize],
    valid: &[bool],
    virtual_pixel: usize,
) -> Vec<usize> {
    assert_eq!(
        pixel_ids.len(),
        valid.len(),
        "`pixel_ids` and `valid` must have the same length."
    );

    pixel_ids
        .par_iter()
        .zip(valid.par_iter())
        .map(|(&pixel, &is_valid)| if is_valid { pixel } else { virtual_pixel })
        .collect()
}

/// Number of flagged samples (`valid == false`).
pub fn count_flagged(valid: &[bool]) -> usize {
    valid.par_iter().filter(|&&is_valid| !is_valid).count()
}

/// Flat index (`sample * n_det + det`) of the first *valid* sample whose pixel
/// lies outside `0..n_pixels`, if any. Flagged samples are ignored, because
/// their pixel is replaced by the virtual pixel anyway.
pub fn first_out_of_range_pixel(
    pixel_ids: &[usize],
    valid: Option<&[bool]>,
    n_pixels: usize,
) -> Option<usize> {
    match valid {
        Some(valid) => pixel_ids
            .par_iter()
            .zip(valid.par_iter())
            .position_first(|(&pixel, &is_valid)| is_valid && pixel >= n_pixels),
        None => pixel_ids
            .par_iter()
            .position_first(|&pixel| pixel >= n_pixels),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redirects_only_flagged_samples() {
        let pixel_ids = [3, 1, 4, 1, 5, 9];
        let valid = [true, false, true, true, false, true];
        let out = redirect_flagged_samples(&pixel_ids, &valid, 10);
        assert_eq!(out, vec![3, 10, 4, 1, 10, 9]);
        assert_eq!(count_flagged(&valid), 2);
    }

    #[test]
    fn range_check_ignores_flagged_samples() {
        let pixel_ids = [0, 99, 2, 77];
        let valid = [true, false, true, true];
        assert_eq!(
            first_out_of_range_pixel(&pixel_ids, Some(&valid), 10),
            Some(3)
        );
        assert_eq!(first_out_of_range_pixel(&pixel_ids, None, 10), Some(1));
        assert_eq!(first_out_of_range_pixel(&[0, 1, 2], None, 10), None);
    }
}
