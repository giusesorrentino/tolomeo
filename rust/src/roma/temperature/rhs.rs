use rayon::prelude::*;
use rustfft::num_complex::Complex;

use crate::roma::config::MapmakingConfig;
use crate::roma::fft::FFTPlans;
use crate::roma::operators::noise_filter::InverseNoiseFilter;
use crate::roma::operators::pointing::PointingOperator;

pub fn build_rhs(
    tod: &[f64],
    pixel_ids: &[usize],
    inv_noiseps: &[f64],
    det_pairs: &[(usize, usize)],
    config: &MapmakingConfig,
    plans: &FFTPlans,
) -> Vec<f64> {
    let pointing = PointingOperator { pixel_ids, config };

    (0..config.n_blocks)
        .into_par_iter()
        .fold(
            || {
                let time_buf = vec![0.0; config.block_len * config.n_det];
                let freq_buf = vec![Complex::new(0.0, 0.0); config.n_freqs * config.n_det];

                let freq_acc = vec![Complex::new(0.0, 0.0); config.n_freqs * config.n_det];
                let local_whitemap = vec![0.0; config.n_pixels];

                let fft = plans.workspace();

                (time_buf, freq_buf, freq_acc, fft, local_whitemap)
            },
            |(mut time_buf, mut freq_buf, mut freq_acc, mut fft, mut local_whitemap), iblock| {
                time_buf.fill(0.0);

                let core_start = iblock * config.valid_len;
                let core_end = (core_start + config.valid_len).min(config.n_samples);

                let read_start_raw = core_start as isize - config.padding as isize;
                let read_end_raw = read_start_raw + config.block_len as isize;

                let left_pad = (-read_start_raw).max(0) as usize;
                let read_start = read_start_raw.max(0) as usize;
                let read_end = (read_end_raw as usize).min(config.n_samples);

                // extracts the raw TOD chunk for each detector and copies it into `time_buf` at
                // offset `left_pad`. reorganizes memory from time-interleaved layout [sample * n_det + det]
                // to detector-contiguous chunks [det * block_len + offset].
                for idet in 0..config.n_det {
                    for j in 0..(read_end - read_start) {
                        time_buf[idet * config.block_len + left_pad + j] =
                            tod[(read_start + j) * config.n_det + idet];
                    }
                }

                let mut noise_filter = InverseNoiseFilter {
                    inv_noiseps,
                    det_pairs,
                    config,
                    fft: &mut fft,
                };

                noise_filter.filter_chunk(&mut time_buf, &mut freq_buf, &mut freq_acc);

                pointing.chunk_to_map(&time_buf, &mut local_whitemap, core_start, core_end);

                (time_buf, freq_buf, freq_acc, fft, local_whitemap)
            },
        )
        .map(|(_, _, _, _, local_whitemap)| local_whitemap)
        .reduce(
            || vec![0.0; config.n_pixels],
            |mut acc, thread_map| {
                for (a, &t) in acc.iter_mut().zip(thread_map.iter()) {
                    *a += t;
                }
                acc
            },
        )
}
