use rayon::prelude::*;
use rustfft::num_complex::Complex;

use crate::roma::config::MapmakingConfig;
use crate::roma::fft::FFTPlans;
use crate::roma::operators::noise_filter::InverseNoiseFilter;
use crate::roma::operators::pointing::PointingOperator;
use crate::roma::solver::LinearOperator;

pub struct TemperatureNormalOperator<'a> {
    pub diag: &'a [f64],
    pub config: &'a MapmakingConfig,
    pub pixel_ids: &'a [usize],
    pub inv_noiseps: &'a [f64],
    pub det_pairs: &'a [(usize, usize)],
    pub plans: FFTPlans,
}

impl<'a> TemperatureNormalOperator<'a> {
    pub fn new(
        diag: &'a [f64],
        config: &'a MapmakingConfig,
        pixel_ids: &'a [usize],
        inv_noiseps: &'a [f64],
        det_pairs: &'a [(usize, usize)],
    ) -> Self {
        let plans = FFTPlans::new(config.block_len);

        Self {
            diag,
            config,
            pixel_ids,
            inv_noiseps,
            det_pairs,
            plans,
        }
    }
}

impl<'a> LinearOperator for TemperatureNormalOperator<'a> {
    fn precondition(&mut self, b: &[f64], x: &mut [f64]) {
        x.par_iter_mut()
            .zip(b.par_iter())
            .zip(self.diag.par_iter())
            .for_each(|((xi, bi), di)| {
                *xi = if *di > 0.0 { bi / di } else { 0.0 };
            });
    }

    /// Computes the linear operator response $y = A \cdot x = (P^T N^{-1} P) \cdot x$.
    ///
    /// Projects the input map (`in_map`) into time-ordered data (TOD), applies frequency-domain
    /// noise whitening ($N^{-1}$), and back-projects the filtered signal into `out_map`.
    ///
    /// # Arguments
    /// * `in_map` - Input sky map vector ($x$).
    /// * `out_map` - Output map vector ($y = A \cdot x$), zeroed and overwritten.
    fn project(&mut self, in_map: &[f64], out_map: &mut [f64]) {
        out_map.fill(0.0);

        let pointing = PointingOperator {
            pixel_ids: self.pixel_ids,
            config: self.config,
        };

        // we process the TOD in chunks ("pieces") because full TODs exceed CPU cache capacity
        // and FFTs introduce circular convolution artifacts at the edges. splitting the TOD
        // into small overlapping chunks (overlap-save) keeps computation fast and mathematically exact.
        let final_map = (0..self.config.n_blocks)
            .into_par_iter()
            .fold(
                || {
                    let time_buf = vec![0.0; self.config.block_len * self.config.n_det];
                    let freq_buf =
                        vec![Complex::new(0.0, 0.0); self.config.n_freqs * self.config.n_det];

                    let freq_acc =
                        vec![Complex::new(0.0, 0.0); self.config.n_freqs * self.config.n_det];
                    let local_map = vec![0.0; out_map.len()];

                    let fft = self.plans.workspace();

                    (time_buf, freq_buf, freq_acc, fft, local_map)
                },
                |(mut time_buf, mut freq_buf, mut freq_acc, mut fft, mut local_map), iblock| {
                    time_buf.fill(0.0);

                    // start and end indices of the valid ("good") chunk
                    let core_start = iblock * self.config.valid_len;
                    let core_end = (core_start + self.config.valid_len).min(self.config.n_samples);

                    let read_start_raw = core_start as isize - self.config.padding as isize;
                    let read_end_raw = read_start_raw + self.config.block_len as isize;

                    // quantity of initial zero-padding
                    let left_pad = (-read_start_raw).max(0) as usize;

                    // real limits of the TOD, clamped to always be in the interval [0, n_samples]
                    let read_start = read_start_raw.max(0) as usize;
                    let read_end = (read_end_raw as usize).min(self.config.n_samples);

                    // reads the pixel values (non so se si capisce) and writes them to the temporary buffer `time_buf`
                    pointing.map_to_chunk(in_map, &mut time_buf, read_start, read_end, left_pad);

                    let mut noise_filter = InverseNoiseFilter {
                        inv_noiseps: self.inv_noiseps,
                        det_pairs: self.det_pairs,
                        config: self.config,
                        fft: &mut fft,
                    };

                    // transform `time_buf` to frequency-domain via FFT, decorrelatse the atmospheric noise and
                    // the 1/f drift by multiplying it by the inverse noise matrix, then reconverts the clean
                    // signal into the time-domain using the iFFT
                    noise_filter.filter_chunk(&mut time_buf, &mut freq_buf, &mut freq_acc);

                    // takes the filtered temporal samples in `time_buf` and accumulats them into the corresponding
                    // pixel in the output map
                    pointing.chunk_to_map(&time_buf, &mut local_map, core_start, core_end);

                    (time_buf, freq_buf, freq_acc, fft, local_map)
                },
            )
            .map(|(_, _, _, _, local_map)| local_map)
            .reduce(
                || vec![0.0; out_map.len()], // Elemento neutro per la somma
                |mut acc_map, thread_map| {
                    for (acc_val, &thread_val) in acc_map.iter_mut().zip(thread_map.iter()) {
                        *acc_val += thread_val;
                    }
                    acc_map
                },
            );

        out_map.copy_from_slice(&final_map);
    }
}
