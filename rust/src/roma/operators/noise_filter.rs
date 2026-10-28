use rustfft::num_complex::Complex;

use crate::roma::config::MapmakingConfig;
use crate::roma::fft::FFTWorkspace;

pub struct InverseNoiseFilter<'a> {
    pub inv_noiseps: &'a [f64],
    pub det_pairs: &'a [(usize, usize)],
    pub config: &'a MapmakingConfig,
    pub fft: &'a mut FFTWorkspace,
}

impl<'a> InverseNoiseFilter<'a> {
    pub fn filter_chunk(
        &mut self,
        time_buf: &mut [f64],
        freq_buf: &mut [Complex<f64>],
        freq_acc: &mut [Complex<f64>],
    ) {
        // temporary buffer to accumulate the final filtered spectra
        freq_acc.fill(Complex::new(0.0, 0.0));

        let real_chunks = time_buf.chunks_exact_mut(self.config.block_len);
        let complex_chunks = freq_buf.chunks_exact_mut(self.config.n_freqs);

        // pushforward to frequency-domain of the TOD chunks
        for (r_chunk, c_chunk) in real_chunks.zip(complex_chunks) {
            self.fft.forward(r_chunk, c_chunk);
        }

        // total number of noise power spectra (detector pairs)
        let n_cols = self.det_pairs.len();

        for i in 0..n_cols {
            // i-th detector pair (f, r)
            let f = self.det_pairs[i].0;
            let r = self.det_pairs[i].1;

            // k-th frequency bin for the i-th detector pair
            for k in 0..self.config.n_freqs {
                let idx_noise = k * n_cols + i;
                let factor = self.inv_noiseps[idx_noise] / (self.config.block_len as f64);

                let idx_sum_f = f * self.config.n_freqs + k;
                let idx_complex_r = r * self.config.n_freqs + k;

                // accumulate noise contribution from detector r to detector f
                freq_acc[idx_sum_f] += freq_buf[idx_complex_r] * factor;

                // exploit symmetry for off-diagonal cross-spectra
                if f != r {
                    let idx_sum_r = r * self.config.n_freqs + k;
                    let idx_complex_f = f * self.config.n_freqs + k;

                    freq_acc[idx_sum_r] += freq_buf[idx_complex_f] * factor;
                }
            }
        }

        // (0..n_cols).into_par_iter()
        //     .fold()
        //     .map()
        //     .reduce()

        freq_buf.copy_from_slice(freq_acc);

        let real_chunks = time_buf.chunks_exact_mut(self.config.block_len);
        let complex_chunks = freq_buf.chunks_exact_mut(self.config.n_freqs);

        // pullback to time-domain the filtered spectral
        for (r_chunk, c_chunk) in real_chunks.zip(complex_chunks) {
            self.fft.backward(c_chunk, r_chunk);
        }
    }
}
