#[derive(Debug, Clone)]
pub struct MapmakingConfig {
    // ---- inputs: they come from the data ----
    pub n_freqs: usize, // number of frequency bins = number of rows of each invps file
    pub n_det: usize,   // number of detectors (mm_nhorns)
    pub n_samples: usize, // samples per detector in the TOD array
    pub n_pixels: usize, // number of pixels represented by the map vectors

    // ---- derived from n_freqs ----
    pub block_len: usize, // = 2*(n_freqs-1): length of each FFT block
    pub valid_len: usize, // = n_freqs-1: "good" samples kept from each block
    pub padding: usize,   // = valid_len/2: context samples on each side of a block

    // ---- derived from the sizes ----
    pub n_blocks: usize, // = ceil(n_samples / valid_len): number of blocks to cover the TOD
}

impl MapmakingConfig {
    pub fn new(n_freqs: usize, n_det: usize, n_samples: usize, n_pixels: usize) -> Self {
        let block_len = 2 * (n_freqs - 1);
        let valid_len = n_freqs - 1;
        let padding = valid_len / 2;

        let n_blocks = n_samples.div_ceil(valid_len);

        Self {
            n_freqs,
            n_det,
            n_samples,
            n_pixels,
            block_len,
            valid_len,
            padding,
            n_blocks,
        }
    }
}
