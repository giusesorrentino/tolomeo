use crate::roma::config::MapmakingConfig;

// manages the pointing matrix operators (P and P^T) to transform data
// between pixel space (sky maps) and time-domain space (TOD chunks).
pub struct PointingOperator<'a> {
    pub pixel_ids: &'a [usize],
    pub config: &'a MapmakingConfig,
}

impl<'a> PointingOperator<'a> {
    // forward projection operator P: Map -> TOD (d = P cdot x).
    // simulates time-ordered data by reading values from an input sky map
    // at the pixel locations defined by the pointing matrix (`pixel_ids`)
    pub fn map_to_chunk(
        &self,
        in_map: &[f64],
        time_buf: &mut [f64],
        read_start: usize,
        read_end: usize,
        left_pad: usize,
    ) {
        let n_det = self.config.n_det;

        for idet in 0..n_det {
            let block_offset = idet * self.config.block_len;

            for sample in read_start..read_end {
                let pixel = self.pixel_ids[sample * n_det + idet];
                time_buf[block_offset + left_pad + (sample - read_start)] = in_map[pixel];
            }
        }
    }

    // transpose projection (binning) operator P^T: TOD -> Map (x = P^T cdot d).
    // accumulates filtered time-ordered samples back into the target map pixels
    // based on detector pointing coordinates.
    pub fn chunk_to_map(
        &self,
        time_buf: &[f64],
        out_map: &mut [f64],
        core_start: usize,
        core_end: usize,
    ) {
        let n_det = self.config.n_det;

        for idet in 0..n_det {
            let block_offset = idet * self.config.block_len;

            for sample in core_start..core_end {
                let pos_in_block = (sample - core_start) + self.config.padding;
                let pixel = self.pixel_ids[sample * n_det + idet];
                out_map[pixel] += time_buf[block_offset + pos_in_block];
            }
        }
    }
}
