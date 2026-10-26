use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::roma::config::MapmakingConfig;
use crate::roma::solver::conjugate_gradient::ResidualNorm;
use crate::roma::temperature::flags::first_out_of_range_pixel;
use crate::roma::temperature::mapmaker::tod2map;

/// `valid` has the same shape as `tod` ([n_samples, n_det]): `true` = use the sample,
/// `false` = flagged. Flagged samples are moved to an extra virtual pixel (as ROMA's
/// `pixel_virt`), so their `pixel_ids` are ignored and never touch the map.
/// `n_pixels` is the number of pixels of the sky map, the virtual pixel is not included.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn tod2map_ffi<'py>(
    py: Python<'py>,
    tod: PyReadonlyArray2<'py, f64>,
    pixel_ids: PyReadonlyArray2<'py, usize>, // devo ancora capire cos'è e perché si chiama pixel_ids
    valid: Option<PyReadonlyArray2<'py, bool>>,
    inv_noiseps: PyReadonlyArray2<'py, f64>,
    det_pairs: PyReadonlyArray2<'py, usize>,
    lag_zero: PyReadonlyArray1<'py, f64>, // questo parametro va calcolato a posteriori, non come input
    n_pixels: usize,
    initial_map: Option<PyReadonlyArray1<'py, f64>>,
    residual_norm: &str,
    tol: f64,
    maxiter: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, f64)> {
    // checked here, so that a wrong value is a ValueError and never reaches the solver
    let residual_norm = match residual_norm {
        "plain" => ResidualNorm::Plain,
        "preconditioned" => ResidualNorm::Preconditioned,
        other => {
            return Err(PyValueError::new_err(format!(
                "`residual_norm` must be \"plain\" or \"preconditioned\", got {other:?}"
            )));
        }
    };

    let tod_shape = tod.shape();
    let pixel_ids_shape = pixel_ids.shape();

    if tod_shape != pixel_ids_shape {
        return Err(PyValueError::new_err(format!(
            "`tod` shape {:?} does not match `pixel_ids` shape {:?}",
            tod_shape, pixel_ids_shape
        )));
    }

    if let Some(valid) = valid.as_ref() {
        if valid.shape() != tod_shape {
            return Err(PyValueError::new_err(format!(
                "`valid` shape {:?} does not match `tod` shape {:?}",
                valid.shape(),
                tod_shape
            )));
        }
    }

    let n_samples = tod_shape[0];
    let n_det = tod_shape[1];
    let n_freqs = inv_noiseps.shape()[0];

    if n_freqs < 2 {
        return Err(PyValueError::new_err(
            "`inv_noiseps` must contain at least two frequency bins",
        ));
    }

    if lag_zero.shape()[0] != n_det {
        return Err(PyValueError::new_err(format!(
            "`lag_zero` length ({}) must match n_det ({})",
            lag_zero.shape()[0],
            n_det
        )));
    }

    let det_pairs_shape = det_pairs.shape();
    if det_pairs_shape[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "`det_pairs` shape {:?} must have second dimension equal to 2",
            det_pairs_shape
        )));
    }

    let n_files = det_pairs_shape[0];
    if inv_noiseps.shape()[1] != n_files {
        return Err(PyValueError::new_err(format!(
            "`inv_noiseps` number of files ({}) does not match `det_pairs` files ({})",
            inv_noiseps.shape()[1],
            n_files
        )));
    }

    if let Some(initial_map) = initial_map.as_ref() {
        if initial_map.shape()[0] != n_pixels {
            return Err(PyValueError::new_err(format!(
                "`initial_map` length ({}) must match `n_pixels` ({})",
                initial_map.shape()[0],
                n_pixels
            )));
        }
    }

    let det_pairs_slice = det_pairs.as_slice()?;
    let det_pairs_pairs: Vec<(usize, usize)> = det_pairs_slice
        .chunks_exact(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .collect();

    let tod_slice = tod.as_slice()?;
    let pixel_ids_slice = pixel_ids.as_slice()?;
    let valid_slice = valid.as_ref().map(|valid| valid.as_slice()).transpose()?;
    let inv_noiseps_slice = inv_noiseps.as_slice()?;
    let lag_zero_slice = lag_zero.as_slice()?;

    // only the samples that are really used must point inside the map: flagged samples are
    // sent to the virtual pixel, whatever their `pixel_ids` is.
    if let Some(flat) = first_out_of_range_pixel(pixel_ids_slice, valid_slice, n_pixels) {
        return Err(PyValueError::new_err(format!(
            "`pixel_ids[{}, {}]` = {} is outside the map (n_pixels = {}). Flag the sample \
             (`flags`) or fix the pixelization.",
            flat / n_det,
            flat % n_det,
            pixel_ids_slice[flat],
            n_pixels
        )));
    }

    let config = MapmakingConfig::new(n_freqs, n_det, n_samples, n_pixels);

    let initial_map_slice = initial_map.as_ref().map(|map| map.as_slice()).transpose()?;

    let mapmaker_result = py.detach(|| {
        catch_unwind(AssertUnwindSafe(|| {
            tod2map(
                tod_slice,
                pixel_ids_slice,
                valid_slice,
                inv_noiseps_slice,
                &det_pairs_pairs,
                lag_zero_slice,
                &config,
                initial_map_slice,
                residual_norm,
                tol,
                maxiter,
            )
        }))
    });

    let result = match mapmaker_result {
        Ok(res) => res,
        Err(_) => {
            return Err(PyValueError::new_err(
                "Il codice Rust è andato in Panic! Controlla il terminale per dettagli.",
            ));
        }
    };

    Ok((
        PyArray1::from_vec(py, result.map),
        result.iterations,
        result.residual,
    ))
}
