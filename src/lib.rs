pub mod logging;
pub mod python;
pub mod roma;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // pyo3_log::init();
    m.add_function(wrap_pyfunction!(python::tod2map_ffi, m)?)?;
    Ok(())
}
