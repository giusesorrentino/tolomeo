#[macro_export]
macro_rules! pyprint {
    ($($arg:tt)*) => {{
        use pyo3::types::PyAnyMethods;

        let msg = format!($($arg)*);
        pyo3::Python::attach(|py| {
            if let Ok(builtins) = py.import("builtins") {
                let _ = builtins.call_method1("print", (msg,));
            }
        });
    }};
}
