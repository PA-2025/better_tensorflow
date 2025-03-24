use pyo3::prelude::*;
mod activation_function;
mod mlp;
mod data_manager;
mod data_converter;

#[pyfunction]
fn predict_perceptron(input: Vec<Vec<f32>>) -> PyResult<i32> {
    Ok(mlp::predict(input))
}

/// A Python module implemented in Rust.
#[pymodule]
fn better_tensorflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(predict_perceptron, m)?)?;
    Ok(())
}
