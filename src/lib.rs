use pyo3::prelude::*;
mod activation_function;
mod perceptron;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn predict_perceptron(input: Vec<Vec<f32>>) -> PyResult<i32> {
    Ok(perceptron::predict(input))
}

/// A Python module implemented in Rust.
#[pymodule]
fn better_tensorflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(predict_perceptron, m)?)?;
    Ok(())
}
