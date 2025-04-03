use pyo3::prelude::*;
mod activation_function;
mod data_converter;
mod data_manager;
mod loss;
mod matrix;
mod mlp;

#[pyfunction]
fn predict_mlp(input: Vec<Vec<f32>>) -> PyResult<i32> {
    Ok(mlp::predict(input))
}

#[pyfunction]
fn train_mlp(
    dataset: Vec<Vec<Vec<Vec<f32>>>>,
    nb_epoch: i32,
    hidden_layers: Vec<i32>,
    training_name: String
) -> PyResult<()> {
    Ok(mlp::training(dataset, nb_epoch, hidden_layers, training_name))
}

/// A Python module implemented in Rust.
#[pymodule]
fn better_tensorflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(predict_mlp, m)?)?;
    m.add_function(wrap_pyfunction!(train_mlp, m)?)?;
    Ok(())
}
