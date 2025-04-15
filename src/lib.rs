use pyo3::prelude::*;
mod activation_function;
mod data_converter;
mod data_manager;
mod loss;
mod matrix;
mod mlp;

#[pyfunction]
fn predict_mlp(input: Vec<Vec<f32>>, is_classification: bool) -> PyResult<f32> {
    Ok(mlp::predict(input, is_classification))
}

#[pyfunction]
fn train_mlp(
    dataset: Vec<Vec<Vec<Vec<f32>>>>,
    dataset_output: Vec<f32>,
    nb_epoch: i32,
    hidden_layers: Vec<i32>,
    training_name: String,
    is_classification: bool,
    verbose: bool
) -> PyResult<()> {
    Ok(mlp::training(
        dataset,
        dataset_output,
        nb_epoch,
        hidden_layers,
        training_name,
        is_classification,
        verbose
    ))
}

/// A Python module implemented in Rust.
#[pymodule]
fn better_tensorflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(predict_mlp, m)?)?;
    m.add_function(wrap_pyfunction!(train_mlp, m)?)?;
    Ok(())
}
