use pyo3::prelude::*;
 // Pour PyResult, PyModule, et wrap_pyfunction!
use pyo3::wrap_pyfunction;
mod activation_function;
mod data_converter;
mod data_manager;
mod database;
mod loss;
mod matrix;
mod mlp;
mod linear;

#[pyfunction]

fn train_linear(
    x_data: Vec<f32>,
    y_data: Vec<f32>,
    mode: String,
    verbose: bool,
    epochs: i32,
    training_name: String,
) -> PyResult<()> {
    Ok(linear::train(x_data, y_data, &mode, verbose, epochs, training_name))
}
#[pyfunction]
fn predict_linear(
    x_data: Vec<f32>,
    m: f32,
    b: f32,
    mode: String,
) -> PyResult<Vec<f32>> {
    Ok(linear::predict(x_data, m, b, &mode))
}

#[pyfunction]
fn predict_mlp(
    input: Vec<f32>,
    all_layers: Vec<Vec<Vec<f32>>>,
    is_classification: bool,
) -> PyResult<i32> {
    Ok(mlp::predict(input, all_layers, is_classification))
}

#[pyfunction]
fn train_mlp(
    dataset: Vec<Vec<Vec<f32>>>,
    dataset_validation: Vec<Vec<Vec<f32>>>,
    dataset_output: Vec<f32>,
    nb_epoch: i32,
    hidden_layers: Vec<i32>,
    training_name: String,
    is_classification: bool,
    verbose: bool,
    save_in_db: bool,
    learning_rate: f32,
) -> PyResult<()> {
    Ok(mlp::training(
        dataset,
        dataset_validation,
        dataset_output,
        nb_epoch,
        hidden_layers,
        training_name,
        is_classification,
        verbose,
        save_in_db,
        learning_rate,
    ))
}

#[pyfunction]
fn convert_matrix_to_array(matrix: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    Ok(matrix::matrix_to_array(matrix))
}

#[pymodule]
fn better_tensorflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(predict_mlp, m)?)?;
    m.add_function(wrap_pyfunction!(predict_linear, m)?)?;
    m.add_function(wrap_pyfunction!(train_mlp, m)?)?;
    m.add_function(wrap_pyfunction!(train_linear, m)?)?;
    m.add_function(wrap_pyfunction!(convert_matrix_to_array, m)?)?;
    Ok(())
}
