use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
mod activation_function;
mod data_converter;
mod data_manager;
mod database;
mod kmeans;
mod linear;
mod loss;
mod math;
mod matrix;
mod mlp;
mod rbf;
mod svm_bis;
mod ols;

#[pyfunction]
fn train_linear(
    x_data: Vec<f32>,
    y_data: Vec<f32>,
    mode: String,
    verbose: bool,
    epochs: i32,
    training_name: String,
) -> PyResult<()> {
    Ok(linear::train(
        x_data,
        y_data,
        &mode,
        verbose,
        epochs,
        training_name,
    ))
}
#[pyfunction]
fn predict_linear(x_data: Vec<f32>, m: f32, b: f32, mode: &str) -> PyResult<Vec<f32>> {
    Ok(linear::predict(&x_data, Some(m), Some(b), &mode))
}

#[pyfunction]
fn predict_rbf(input_data: Vec<f32>, gamma: f32, is_classification: bool) -> PyResult<f32> {
    Ok(rbf::predict_rbf(input_data, gamma, is_classification))
}

#[pyfunction]
fn train_rbf(
    dataset_input: Vec<Vec<Vec<f32>>>,
    dataset_validation: Vec<Vec<Vec<f32>>>,
    output_dataset: Vec<f32>,
    number_clusters: i32,
    gamma: f32,
    is_classification: bool,
    save_in_db: bool,
    training_name: String,
) -> PyResult<(f32)> {
    Ok(rbf::train_rbf(
        dataset_input,
        dataset_validation,
        output_dataset,
        number_clusters,
        gamma,
        is_classification,
        save_in_db,
        training_name,
    ))
}

#[pyfunction]
fn predict_mlp(
    input: Vec<f32>,
    all_layers: Vec<Vec<Vec<f32>>>,
    is_classification: bool,
    verbose: bool,
) -> PyResult<f32> {
    Ok(mlp::predict(input, all_layers, is_classification, verbose))
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
    nb_epoch_to_save: i32,
) -> PyResult<(f32)> {
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
        nb_epoch_to_save,
    ))
}

#[pyfunction]
fn convert_matrix_to_array(matrix: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    Ok(matrix::matrix_to_array(matrix))
}
#[pyfunction]
fn convert_image_to_array(image: Vec<Vec<Vec<f32>>>) -> PyResult<Vec<f32>> {
    Ok(matrix::convert_image_to_array(image))
}

#[pyfunction]
fn load_linear_weights() -> PyResult<(f32, f32)> {
    Ok(data_converter::import_weights_linear())
}
#[pyfunction]
fn export_linear_weights(m: f32, b: f32) -> PyResult<()> {
    Ok(data_converter::export_weights_linear(m, b))
}


#[pyfunction]
fn train_ols(x_data: Vec<Vec<f32>>, y_data: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(ols::train_ols(x_data, y_data))
}

#[pyfunction]
fn predict_ols(x_data: Vec<Vec<f32>>, weights: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(ols::predict_ols(x_data, weights))
}

#[pyfunction]
fn train_ols_robust(x_data: Vec<Vec<f32>>, y_data: Vec<f32>) -> PyResult<Vec<f32>> {
    ols::train_ols_robust(x_data, y_data)
}

#[pyfunction]
fn import_weights_ols_from_file() -> PyResult<Vec<f32>> {
    let weights = data_converter::import_weights_ols();
    Ok(weights)
}


#[pymodule]
fn better_tensorflow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(predict_mlp, m)?)?;
    m.add_function(wrap_pyfunction!(predict_linear, m)?)?;
    m.add_function(wrap_pyfunction!(train_mlp, m)?)?;
    m.add_function(wrap_pyfunction!(train_linear, m)?)?;
    m.add_function(wrap_pyfunction!(convert_matrix_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(convert_image_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(load_linear_weights, m)?)?;
    m.add_function(wrap_pyfunction!(export_linear_weights, m)?)?;
    m.add_function(wrap_pyfunction!(train_rbf, m)?)?;
    m.add_function(wrap_pyfunction!(predict_rbf, m)?)?;
    m.add_class::<svm_bis::KernelSVM>()?;

    m.add_function(wrap_pyfunction!(train_ols, m)?)?;
    m.add_function(wrap_pyfunction!(predict_ols, m)?)?;
    m.add_function(wrap_pyfunction!(train_ols_robust, m)?)?;
    m.add_function(wrap_pyfunction!(import_weights_ols_from_file, m)?)?;


    Ok(())
}
