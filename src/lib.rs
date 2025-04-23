use pyo3::prelude::*;
mod activation_function;
mod data_converter;
mod data_manager;
mod loss;
mod matrix;
mod mlp;

#[pyfunction]
fn predict_mlp(input: Vec<f32>,all_layers:Vec<Vec<Vec<f32>>>, is_classification: bool) -> PyResult<i32> {
    Ok(mlp::predict(input,all_layers ,is_classification))
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
    m.add_function(wrap_pyfunction!(train_mlp, m)?)?;
    m.add_function(wrap_pyfunction!(convert_matrix_to_array, m)?)?;
    Ok(())
}
