use pyo3::prelude::*;
use rand::Rng;

// À adapter avec les bons modules si nécessaire
use crate::data_converter::{import_weights_linear, export_weights_linear};
use crate::database::insert_training_score;
use crate::loss;

/// fct de seuil (perceptron)
fn step_function(z: f32) -> f32 {
    if z >= 0.0 { 1.0 } else { 0.0 }
}

/// predic for reg & Class
pub fn predict_rosenblatt(x_data: &[f32], m: Option<f32>, b: Option<f32>, mode: &str) -> Vec<f32> {
    let (m, b) = match (m, b) {
        (Some(m), Some(b)) => (m, b),
        _ => import_weights_linear(),
    };

    x_data.iter().map(|&x| {
        let z = m * x + b;
        match mode {
            "classification" => step_function(z),
            "regression" => z,
            _ => panic!("Mode inconnu : {}", mode),
        }
    }).collect()
}

/// Rosenblatt Training
pub fn train_rosenblatt(
    x_data: Vec<f32>,
    y_data: Vec<f32>,
    mode: &str,
    verbose: bool,
    epochs: i32,
    training_name: String,
) {
    assert_eq!(x_data.len(), y_data.len(), "x_data et y_data doivent avoir la même taille");

    let mut rng = rand::thread_rng();
    let mut m = rng.gen_range(0.0..1.0);
    let mut b = rng.gen_range(0.0..1.0);
    let learning_rate = 0.01;

    for epoch in 0..epochs {
        let mut total_error = 0.0;

        for (&x, &y) in x_data.iter().zip(&y_data) {
            let z = m * x + b;
            let (y_pred, error) = match mode {
                "classification" => {
                    let y_pred = step_function(z);
                    let error = y - y_pred;
                    if error != 0.0 {
                        m += learning_rate * error * x;
                        b += learning_rate * error;
                    }
                    (y_pred, error.abs())
                }
                "regression" => {
                    let y_pred = z;
                    let error = y - y_pred;
                    m += learning_rate * error * x;
                    b += learning_rate * error;
                    (y_pred, error.powi(2))
                }
                _ => panic!("Mode inconnu : {}", mode),
            };

            total_error += error;
        }

        // Score Saving
        if epoch % 200 == 0 || epoch == epochs - 1 {
            let y_pred = predict_rosenblatt(&x_data, Some(m), Some(b), mode);
            let score = match mode {
                "classification" => {
                    let correct = y_data.iter().zip(&y_pred).filter(|(&yt, &yp)| yt == yp).count();
                    1.0 - (correct as f32 / y_data.len() as f32)
                }
                "regression" => loss::mse(&y_data, &y_pred),
                _ => unreachable!(),
            };

            let _ = insert_training_score(training_name.clone(), score, 0.0, epoch);

            if verbose {
                match mode {
                    "classification" => println!(
                        "Epoch {}: y = {:.3}x + {:.3}, Accuracy: {:.2}%",
                        epoch, m, b, (1.0 - score) * 100.0
                    ),
                    "regression" => println!(
                        "Epoch {}: y = {:.3}x + {:.3}, MSE: {:.5}",
                        epoch, m, b, score
                    ),
                    _ => {}
                }
            }
        }
    }

    // Show final
    println!("Modèle entraîné : y = {:.3}x + {:.3}", m, b);

    let y_pred = predict_rosenblatt(&x_data, Some(m), Some(b), mode);
    let final_score = match mode {
        "classification" => {
            let acc = y_data.iter().zip(&y_pred).filter(|(&yt, &yp)| yt == yp).count() as f32 / y_data.len() as f32;
            println!("Exactitude finale : {:.2}%", acc * 100.0);
            1.0 - acc
        }
        "regression" => {
            let mse = loss::mse(&y_data, &y_pred);
            println!("Erreur quadratique moyenne finale : {:.5}", mse);
            mse
        }
        _ => unreachable!(),
    };

    let _ = insert_training_score(training_name, final_score, 0.0, epochs);
    export_weights_linear(m, b);
}

// ----------- WRAPPERS PYTHON ----------------

#[pyfunction]
fn train_rosenblatt_py(
    x_data: Vec<f32>,
    y_data: Vec<f32>,
    mode: String,
    verbose: bool,
    epochs: i32,
    training_name: String,
) {
    train_rosenblatt(x_data, y_data, &mode, verbose, epochs, training_name);
}

#[pyfunction]
fn predict_rosenblatt_py(
    x_data: Vec<f32>,
    m: Option<f32>,
    b: Option<f32>,
    mode: String,
) -> Vec<f32> {
    predict_rosenblatt(&x_data, m, b, &mode)
}

#[pymodule]
fn rosenblatt_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(train_rosenblatt_py, m)?)?;
    m.add_function(wrap_pyfunction!(predict_rosenblatt_py, m)?)?;
    Ok(())
}
