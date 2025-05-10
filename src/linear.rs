use pyo3::prelude::*;
use rand::Rng;

/// Fonction sigmoïde
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}


pub fn predict(x_data: Vec<f32>, m: f32, b: f32, mode: &str) -> Vec<f32> {
    x_data.iter().map(|&x| {
        let linear_result = m * x + b;
        match mode {
            "classification" => sigmoid(linear_result),  // Applique la sigmoïde pour la classification
            "regression" => linear_result,               // Pour la régression, on garde le résultat linéaire
            _ => panic!("Mode inconnu : {}", mode),
        }
    }).collect()
}

/// Perte logistique (log loss)
pub fn logistic_loss(y_true: &Vec<f32>, y_pred: &Vec<f32>) -> f32 {
    y_true.iter()
        .zip(y_pred.iter())
        .map(|(&y, &p)| {
            let p = p.clamp(1e-7, 1.0 - 1e-7); // éviter les log(0)
            -y * p.ln() - (1.0 - y) * (1.0 - p).ln()
        })
        .sum::<f32>() / y_true.len() as f32
}

///  (MSE)
pub fn mse(y_true: &Vec<f32>, y_pred: &Vec<f32>) -> f32 {
    y_true.iter()
        .zip(y_pred.iter())
        .map(|(&y, &p)| (y - p).powi(2))
        .sum::<f32>() / y_true.len() as f32
}

pub fn train(
    x_data: Vec<f32>,
    y_data: Vec<f32>,
    mode: &str,
    verbose: bool,
    epochs: i32,
    training_name: String,
) {
    let mut rng = rand::thread_rng();
    let mut m = rng.gen_range(0.0..1.0);
    let mut b = rng.gen_range(0.0..1.0);
    let learning_rate = 0.01;

    for epoch in 0..epochs {
        let mut m_gradient = 0.0;
        let mut b_gradient = 0.0;
        let n = x_data.len() as f32;

        for (&x, &y) in x_data.iter().zip(y_data.iter()) {
            let (y_pred, error) = match mode {
                "classification" => {
                    let z = m * x + b;
                    let pred = sigmoid(z);
                    (pred, pred - y)
                }
                "regression" => {
                    let pred = m * x + b;
                    (pred, pred - y)
                }
                _ => panic!("Mode inconnu: {}", mode),
            };

            m_gradient += error * x;
            b_gradient += error;
        }

        m -= (m_gradient / n) * learning_rate;
        b -= (b_gradient / n) * learning_rate;

        if epoch % 200 == 0 {
            let y_pred = match mode {
                "classification" => predict(x_data.clone(), m, b, "classification"),
                "regression" => predict(x_data.clone(), m, b, "regression"),
                _ => panic!("Mode inconnu: {}", mode),
            };

            let loss = match mode {
                "classification" => logistic_loss(&y_data, &y_pred),
                "regression" => mse(&y_data, &y_pred),
                _ => 0.0,
            };

            if verbose {
                println!("Epoch {}: y = {:.3}x + {:.3}, Loss: {:.5}", epoch, m, b, loss);
            }
        }
    }

    println!("Modèle entraîné : y = {:.3}x + {:.3}", m, b);

    let y_pred = match mode {
        "classification" => predict(x_data.clone(), m, b, "classification"),
        "regression" => predict(x_data.clone(), m, b, "regression"),
        _ => panic!("Mode inconnu: {}", mode),
    };

    let final_loss = match mode {
        "classification" => logistic_loss(&y_data, &y_pred),
        "regression" => mse(&y_data, &y_pred),
        _ => 0.0,
    };

    match mode {
        "classification" => println!("Perte logistique finale : {:.5}", final_loss),
        "regression" => println!("Erreur quadratique moyenne : {:.5}", final_loss),
        _ => (),
    };
}


