use pyo3::prelude::*;
use rand::Rng;
use crate::data_converter::import_weights_linear;
use crate::data_converter::export_weights_linear;
use crate::{data_converter, loss};
use crate::database::insert_training_score;

/// Fonction sigmoïde
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

pub fn predict(x_data: &Vec<f32>, m: Option<f32>, b: Option<f32>, mode: &str) -> Vec<f32> {
    let (m, b) = match (m, b) {
        (Some(m), Some(b)) => (m, b),
        _ => import_weights_linear(),
    };

    // Calcul des prédictions linéaires
    let predictions: Vec<f32> = x_data.iter()
        .map(|&x| m * x + b)  // Prédiction linéaire
        .collect();

    // Si c'est un problème de classification, on applique une fonction sigmoïde
    if mode == "classification" {
        predictions.into_iter().map(|x| sigmoid(x)).collect() // Transformation avec sigmoid pour classification
    } else {
        predictions  // Sinon, on retourne les prédictions pour la régression
    }
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

/// Fonction d'entraînement pour la régression ou la classification
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

    // Boucle sur les époques pour l'entraînement
    for epoch in 0..epochs {
        let mut m_gradient = 0.0;
        let mut b_gradient = 0.0;
        let n = x_data.len() as f32;

        // Calcul des gradients sur les données d'entraînement
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

            // Mise à jour des gradients
            m_gradient += error * x;
            b_gradient += error;
        }

        // Mise à jour des poids (m et b)
        m -= (m_gradient / n) * learning_rate;
        b -= (b_gradient / n) * learning_rate;

        // Affichage des résultats tous les 200 epochs
        if epoch % 200 == 0 {
            let y_pred = match mode {
                "classification" => predict(&x_data.clone(), Some(m), Some(b), "classification"),
                "regression" => predict(&x_data.clone(), Some(m), Some(b), "regression"),
                _ => panic!("Mode inconnu: {}", mode),
            };

            let loss = match mode {
                "classification" => logistic_loss(&y_data, &y_pred),
                "regression" => loss::mse(y_data.clone(), y_pred),
                _ => 0.0,
            };

            // Enregistrement des résultats dans la base de données
            insert_training_score(training_name.clone(), loss, 0.0, epoch).expect("Error inserting into database");

            if verbose {
                println!("Epoch {}: y = {:.3}x + {:.3}, Loss: {:.5}", epoch, m, b, loss);
            }
        }
    }

    // Affichage des résultats finaux
    println!("Modèle entraîné : y = {:.3}x + {:.3}", m, b);

    let y_pred = match mode {
        "classification" => predict(&x_data.clone(), Some(m), Some(b), "classification"),
        "regression" => predict(&x_data.clone(), Some(m), Some(b), "regression"),
        _ => panic!("Mode inconnu: {}", mode),
    };

    let final_loss = match mode {
        "classification" => logistic_loss(&y_data, &y_pred),
        "regression" => loss::mse(y_data.clone(), y_pred),
        _ => 0.0,
    };

    match mode {
        "classification" => println!("Perte logistique finale : {:.5}", final_loss),
        "regression" => println!("Erreur quadratique moyenne : {:.5}", final_loss),
        _ => (),
    };

    // Enregistrer le score final d'entraînement dans la base de données
    insert_training_score(training_name, final_loss, 0.0, epochs).expect("Error inserting final score into database");
    data_converter::export_weights_linear(m, b);
}
