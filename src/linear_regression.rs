use rand::Rng;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use crate::loss;
use crate::data_manager;



pub fn train(x_data : Vec<f32>, y_data : Vec<f32>, verbose : bool, epochs : i32, training_name : String){

    let mut rng = rand::thread_rng();
    let mut m = rng.gen_range(0.0..1.0);
    let mut b = rng.gen_range(0.0..1.0);

    let learning_rate = 0.01;

    for epoch in 0..epochs {
        let mut m_gradient = 0.0;
        let mut b_gradient = 0.0;
        let n = x_data.len() as f32;

        for (&x, &y) in x_data.iter().zip(y_data.iter()) {
            let y_pred = m * x + b;
            m_gradient += -2.0 * x * (y - y_pred);
            b_gradient += -2.0 * (y - y_pred);
        }

        m -= (m_gradient / n) * learning_rate;
        b -= (b_gradient / n) * learning_rate;



        if epoch % 200 == 0 {
            let y_pred: Vec<f32> = predict(x_data.clone(), m, b);
            let mse = loss::mse(y_data.clone(), y_pred);
            data_manager::add_text_to_file(training_name.clone(), mse.to_string() + "\n")
                .expect("Error: error during write train data");
            if verbose {
                println!("Epoch {}: y = {:.3}x + {:.3}", epoch, m, b);
            }
        }
    }

    println!("Modèle entraîné : y = {:.3}x + {:.3}", m, b);

    let y_pred: Vec<f32> = predict(x_data,m,b);
    let mse = loss::mse(y_data, y_pred);
    println!("Erreur quadratique moyenne : {:.5}", mse);
}

pub fn predict(x_data : Vec<f32>,m : f32, b: f32) -> Vec<f32> {
    return x_data.iter().map(|&x| m * x + b).collect();
}
