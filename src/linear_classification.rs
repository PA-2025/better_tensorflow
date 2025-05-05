use rand::Rng;

/// Fonction sigmoïde
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Fonction de prédiction
fn predict(x_data: &Vec<f32>, m: f32, b: f32) -> Vec<f32> {
    x_data.iter().map(|&x| sigmoid(m * x + b)).collect()
}

/// Fonction de calcul de la perte logistique
fn logistic_loss(y_true: &Vec<f32>, y_pred: &Vec<f32>) -> f32 {
    y_true.iter()
        .zip(y_pred.iter())
        .map(|(&y, &p)| {
            let p = p.clamp(1e-7, 1.0 - 1e-7); // Évite les log(0)
            -y * p.ln() - (1.0 - y) * (1.0 - p).ln()
        })
        .sum::<f32>() / y_true.len() as f32
}

/// Fonction d'entraînement
pub fn train(
    x_data: Vec<f32>,
    y_data: Vec<f32>,
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
            let z = m * x + b;
            let y_pred = sigmoid(z);
            let error = y_pred - y;
            m_gradient += error * x;
            b_gradient += error;
        }

        m -= (m_gradient / n) * learning_rate;
        b -= (b_gradient / n) * learning_rate;

        if epoch % 200 == 0 {
            let y_pred = predict(&x_data, m, b);
            let loss = logistic_loss(&y_data, &y_pred);
            println!("Epoch {}: y = {:.3}x + {:.3}, Loss: {:.5}", epoch, m, b, loss);
        }
    }

    println!("Modèle entraîné : y = {:.3}x + {:.3}", m, b);
    let y_pred = predict(&x_data, m, b);
    let loss = logistic_loss(&y_data, &y_pred);
    println!("Perte logistique finale : {:.5}", loss);
}
