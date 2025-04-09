use rand::Rng;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pymodule]
fn better_tensorflow(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

fn mean_squared_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true.iter()
        .zip(y_pred.iter())
        .map(|(yt, yp)| (yt - yp).powi(2))
        .sum::<f64>() / y_true.len() as f64
}

fn main() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y_data = vec![2.0, 4.1, 6.0, 8.2, 10.1];

    let mut rng = rand::thread_rng();
    let mut m = rng.gen_range(0.0..1.0);
    let mut b = rng.gen_range(0.0..1.0);

    let learning_rate = 0.01;
    let epochs = 1000;

    for epoch in 0..epochs {
        let mut m_gradient = 0.0;
        let mut b_gradient = 0.0;
        let n = x_data.len() as f64;

        for (&x, &y) in x_data.iter().zip(y_data.iter()) {
            let y_pred = m * x + b;
            m_gradient += -2.0 * x * (y - y_pred);
            b_gradient += -2.0 * (y - y_pred);
        }

        m -= (m_gradient / n) * learning_rate;
        b -= (b_gradient / n) * learning_rate;


        if epoch % 200 == 0 {
            println!("Epoch {}: y = {:.3}x + {:.3}", epoch, m, b);
        }
    }

    println!("Modèle entraîné : y = {:.3}x + {:.3}", m, b);

    let y_pred: Vec<f64> = x_data.iter().map(|&x| m * x + b).collect();
    let mse = mean_squared_error(&y_data, &y_pred);
    println!("Erreur quadratique moyenne : {:.5}", mse);
}
