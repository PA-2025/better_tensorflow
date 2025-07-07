use crate::data_converter;
use crate::math::{xtx, xty, inverse, multiply_matrix_vector};

/// Trains linear regression weights: W = (X^T X)^(-1) X^T Y
pub fn train_ols(x: Vec<Vec<f32>>, y: Vec<f32>) -> Vec<f32> {
    let xtx_ = xtx(&x);
    let xty_ = xty(&x, &y);
    let xtx_inv = inverse(&xtx_);
    let weights = multiply_matrix_vector(&xtx_inv, &xty_);

    data_converter::export_weights_ols(&weights);
    weights
}

/// Predicts values using trained weights
pub fn predict_ols(x: Vec<Vec<f32>>, weights: Vec<f32>) -> Vec<f32> {
    let mut predictions = Vec::new();

    for row in x {
        let mut sum = 0.0;
        for i in 0..row.len() {
            sum += row[i] * weights[i];
        }
        predictions.push(sum);
    }

    predictions
}
