use crate::data_converter;
use crate::math::{xtx, xty, inverse, multiply_matrix_vector};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
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


pub fn train_ols_robust(x: Vec<Vec<f32>>, y: Vec<f32>) -> PyResult<Vec<f32>> {
    let xtx_ = xtx(&x);
    let xty_ = xty(&x, &y);

    // Try normal inversion
    let xtx_inv = match crate::math::try_inverse(&xtx_) {
        Some(inv) => {
            println!("Inversion succeeded");
            inv
        }
        None => {
            println!("Inversion failed, adding λ=1e-4");

            // Try with small λ
            let lambda = 1e-4;
            let mut xtx_reg = xtx_.clone();
            for i in 0..xtx_reg.len() {
                xtx_reg[i][i] += lambda;
            }

            match crate::math::try_inverse(&xtx_reg) {
                Some(inv) => {
                    println!("Inversion succeeded with λ=1e-4");
                    inv
                }
                None => {
                    println!("Inversion failed again, trying with λ=1e-2");

                    // Try with larger λ
                    let lambda_big = 1e-2;
                    let mut xtx_reg_big = xtx_.clone();
                    for i in 0..xtx_reg_big.len() {
                        xtx_reg_big[i][i] += lambda_big;
                    }

                    match crate::math::try_inverse(&xtx_reg_big) {
                        Some(inv) => {
                            println!("Inversion succeeded with λ=1e-2");
                            inv
                        }
                        None => {
                            return Err(PyValueError::new_err(
                                "La matrice X^T X est trop mal conditionnée même après régularisation (λ).",
                            ));
                        }
                    }
                }
            }
        }
    };

    let weights = multiply_matrix_vector(&xtx_inv, &xty_);
    data_converter::export_weights_ols(&weights);
    Ok(weights)
}
