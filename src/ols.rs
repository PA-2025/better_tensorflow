use crate::data_converter;

/// Multiplie une matrice par son transposée (X^T * X)
fn xtx(x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let rows = x.len();
    let cols = x[0].len();
    let mut result = vec![vec![0.0; cols]; cols];

    for i in 0..cols {
        for j in 0..cols {
            for k in 0..rows {
                result[i][j] += x[k][i] * x[k][j];
            }
        }
    }

    result
}

/// Multiplie une transposée par un vecteur (X^T * Y)
fn xty(x: &Vec<Vec<f32>>, y: &Vec<f32>) -> Vec<f32> {
    let cols = x[0].len();
    let mut result = vec![0.0; cols];

    for i in 0..cols {
        for j in 0..x.len() {
            result[i] += x[j][i] * y[j];
        }
    }

    result
}

/// Inversion de matrice 2x2 ou 3x3
fn inverse(matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let n = matrix.len();
    let mut result = vec![vec![0.0; n]; n];
    let mut a = matrix.clone();

    for i in 0..n {
        result[i][i] = 1.0;
    }

    for i in 0..n {
        let mut pivot = a[i][i];
        if pivot == 0.0 {
            panic!("Matrix not invertible");
        }

        for j in 0..n {
            a[i][j] /= pivot;
            result[i][j] /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = a[k][i];
                for j in 0..n {
                    a[k][j] -= factor * a[i][j];
                    result[k][j] -= factor * result[i][j];
                }
            }
        }
    }

    result
}

fn multiply_matrix_vector(matrix: &Vec<Vec<f32>>, vector: &Vec<f32>) -> Vec<f32> {
    let mut result = vec![0.0; matrix.len()];

    for i in 0..matrix.len() {
        for j in 0..vector.len() {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    result
}

/// Apprend les poids W pour la régression linéaire : W = (X^T X)^(-1) X^T Y
pub fn train_ols(x: Vec<Vec<f32>>, y: Vec<f32>) -> Vec<f32> {
    let xtx_ = xtx(&x);
    let xty_ = xty(&x, &y);
    let xtx_inv = inverse(&xtx_);
    let weights = multiply_matrix_vector(&xtx_inv, &xty_);

    data_converter::export_weights_ols(&weights);
    weights
}

/// Prédit des valeurs à partir des poids appris
pub fn predict_ols(x: Vec<Vec<f32>>, weights: Vec<f32>) -> Vec<f32> {
    x.iter()
        .map(|row| {
            row.iter()
                .zip(weights.iter())
                .map(|(xi, wi)| xi * wi)
                .sum()
        })
        .collect()
}
