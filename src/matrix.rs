pub fn matrix_to_array(matrix: Vec<Vec<f32>>) -> Vec<f32> {
    matrix.into_iter().flatten().collect()
}

pub fn convert_image_to_array(image: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    image.into_iter().flatten().flatten().collect()
}

pub fn sum(input_array: Vec<f32>, weights_array: Vec<f32>) -> f32 {
    input_array.iter().zip(weights_array.iter()).map(|(x, w)| x * w).sum()
}

pub fn multiply_matrix_vector(matrix: &Vec<Vec<f32>>, vector: &Vec<f32>) -> Vec<f32> {
    matrix.iter()
        .map(|row| row.iter().zip(vector.iter()).map(|(x, w)| x * w).sum())
        .collect()
}

pub fn transpose(matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut result = vec![vec![0.0; rows]; cols];

    for i in 0..rows {
        for j in 0..cols {
            result[j][i] = matrix[i][j];
        }
    }

    result
}

pub fn multiply_matrices(a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let rows = a.len();
    let cols = b[0].len();
    let shared = a[0].len();
    assert_eq!(shared, b.len(), "Dimensions incompatibles pour multiplication");

    let mut result = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            for k in 0..shared {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

pub fn inverse(matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let n = matrix.len();
    assert_eq!(n, matrix[0].len(), "La matrice doit être carrée pour être inversée");

    let mut a = matrix.clone();
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        inv[i][i] = 1.0;
    }

    for i in 0..n {
        let diag = a[i][i];
        if diag == 0.0 {
            panic!("Pivot nul détecté, matrice non inversible");
        }

        for j in 0..n {
            a[i][j] /= diag;
            inv[i][j] /= diag;
        }

        for k in 0..n {
            if k != i {
                let factor = a[k][i];
                for j in 0..n {
                    a[k][j] -= factor * a[i][j];
                    inv[k][j] -= factor * inv[i][j];
                }
            }
        }
    }

    inv
}

pub fn regularized_pseudo_inverse(phi: &Vec<Vec<f32>>, lambda: f32) -> Vec<Vec<f32>> {
    let phi_t = transpose(phi);
    let phi_t_phi = multiply_matrices(&phi_t, phi);

    let mut regularized = phi_t_phi.clone();
    for i in 0..regularized.len() {
        regularized[i][i] += lambda;
    }

    let inv = inverse(&regularized);
    multiply_matrices(&inv, &phi_t)
}


pub fn matrix_dataset_to_array(dataset: Vec<Vec<Vec<f32>>>) -> Vec<(Vec<f32>, usize)> {
    let mut flattened_data = Vec::new();

    for (class_idx, class_data) in dataset.iter().enumerate() {
        for data_point in class_data {
            flattened_data.push((data_point.clone(), class_idx));
        }
    }

    flattened_data
}