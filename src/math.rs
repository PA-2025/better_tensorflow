pub fn euclidean_distance_sq(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - yi).powi(2))
        .sum()
}

pub fn gaussian_kernel(x: &Vec<f32>, center: &Vec<f32>, gamma: f32) -> f32 {
    let dist_sq = euclidean_distance_sq(x, center);
    (-gamma * dist_sq).exp()
}


pub fn xtx(x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
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

pub fn xty(x: &Vec<Vec<f32>>, y: &Vec<f32>) -> Vec<f32> {
    let cols = x[0].len();
    let mut result = vec![0.0; cols];

    for i in 0..cols {
        for j in 0..x.len() {
            result[i] += x[j][i] * y[j];
        }
    }

    result
}

pub fn inverse(matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let n = matrix.len();
    let mut result = vec![vec![0.0; n]; n];
    let mut a = matrix.clone();

    for i in 0..n {
        result[i][i] = 1.0;
    }

    for i in 0..n {
        let pivot = a[i][i];
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

pub fn multiply_matrix_vector(matrix: &Vec<Vec<f32>>, vector: &Vec<f32>) -> Vec<f32> {
    let mut result = vec![0.0; matrix.len()];

    for i in 0..matrix.len() {
        for j in 0..vector.len() {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    result
}
