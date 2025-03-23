use crate::activation_function;
use rand::Rng;

pub fn init_weights(width: usize, height: usize) -> Vec<Vec<f32>> {
    let mut matrix = vec![];
    for _ in 0..width {
        let mut row = vec![];
        for _ in 0..height {
            row.push(rand::thread_rng().gen_range(0..100) as f32);
        }
        matrix.push(row);
    }
    matrix
}

pub fn sum(matrix_input: Vec<Vec<f32>>, matrix_k: Vec<Vec<f32>>) -> f32 {
    let mut result: f32 = 0.;
    for i in 0..matrix_input.len() {
        for j in 0..matrix_input[i].len() {
            result += matrix_input[i][j] * matrix_k[i][j];
        }
    }
    result
}

pub fn predict(matrix: Vec<Vec<f32>>) -> i32 {
    let matrix_k: Vec<Vec<f32>> = init_weights(matrix.len(), matrix[0].len());
    let sum: f32 = sum(matrix, matrix_k);
    activation_function::sigmoid(sum)
}
