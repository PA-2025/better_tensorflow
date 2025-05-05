pub fn matrix_to_array(matrix: Vec<Vec<f32>>) -> Vec<f32> {
    let mut array = vec![];
    for i in 0..matrix.len() {
        for j in 0..matrix[i].len() {
            array.push(matrix[i][j]);
        }
    }
    array
}

pub fn sum(input_array: Vec<f32>, weights_array: Vec<f32>) -> f32 {
    let mut result: f32 = 0.;
    for i in 0..input_array.len() {
        result += input_array[i] * weights_array[i];
    }
    result
}