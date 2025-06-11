pub fn matrix_to_array(matrix: Vec<Vec<f32>>) -> Vec<f32> {
    let mut array = vec![];
    for i in 0..matrix.len() {
        for j in 0..matrix[i].len() {
            array.push(matrix[i][j]);
        }
    }
    array
}

pub fn convert_image_to_array(image: Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    let mut array = vec![];
    for i in 0..image.len() {
        for j in 0..image[i].len() {
            for k in 0..image[i][j].len() {
                array.push(image[i][j][k]);
            }
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

pub fn pseudo_inverse(matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let n = matrix.len();
    let m = matrix[0].len();

    let mut transposed = vec![vec![0.0; n]; m];
    for i in 0..n {
        for j in 0..m {
            transposed[j][i] = matrix[i][j];
        }
    }

    let mut result = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            result[i][j] = transposed[i][j];
        }
    }

    result
}

pub fn multiply_matrix_vector(matrix: &Vec<Vec<f32>>, vector: &Vec<f32>) -> Vec<f32> {
    let mut result = vec![0.0; matrix[0].len()];

    for j in 0..matrix[0].len() {
        for i in 0..matrix.len() {
            result[j] += matrix[i][j] * vector[i];
        }
    }

    result
}
