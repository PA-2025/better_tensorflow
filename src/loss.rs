pub fn mse(neural_output_array: Vec<f32>,dataset_output_array: Vec<f32>) -> f32  {
    let mut result = 0.;
    for i in 0..neural_output_array.len() {
        result += f32::powi(neural_output_array[i] - dataset_output_array[i],2);
    }
    result
}