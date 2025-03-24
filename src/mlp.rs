use crate::activation_function;
use crate::data_converter;
use rand::Rng;

pub fn init_weights(dim: i32) -> Vec<f32> {
    let mut w = vec![];
    for _ in 0..dim {
        w.push(rand::thread_rng().gen_range(0..100) as f32);
    }
    w
}

pub fn sum(input_array: Vec<f32>, weights_array: Vec<f32>) -> f32 {
    let mut result: f32 = 0.;
    for i in 0..input_array.len() {
        result += input_array[i] * weights_array[i];
    }
    result
}

pub fn matrix_to_array(matrix: Vec<Vec<f32>>) -> Vec<f32> {
    let mut array = vec![];
    for i in 0..matrix.len() {
        for j in 0..matrix[i].len() {
            array.push(matrix[i][j]);
        }
    }
    array
}

pub fn predict(matrix: Vec<Vec<f32>>) -> i32 {
    let layers: Vec<i32> = vec![(matrix.len() * matrix[0].len()) as i32, 4, 5, 7, 3];
    let all_layers = init_layers(layers);
    // let all_layers = data_converter::load_weights_mlp();
    println!("{}",all_layers.len());
    println!("{}",all_layers[0].len());
    println!("{}",all_layers[0][0].len());
    let mut results_layer: Vec<Vec<i32>> = vec![];
    for layers_index in 0..all_layers.len() {
        let mut result_neural = vec![];
        for neural_index_in_layers in 0..all_layers[layers_index].len() {
            let mut result = 0.;
            if layers_index == 0 {
                result = sum(
                    matrix_to_array(matrix.clone()),
                    all_layers[layers_index][neural_index_in_layers].clone(),
                );
            } else {
                result = sum(
                    results_layer.last().unwrap().iter().map(|&x| x as f32).collect::<Vec<f32>>(),
                    all_layers[layers_index][neural_index_in_layers].clone(),
                );
            }
            result_neural.push(activation_function::sigmoid(result));
        }
        results_layer.push(result_neural);
    }
    data_converter::export_weights_mlp(all_layers);
    let mut index_good_neural: i32 = 0;
    for index_neural in 0..results_layer.last().unwrap().len() {
        if results_layer.last().unwrap()[index_neural] == 1 {
            index_good_neural = index_neural as i32;
        }
    }
    index_good_neural
}

pub fn init_layers(nb_layers: Vec<i32>) -> Vec<Vec<Vec<f32>>> {
    let mut all_random_weights = vec![];
    for nb_layer_index in 1..nb_layers.len() {
        let mut layers = vec![];
        for _ in 0..nb_layers[nb_layer_index] {
            layers.push(init_weights(nb_layers[nb_layer_index - 1]));
        }
        all_random_weights.push(layers)
    }
    all_random_weights
}
