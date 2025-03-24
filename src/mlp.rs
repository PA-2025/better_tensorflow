use crate::activation_function;
use crate::data_converter;
use crate::loss;
use crate::matrix;
use rand::Rng;
use std::process::Output;

pub fn init_weights(dim: i32) -> Vec<f32> {
    let mut w = vec![];
    for _ in 0..dim {
        w.push(rand::thread_rng().gen_range(0..100) as f32);
    }
    w
}

pub fn forward_propagation(all_layers: Vec<Vec<Vec<f32>>>, matrix: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut results_layer: Vec<Vec<f32>> = vec![];
    for layers_index in 0..all_layers.len() {
        let mut result_neural = vec![];
        for neural_index_in_layers in 0..all_layers[layers_index].len() {
            let mut result = 0.;
            if layers_index == 0 {
                result = matrix::sum(
                    matrix::matrix_to_array(matrix.clone()),
                    all_layers[layers_index][neural_index_in_layers].clone(),
                );
            } else {
                result = matrix::sum(
                    results_layer
                        .last()
                        .unwrap()
                        .iter()
                        .map(|&x| x as f32)
                        .collect::<Vec<f32>>(),
                    all_layers[layers_index][neural_index_in_layers].clone(),
                );
            }
            result_neural.push(activation_function::sigmoid(result));
        }
        results_layer.push(result_neural);
    }
    results_layer
}

pub fn back_propagation(
    output: Vec<f32>,
    all_layers: Vec<Vec<Vec<f32>>>,
    result_fastforward: Vec<Vec<f32>>,
) -> Vec<Vec<Vec<f32>>> {
    let mut updated_layers = all_layers.clone();
    let learning_rate = 0.01;


    updated_layers
}
pub fn predict(matrix: Vec<Vec<f32>>) -> i32 {
    let all_layers = data_converter::load_weights_mlp();
    let mut results_layer = forward_propagation(all_layers.clone(), matrix);
    println!("{:?}", results_layer);
    let mut index_good_neural: i32 = 0;
    let mut max_result_neural = 0.;
    for index_neural in 0..results_layer.last().unwrap().len() {
        if results_layer.last().unwrap()[index_neural] > max_result_neural {
            max_result_neural = results_layer.last().unwrap()[index_neural];
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

pub fn training(dataset: Vec<Vec<Vec<Vec<f32>>>>, nb_epoch: i32, hidden_layers: Vec<i32>) {
    let mut layers: Vec<i32> = vec![(dataset[0][0].len() * dataset[0][0][0].len()) as i32];
    for i in 0..hidden_layers.len() {
        layers.push(hidden_layers[i]);
    }
    layers.push(dataset.len() as i32);
    let mut all_layers = init_layers(layers);
    for epoch in 0..nb_epoch {
        println!("Epoch : {} / {}", epoch + 1, nb_epoch);
        for index_cat in 0..dataset.len() {
            let mut need_result_output_neural = vec![];
            for i in 0..dataset.len() {
                if index_cat == i {
                    need_result_output_neural.push(1.);
                } else {
                    need_result_output_neural.push(0.);
                }
            }
            for index_data in 0..dataset[index_cat].len() {
                let result_layers =
                    forward_propagation(all_layers.clone(), dataset[index_cat][index_data].clone());
                let mse = loss::mse(
                    result_layers.last().unwrap().clone(),
                    need_result_output_neural.clone(),
                );
                all_layers =
                    back_propagation(need_result_output_neural.clone(), all_layers, result_layers);
            }
        }
    }
    data_converter::export_weights_mlp(all_layers.clone());
}
