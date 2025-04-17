use crate::activation_function;
use crate::data_converter;
use crate::data_manager;
use crate::loss;
use crate::matrix;
use rand::Rng;
use std::process::Output;

pub fn init_weights(dim: i32) -> Vec<f32> {
    let mut w = vec![];
    for _ in 0..dim {
        w.push(rand::thread_rng().gen_range(-1..1) as f32);
    }
    w
}

pub fn forward_propagation(
    all_layers: Vec<Vec<Vec<f32>>>,
    data: Vec<f32>,
    is_classification: bool,
) -> Vec<Vec<f32>> {
    let mut results_layer: Vec<Vec<f32>> = vec![];
    for layers_index in 0..all_layers.len() {
        let mut result_neural = vec![];
        for neural_index_in_layers in 0..all_layers[layers_index].len() {
            let mut result = 0.;
            if layers_index == 0 {
                result = matrix::sum(
                    data.clone(),
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
            if (layers_index == all_layers.len() - 1) {
                if (is_classification) {
                    result_neural.push(activation_function::sigmoid(result));
                } else {
                    result_neural.push(result);
                }
            } else {
                result_neural.push(activation_function::sigmoid(result));
            }
        }
        results_layer.push(result_neural);
    }
    results_layer
}

pub fn back_propagation(
    output: Vec<f32>,
    all_layers: Vec<Vec<Vec<f32>>>,
    result_fastforward: Vec<Vec<f32>>,
    input_data: Vec<f32>,
    learning_rage: f32,
) -> Vec<Vec<Vec<f32>>> {
    let mut updated_layers = all_layers.clone();

    for layers_index in (0..all_layers.len()).rev() {
        for neural_index in 0..all_layers[layers_index].len() {
            if layers_index == all_layers.len() - 1 {
                let mut o = vec![];
                for i in 0..result_fastforward[layers_index - 1].len() {
                    o.push(result_fastforward[layers_index - 1][i]);
                }
                updated_layers[layers_index][neural_index] = update_weight(
                    updated_layers[layers_index][neural_index].clone(),
                    output.clone(),
                    o,
                    result_fastforward[layers_index][neural_index],
                    learning_rage,
                );
            } else if layers_index == 0 {
                let mut a = vec![];
                for i in 0..result_fastforward[layers_index + 1].len() {
                    a.push(result_fastforward[layers_index + 1][i]);
                }
                updated_layers[layers_index][neural_index] = update_weight(
                    updated_layers[layers_index][neural_index].clone(),
                    a,
                    input_data.clone(),
                    result_fastforward[layers_index][neural_index],
                    learning_rage,
                );
            } else {
                let mut a = vec![];
                let mut o = vec![];
                for i in 0..result_fastforward[layers_index + 1].len() {
                    a.push(result_fastforward[layers_index + 1][i]);
                }
                for i in 0..result_fastforward[layers_index - 1].len() {
                    o.push(result_fastforward[layers_index - 1][i]);
                }
                updated_layers[layers_index][neural_index] = update_weight(
                    updated_layers[layers_index][neural_index].clone(),
                    a,
                    o,
                    result_fastforward[layers_index][neural_index],
                    learning_rage,
                );
            }
        }
    }

    updated_layers
}

pub fn update_weight(
    w: Vec<f32>,
    y: Vec<f32>,
    values_before_w: Vec<f32>,
    result_w_x_values_before_w: f32,
    learning_rate: f32,
) -> Vec<f32> {
    let mut updated_w = w.clone();
    for i in 0..w.len() {
        for j in 0..y.len() {
            let gradient = (result_w_x_values_before_w - y[j]) * values_before_w[i];
            updated_w[i] -= learning_rate * gradient;
        }
    }

    updated_w
}

pub fn predict(data: Vec<f32>, is_classification: bool) -> f32 {
    let all_layers = data_converter::load_weights_mlp();
    let mut results_layer = forward_propagation(all_layers.clone(), data, is_classification);
    println!("{:?}", results_layer);
    if !is_classification {
        if results_layer.last().unwrap().len() != 0 {
            return *results_layer.last().unwrap().last().unwrap();
        }
        return -1.;
    }
    let mut index_good_neural = 0;
    let mut max_result_neural = 0.;
    for index_neural in 0..results_layer.last().unwrap().len() {
        if results_layer.last().unwrap()[index_neural] > max_result_neural {
            max_result_neural = results_layer.last().unwrap()[index_neural];
            index_good_neural = index_neural as i32;
        }
    }
    index_good_neural as f32
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

pub fn training(
    dataset_input: Vec<Vec<Vec<f32>>>,
    output_dataset: Vec<f32>,
    nb_epoch: i32,
    hidden_layers: Vec<i32>,
    training_name: String,
    is_classification: bool,
    verbose: bool,
    learning_rate: f32,
) {
    let mut layers: Vec<i32> = vec![(dataset_input[0][0].len() * dataset_input[0][0].len()) as i32];
    for i in 0..hidden_layers.len() {
        layers.push(hidden_layers[i]);
    }
    layers.push(dataset_input.len() as i32);
    let mut all_layers = init_layers(layers);
    for epoch in 0..nb_epoch {
        if (verbose) {
            println!("Epoch : {} / {}", epoch + 1, nb_epoch);
        }
        let mut mse = 0.;
        let mut accuracy = 0.;
        for index_cat in 0..dataset_input.len() {
            let mut need_result_output_neural = vec![];
            for i in 0..dataset_input.len() {
                if index_cat == i {
                    need_result_output_neural.push(1.);
                } else {
                    need_result_output_neural.push(0.);
                }
            }
            let random_index: usize =
                rand::thread_rng().gen_range(0..dataset_input[index_cat].len()) as usize;
            for index_data in random_index..random_index + 1 {
                let result_layers = forward_propagation(
                    all_layers.clone(),
                    dataset_input[index_cat][index_data].clone(),
                    is_classification,
                );
                mse = loss::mse(
                    result_layers.last().unwrap().clone(),
                    need_result_output_neural.clone(),
                );
                all_layers = back_propagation(
                    if is_classification {
                        need_result_output_neural.clone()
                    } else {
                        output_dataset.clone()
                    },
                    all_layers,
                    result_layers,
                    dataset_input[index_cat][index_data].clone(),
                    learning_rate,
                );
            }
        }
        data_manager::add_text_to_file(training_name.clone(), mse.to_string() + "\n")
            .expect("Error: error during write train data");
    }
    data_converter::export_weights_mlp(all_layers.clone());
}
