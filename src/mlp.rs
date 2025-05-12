use crate::activation_function;
use crate::data_converter;
use crate::data_manager;
use crate::database;
use crate::loss;
use crate::matrix;
use rand::Rng;

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
            let result;
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
    is_classification: bool,
    output: Vec<f32>,
    all_layers: Vec<Vec<Vec<f32>>>,
    result_fastforward: Vec<Vec<f32>>,
    delta: &mut Vec<Vec<f32>>,
    input_data: Vec<f32>,
    learning_rage: f32,
) -> Vec<Vec<Vec<f32>>> {
    let mut updated_layers = all_layers.clone();

    for j in 0..all_layers.last().unwrap().len() {
        delta[all_layers.len() - 1][j] = result_fastforward.last().unwrap()[j] - output[j];
        if is_classification {
            delta[all_layers.len() - 1][j] *=
                1. - result_fastforward[all_layers.len() - 1][j].powf(2.);
        }
    }

    for layers_index in (1..all_layers.len()).rev() {
        for back_neural_index in 0..all_layers[layers_index - 1].len() {
            let mut total = 0.;
            for actual_neural_index in 0..all_layers[layers_index].len() {
                total += all_layers[layers_index][actual_neural_index][back_neural_index]
                    * delta[layers_index][actual_neural_index];
            }
            total *= 1. - result_fastforward[layers_index - 1][back_neural_index].powf(2.);
            delta[layers_index - 1][back_neural_index] = total;
        }
    }
    for layer_index in 0..all_layers.len() {
        for neural_index in 0..all_layers[layer_index].len() {
            for weight_index in 0..all_layers[layer_index][neural_index].len() {
                updated_layers[layer_index][neural_index][weight_index] -= learning_rage
                    * delta[layer_index][neural_index]
                    * if layer_index == 0 {
                        input_data[weight_index]
                    } else {
                        result_fastforward[layer_index - 1][weight_index]
                    };
            }
        }
    }

    updated_layers
}

pub fn predict(data: Vec<f32>, mut all_layers: Vec<Vec<Vec<f32>>>, is_classification: bool, verbose: bool) -> i32 {
    if all_layers.len() == 0 {
        all_layers = data_converter::load_weights_mlp();
    }
    let results_layer = forward_propagation(all_layers.clone(), data, is_classification);
    if verbose {
        println!("{:?}", results_layer);
    }
    if !is_classification {
        if results_layer.last().unwrap().len() != 0 {
            return *results_layer.last().unwrap().last().unwrap() as i32;
        }
        return -1;
    }
    let mut index_good_neural = 0;
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

pub fn compute_accuracy_score(
    dataset_validation: Vec<Vec<Vec<f32>>>,
    all_layers: Vec<Vec<Vec<f32>>>,
) -> f32 {
    let mut score = 0;
    let mut total = 0;
    for index_cat in 0..dataset_validation.len() {
        for index_data in 0..dataset_validation[index_cat].len() {
            if predict(
                dataset_validation[index_cat][index_data].clone(),
                all_layers.clone(),
                true,
                false
            ) == index_cat as i32
            {
                score += 1
            }
            total += 1;
        }
    }
    (score * 100 / total ) as f32
}

pub fn training(
    dataset_input: Vec<Vec<Vec<f32>>>,
    dataset_validation: Vec<Vec<Vec<f32>>>,
    output_dataset: Vec<f32>,
    nb_epoch: i32,
    hidden_layers: Vec<i32>,
    training_name: String,
    is_classification: bool,
    verbose: bool,
    save_in_db: bool,
    learning_rate: f32,
) {
    let mut layers: Vec<i32> = vec![dataset_input[0][0].len() as i32];
    for i in 0..hidden_layers.len() {
        layers.push(hidden_layers[i]);
    }
    layers.push(dataset_input.len() as i32);
    let mut all_layers = init_layers(layers);

    let mut delta: Vec<Vec<f32>> = vec![];

    for i in 0..all_layers.len() {
        delta.push(vec![]);
        for j in 0..all_layers[i].len() {
            delta[i].push(0.);
        }
    }

    for epoch in 0..nb_epoch {
        if (verbose) {
            println!("Epoch : {} / {}", epoch + 1, nb_epoch);
        }
        let index_cat = rand::thread_rng().gen_range(0..dataset_input.len());
        let mut need_result_output_neural = vec![];
        for i in 0..dataset_input.len() {
            if index_cat == i {
                need_result_output_neural.push(1.);
            } else {
                need_result_output_neural.push(0.);
            }
        }
        let index_data: usize = rand::thread_rng().gen_range(0..dataset_input[index_cat].len());

        let result_layers = forward_propagation(
            all_layers.clone(),
            dataset_input[index_cat][index_data].clone(),
            is_classification,
        );

        all_layers = back_propagation(
            is_classification,
            if is_classification {
                need_result_output_neural.clone()
            } else {
                output_dataset.clone()
            },
            all_layers,
            result_layers.clone(),
            &mut delta,
            dataset_input[index_cat][index_data].clone(),
            learning_rate,
        );

        if save_in_db {
            let mut accuracy = 0.;
            if dataset_validation.len() != 0 && is_classification {
                accuracy = compute_accuracy_score(dataset_validation.clone(), all_layers.clone());
            }
            let mse = loss::mse(
                result_layers.last().unwrap().clone(),
                need_result_output_neural.clone(),
            );

            database::insert_training_score(training_name.clone(), mse, accuracy, epoch)
                .expect("Error during save record");
        }

        /*data_manager::add_text_to_file(training_name.clone(), mse.to_string() + "\n")
            .expect("Error: error during write train data");
        data_manager::add_text_to_file(training_name.clone() + "_acc", accuracy.to_string() + "\n")
            .expect("Error: error during write train data");*/
    }
    data_converter::export_weights_mlp(all_layers.clone());
}
