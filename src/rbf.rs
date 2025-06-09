use crate::activation_function;
use crate::data_converter;
use crate::database;
use crate::loss;
use crate::matrix;
use rand::Rng;

pub fn init_centers_and_sigmas(nb_neurons: usize, input_dim: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut centers = vec![];
    let mut sigmas = vec![];

    for _ in 0..nb_neurons {
        let mut center = vec![];
        for _ in 0..input_dim {
            center.push(rand::thread_rng().gen_range(-1.0..1.0));
        }
        centers.push(center);
        sigmas.push(rand::thread_rng().gen_range(0.1..1.0));
    }

    (centers, sigmas)
}

pub fn init_output_weights(nb_outputs: usize, nb_hidden: usize) -> Vec<Vec<f32>> {
    (0..nb_outputs)
        .map(|_| {
            (0..=nb_hidden) // +1 pour biais
                .map(|_| rand::thread_rng().gen_range(-1.0..1.0))
                .collect()
        })
        .collect()
}

pub fn forward_rbf(
    input: Vec<f32>,
    centers: &Vec<Vec<f32>>,
    sigmas: &Vec<f32>,
    output_weights: &Vec<Vec<f32>>,
    is_classification: bool,
) -> Vec<f32> {
    let mut hidden_activations = vec![];

    for (i, center) in centers.iter().enumerate() {
        let mut dist = 0.0;
        for j in 0..input.len() {
            dist += (input[j] - center[j]).powi(2);
        }
        let r = (-(dist) / (2.0 * sigmas[i].powi(2))).exp();
        hidden_activations.push(r);
    }

    // Ajout biais
    let mut activations_with_bias = hidden_activations.clone();
    activations_with_bias.push(1.0);

    let mut outputs = vec![];
    for neuron_weights in output_weights.iter() {
        let sum = matrix::sum(activations_with_bias.clone(), neuron_weights.clone());
        let activated = if is_classification {
            activation_function::sigmoid(sum)
        } else {
            sum
        };
        outputs.push(activated);
    }

    outputs
}

pub fn train_rbf(
    dataset_input: Vec<Vec<Vec<f32>>>,
    dataset_validation: Vec<Vec<Vec<f32>>>,
    nb_epoch: i32,
    nb_hidden: usize,
    training_name: String,
    is_classification: bool,
    verbose: bool,
    save_in_db: bool,
    learning_rate: f32,
    nb_epoch_to_save: i32,
) {
    let input_dim = dataset_input[0][0].len();
    let nb_outputs = dataset_input.len();

    let (centers, sigmas) = init_centers_and_sigmas(nb_hidden, input_dim);
    let mut output_weights = init_output_weights(nb_outputs, nb_hidden);

    for epoch in 0..nb_epoch {
        if verbose {
            println!("Epoch : {} / {}", epoch + 1, nb_epoch);
        }

        let class_idx = rand::thread_rng().gen_range(0..dataset_input.len());
        let data_idx = rand::thread_rng().gen_range(0..dataset_input[class_idx].len());
        let input = dataset_input[class_idx][data_idx].clone();

        let mut expected_output = vec![0.0; nb_outputs];
        expected_output[class_idx] = 1.0;

        let predicted = forward_rbf(
            input.clone(),
            &centers,
            &sigmas,
            &output_weights,
            is_classification,
        );

        let mut errors: Vec<f32> = vec![0.0; nb_outputs];
        for i in 0..nb_outputs {
            errors[i] = predicted[i] - expected_output[i];
            if is_classification {
                errors[i] *= predicted[i] * (1.0 - predicted[i]);
            }
        }

        let mut hidden_activations = vec![];
        for (i, center) in centers.iter().enumerate() {
            let mut dist = 0.0;
            for j in 0..input.len() {
                dist += (input[j] - center[j]).powi(2);
            }
            hidden_activations.push((-(dist) / (2.0 * sigmas[i].powi(2))).exp());
        }

        let mut hidden_with_bias = hidden_activations.clone();
        hidden_with_bias.push(1.0);

        for i in 0..nb_outputs {
            for j in 0..output_weights[i].len() {
                output_weights[i][j] -= learning_rate * errors[i] * hidden_with_bias[j];
            }
        }

        if save_in_db && epoch % nb_epoch_to_save == 0 {
            let mut accuracy = 0.0;
            if is_classification && !dataset_validation.is_empty() {
                accuracy = compute_accuracy_score_rbf(
                    dataset_validation.clone(),
                    &centers,
                    &sigmas,
                    &output_weights,
                );
            }
            let mse = loss::mse(predicted, expected_output);
            database::insert_training_score(training_name.clone(), mse, accuracy, epoch)
                .expect("Error saving DB");
        }
    }
    data_converter::export_weights_rbf(&centers, &sigmas, &output_weights);
}

pub fn predict_rbf(data: Vec<f32>, is_classification: bool, verbose: bool) -> i32 {
    let (centers, sigmas, output_weights) = data_converter::load_weights_rbf();
    predict(
        data,
        &centers,
        &sigmas,
        &output_weights,
        is_classification,
        verbose,
    )
}

fn predict(
    data: Vec<f32>,
    centers: &Vec<Vec<f32>>,
    sigmas: &Vec<f32>,
    output_weights: &Vec<Vec<f32>>,
    is_classification: bool,
    verbose: bool,
) -> i32 {
    let result = forward_rbf(data, centers, sigmas, output_weights, is_classification);

    if verbose {
        println!("{:?}", result);
    }

    if !is_classification {
        return result[0] as i32;
    }

    let mut max = result[0];
    let mut index = 0;
    for i in 1..result.len() {
        if result[i] > max {
            max = result[i];
            index = i;
        }
    }

    index as i32
}

pub fn compute_accuracy_score_rbf(
    dataset_validation: Vec<Vec<Vec<f32>>>,
    centers: &Vec<Vec<f32>>,
    sigmas: &Vec<f32>,
    output_weights: &Vec<Vec<f32>>,
) -> f32 {
    let mut correct = 0;
    let mut total = 0;

    for class_idx in 0..dataset_validation.len() {
        for input in &dataset_validation[class_idx] {
            let pred = predict(input.clone(), centers, sigmas, output_weights, true, false);
            if pred == class_idx as i32 {
                correct += 1;
            }
            total += 1;
        }
    }

    println!("{} / {}", correct, total);
    (correct as f32 * 100.0 / total as f32)
}
