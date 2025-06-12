use crate::{data_converter, database, kmeans, math, matrix};

pub fn forward_propagation_rbf(
    input: Vec<f32>,
    centers: Vec<Vec<f32>>,
    weights: Vec<f32>,
    gamma: f32,
    is_classification: bool,
) -> f32 {
    let result: f32 = centers
        .iter()
        .zip(weights.iter())
        .map(|(center, w)| w * math::gaussian_kernel(&input, center, gamma))
        .sum();

    if is_classification {
        if result >= 0.0 {
            1.0
        } else {
            -1.0
        }
    } else {
        result
    }
}

pub fn train_rbf(
    dataset_input: Vec<Vec<Vec<f32>>>,
    dataset_validation: Vec<Vec<Vec<f32>>>,
    output_dataset: Vec<f32>,
    gamma: f32,
    is_classification: bool,
    save_in_db: bool,
    training_name: String,
) -> f32 {
    let num_centers: usize = dataset_input.iter().map(|v| v.len()).sum();
    let centers = kmeans::kmeans(dataset_input.clone(), num_centers, 100);

    let mut matrix = Vec::new();
    let mut target_vector = Vec::new();
    let mut global_index = 0;

    for (index_cat, category_data) in dataset_input.iter().enumerate() {
        for input_data in category_data {
            let row: Vec<f32> = centers
                .iter()
                .map(|center| math::gaussian_kernel(input_data, center, gamma))
                .collect();
            matrix.push(row);

            let target = if is_classification {
                if index_cat == 0 { 1.0 } else { -1.0 }
            } else {
                output_dataset[global_index]
            };
            target_vector.push(target);
            global_index += 1;
        }
    }

    let reg_pinv = matrix::regularized_pseudo_inverse(&matrix, 0.1);
    let weights = matrix::multiply_matrix_vector(&reg_pinv, &target_vector);

    let accuracy = compute_accuracy_score(
        &dataset_validation,
        &centers,
        &weights,
        gamma,
        is_classification,
    );

    if save_in_db {
        database::insert_training_score(training_name, 0., accuracy, 0)
            .expect("Error during save record");
    }

    data_converter::export_weights_rbf(&centers, &weights);
    accuracy
}

pub fn compute_accuracy_score(
    dataset_validation: &Vec<Vec<Vec<f32>>>,
    centers: &Vec<Vec<f32>>,
    weights: &Vec<f32>,
    gamma: f32,
    is_classification: bool,
) -> f32 {
    let mut score = 0;
    let mut total = 0;

    for (index_cat, category_data) in dataset_validation.iter().enumerate() {
        for input_data in category_data {
            let prediction = forward_propagation_rbf(
                input_data.clone(),
                centers.clone(),
                weights.clone(),
                gamma,
                is_classification,
            );

            let correct = if is_classification {
                prediction == if index_cat == 0 { 1.0 } else { -1.0 }
            } else {
                let true_value = input_data.last().unwrap();
                (prediction - true_value).abs() < 1e-2
            };

            if correct {
                score += 1;
            }
            total += 1;
        }
    }

    (score as f32 / total as f32) * 100.0
}

pub fn predict_rbf(input_data: Vec<f32>, gamma: f32, is_classification: bool) -> f32 {
    let (centers, weights) = data_converter::load_weights_rbf();
    forward_propagation_rbf(input_data, centers, weights, gamma, is_classification)
}
