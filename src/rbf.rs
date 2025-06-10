use crate::{data_converter, database, loss};
use rand::Rng;

fn euclidean_distance_sq(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - yi).powi(2))
        .sum()
}

fn flatten_dataset(dataset: Vec<Vec<Vec<f32>>>) -> Vec<(Vec<f32>, usize)> {
    let mut flattened_data = Vec::new();

    for (class_idx, class_data) in dataset.iter().enumerate() {
        for data_point in class_data {
            flattened_data.push((data_point.clone(), class_idx));
        }
    }

    flattened_data
}

pub fn lloyd_algorithm(
    dataset: Vec<Vec<Vec<f32>>>,
    num_centers: usize,
    max_iter: usize,
) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let flattened_data = flatten_dataset(dataset);
    let mut centers = Vec::new();
    let mut labels = vec![0; flattened_data.len()];

    for _ in 0..num_centers {
        let random_index = rng.gen_range(0..flattened_data.len());
        centers.push(flattened_data[random_index].0.clone());
    }

    let mut converged = false;
    let mut iter = 0;

    while !converged && iter < max_iter {
        converged = true;

        for (i, (data_point, _)) in flattened_data.iter().enumerate() {
            let mut min_dist = f32::MAX;
            let mut closest_center = 0;

            for (j, center) in centers.iter().enumerate() {
                let dist = euclidean_distance_sq(data_point, center);
                if dist < min_dist {
                    min_dist = dist;
                    closest_center = j;
                }
            }

            if labels[i] != closest_center {
                labels[i] = closest_center;
                converged = false;
            }
        }

        let mut new_centers = vec![vec![0.0; flattened_data[0].0.len()]; num_centers];
        let mut count = vec![0; num_centers];

        for (i, (data_point, label)) in flattened_data.iter().enumerate() {
            for j in 0..data_point.len() {
                new_centers[*label][j] += data_point[j];
            }
            count[*label] += 1;
        }

        for i in 0..num_centers {
            if count[i] > 0 {
                for j in 0..new_centers[i].len() {
                    new_centers[i][j] /= count[i] as f32;
                }
            }
        }

        centers = new_centers;
        iter += 1;
    }

    centers
}
pub fn gaussian_kernel(x: &Vec<f32>, center: &Vec<f32>, gamma: f32) -> f32 {
    let dist_sq = euclidean_distance_sq(x, center);
    (-gamma * dist_sq).exp()
}

pub fn forward_propagation_rbf(
    dataset: Vec<f32>,
    centers: Vec<Vec<f32>>,
    weights: Vec<f32>,
    gamma: f32,
    is_classification: bool,
) -> f32 {
    let mut result = 0.0;

    for (i, center) in centers.iter().enumerate() {
        let activation = gaussian_kernel(&dataset, center, gamma);
        result += weights[i] * activation;
    }

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
    let num_centers = dataset_input.len();
    let centers = lloyd_algorithm(dataset_input.clone(), num_centers, 100);

    let mut matrix = Vec::new();
    let mut target_vector = Vec::new();
    let mut global_index = 0;

    for (index_cat, category_data) in dataset_input.iter().enumerate() {
        for input_data in category_data {
            let row: Vec<f32> = centers
                .iter()
                .map(|center| gaussian_kernel(input_data, center, gamma))
                .collect();
            matrix.push(row);

            let target = if is_classification {
                if index_cat == 0 {
                    1.0
                } else {
                    -1.0
                }
            } else {
                output_dataset[global_index]
            };
            target_vector.push(target);
            global_index += 1;
        }
    }

    let pseudo_inv = pseudo_inverse(&matrix);
    let weights = multiply_matrix_vector(&pseudo_inv, &target_vector);

    let accuracy = compute_accuracy_score(
        dataset_validation.clone(),
        centers.clone(),
        weights.clone(),
        gamma,
        is_classification,
    );

    if save_in_db {
        database::insert_training_score(training_name.clone(), 0., accuracy, 0)
            .expect("Error during save record");
    }

    data_converter::export_weights_rbf(&centers, &weights);
    accuracy
}

pub fn compute_accuracy_score(
    dataset_validation: Vec<Vec<Vec<f32>>>,
    centers: Vec<Vec<f32>>,
    weights: Vec<f32>,
    gamma: f32,
    is_classification: bool,
) -> f32 {
    let mut score = 0;
    let mut total = 0;
    let mut global_index = 0;

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
            global_index += 1;
        }
    }

    (score as f32 / total as f32) * 100.0
}

pub fn predict_rbf(input_data: Vec<f32>, gamma: f32, is_classification: bool) -> f32 {
    let (centers, weights) = data_converter::load_weights_rbf();

    let output = forward_propagation_rbf(input_data, centers, weights, gamma, is_classification);

    output
}

fn pseudo_inverse(matrix: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
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

fn multiply_matrix_vector(matrix: &Vec<Vec<f32>>, vector: &Vec<f32>) -> Vec<f32> {
    let mut result = vec![0.0; matrix[0].len()];

    for j in 0..matrix[0].len() {
        for i in 0..matrix.len() {
            result[j] += matrix[i][j] * vector[i];
        }
    }

    result
}
