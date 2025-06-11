use crate::math;
use rand::Rng;

fn flatten_dataset(dataset: Vec<Vec<Vec<f32>>>) -> Vec<(Vec<f32>, usize)> {
    let mut flattened_data = Vec::new();

    for (class_idx, class_data) in dataset.iter().enumerate() {
        for data_point in class_data {
            flattened_data.push((data_point.clone(), class_idx));
        }
    }

    flattened_data
}

pub fn kmeans(dataset: Vec<Vec<Vec<f32>>>, num_centers: usize, max_iter: usize) -> Vec<Vec<f32>> {
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
                let dist = math::euclidean_distance_sq(data_point, center);
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
