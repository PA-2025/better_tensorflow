use crate::math;
use crate::matrix;
use rand::Rng;



pub fn kmeans(dataset: Vec<Vec<Vec<f32>>>, number_clusters: usize, max_iter: usize) -> Vec<Vec<f32>> {
    let array_dataset = matrix::matrix_dataset_to_array(dataset);
    let mut centers = vec![];
    let mut labels = vec![0; array_dataset.len()];

    for _ in 0..number_clusters {
        let random_index = rand::thread_rng().gen_range(0..array_dataset.len());
        centers.push(array_dataset[random_index].0.clone());
    }

    let mut converged = false;
    let mut iter = 0;

    while !converged && iter < max_iter {
        converged = true;

        for (i, (data_point, _)) in array_dataset.iter().enumerate() {
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

        let mut new_centers = vec![];
        for _ in 0..number_clusters {
            new_centers.push(vec![0.0; array_dataset[0].0.len()]);
        }
        let mut count = vec![0; number_clusters];

        for (i, (data_point, label)) in array_dataset.iter().enumerate() {
            for j in 0..data_point.len() {
                new_centers[*label][j] += data_point[j];
            }
            count[*label] += 1;
        }

        for i in 0..number_clusters {
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
