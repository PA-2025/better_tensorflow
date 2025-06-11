pub fn euclidean_distance_sq(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (xi - yi).powi(2))
        .sum()
}

pub fn gaussian_kernel(x: &Vec<f32>, center: &Vec<f32>, gamma: f32) -> f32 {
    let dist_sq = euclidean_distance_sq(x, center);
    (-gamma * dist_sq).exp()
}