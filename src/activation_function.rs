pub fn sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-x).exp()))
}

pub fn sigmoid_derivative(x: f32) -> f32 {
    sigmoid(x) * (1. - sigmoid(x))
}