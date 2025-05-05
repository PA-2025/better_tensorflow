pub fn sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-x).exp()))
}

pub fn tanh(x: f32) -> f32 {
    (2. / 1. + (-2. * x).exp()) - 1.
}