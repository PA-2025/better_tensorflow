pub fn sigmoid(x: f32) -> i32 {
    (1.0 / (1.0 + (-x).exp())) as i32
}
