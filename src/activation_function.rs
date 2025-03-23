pub fn sigmoid(x: f32) -> i32 {
    (1.0 / (1.0 + (-x).exp())) as i32
}

pub fn relu(x: i32) -> i32 {
    x * (x > 0) as i32
}