use std::fs::File;
use std::io::{BufRead, BufWriter};
use std::io::Write;
use crate::data_manager;

pub fn export_weights_mlp(data: Vec<Vec<Vec<f32>>>) {
    let mut result_str = String::from("");
    result_str.push('[');
    for i in 0..data.len() {
        result_str.push('[');
        for j in 0..data[i].len() {
            result_str.push('[');
            for k in 0..data[i][j].len() {
                result_str.push_str(&data[i][j][k].to_string());
                result_str.push(',');
            }
            result_str.push(']');
        }
        result_str.push(']');
    }
    result_str.push(']');
    data_manager::import_text_to_file("w_mlp.weight", result_str)
        .expect("Error during save weights");
}

pub fn load_weights_mlp() -> Vec<Vec<Vec<f32>>> {
    let mut result = vec![];
    let content = data_manager::load_text_to_file("w_mlp.weight");
    let split_layers = content.split("]]");
    for layers in split_layers {
        if layers == "]" {
            break;
        }
        let mut neural_array = vec![];
        let split_neural = layers.split("]");
        for neural in split_neural {
            let mut neural_weight_array = vec![];
            for numbers in neural.split(",") {
                if !numbers.replace("[", "").is_empty() {
                    neural_weight_array.push(
                        numbers
                            .replace("[", "")
                            .trim()
                            .parse::<f32>()
                            .expect("Error parsing string to f32"),
                    );
                }
            }
            neural_array.push(neural_weight_array);
        }
        result.push(neural_array);
    }
    result
}
pub fn export_weights_linear(m: f32, b: f32) {
    let data = vec![vec![vec![m, b]]]; // Même format que pour le MLP : [[[m, b]]]
    let mut result_str = String::from("[");
    for layer in &data {
        result_str.push('[');
        for neuron in layer {
            result_str.push('[');
            for weight in neuron {
                result_str.push_str(&weight.to_string());
                result_str.push(',');
            }
            result_str.push(']');
        }
        result_str.push(']');
    }
    result_str.push(']');
    data_manager::import_text_to_file("w_linear.weight", result_str)
        .expect("Error during save weights");
}

pub fn import_weights_linear() -> (f32, f32) {
    let content = data_manager::load_text_to_file("w_linear.weight");
    let split_layers = content.split("]]");
    for layers in split_layers {
        if layers == "]" {
            break;
        }
        let split_neural = layers.split("]");
        for neural in split_neural {
            let mut values = vec![];
            for numbers in neural.split(",") {
                if !numbers.replace("[", "").is_empty() {
                    values.push(
                        numbers
                            .replace("[", "")
                            .trim()
                            .parse::<f32>()
                            .expect("Error parsing string to f32"),
                    );
                }
            }
            if values.len() == 2 {
                return (values[0], values[1]); // Retourne m et b
            }
        }
    }
    panic!("Invalid weight format for linear model");
}

pub fn export_weights_rbf(centers: &Vec<Vec<f32>>, sigmas: &Vec<f32>, output_w: &Vec<Vec<f32>>) {
    let mut result_str = String::from("");
    result_str.push('[');
    for i in 0..centers.len() {
        result_str.push('[');
        for j in 0..centers[i].len() {
            result_str.push_str(&centers[i][j].to_string());
            result_str.push(',');
        }
        result_str.push(']');
    }
    result_str.push(']');
    result_str.push('[');
    for i in 0..sigmas.len() {
        result_str.push_str(&sigmas[i].to_string());
        result_str.push(',');
    }
    result_str.push(']');
    result_str.push('[');
    for i in 0..output_w.len() {
        result_str.push('[');
        for j in 0..output_w[i].len() {
            result_str.push_str(&output_w[i][j].to_string());
            result_str.push(',');
        }
        result_str.push(']');
    }
    result_str.push(']');
    data_manager::import_text_to_file("w_rbf.weight", result_str)
        .expect("Error during save weights");
}

pub fn load_weights_rbf() -> (Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>) {
    let content = data_manager::load_text_to_file("w_rbf.weight");
    let mut sections = vec![];
    let mut current = String::new();
    let mut count = 0;

    for c in content.chars() {
        if c == '[' {
            count += 1;
            if count == 1 {
                continue;
            }
        } else if c == ']' {
            count -= 1;
            if count == 0 {
                sections.push(current.clone());
                current.clear();
                continue;
            }
        }

        if count >= 1 {
            current.push(c);
        }
    }

    let centers_str = &sections[0];
    let mut centers = vec![];
    for line in centers_str.split("],") {
        let mut center = vec![];
        for val in line.replace("[", "").replace("]", "").split(',') {
            if !val.trim().is_empty() {
                center.push(
                    val.trim()
                        .parse::<f32>()
                        .expect("Error parsing center value"),
                );
            }
        }
        if !center.is_empty() {
            centers.push(center);
        }
    }

    let sigmas_str = &sections[1];
    let sigmas: Vec<f32> = sigmas_str
        .split(',')
        .filter_map(|val| {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                Some(trimmed.parse::<f32>().expect("Error parsing sigma value"))
            } else {
                None
            }
        })
        .collect();

    let output_str = &sections[2];
    let mut output_weights = vec![];
    for line in output_str.split("],") {
        let mut weights = vec![];
        for val in line.replace("[", "").replace("]", "").split(',') {
            if !val.trim().is_empty() {
                weights.push(
                    val.trim()
                        .parse::<f32>()
                        .expect("Error parsing output weight"),
                );
            }
        }
        if !weights.is_empty() {
            output_weights.push(weights);
        }
    }

    (centers, sigmas, output_weights)
}
