use std::fs::File;
use std::io::{BufRead, BufWriter};
use std::io::Write;
use crate::data_manager;
use ndarray::Array1;


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
pub fn export_weights_svm(weights: &Array1<f64>, bias: f64) {
    let mut result_str = String::from("[[");
    for w in weights {
        result_str.push_str(&w.to_string());
        result_str.push(',');
    }
    result_str.push_str(&bias.to_string());
    result_str.push_str("]]");

    data_manager::import_text_to_file("w_svm.weight", result_str)
        .expect("Error during save weights for SVM");
}
pub fn import_weights_svm() -> (Array1<f64>, f64) {
    let content = data_manager::load_text_to_file("w_svm.weight");
    let clean = content.replace("[[", "").replace("]]", "");
    let parts: Vec<&str> = clean.split(',').collect();

    if parts.len() < 2 {
        panic!("Invalid weight format for SVM");
    }

    let weights: Vec<f64> = parts[..parts.len() - 1]
        .iter()
        .map(|s| s.trim().parse::<f64>().expect("Error parsing weight"))
        .collect();

    let bias = parts[parts.len() - 1]
        .trim()
        .parse::<f64>()
        .expect("Error parsing bias");

    (Array1::from(weights), bias)
}
