use std::fs::File;
use std::io::{BufRead, BufWriter};
use std::io::Write;
use crate::data_manager;
use ndarray::{Array1, Array2};

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
    let data = vec![vec![vec![m, b]]];
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
                return (values[0], values[1]);
            }
        }
    }
    panic!("Invalid weight format for linear model");
}

pub fn export_weights_rbf(centers: &Vec<Vec<f32>>, w: &Vec<f32>) {
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
    for i in 0..w.len() {
        result_str.push_str(&w[i].to_string());
        result_str.push(',');
    }
    result_str.push(']');
    data_manager::import_text_to_file("w_rbf.weight", result_str)
        .expect("Error during save weights");
}

pub fn load_weights_rbf() -> (Vec<Vec<f32>>, Vec<f32>) {
    let content = data_manager::load_text_to_file("w_rbf.weight");
    let mut sections = vec![];
    let mut current = String::new();
    let mut count = 0;

    for c in content.chars() {
        if c == '[' {
            if count > 0 {
                current.push(c);
            }
            count += 1;
        } else if c == ']' {
            count -= 1;
            if count > 0 {
                current.push(c);
            } else if count == 0 {
                sections.push(current.clone());
                current.clear();
            }
        } else {
            if count > 0 {
                current.push(c);
            }
        }
    }

    let centers_str = &sections[0];
    let mut centers = vec![];

    for part in centers_str
        .split("][")
        .map(|s| s.replace('[', "").replace(']', ""))
    {
        let center: Vec<f32> = part
            .split(',')
            .filter_map(|val| {
                let trimmed = val.trim();
                if !trimmed.is_empty() {
                    Some(trimmed.parse::<f32>().expect("Error"))
                } else {
                    None
                }
            })
            .collect();

        if !center.is_empty() {
            centers.push(center);
        }
    }

    let w_str = &sections[1];
    let w: Vec<f32> = w_str
        .split(',')
        .filter_map(|val| {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                Some(trimmed.parse::<f32>().expect("Error"))
            } else {
                None
            }
        })
        .collect();

    (centers, w)
}

pub fn export_weights_svm(
    alpha: &Vec<f64>,
    bias: f64,
    support_vectors: &Vec<Array1<f64>>,
    support_labels: &Vec<f64>,
    path: &str,
    kernel_type: &str,
    param: f64,
    lr: f64,
    lambda: f64,
) {
    let mut result_str = String::new();

    result_str.push_str(&format!("kernel:{}\n", kernel_type));
    result_str.push_str(&format!("param:{}\n", param));
    result_str.push_str(&format!("lr:{}\n", lr));
    result_str.push_str(&format!("lambda:{}\n", lambda));
    result_str.push_str(&format!("bias:{}\n", bias));

    result_str.push_str("alpha:[");
    result_str.push_str(&alpha.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
    result_str.push_str("]\n");

    result_str.push_str("support_vectors:[");
    for vec in support_vectors {
        result_str.push('[');
        result_str.push_str(&vec.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
        result_str.push_str("],");
    }
    result_str.push_str("]\n");

    result_str.push_str("support_labels:[");
    result_str.push_str(&support_labels.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
    result_str.push_str("]\n");

    data_manager::import_text_to_file(path, result_str)
        .expect("Error during save weights");
}

pub fn import_weights_svm(path: &str) -> (Vec<f64>, f64, Vec<Array1<f64>>, Vec<f64>) {
    let content = data_manager::load_text_to_file(path);

    let bias_line = content.lines().find(|l| l.starts_with("bias:")).expect("Missing bias");
    let bias: f64 = bias_line["bias:".len()..].parse().expect("Invalid bias");

    let alpha_line = content.lines().find(|l| l.starts_with("alpha:[")).expect("Missing alpha");
    let alpha_str = &alpha_line["alpha:[".len()..alpha_line.len() - 1];
    let alpha: Vec<f64> = alpha_str.split(',').map(|s| s.trim().parse().unwrap()).collect();

    let sv_line = content.lines().find(|l| l.starts_with("support_vectors:[")).expect("Missing vectors");
    let sv_str = &sv_line["support_vectors:[".len()..sv_line.len() - 1];
    let support_vectors: Vec<Array1<f64>> = sv_str.split("],")
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let s_clean = s.trim().trim_start_matches('[');
            let values: Vec<f64> = s_clean.split(',').map(|v| v.trim().parse().unwrap()).collect();
            Array1::from(values)
        })
        .collect();

    let label_line = content.lines().find(|l| l.starts_with("support_labels:[")).expect("Missing labels");
    let label_str = &label_line["support_labels:[".len()..label_line.len() - 1];
    let support_labels: Vec<f64> = label_str.split(',').map(|v| v.trim().parse().unwrap()).collect();

    (alpha, bias, support_vectors, support_labels)
}

pub fn export_weights_ols(weights: &Vec<f32>) {
    let mut result_str = String::from("[");
    for w in weights {
        result_str.push_str(&w.to_string());
        result_str.push(',');
    }
    result_str.push(']');
    data_manager::import_text_to_file("weights_ols.weights", result_str)
        .expect("Error during save weights");
}

pub fn import_weights_ols() -> Vec<f32> {
    let content = data_manager::load_text_to_file("weights_ols.weights");
    let trimmed = content.trim_matches(|c| c == '[' || c == ']');
    trimmed
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect()
}