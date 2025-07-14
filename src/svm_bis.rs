use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use crate::data_converter::{export_weights_svm, import_weights_svm};

enum Kernel {
    RBF(f64),
    Polynomial(u32),
}

impl Kernel {
    fn compute(&self, x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        match self {
            Kernel::RBF(gamma) => {
                let diff = x - y;
                let dist_sq = diff.dot(&diff);
                (-gamma * dist_sq).exp()
            }
            Kernel::Polynomial(degree) => {
                let dot = x.dot(y);
                (1.0 + dot).powi(*degree as i32)
            }
        }
    }
}

#[pyclass]
pub struct KernelSVM {
    alpha: Vec<f64>,
    bias: f64,
    lr: f64,
    lambda_svm: f64,
    epochs: usize,
    kernel: Kernel,
    kernel_type: String,
    param: f64,
    support_vectors: Vec<Array1<f64>>,
    support_labels: Vec<f64>,
}

#[pymethods]
impl KernelSVM {
    #[new]
    pub fn new(kernel_type: &str, param: f64, lr: f64, lambda_svm: f64, epochs: usize) -> PyResult<Self> {
        let kernel = match kernel_type {
            "rbf" => Kernel::RBF(param),
            "poly" => Kernel::Polynomial(param as u32),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid kernel")),
        };

        Ok(Self {
            alpha: vec![],
            bias: 0.0,
            lr,
            lambda_svm,
            epochs,
            kernel,
            kernel_type: kernel_type.to_string(),
            param,
            support_vectors: vec![],
            support_labels: vec![],
        })
    }

    pub fn load_weights_from(&mut self, path: &str) {
        let (alpha, bias, support_vectors, support_labels) = import_weights_svm(path);
        self.alpha = alpha;
        self.bias = bias;
        self.support_vectors = support_vectors;
        self.support_labels = support_labels;
    }

    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<i32>, path: &str, x_val: Option<Vec<Vec<f64>>>, y_val: Option<Vec<i32>>) {
        let x = Array2::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect()).unwrap();
        let y = Array1::from_iter(y.iter().map(|&v| v as f64));
        let n = x.nrows();

        self.alpha = vec![0.0; n];
        self.support_vectors = x.outer_iter().map(|row| row.to_owned()).collect();
        self.support_labels = y.to_vec();

        let mut indices: Vec<usize> = (0..n).collect();

        for _ in 0..self.epochs {
            indices.shuffle(&mut thread_rng());

            for &i in &indices {
                let xi = &self.support_vectors[i];
                let yi = self.support_labels[i];

                let mut sum = 0.0;
                for j in 0..n {
                    let xj = &self.support_vectors[j];
                    let yj = self.support_labels[j];
                    sum += self.alpha[j] * yj * self.kernel.compute(xi, xj);
                }

                let margin = yi * (sum + self.bias);

                if margin < 1.0 {
                    self.alpha[i] += self.lr * (1.0 - margin);
                    self.bias += self.lr * yi;
                } else {
                    self.alpha[i] *= 1.0 - self.lr * self.lambda_svm;
                }
            }
        }

        let (kernel_type, param) = match &self.kernel {
            Kernel::RBF(gamma) => ("rbf", *gamma),
            Kernel::Polynomial(degree) => ("poly", *degree as f64),
        };

        export_weights_svm(
            &self.alpha,
            self.bias,
            &self.support_vectors,
            &self.support_labels,
            path,
            kernel_type,
            param,
            self.lr,
            self.lambda_svm,
            self.epochs,
        );


        let accuracy = self.evaluate(
            x.outer_iter().map(|v| v.to_vec()).collect(),
            y.iter().map(|&v| v as i32).collect(),
        );
        println!("Training accuracy: {:.2}%", accuracy * 100.0);

        if let (Some(xv), Some(yv)) = (x_val, y_val) {
            let val_acc = self.evaluate(xv, yv);
            println!("Validation accuracy: {:.2}%", val_acc * 100.0);
        }
    }

    pub fn predict(&self, x: Vec<Vec<f64>>) -> Vec<i32> {
        let x = Array2::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect())
            .expect("Erreur dans le format des données d'entrée");

        x.outer_iter()
            .map(|xi| {
                let mut sum = 0.0;
                for (alpha_i, (xj, &yj)) in self.alpha.iter().zip(self.support_vectors.iter().zip(&self.support_labels)) {
                    sum += alpha_i * yj * self.kernel.compute(&xi.to_owned(), xj);
                }
                if sum + self.bias >= 0.0 {
                    1
                } else {
                    -1
                }
            })
            .collect()
    }

    pub fn evaluate(&self, x: Vec<Vec<f64>>, y: Vec<i32>) -> f64 {
        let x = Array2::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect())
            .expect("Erreur dans les données");
        let y = Array1::from_iter(y.iter().map(|&v| v as f64));

        let mut correct = 0;
        let total = x.nrows();

        for (xi, &yi) in x.outer_iter().zip(y.iter()) {
            let mut sum = 0.0;
            for (alpha_i, (xj, &yj)) in self.alpha.iter().zip(self.support_vectors.iter().zip(&self.support_labels)) {
                sum += alpha_i * yj * self.kernel.compute(&xi.to_owned(), xj);
            }

            let prediction = if sum + self.bias >= 0.0 { 1.0 } else { -1.0 };
            if (prediction - yi).abs() < std::f64::EPSILON {
                correct += 1;
            }
        }

        correct as f64 / total as f64
    }

    pub fn get_bias(&self) -> f64 {
        self.bias
    }

    pub fn get_alpha(&self) -> Vec<f64> {
        self.alpha.clone()
    }
}