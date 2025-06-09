use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::data_manager; // déjà là normalement
use crate::data_converter::{export_weights_svm};
use crate::data_converter::{import_weights_svm};

/// SVM linéaire avec SGD
#[pyclass]
pub struct LinearSVM {
    pub weights: Array1<f64>,
    pub bias: f64,
    pub lr: f64,
    pub epochs: usize,
    pub svm_lambda: f64,
}

#[pymethods]
impl LinearSVM {
    #[new]
    pub fn new(lr: f64, epochs: usize, svm_lambda: f64) -> Self {
        Self {
            weights: Array1::zeros(1),
            bias: 0.0,
            lr,
            epochs,
            svm_lambda,
        }
    }

    /// Entraînement du SVM
    pub fn fit(&mut self, x: Vec<Vec<f64>>, y: Vec<i32>) {
        let x = Array2::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect()).unwrap();
        let y = Array1::from_iter(y.into_iter().map(|v| v as f64));

        let n_samples = x.nrows();
        let n_features = x.ncols();

        self.weights = Array1::zeros(n_features);
        self.bias = 0.0;

        let mut indices: Vec<usize> = (0..n_samples).collect();

        for _ in 0..self.epochs {
            indices.shuffle(&mut thread_rng());

            for &i in &indices {
                let xi = x.row(i);
                let yi = y[i];

                let condition = yi * (self.weights.dot(&xi) + self.bias);

                if condition >= 1.0 {
                    self.weights = &self.weights - &(self.lr * self.svm_lambda * &self.weights);
                } else {
                    self.weights = &self.weights - self.lr * (self.svm_lambda * &self.weights - yi * &xi);
                    self.bias += self.lr * yi;
                }
            }
        }


        export_weights_svm(&self.weights, self.bias);
    }


    /// Prédiction
    pub fn predict(&self, x: Vec<Vec<f64>>) -> Vec<i32> {
        let x = Array2::from_shape_vec((x.len(), x[0].len()), x.into_iter().flatten().collect()).unwrap();
        x.outer_iter()
            .map(|xi| if self.weights.dot(&xi) + self.bias >= 0.0 { 1 } else { -1 })
            .collect()
    }
    pub fn load_weights(&mut self) {
        let (w, b) = import_weights_svm();
        self.weights = w;
        self.bias = b;
    }
    pub fn get_weights(&self) -> Vec<f64> {
        self.weights.to_vec()
    }

    /// Retourne le biais
    pub fn get_bias(&self) -> f64 {
        self.bias
    }

}


