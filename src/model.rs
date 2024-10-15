// src/model.rs
use crate::layers::DenseLayer;
use crate::optimizer::SGD;
use ndarray::Array1;

pub struct NeuralNetwork {
    pub layers: Vec<DenseLayer>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        NeuralNetwork { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: DenseLayer) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn backward(&mut self, output_error: &Array1<f64>, optimizer: &SGD) {
        let mut error = output_error.clone();
        for layer in self.layers.iter_mut().rev() {
            let (input_error, weight_gradients, bias_gradients) = layer.backward(&error);
            optimizer.update(
                &mut layer.weights,
                &mut layer.biases,
                &weight_gradients,
                &bias_gradients,
            );
            error = input_error;
        }
    }
}
