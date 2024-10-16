// src/model.rs
use crate::{layers::DenseLayer, optimizer::Optimizer};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Read, io::Write};

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<DenseLayer>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        NeuralNetwork { layers: Vec::new() }
    }

    pub fn save(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = serde_json::to_string(&self)?;
        let mut file = File::create(filename)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    pub fn load(filename: &str) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut file = File::open(filename)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let mut deserialized: NeuralNetwork = serde_json::from_str(&contents)?;

        // Restaura as ativações nas camadas após desserialização
        for layer in &mut deserialized.layers {
            layer.restore_activation();
        }

        Ok(deserialized)
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

    pub fn backward(&mut self, output_error: &Array1<f64>, optimizer: &mut dyn Optimizer) {
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
