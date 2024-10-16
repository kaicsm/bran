// src/model.rs
use crate::layers::DenseLayer;
use crate::optimizer::Optimizer;
use ndarray::{Array2};
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Write, io::Read};

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<DenseLayer>,
}

impl NeuralNetwork {
    /// Cria uma nova rede neural vazia.
    pub fn new() -> Self {
        NeuralNetwork { layers: Vec::new() }
    }

    /// Salva o modelo utilizando a serialização binária `bincode` para melhor performance.
    pub fn save(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded: Vec<u8> = bincode::serialize(&self)?;
        let mut file = File::create(filename)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    /// Carrega o modelo a partir de um arquivo serializado com `bincode`.
    pub fn load(filename: &str) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let mut deserialized: NeuralNetwork = bincode::deserialize(&buffer)?;
        
        // Restaura as ativações nas camadas após desserialização
        for layer in &mut deserialized.layers {
            layer.restore_activation();
        }

        Ok(deserialized)
    }

    /// Adiciona uma nova camada à rede neural.
    pub fn add_layer(&mut self, layer: DenseLayer) {
        self.layers.push(layer);
    }

    /// Realiza a passagem forward para um lote de entradas (`Array2<f32>`).
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    /// Realiza a passagem backward para um lote de erros de saída (`Array2<f32>`).
    /// Atualiza os pesos e vieses utilizando o otimizador fornecido.
    pub fn backward(&mut self, output_error: &Array2<f32>, optimizer: &mut dyn Optimizer) -> Result<(), Box<dyn std::error::Error>> {
        let mut error = output_error.clone();
        for layer in self.layers.iter_mut().rev() {
            error = layer.backward(&error, optimizer);
        }
        Ok(())
    }
}

