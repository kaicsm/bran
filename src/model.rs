// src/model.rs
use crate::layers::DenseLayer;
use crate::loss::Loss; // Importa o trait Loss
use crate::optimizer::Optimizer;
use crate::visualization::TrainingStats; // Importa TrainingStats para coletar estatísticas
use ndarray::{s, Array2};
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Read, io::Write};

use std::sync::{Arc, Mutex};

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
    pub fn backward(
        &mut self,
        output_error: &Array2<f32>,
        optimizer: &mut dyn Optimizer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut error = output_error.clone();
        for layer in self.layers.iter_mut().rev() {
            error = layer.backward(&error, optimizer);
        }
        Ok(())
    }

    // Adicione `stats: Arc<Mutex<TrainingStats>>` como parâmetro
    pub fn train(
        &mut self,
        x_train: &Array2<f32>,
        y_train: &Array2<f32>,
        epochs: usize,
        batch_size: usize,
        loss_fn: &dyn Loss,
        optimizer: &mut dyn Optimizer,
        stats: Arc<Mutex<TrainingStats>>,
    ) {
        let n_samples = x_train.shape()[0];

        for epoch in 0..epochs {
            for i in (0..n_samples).step_by(batch_size) {
                let end = usize::min(i + batch_size, n_samples);
                let x_batch = x_train.slice(s![i..end, ..]).to_owned();
                let y_batch = y_train.slice(s![i..end, ..]).to_owned();

                let output = self.forward(&x_batch);
                let error = loss_fn.derivative(&output, &y_batch);

                self.backward(&error, optimizer).unwrap();
            }

            // Coleta estatísticas ao final de cada época
            let output = self.forward(&x_train);
            let loss = loss_fn.loss(&output, &y_train);

            // Atualiza as estatísticas dentro do Mutex
            {
                let mut stats = stats.lock().unwrap();
                stats.log_epoch(epoch as f32 + 1.0, loss);
            }

            println!("Epoch {}/{} - Loss: {:.6}", epoch + 1, epochs, loss);
        }
    }
}
