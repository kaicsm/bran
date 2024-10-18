// bran/src/model.rs

use crate::layers::DenseLayer;
use crate::loss::Loss;
use crate::optimizer::Optimizer;
use crate::visualization::TrainingStats;
use ndarray::{parallel::prelude::*, s, Array2};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::{fs::File, io::Read, io::Write};

/// Estrutura principal que representa uma rede neural.
#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<DenseLayer>,
}

impl NeuralNetwork {
    /// Cria uma nova rede neural vazia.
    pub fn new() -> Self {
        NeuralNetwork { layers: Vec::new() }
    }

    /// Salva o modelo utilizando a serialização binária `bincode` para melhorar a performance.
    ///
    /// # Parâmetros
    /// - `filename`: O caminho onde o modelo será salvo.
    ///
    /// # Retornos
    /// - `Result<(), Box<dyn std::error::Error>>`: Retorna Ok se o modelo foi salvo com sucesso, ou um erro caso contrário.
    pub fn save(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded: Vec<u8> = bincode::serialize(&self)?;
        let mut file = File::create(filename)?;
        file.write_all(&encoded)?;
        Ok(())
    }

    /// Carrega um modelo a partir de um arquivo serializado com `bincode`.
    ///
    /// # Parâmetros
    /// - `filename`: O caminho de onde o modelo será carregado.
    ///
    /// # Retornos
    /// - `Result<NeuralNetwork, Box<dyn std::error::Error>>`: A rede neural carregada ou um erro, caso a operação falhe.
    pub fn load(filename: &str) -> Result<NeuralNetwork, Box<dyn std::error::Error>> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let mut deserialized: NeuralNetwork = bincode::deserialize(&buffer)?;

        // Restaura as ativações nas camadas após a desserialização
        for layer in &mut deserialized.layers {
            layer.restore_activation();
        }

        Ok(deserialized)
    }

    /// Adiciona uma nova camada à rede neural.
    ///
    /// # Parâmetros
    /// - `layer`: A nova camada a ser adicionada à rede.
    pub fn add_layer(&mut self, layer: DenseLayer) {
        self.layers.push(layer);
    }

    /// Executa a propagação forward para um lote de entradas.
    ///
    /// Os dados são propagados através de todas as camadas da rede.
    ///
    /// # Parâmetros
    /// - `input`: Os dados de entrada para a rede.
    ///
    /// # Retornos
    /// - `Array2<f32>`: A saída gerada pela última camada da rede.
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    /// Executa a retropropagação (backward) para um lote de erros de saída.
    ///
    /// Atualiza os pesos e vieses de todas as camadas usando o otimizador fornecido.
    ///
    /// # Parâmetros
    /// - `output_error`: O erro da saída que será propagado de volta.
    /// - `optimizer`: O otimizador que será usado para atualizar os pesos.
    ///
    /// # Retornos
    /// - `Result<(), Box<dyn std::error::Error>>`: Retorna Ok se a operação foi bem-sucedida.
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

    /// Treina a rede neural utilizando os dados fornecidos.
    ///
    /// # Parâmetros
    /// - `neural_net`: A rede neural encapsulada em `Arc<Mutex<Self>>` para permitir acesso concorrente seguro.
    /// - `x_train`: Dados de entrada para o treinamento.
    /// - `y_train`: Rótulos/alvos correspondentes.
    /// - `epochs`: Número de épocas de treinamento.
    /// - `batch_size`: Tamanho do lote para o treinamento em mini-lotes.
    /// - `loss_fn`: Função de perda a ser usada, encapsulada em `Arc` para ser segura para threads.
    /// - `optimizer`: Otimizador para a atualização dos pesos, também encapsulado em `Arc<Mutex>` para garantir acesso seguro entre threads.
    /// - `stats`: Estrutura para coletar estatísticas de treinamento, encapsulada em `Arc<Mutex>`.
    pub fn train(
        neural_net: Arc<Mutex<Self>>,
        x_train: &Array2<f32>,
        y_train: &Array2<f32>,
        epochs: usize,
        batch_size: usize,
        loss_fn: Arc<dyn Loss + Sync + Send>,
        optimizer: Arc<Mutex<dyn Optimizer + Send>>,
        stats: Arc<Mutex<TrainingStats>>,
    ) {
        let n_samples = x_train.shape()[0];

        for epoch in 0..epochs {
            // Itera sobre mini-lotes em paralelo
            (0..n_samples)
                .into_par_iter()
                .step_by(batch_size)
                .for_each(|i| {
                    let end = usize::min(i + batch_size, n_samples);
                    let x_batch = x_train.slice(s![i..end, ..]).to_owned();
                    let y_batch = y_train.slice(s![i..end, ..]).to_owned();

                    // Passagem forward e backward
                    {
                        let mut neural_net = neural_net.lock().unwrap(); // Escopo do lock é limitado
                        let output = neural_net.forward(&x_batch);
                        let error = loss_fn.derivative(&output, &y_batch);

                        // Atualiza os pesos e vieses
                        let mut optimizer = optimizer.lock().unwrap();
                        neural_net.backward(&error, &mut *optimizer).unwrap();
                    }
                });

            // Calcula a perda e a acurácia após a época
            let (loss, accuracy) = {
                let mut neural_net = neural_net.lock().unwrap();
                let output = neural_net.forward(&x_train);
                let loss = loss_fn.loss(&output, &y_train);
                let accuracy = calculate_accuracy(&y_train, &output);
                (loss, accuracy)
            };

            // Atualiza as estatísticas de treinamento
            {
                let mut stats = stats.lock().unwrap();
                stats.log_epoch(epoch as f32 + 1.0, loss, accuracy);
            }

            println!(
                "Epoch {}/{} - Loss: {:.6} - Accuracy: {:.4}",
                epoch + 1,
                epochs,
                loss,
                accuracy
            );
        }
    }
}

/// Função para calcular a acurácia entre as saídas previstas e os rótulos reais.
fn calculate_accuracy(y_true: &Array2<f32>, y_pred: &Array2<f32>) -> f32 {
    let mut correct = 0;
    let total = y_true.shape()[0];

    for (true_val, pred_val) in y_true.iter().zip(y_pred.iter()) {
        if true_val.round() == pred_val.round() {
            correct += 1;
        }
    }

    correct as f32 / total as f32
}
