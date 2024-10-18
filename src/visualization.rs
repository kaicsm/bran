// bran/src/visualization.rs

use serde::{Deserialize, Serialize};

/// Estrutura para armazenar estatísticas de treinamento.
/// Isso inclui o número de épocas, os valores de perda correspondentes e as acurácias.
#[derive(Serialize, Deserialize, Clone)]
pub struct TrainingStats {
    /// Vetor para armazenar os números das épocas.
    pub epochs: Vec<f32>,
    /// Vetor para armazenar os valores de perda correspondentes a cada época.
    pub losses: Vec<f32>,
    /// Vetor para armazenar os valores de acurácia correspondentes a cada época.
    pub accuracies: Vec<f32>,
}

impl TrainingStats {
    /// Cria uma nova instância de TrainingStats com vetores vazios.
    pub fn new() -> Self {
        TrainingStats {
            epochs: Vec::new(),
            losses: Vec::new(),
            accuracies: Vec::new(),
        }
    }

    /// Limpa todos os dados armazenados, resetando as estatísticas.
    pub fn reset(&mut self) {
        self.epochs.clear();
        self.losses.clear();
        self.accuracies.clear();
    }

    /// Registra os dados de uma época de treinamento.
    ///
    /// # Parâmetros
    ///
    /// - `epoch`: O número da época atual.
    /// - `loss`: O valor de perda calculado para esta época.
    /// - `accuracy`: O valor da acurácia calculado para esta época.
    pub fn log_epoch(&mut self, epoch: f32, loss: f32, accuracy: f32) {
        self.epochs.push(epoch);
        self.losses.push(loss);
        self.accuracies.push(accuracy);
    }
}
