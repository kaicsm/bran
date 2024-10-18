// bran/src/loss.rs

use ndarray::prelude::*;

/// Trait que define métodos para funções de perda.
pub trait Loss {
    /// Calcula a perda entre a predição e o alvo.
    ///
    /// # Parâmetros
    /// * `predicted` - Array 2D contendo as predições do modelo
    /// * `target` - Array 2D contendo os valores alvo reais
    ///
    /// # Retorno
    /// Retorna um valor float representando a perda total
    fn loss(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> f32;

    /// Calcula o gradiente da perda em relação à predição.
    ///
    /// # Parâmetros
    /// * `predicted` - Array 2D contendo as predições do modelo
    /// * `target` - Array 2D contendo os valores alvo reais
    ///
    /// # Retorno
    /// Retorna um Array 2D contendo os gradientes da perda
    fn derivative(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> Array2<f32>;
}

/// Implementação da função de perda Mean Squared Error (MSE).
pub struct MeanSquaredError;

impl Loss for MeanSquaredError {
    fn loss(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> f32 {
        // Calcula a diferença entre predições e alvos
        let diff = predicted - target;
        // Eleva as diferenças ao quadrado
        let squared_diff = diff.mapv(|x| x.powi(2));
        // Calcula a média dos quadrados das diferenças
        squared_diff.sum() / (2 * target.len()) as f32
    }

    fn derivative(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
        // Calcula o gradiente da MSE
        (predicted - target) / target.len() as f32
    }
}

/// Implementação da função de perda Cross-Entropy Loss.
pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn loss(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> f32 {
        // Adiciona epsilon para evitar log(0)
        let epsilon = 1e-10;
        // Limita as predições entre epsilon e 1-epsilon para estabilidade numérica
        let predicted = predicted.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
        // Calcula a perda de entropia cruzada
        let loss = -(target * predicted.mapv(|x| x.ln())
            + (1.0 - target) * (1.0 - &predicted).mapv(|x| x.ln()));
        // Retorna a média da perda
        loss.sum() / target.len() as f32
    }

    fn derivative(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
        // Adiciona epsilon para evitar divisão por zero
        let epsilon = 1e-10;
        // Calcula o gradiente da entropia cruzada
        (predicted - target) / ((predicted * (1.0 - predicted)) + epsilon)
    }
}
