// src/loss.rs
use ndarray::prelude::*;

/// Trait que define métodos para funções de perda.
pub trait Loss {
    /// Calcula a perda entre a predição e o alvo.
    fn loss(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> f32;

    /// Calcula o gradiente da perda em relação à predição.
    fn derivative(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> Array2<f32>;
}

/// Implementação da função de perda Mean Squared Error (MSE).
pub struct MeanSquaredError;

impl Loss for MeanSquaredError {
    fn loss(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> f32 {
        let diff = predicted - target;
        let squared_diff = diff.mapv(|x| x.powi(2));
        squared_diff.sum() / (2 * target.len()) as f32
    }

    fn derivative(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
        (predicted - target) / target.len() as f32
    }
}

/// Implementação da função de perda Cross-Entropy Loss.
pub struct CrossEntropyLoss;

impl Loss for CrossEntropyLoss {
    fn loss(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> f32 {
        // Adiciona epsilon para evitar log(0)
        let epsilon = 1e-10;
        let predicted = predicted.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
        let loss = -(target * predicted.mapv(|x| x.ln())
            + (1.0 - target) * (1.0 - &predicted).mapv(|x| x.ln()));
        loss.sum() / target.len() as f32
    }

    fn derivative(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
        // Adiciona epsilon para evitar divisão por zero
        let epsilon = 1e-10;
        (predicted - target) / ((predicted * (1.0 - predicted)) + epsilon)
    }
}
