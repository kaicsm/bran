// src/activations.rs
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};

/// Trait que define métodos para funções de ativação.
pub trait Activation {
    /// Aplica a função de ativação a um único valor.
    fn activate(&self, x: f32) -> f32;
    
    /// Calcula a derivada da função de ativação para um único valor.
    fn derivative(&self, x: f32) -> f32;

    /// Aplica a função de ativação a uma matriz (`Array2<f32>`).
    fn activate_array(&self, x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|elem| self.activate(elem))
    }

    /// Calcula a derivada da função de ativação para uma matriz (`Array2<f32>`).
    fn derivative_array(&self, x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|elem| self.derivative(elem))
    }
}

/// Estrutura para a função de ativação ReLU.
pub struct ReLU;

impl Activation for ReLU {
    fn activate(&self, x: f32) -> f32 {
        x.max(0.0)
    }

    fn derivative(&self, x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

/// Estrutura para a função de ativação Sigmoid.
pub struct Sigmoid;

impl Activation for Sigmoid {
    fn activate(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(&self, x: f32) -> f32 {
        let sig = self.activate(x);
        sig * (1.0 - sig)
    }
}

/// Enum para representar os diferentes tipos de ativação.
#[derive(Serialize, Deserialize, Clone)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
}

impl Activation for ActivationType {
    fn activate(&self, x: f32) -> f32 {
        match self {
            ActivationType::ReLU => ReLU.activate(x),
            ActivationType::Sigmoid => Sigmoid.activate(x),
        }
    }

    fn derivative(&self, x: f32) -> f32 {
        match self {
            ActivationType::ReLU => ReLU.derivative(x),
            ActivationType::Sigmoid => Sigmoid.derivative(x),
        }
    }

    fn activate_array(&self, x: &Array2<f32>) -> Array2<f32> {
        match self {
            ActivationType::ReLU => ReLU.activate_array(x),
            ActivationType::Sigmoid => Sigmoid.activate_array(x),
        }
    }

    fn derivative_array(&self, x: &Array2<f32>) -> Array2<f32> {
        match self {
            ActivationType::ReLU => ReLU.derivative_array(x),
            ActivationType::Sigmoid => Sigmoid.derivative_array(x),
        }
    }
}

