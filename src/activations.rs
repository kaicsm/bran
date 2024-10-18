// bran/src/activations.rs

use ndarray::prelude::*;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

static RELU: Lazy<ReLU> = Lazy::new(|| ReLU);
static SIGMOID: Lazy<Sigmoid> = Lazy::new(|| Sigmoid);
static LINEAR: Lazy<Linear> = Lazy::new(|| Linear);
static TANH: Lazy<Tanh> = Lazy::new(|| Tanh);

/// Trait que define métodos para funções de ativação.
/// Implementa o cálculo da função de ativação e sua derivada,
/// tanto para um único valor quanto para uma matriz `Array2<f32>`.
pub trait Activation: Send + Sync {
    /// Aplica a função de ativação a um único valor `x`.
    ///
    /// # Argumentos
    ///
    /// * `x` - Um valor do tipo `f32` ao qual será aplicada a função de ativação.
    ///
    /// # Retorno
    ///
    /// Retorna o resultado da função de ativação aplicada a `x`.
    fn activate(&self, x: f32) -> f32;

    /// Calcula a derivada da função de ativação para um único valor `x`.
    ///
    /// # Argumentos
    ///
    /// * `x` - Um valor do tipo `f32` ao qual será aplicada a função derivada.
    ///
    /// # Retorno
    ///
    /// Retorna o valor da derivada da função de ativação em `x`.
    fn derivative(&self, x: f32) -> f32;

    /// Aplica a função de ativação a cada elemento de uma matriz (`Array2<f32>`).
    /// Esta versão utiliza paralelismo para maior eficiência em grandes matrizes.
    ///
    /// # Argumentos
    ///
    /// * `x` - Referência para uma matriz `Array2<f32>` cujos elementos terão a função de ativação aplicada.
    ///
    /// # Retorno
    ///
    /// Retorna uma nova matriz com a função de ativação aplicada a cada elemento.
    fn activate_array(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut result = x.clone(); // Clona a matriz de entrada
        result.par_mapv_inplace(|elem| self.activate(elem)); // Aplica a ativação de forma paralela
        result
    }

    /// Calcula a derivada da função de ativação para cada elemento de uma matriz (`Array2<f32>`).
    /// Esta versão utiliza paralelismo para maior eficiência em grandes matrizes.
    ///
    /// # Argumentos
    ///
    /// * `x` - Referência para uma matriz `Array2<f32>` cujos elementos terão a derivada da função aplicada.
    ///
    /// # Retorno
    ///
    /// Retorna uma nova matriz com a derivada da função de ativação aplicada a cada elemento.
    fn derivative_array(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut result = x.clone(); // Clona a matriz de entrada
        result.par_mapv_inplace(|elem| self.derivative(elem)); // Aplica a derivada de forma paralela
        result
    }
}

/// Implementação da função de ativação ReLU (Rectified Linear Unit).
/// ReLU retorna o valor original se for positivo, e 0 caso contrário.
pub struct ReLU;

impl Activation for ReLU {
    /// Implementa a função ReLU: `f(x) = max(0, x)`.
    fn activate(&self, x: f32) -> f32 {
        x.max(0.0)
    }

    /// Calcula a derivada da função ReLU: `f'(x) = 1` se `x > 0`, `0` caso contrário.
    fn derivative(&self, x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

/// Implementação da função de ativação Sigmoid.
/// Sigmoid suaviza a entrada, retornando valores entre 0 e 1.
pub struct Sigmoid;

impl Activation for Sigmoid {
    /// Implementa a função Sigmoid: `f(x) = 1 / (1 + e^(-x))`.
    fn activate(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Calcula a derivada da função Sigmoid: `f'(x) = f(x) * (1 - f(x))`.
    fn derivative(&self, x: f32) -> f32 {
        let sig = self.activate(x);
        sig * (1.0 - sig)
    }
}

/// Implementação da função de ativação Tanh (Tangente Hiperbólica).
/// Tanh mapeia a entrada para valores entre -1 e 1.
pub struct Tanh;

impl Activation for Tanh {
    /// Implementa a função Tanh: `f(x) = tanh(x)`.
    fn activate(&self, x: f32) -> f32 {
        x.tanh()
    }

    /// Calcula a derivada da função Tanh: `f'(x) = 1 - tanh^2(x)`.
    fn derivative(&self, x: f32) -> f32 {
        let tanh_x = self.activate(x);
        1.0 - tanh_x.powi(2)
    }
}

/// Implementação da função de ativação Linear.
/// A função Linear retorna o próprio valor de entrada sem modificações.
pub struct Linear;

impl Activation for Linear {
    /// Implementa a função Linear: `f(x) = x`.
    fn activate(&self, x: f32) -> f32 {
        x
    }

    /// A derivada da função Linear é sempre 1: `f'(x) = 1`.
    fn derivative(&self, _x: f32) -> f32 {
        1.0
    }
}

/// Enum que representa diferentes tipos de funções de ativação.
/// Facilita a serialização, desserialização e a troca dinâmica de funções de ativação.
#[derive(Serialize, Deserialize, Clone)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Linear,
    Tanh,
}

impl Activation for ActivationType {
    /// Aplica a função de ativação correspondente com base no tipo especificado.
    fn activate(&self, x: f32) -> f32 {
        match self {
            ActivationType::ReLU => RELU.activate(x),
            ActivationType::Sigmoid => SIGMOID.activate(x),
            ActivationType::Linear => LINEAR.activate(x),
            ActivationType::Tanh => TANH.activate(x),
        }
    }

    /// Calcula a derivada da função de ativação correspondente com base no tipo.
    fn derivative(&self, x: f32) -> f32 {
        match self {
            ActivationType::ReLU => RELU.derivative(x),
            ActivationType::Sigmoid => SIGMOID.derivative(x),
            ActivationType::Linear => LINEAR.derivative(x),
            ActivationType::Tanh => TANH.derivative(x),
        }
    }

    /// Aplica a função de ativação correspondente a uma matriz (`Array2<f32>`) em paralelo.
    fn activate_array(&self, x: &Array2<f32>) -> Array2<f32> {
        match self {
            ActivationType::ReLU => RELU.activate_array(x),
            ActivationType::Sigmoid => SIGMOID.activate_array(x),
            ActivationType::Linear => LINEAR.activate_array(x),
            ActivationType::Tanh => TANH.activate_array(x),
        }
    }

    /// Calcula a derivada da função de ativação correspondente para cada elemento de uma matriz (`Array2<f32>`) em paralelo.
    fn derivative_array(&self, x: &Array2<f32>) -> Array2<f32> {
        match self {
            ActivationType::ReLU => RELU.derivative_array(x),
            ActivationType::Sigmoid => SIGMOID.derivative_array(x),
            ActivationType::Linear => LINEAR.derivative_array(x),
            ActivationType::Tanh => TANH.derivative_array(x),
        }
    }
}
