// src/activations.rs
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};

pub trait Activation {
    fn activate(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;

    // Funções para aplicar em Arrays de forma paralela
    fn activate_array(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|elem| self.activate(elem))
    }

    fn derivative_array(&self, x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|elem| self.derivative(elem))
    }
}

pub struct ReLU;
impl Activation for ReLU {
    fn activate(&self, x: f64) -> f64 {
        x.max(0.0)
    }
    fn derivative(&self, x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

pub struct Sigmoid;
impl Activation for Sigmoid {
    fn activate(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    fn derivative(&self, x: f64) -> f64 {
        let sig = self.activate(x);
        sig * (1.0 - sig)
    }
}

pub struct Linear;
impl Activation for Linear {
    fn activate(&self, x: f64) -> f64 {
        x
    }
    fn derivative(&self, _x: f64) -> f64 {
        1.0
    }
}

// Enum para representar os diferentes tipos de ativação
#[derive(Serialize, Deserialize, Clone)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Linear,
}

impl Activation for ActivationType {
    fn activate(&self, x: f64) -> f64 {
        match self {
            ActivationType::ReLU => ReLU.activate(x),
            ActivationType::Sigmoid => Sigmoid.activate(x),
            ActivationType::Linear => Linear.activate(x),
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationType::ReLU => ReLU.derivative(x),
            ActivationType::Sigmoid => Sigmoid.derivative(x),
            ActivationType::Linear => Linear.derivative(x),
        }
    }

    fn activate_array(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            ActivationType::ReLU => ReLU.activate_array(x),
            ActivationType::Sigmoid => Sigmoid.activate_array(x),
            ActivationType::Linear => Linear.activate_array(x),
        }
    }

    fn derivative_array(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            ActivationType::ReLU => ReLU.derivative_array(x),
            ActivationType::Sigmoid => Sigmoid.derivative_array(x),
            ActivationType::Linear => Linear.derivative_array(x),
        }
    }
}
