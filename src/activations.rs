use ndarray::prelude::*;

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

// Enum para representar os diferentes tipos de ativação
pub enum ActivationType {
    ReLU,
    Sigmoid,
}

impl Activation for ActivationType {
    fn activate(&self, x: f64) -> f64 {
        match self {
            ActivationType::ReLU => ReLU.activate(x),
            ActivationType::Sigmoid => Sigmoid.activate(x),
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationType::ReLU => ReLU.derivative(x),
            ActivationType::Sigmoid => Sigmoid.derivative(x),
        }
    }

    fn activate_array(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            ActivationType::ReLU => ActivationType::ReLU.activate_array(x),
            ActivationType::Sigmoid => ActivationType::Sigmoid.activate_array(x),
        }
    }

    fn derivative_array(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            ActivationType::ReLU => ActivationType::ReLU.derivative_array(x),
            ActivationType::Sigmoid => ActivationType::Sigmoid.derivative_array(x),
        }
    }
}
