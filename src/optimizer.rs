// src/optimizer.rs
use ndarray::{Array1, Array2};

pub struct SGD {
    pub learning_rate: f64,
    pub l2_reg: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        SGD {
            learning_rate,
            l2_reg: 1e-5,
        }
    }

    pub fn update(
        &self,
        weights: &mut Array2<f64>,
        biases: &mut Array1<f64>,
        weight_grads: &Array2<f64>,
        bias_grads: &Array1<f64>,
    ) {
        // Regularização L2 aplicada apenas aos pesos
        *weights -=
            &(weight_grads * self.learning_rate + &*weights * self.l2_reg * self.learning_rate);

        // Atualização dos vieses, sem regularização L2
        *biases -= &(bias_grads * self.learning_rate);
    }
}
