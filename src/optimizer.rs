// src/optimizer.rs
use ndarray::{Array1, Array2};

pub trait Optimizer {
    fn update(
        &mut self,
        weights: &mut Array2<f64>,
        biases: &mut Array1<f64>,
        weight_grads: &Array2<f64>,
        bias_grads: &Array1<f64>,
    );
}

pub struct SGD {
    pub learning_rate: f64,
    pub l2_reg: f64,
}

impl SGD {
    #[allow(dead_code)]
    pub fn new(learning_rate: f64) -> Self {
        SGD {
            learning_rate,
            l2_reg: 1e-5,
        }
    }
}

impl Optimizer for SGD {
    fn update(
        &mut self,
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

pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub l2_reg: f64,
    m_w: Array2<f64>,
    v_w: Array2<f64>,
    m_b: Array1<f64>,
    v_b: Array1<f64>,
    t: usize,
}

impl Adam {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            l2_reg: 1e-5,
            m_w: Array2::zeros((0, 0)),
            v_w: Array2::zeros((0, 0)),
            m_b: Array1::zeros(0),
            v_b: Array1::zeros(0),
            t: 0,
        }
    }
}

use rayon::prelude::*;

impl Optimizer for Adam {
    fn update(
        &mut self,
        weights: &mut Array2<f64>,
        biases: &mut Array1<f64>,
        weight_grads: &Array2<f64>,
        bias_grads: &Array1<f64>,
    ) {
        self.t += 1;

        // Inicializa m e v se ainda não foram inicializados
        if self.m_w.shape() != weights.shape() {
            self.m_w = Array2::zeros(weights.raw_dim());
            self.v_w = Array2::zeros(weights.raw_dim());
            self.m_b = Array1::zeros(biases.raw_dim());
            self.v_b = Array1::zeros(biases.raw_dim());
        }

        // Incorpora a regularização L2 nos gradientes
        let weight_grads_reg = weight_grads + &(weights.view().mapv(|w| w * self.l2_reg));

        // Atualiza momentos para pesos (operação in-place)
        self.m_w.zip_mut_with(&weight_grads_reg, |m, g| {
            *m = *m * self.beta1 + *g * (1.0 - self.beta1);
        });
        self.v_w.zip_mut_with(&weight_grads_reg, |v, g| {
            *v = *v * self.beta2 + (*g * *g) * (1.0 - self.beta2);
        });

        // Atualiza momentos para vieses (operação in-place)
        self.m_b.zip_mut_with(bias_grads, |m, g| {
            *m = *m * self.beta1 + *g * (1.0 - self.beta1);
        });
        self.v_b.zip_mut_with(bias_grads, |v, g| {
            *v = *v * self.beta2 + (*g * *g) * (1.0 - self.beta2);
        });

        // Calcula correções de viés
        let bias_correction1 = 1.0 - self.beta1.powf(self.t as f64);
        let bias_correction2 = 1.0 - self.beta2.powf(self.t as f64);

        let m_w_hat = &self.m_w / bias_correction1;
        let v_w_hat = &self.v_w / bias_correction2;
        let m_b_hat = &self.m_b / bias_correction1;
        let v_b_hat = &self.v_b / bias_correction2;

        // Convertendo os arrays para slices para aplicar a paralelização
        if let (Some(w_slice), Some(m_w_slice), Some(v_w_slice)) = (
            weights.as_slice_mut(),
            m_w_hat.as_slice(),
            v_w_hat.as_slice(),
        ) {
            w_slice
                .par_iter_mut()
                .zip(m_w_slice.par_iter().zip(v_w_slice.par_iter()))
                .for_each(|(w, (m, v))| {
                    *w -= self.learning_rate * m / (v.sqrt() + self.epsilon);
                });
        }

        if let (Some(b_slice), Some(m_b_slice), Some(v_b_slice)) = (
            biases.as_slice_mut(),
            m_b_hat.as_slice(),
            v_b_hat.as_slice(),
        ) {
            b_slice
                .par_iter_mut()
                .zip(m_b_slice.par_iter().zip(v_b_slice.par_iter()))
                .for_each(|(b, (m, v))| {
                    *b -= self.learning_rate * m / (v.sqrt() + self.epsilon);
                });
        }
    }
}
