// bran/src/optimizer.rs

use ndarray::{Array1, Array2, Zip};

/// Define a interface para otimizadores que ajustam pesos e vieses durante o treinamento de redes neurais.
pub trait Optimizer {
    /// Atualiza os pesos e vieses da camada com base nos gradientes e momentos fornecidos.
    ///
    /// # Parâmetros
    ///
    /// - `weights`: Matriz de pesos.
    /// - `biases`: Vetor de vieses.
    /// - `weight_grads`: Gradientes dos pesos.
    /// - `bias_grads`: Gradientes dos vieses.
    /// - `m_w`: Momento m dos pesos.
    /// - `v_w`: Momento v dos pesos.
    /// - `m_b`: Momento m dos vieses.
    /// - `v_b`: Momento v dos vieses.
    fn update(
        &mut self,
        weights: &mut Array2<f32>,
        biases: &mut Array1<f32>,
        weight_grads: &Array2<f32>,
        bias_grads: &Array1<f32>,
        m_w: &mut Array2<f32>,
        v_w: &mut Array2<f32>,
        m_b: &mut Array1<f32>,
        v_b: &mut Array1<f32>,
    );
}

/// Otimizador SGD (Stochastic Gradient Descent) com suporte para regularização L2.
pub struct SGD {
    pub learning_rate: f32,
    pub l2_reg: f32,
}

impl SGD {
    /// Cria uma nova instância do SGD.
    pub fn new(learning_rate: f32, l2_reg: f32) -> Self {
        SGD {
            learning_rate,
            l2_reg,
        }
    }
}

impl Optimizer for SGD {
    fn update(
        &mut self,
        weights: &mut Array2<f32>,
        biases: &mut Array1<f32>,
        weight_grads: &Array2<f32>,
        bias_grads: &Array1<f32>,
        _m_w: &mut Array2<f32>,
        _v_w: &mut Array2<f32>,
        _m_b: &mut Array1<f32>,
        _v_b: &mut Array1<f32>,
    ) {
        // Atualiza pesos com regularização L2
        Zip::from(weights).and(weight_grads).par_for_each(|w, &wg| {
            *w -= self.learning_rate * (wg + self.l2_reg * *w);
        });

        // Atualiza vieses
        Zip::from(biases).and(bias_grads).par_for_each(|b, &bg| {
            *b -= self.learning_rate * bg;
        });
    }
}

/// Otimizador Adam (Adaptive Moment Estimation) com suporte para regularização L2.
pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub l2_reg: f32,
    pub t: usize, // Contador de iterações
}

impl Adam {
    /// Cria uma nova instância do otimizador Adam.
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32, l2_reg: f32) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            l2_reg,
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn update(
        &mut self,
        weights: &mut Array2<f32>,
        biases: &mut Array1<f32>,
        weight_grads: &Array2<f32>,
        bias_grads: &Array1<f32>,
        m_w: &mut Array2<f32>,
        v_w: &mut Array2<f32>,
        m_b: &mut Array1<f32>,
        v_b: &mut Array1<f32>,
    ) {
        self.t += 1; // Incrementa o contador de iterações

        // Atualiza os momentos dos pesos
        Zip::from(m_w.view_mut())
            .and(weight_grads.view())
            .par_for_each(|mw, &wg| {
                *mw = self.beta1 * *mw + (1.0 - self.beta1) * wg;
            });
        Zip::from(v_w.view_mut())
            .and(weight_grads.view())
            .par_for_each(|vw, &wg| {
                *vw = self.beta2 * *vw + (1.0 - self.beta2) * wg * wg;
            });

        // Atualiza os momentos dos vieses
        Zip::from(m_b.view_mut())
            .and(bias_grads.view())
            .par_for_each(|mb, &bg| {
                *mb = self.beta1 * *mb + (1.0 - self.beta1) * bg;
            });
        Zip::from(v_b.view_mut())
            .and(bias_grads.view())
            .par_for_each(|vb, &bg| {
                *vb = self.beta2 * *vb + (1.0 - self.beta2) * bg * bg;
            });

        // Correção de viés
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        let m_w_hat = &*m_w / bias_correction1;
        let v_w_hat = &*v_w / bias_correction2;
        let m_b_hat = &*m_b / bias_correction1;
        let v_b_hat = &*v_b / bias_correction2;

        // Atualiza os pesos com regularização L2
        Zip::from(weights.view_mut())
            .and(m_w_hat.view())
            .and(v_w_hat.view())
            .par_for_each(|w, &mw, &vw| {
                *w -= self.learning_rate * mw / (vw.sqrt() + self.epsilon)
                    + self.l2_reg * self.learning_rate * *w;
            });

        // Atualiza os vieses
        Zip::from(biases.view_mut())
            .and(m_b_hat.view())
            .and(v_b_hat.view())
            .par_for_each(|b, &mb, &vb| {
                *b -= self.learning_rate * mb / (vb.sqrt() + self.epsilon);
            });
    }
}
