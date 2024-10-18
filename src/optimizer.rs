// src/optimizer.rs

use ndarray::{Array1, Array2};

/// Trait que define métodos para otimizadores.
pub trait Optimizer {
    /// Atualiza os pesos e vieses utilizando os gradientes e estados do otimizador.
    ///
    /// # Parâmetros
    ///
    /// - `weights`: Referência mutável para os pesos da camada.
    /// - `biases`: Referência mutável para os vieses da camada.
    /// - `weight_grads`: Referência aos gradientes dos pesos.
    /// - `bias_grads`: Referência aos gradientes dos vieses.
    /// - `m_w`: Referência mutável para o momento m dos pesos.
    /// - `v_w`: Referência mutável para o momento v dos pesos.
    /// - `m_b`: Referência mutável para o momento m dos vieses.
    /// - `v_b`: Referência mutável para o momento v dos vieses.
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

/// Estrutura para o otimizador SGD (Stochastic Gradient Descent) com regularização L2.
pub struct SGD {
    pub learning_rate: f32,
    pub l2_reg: f32,
}

impl SGD {
    /// Cria um novo otimizador SGD com a taxa de aprendizado e regularização L2 configuráveis.
    ///
    /// # Parâmetros
    ///
    /// - `learning_rate`: Taxa de aprendizado para as atualizações dos parâmetros.
    /// - `l2_reg`: Fator de regularização L2 para evitar overfitting.
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
        _m_w: &mut Array2<f32>, // SGD não utiliza momentos
        _v_w: &mut Array2<f32>,
        _m_b: &mut Array1<f32>,
        _v_b: &mut Array1<f32>,
    ) {
        // Aplicação da regularização L2 apenas nos pesos
        *weights -=
            &(weight_grads * self.learning_rate + &(*weights) * self.l2_reg * self.learning_rate);

        // Atualização dos vieses sem regularização L2
        *biases -= &(bias_grads * self.learning_rate);
    }
}

/// Estrutura para o otimizador Adam com regularização L2.
pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub l2_reg: f32,
    pub t: usize, // Passo de tempo para correção de viés
}

impl Adam {
    /// Cria um novo otimizador Adam com parâmetros configuráveis.
    ///
    /// # Parâmetros
    ///
    /// - `learning_rate`: Taxa de aprendizado para as atualizações dos parâmetros.
    /// - `beta1`: Fator de decaimento para o momento m.
    /// - `beta2`: Fator de decaimento para o momento v.
    /// - `epsilon`: Pequeno valor para evitar divisão por zero.
    /// - `l2_reg`: Fator de regularização L2 para evitar overfitting.
    #[allow(dead_code)]
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
        self.t += 1;

        // Atualiza momentos para pesos
        *m_w *= self.beta1;
        *m_w += &(weight_grads * (1.0 - self.beta1));
        *v_w *= self.beta2;
        *v_w += &(weight_grads.mapv(|g| g * g) * (1.0 - self.beta2));

        // Atualiza momentos para vieses
        *m_b *= self.beta1;
        *m_b += &(bias_grads * (1.0 - self.beta1));
        *v_b *= self.beta2;
        *v_b += &(bias_grads.mapv(|g| g * g) * (1.0 - self.beta2));

        // Correção de viés
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        let m_w_hat = &*m_w / bias_correction1;
        let v_w_hat = &*v_w / bias_correction2;
        let m_b_hat = &*m_b / bias_correction1;
        let v_b_hat = &*v_b / bias_correction2;

        // Atualiza pesos com regularização L2
        *weights -= &(m_w_hat * self.learning_rate / (v_w_hat.mapv(|v| v.sqrt()) + self.epsilon)
            + &(*weights) * self.l2_reg * self.learning_rate);

        // Atualiza vieses sem regularização L2
        *biases -= &(m_b_hat * self.learning_rate / (v_b_hat.mapv(|v| v.sqrt()) + self.epsilon));
    }
}
