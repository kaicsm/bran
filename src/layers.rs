// src/layers.rs

use crate::activations::{Activation, ActivationType};
use ndarray::{Array1, Array2, Axis};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Serialize)]
pub struct DenseLayer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation_type: ActivationType,

    #[serde(skip)] // Ignora a ativação, pois é um trait object
    pub activation: Option<Box<dyn Activation>>,

    #[serde(skip)]
    pub input: Option<Array2<f32>>, // Agora suporta lotes

    #[serde(skip)]
    pub output: Option<Array2<f32>>, // Agora suporta lotes

    #[serde(skip)] // Estado do otimizador Adam
    pub m_w: Array2<f32>,

    #[serde(skip)] // Estado do otimizador Adam
    pub v_w: Array2<f32>,

    #[serde(skip)] // Estado do otimizador Adam
    pub m_b: Array1<f32>,

    #[serde(skip)] // Estado do otimizador Adam
    pub v_b: Array1<f32>,
}

impl DenseLayer {
    /// Cria uma nova camada densa com inicialização de pesos e vieses
    pub fn new(input_size: usize, output_size: usize, activation_type: ActivationType) -> Self {
        let mut rng = thread_rng();
        let std_dev = (2.0 / (input_size + output_size) as f32).sqrt();
        let dist = Uniform::new(-std_dev, std_dev);
        let weights = Array2::from_shape_fn((output_size, input_size), |_| dist.sample(&mut rng));
        let biases = Array1::zeros(output_size);

        // Inicializa os momentos do Adam com zeros
        let m_w = Array2::zeros((output_size, input_size));
        let v_w = Array2::zeros((output_size, input_size));
        let m_b = Array1::zeros(output_size);
        let v_b = Array1::zeros(output_size);

        DenseLayer {
            weights,
            biases,
            activation: Some(Box::new(activation_type.clone())), // Instancia a ativação
            activation_type: activation_type.clone(), // Guarda o tipo de ativação para serialização
            input: None,
            output: None,
            m_w,
            v_w,
            m_b,
            v_b,
        }
    }

    /// Restaura a ativação após a desserialização
    pub fn restore_activation(&mut self) {
        self.activation = Some(Box::new(self.activation_type.clone()));
    }

    /// Realiza a passagem forward para um lote de entradas
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.input = Some(input.clone());
        // Cálculo z = input * weights^T + biases
        let z = input.dot(&self.weights.t()) + &self.biases;
        // Aplica a função de ativação em lote
        let output = self.activation.as_ref().unwrap().activate_array(&z);
        self.output = Some(output.clone());
        output
    }

    /// Realiza a passagem backward para um lote de erros de saída
    pub fn backward(
        &mut self,
        output_error: &Array2<f32>,
        optimizer: &mut dyn crate::optimizer::Optimizer, // Usando trait genérico
    ) -> Array2<f32> {
        let input = self.input.as_ref().unwrap();
        let output = self.output.as_ref().unwrap();

        // Calcula o delta aplicando a derivada da ativação
        let activation_derivative = self.activation.as_ref().unwrap().derivative_array(&output);
        let delta = output_error * &activation_derivative;

        // Calcula o erro para a camada anterior
        let input_error = delta.dot(&self.weights);

        // Calcula os gradientes dos pesos e vieses
        let weight_gradients = delta.t().dot(input);
        let bias_gradients = delta.sum_axis(Axis(0));

        // Atualiza os pesos e vieses utilizando o otimizador
        optimizer.update(
            &mut self.weights,
            &mut self.biases,
            &weight_gradients,
            &bias_gradients,
            &mut self.m_w,
            &mut self.v_w,
            &mut self.m_b,
            &mut self.v_b,
        );

        input_error
    }
}

// Implementação manual da deserialização para lidar com o campo "activation" e inicializar os estados do otimizador
impl<'de> Deserialize<'de> for DenseLayer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct DenseLayerData {
            weights: Array2<f32>,
            biases: Array1<f32>,
            activation_type: ActivationType,
        }

        let data = DenseLayerData::deserialize(deserializer)?;

        // Inicializa os momentos do Adam com zeros após a desserialização
        let m_w = Array2::zeros(data.weights.raw_dim());
        let v_w = Array2::zeros(data.weights.raw_dim());
        let m_b = Array1::zeros(data.biases.len());
        let v_b = Array1::zeros(data.biases.len());

        let mut layer = DenseLayer {
            weights: data.weights,
            biases: data.biases,
            activation_type: data.activation_type.clone(),
            activation: None, // Inicialmente None, será restaurado a seguir
            input: None,
            output: None,
            m_w,
            v_w,
            m_b,
            v_b,
        };

        // Restaurar a ativação após a desserialização
        layer.restore_activation();

        Ok(layer)
    }
}
