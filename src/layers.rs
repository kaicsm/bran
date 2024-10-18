// bran/src/layers.rs

use crate::activations::{Activation, ActivationType};
use ndarray::{Array1, Array2, Axis};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use serde::{Deserialize, Deserializer, Serialize};

/// Representa uma camada densa (totalmente conectada) em uma rede neural.
#[derive(Serialize)]
pub struct DenseLayer {
    /// Matriz de pesos da camada.
    pub weights: Array2<f32>,
    /// Vetor de vieses da camada.
    pub biases: Array1<f32>,
    /// Tipo de função de ativação usada pela camada.
    pub activation_type: ActivationType,

    /// Função de ativação da camada (ignorada na serialização).
    #[serde(skip)]
    pub activation: Option<Box<dyn Activation>>,

    /// Última entrada processada pela camada (ignorada na serialização).
    #[serde(skip)]
    pub input: Option<Array2<f32>>,

    /// Última saída produzida pela camada (ignorada na serialização).
    #[serde(skip)]
    pub output: Option<Array2<f32>>,

    // Campos para o otimizador Adam (todos ignorados na serialização)
    /// Momento de primeira ordem para pesos (Adam).
    #[serde(skip)]
    pub m_w: Array2<f32>,
    /// Momento de segunda ordem para pesos (Adam).
    #[serde(skip)]
    pub v_w: Array2<f32>,
    /// Momento de primeira ordem para vieses (Adam).
    #[serde(skip)]
    pub m_b: Array1<f32>,
    /// Momento de segunda ordem para vieses (Adam).
    #[serde(skip)]
    pub v_b: Array1<f32>,
}

impl DenseLayer {
    /// Cria uma nova camada densa com inicialização de pesos e vieses.
    ///
    /// # Argumentos
    ///
    /// * `input_size` - Número de neurônios na camada de entrada.
    /// * `output_size` - Número de neurônios nesta camada.
    /// * `activation_type` - Tipo de função de ativação a ser usada.
    ///
    /// # Retorno
    ///
    /// Retorna uma nova instância de `DenseLayer`.
    pub fn new(input_size: usize, output_size: usize, activation_type: ActivationType) -> Self {
        let mut rng = thread_rng();
        let std_dev = (2.0 / (input_size + output_size) as f32).sqrt();
        let dist = Uniform::new(-std_dev, std_dev);
        let weights = Array2::from_shape_fn((output_size, input_size), |_| dist.sample(&mut rng));
        let biases = Array1::zeros(output_size);

        let m_w = Array2::zeros((output_size, input_size));
        let v_w = Array2::zeros((output_size, input_size));
        let m_b = Array1::zeros(output_size);
        let v_b = Array1::zeros(output_size);

        DenseLayer {
            weights,
            biases,
            activation: Some(Box::new(activation_type.clone())),
            activation_type: activation_type.clone(),
            input: None,
            output: None,
            m_w,
            v_w,
            m_b,
            v_b,
        }
    }

    /// Restaura a função de ativação após a desserialização.
    pub fn restore_activation(&mut self) {
        self.activation = Some(Box::new(self.activation_type.clone()));
    }

    /// Realiza a passagem forward para um lote de entradas.
    ///
    /// # Argumentos
    ///
    /// * `input` - Array 2D contendo o lote de entradas.
    ///
    /// # Retorno
    ///
    /// Retorna um Array 2D contendo as saídas da camada para o lote de entradas.
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.input = Some(input.clone());
        let z = input.dot(&self.weights.t()) + &self.biases;
        let output = self.activation.as_ref().unwrap().activate_array(&z);
        self.output = Some(output.clone());
        output
    }

    /// Realiza a passagem backward para um lote de erros de saída.
    ///
    /// # Argumentos
    ///
    /// * `output_error` - Array 2D contendo os erros da camada seguinte.
    /// * `optimizer` - Otimizador usado para atualizar os pesos e vieses.
    ///
    /// # Retorno
    ///
    /// Retorna um Array 2D contendo os erros propagados para a camada anterior.
    pub fn backward(
        &mut self,
        output_error: &Array2<f32>,
        optimizer: &mut dyn crate::optimizer::Optimizer,
    ) -> Array2<f32> {
        let input = self.input.as_ref().unwrap();
        let output = self.output.as_ref().unwrap();

        let activation_derivative = self.activation.as_ref().unwrap().derivative_array(&output);
        let delta = output_error * &activation_derivative;

        let input_error = delta.dot(&self.weights);

        let weight_gradients = delta.t().dot(input);
        let bias_gradients = delta.sum_axis(Axis(0));

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

/// Implementação manual da deserialização para DenseLayer.
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

        let m_w = Array2::zeros(data.weights.raw_dim());
        let v_w = Array2::zeros(data.weights.raw_dim());
        let m_b = Array1::zeros(data.biases.len());
        let v_b = Array1::zeros(data.biases.len());

        let mut layer = DenseLayer {
            weights: data.weights,
            biases: data.biases,
            activation_type: data.activation_type.clone(),
            activation: None,
            input: None,
            output: None,
            m_w,
            v_w,
            m_b,
            v_b,
        };

        layer.restore_activation();

        Ok(layer)
    }
}
