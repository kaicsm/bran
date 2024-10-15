// src/layers.rs
use crate::activations::{Activation, ActivationType};
use ndarray::{Array1, Array2};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Serialize)]
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub activation_type: ActivationType,
    
    #[serde(skip)] // Ignora a ativação, pois é um trait object
    pub activation: Option<Box<dyn Activation>>,
    
    #[serde(skip)]
    pub input: Option<Array1<f64>>,
    
    #[serde(skip)]
    pub output: Option<Array1<f64>>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, activation_type: ActivationType) -> Self {
        let mut rng = thread_rng();
        let std_dev = (2.0 / (input_size + output_size) as f64).sqrt();
        let dist = Uniform::new(-std_dev, std_dev);
        let weights = Array2::from_shape_fn((output_size, input_size), |_| dist.sample(&mut rng));
        let biases = Array1::zeros(output_size);

        DenseLayer {
            weights,
            biases,
            activation: Some(Box::new(activation_type.clone())), // Instancia a ativação
            activation_type, // Guarda o tipo de ativação para serialização
            input: None,
            output: None,
        }
    }

    pub fn restore_activation(&mut self) {
        self.activation = Some(Box::new(self.activation_type.clone()));
    }

    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.input = Some(input.clone());
        let z = input.dot(&self.weights.t()) + &self.biases;
        let output = z.mapv(|x| self.activation.as_ref().unwrap().activate(x));
        self.output = Some(output.clone());
        output
    }

    pub fn backward(
        &mut self,
        output_error: &Array1<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
        let input = self.input.as_ref().unwrap();
        let output = self.output.as_ref().unwrap();

        let delta = output_error * &output.mapv(|x| self.activation.as_ref().unwrap().derivative(x));
        let input_error = delta.dot(&self.weights);

        let weight_gradients = delta
            .view()
            .insert_axis(ndarray::Axis(1))
            .dot(&input.view().insert_axis(ndarray::Axis(0)));
        let bias_gradients = delta.clone();

        (input_error, weight_gradients, bias_gradients)
    }
}

// Implementação manual da deserialização para lidar com o campo "activation"
impl<'de> Deserialize<'de> for DenseLayer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct DenseLayerData {
            weights: Array2<f64>,
            biases: Array1<f64>,
            activation_type: ActivationType,
        }

        let data = DenseLayerData::deserialize(deserializer)?;

        let mut layer = DenseLayer {
            weights: data.weights,
            biases: data.biases,
            activation_type: data.activation_type.clone(),
            activation: None, // Inicialmente None, será restaurado a seguir
            input: None,
            output: None,
        };

        // Restaurar a ativação após a desserialização
        layer.restore_activation();

        Ok(layer)
    }
}

