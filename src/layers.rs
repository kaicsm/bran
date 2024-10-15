// src/layers.rs
use crate::activations::{Activation, ActivationType};
use ndarray::{Array1, Array2};
use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub activation: Box<dyn Activation>,
    input: Option<Array1<f64>>,
    output: Option<Array1<f64>>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, activation: ActivationType) -> Self {
        let mut rng = thread_rng();
        let std_dev = (2.0 / (input_size + output_size) as f64).sqrt();
        let dist = Uniform::new(-std_dev, std_dev);
        let weights = Array2::from_shape_fn((output_size, input_size), |_| dist.sample(&mut rng));
        let biases = Array1::zeros(output_size);

        DenseLayer {
            weights,
            biases,
            activation: Box::new(activation),
            input: None,
            output: None,
        }
    }

    pub fn forward(&mut self, input: &Array1<f64>) -> Array1<f64> {
        self.input = Some(input.clone());
        let z = input.dot(&self.weights.t()) + &self.biases;
        let output = z.mapv(|x| self.activation.activate(x));
        self.output = Some(output.clone());
        output
    }

    pub fn backward(
        &mut self,
        output_error: &Array1<f64>,
    ) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
        let input = self.input.as_ref().unwrap();
        let output = self.output.as_ref().unwrap();

        let delta = output_error * &output.mapv(|x| self.activation.derivative(x));
        let input_error = delta.dot(&self.weights);

        let weight_gradients = delta
            .view()
            .insert_axis(ndarray::Axis(1))
            .dot(&input.view().insert_axis(ndarray::Axis(0)));
        let bias_gradients = delta.clone();

        (input_error, weight_gradients, bias_gradients)
    }
}
