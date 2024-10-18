// bran/src/lib.rs

pub mod activations;
pub mod layers;
pub mod loss;
pub mod model;
pub mod optimizer;
pub mod prelude;
pub mod visualization;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, array};
    use prelude::*;

    #[test]
    fn test_activation_relu() {
        let relu = ActivationType::ReLU;
        assert_eq!(relu.activate(1.0), 1.0);
        assert_eq!(relu.activate(-1.0), 0.0);
        assert_eq!(relu.derivative(1.0), 1.0);
        assert_eq!(relu.derivative(-1.0), 0.0);
    }

    #[test]
    fn test_activation_sigmoid() {
        let sigmoid = ActivationType::Sigmoid;
        assert_abs_diff_eq!(sigmoid.activate(0.0), 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(sigmoid.derivative(0.0), 0.25, epsilon = 1e-6);
    }

    #[test]
    fn test_dense_layer() {
        let mut layer = DenseLayer::new(2, 3, ActivationType::ReLU);
        let input = arr2(&[[1.0, 2.0]]);
        let output = layer.forward(&input);
        assert_eq!(output.shape(), &[1, 3]);
    }

    #[test]
    fn test_neural_network() {
        let mut nn = NeuralNetwork::new();
        nn.add_layer(DenseLayer::new(2, 3, ActivationType::ReLU));
        nn.add_layer(DenseLayer::new(3, 1, ActivationType::Sigmoid));
        let input = arr2(&[[1.0, 2.0]]);
        let output = nn.forward(&input);
        assert_eq!(output.shape(), &[1, 1]);
    }

    #[test]
    fn test_sgd_optimizer() {
        let mut sgd = SGD::new(0.01, 0.0);
        let mut weights = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mut biases = arr1(&[0.1, 0.2]);
        let weight_grads = arr2(&[[0.1, 0.2], [0.3, 0.4]]);
        let bias_grads = arr1(&[0.01, 0.02]);
        sgd.update(
            &mut weights,
            &mut biases,
            &weight_grads,
            &bias_grads,
            &mut arr2(&[[0.0, 0.0], [0.0, 0.0]]),
            &mut arr2(&[[0.0, 0.0], [0.0, 0.0]]),
            &mut arr1(&[0.0, 0.0]),
            &mut arr1(&[0.0, 0.0]),
        );
        assert_abs_diff_eq!(weights[[0, 0]], 0.999, epsilon = 1e-6);
        assert_abs_diff_eq!(biases[0], 0.0999, epsilon = 1e-6);
    }

    #[test]
    fn test_adam_optimizer() {
        // Configuração inicial
        let mut weights = array![[0.5, 0.3], [0.7, 0.9]];
        let mut biases = array![0.1, 0.2];
        let weight_grads = array![[0.05, 0.02], [0.03, 0.04]];
        let bias_grads = array![0.01, 0.02];
        let mut m_w = array![[0.0, 0.0], [0.0, 0.0]];
        let mut v_w = array![[0.0, 0.0], [0.0, 0.0]];
        let mut m_b = array![0.0, 0.0];
        let mut v_b = array![0.0, 0.0];

        // Cria o otimizador Adam
        let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0);

        // Executa várias atualizações do Adam
        for _ in 0..10 {
            adam.update(
                &mut weights,
                &mut biases,
                &weight_grads,
                &bias_grads,
                &mut m_w,
                &mut v_w,
                &mut m_b,
                &mut v_b,
            );
        }
    }

    #[test]
    fn test_mse_loss() {
        let mse = MeanSquaredError;
        let predicted = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let target = arr2(&[[1.5, 2.5], [3.5, 4.5]]);
        let loss = mse.loss(&predicted, &target);
        assert_abs_diff_eq!(loss, 0.125, epsilon = 1e-6);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let ce = CrossEntropyLoss;
        let predicted = arr2(&[[0.6, 0.4], [0.3, 0.7]]);
        let target = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let loss = ce.loss(&predicted, &target);
        assert_abs_diff_eq!(loss, 0.4337, epsilon = 1e-4);
    }

    #[test]
    fn test_training_stats() {
        use visualization::TrainingStats;
        let mut stats = TrainingStats::new();
        stats.log_epoch(1.0, 0.5, 0.8);
        stats.log_epoch(2.0, 0.3, 0.9);
        assert_eq!(stats.epochs, vec![1.0, 2.0]);
        assert_eq!(stats.losses, vec![0.5, 0.3]);
    }
}
