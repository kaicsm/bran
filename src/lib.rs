pub mod activations;
pub mod layers;
pub mod model;
pub mod optimizer;

// Re-exporte os itens principais para facilitar o uso
pub use activations::ActivationType;
pub use layers::DenseLayer;
pub use model::NeuralNetwork;
pub use optimizer::SGD;
