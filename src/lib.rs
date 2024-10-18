// bran/src/lib.rs

pub mod activations;
pub mod layers;
pub mod loss;
pub mod model;
pub mod optimizer;
pub mod visualization;

// Re-exporte os itens principais para facilitar o uso
pub use activations::ActivationType;
pub use layers::DenseLayer;
pub use model::NeuralNetwork;
pub use optimizer::SGD;

pub use ndarray;
pub use rand;
