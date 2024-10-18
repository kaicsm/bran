// bran/src/prelude.rs

// Re-exporte os itens principais para facilitar o uso
pub use crate::activations::{Activation, ActivationType};
pub use crate::layers::DenseLayer;
pub use crate::loss::{CrossEntropyLoss, Loss, MeanSquaredError};
pub use crate::model::NeuralNetwork;
pub use crate::optimizer::{Adam, Optimizer, SGD};
pub use crate::visualization::TrainingStats;

pub use ndarray;
pub use rand;
