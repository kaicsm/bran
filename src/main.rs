mod activations;
mod layers;
mod model;
mod optimizer;

use activations::ActivationType;
use layers::DenseLayer;
use model::NeuralNetwork;
use ndarray::Array1;
use optimizer::SGD;

fn main() {
    // Criação do modelo
    let mut network = NeuralNetwork::new();
    network.add_layer(DenseLayer::new(2, 8, ActivationType::ReLU)); // Aumentamos para 8 neurônios
    network.add_layer(DenseLayer::new(8, 1, ActivationType::Sigmoid));

    // Optimizador
    let optimizer = SGD::new(0.05);

    // Dados de treinamento (exemplo simples: XOR)
    let inputs = vec![
        Array1::from(vec![0.0, 0.0]),
        Array1::from(vec![0.0, 1.0]),
        Array1::from(vec![1.0, 0.0]),
        Array1::from(vec![1.0, 1.0]),
    ];
    let targets = vec![
        Array1::from(vec![0.0]),
        Array1::from(vec![1.0]),
        Array1::from(vec![1.0]),
        Array1::from(vec![0.0]),
    ];

    // Ciclo de treinamento
    for epoch in 0..15000 {
        let mut total_loss = 0.0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            // Forward pass
            let output = network.forward(input);

            // Calculando o erro (perda) - MSE (Mean Squared Error)
            let error = &output - target;
            let loss = error.mapv(|e| e.powi(2)).sum();
            total_loss += loss;

            // Backpropagation e atualização dos pesos
            network.backward(&error, &optimizer);
        }

        // A cada 1000 épocas, mostrar a perda
        if epoch % 1000 == 0 {
            println!(
                "Epoch {}: Loss = {:.4}",
                epoch,
                total_loss / inputs.len() as f64
            );
        }
    }

    // Teste após o treinamento
    println!("\nResultados finais:");
    for input in &inputs {
        let output = network.forward(input);
        println!("Input: {:?}, Output: {:.4}", input, output[0]);
    }
}
