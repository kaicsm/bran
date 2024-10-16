// src/main.rs
mod activations;
mod layers;
mod model;
mod optimizer;

use activations::ActivationType;
use layers::DenseLayer;
use model::NeuralNetwork;
use optimizer::SGD;
use ndarray::{array, s};
use std::path::Path;

fn main() {
    let model_path = "model.bin";
    
    // Carregar o modelo se o arquivo existir
    let mut network = if Path::new(model_path).exists() {
        println!("Carregando o modelo salvo...");
        NeuralNetwork::load(model_path).expect("Erro ao carregar o modelo")
    } else {
        // Caso contrário, crie um novo modelo
        println!("Criando um novo modelo...");
        let mut network = NeuralNetwork::new();
        network.add_layer(DenseLayer::new(2, 8, ActivationType::ReLU)); // Aumentamos para 8 neurônios
        network.add_layer(DenseLayer::new(8, 1, ActivationType::Sigmoid));
        network
    };

    // Optimizador
    let mut optimizer = SGD::new(0.05, 1e-5);

    // Dados de treinamento (exemplo simples: XOR)
    let inputs = array![
        [0.0f32, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];

    let targets = array![
        [0.0f32],
        [1.0],
        [1.0],
        [0.0],
    ];

    // Ciclo de treinamento
    for epoch in 0..100000 {
        // Forward pass
        let outputs = network.forward(&inputs);

        // Calculando o erro (perda) - MSE (Mean Squared Error)
        let errors = &outputs - &targets;
        let loss = errors.mapv(|e| e.powi(2)).sum() / inputs.nrows() as f32;

        // Backpropagation e atualização dos pesos
        network.backward(&errors, &mut optimizer).unwrap();

        // A cada 1000 épocas, mostrar a perda
        if epoch % 1000 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss);
        }
    }

    // Teste após o treinamento
    println!("\nResultados finais:");
    for i in 0..inputs.nrows() {
        let input = inputs.slice(s![i..i+1, ..]).to_owned();  // Cópia da fatia para Array2<f32>
        let output = network.forward(&input); // Agora passamos a cópia
        println!("Input: {:?}, Output: {:.4}", input, output[[0, 0]]);
    }

    // Salvar o modelo após o treinamento
    println!("\nSalvando o modelo...");
    network.save(model_path).expect("Erro ao salvar o modelo");
    println!("Modelo salvo em '{}'", model_path);
}

