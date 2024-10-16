// src/layers.rs
mod activations;
mod layers;
mod model;
mod optimizer;

use std::path::Path;

use activations::ActivationType;
use layers::DenseLayer;
use model::NeuralNetwork;

use ndarray::Array1;
use optimizer::Adam;
use rand::Rng;

use rand::seq::SliceRandom;

fn normalize_input(input: &Array1<f64>) -> Array1<f64> {
    let mean = input.mean().unwrap();
    let std = input.std(0.0);
    (input - mean) / std
}

fn generate_derivative_data(num_samples: usize) -> (Vec<Array1<f64>>, Vec<Array1<f64>>) {
    let mut rng = rand::thread_rng();
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for _ in 0..num_samples {
        // Gerar coeficientes aleatórios para um polinômio de segundo grau: ax^2 + bx + c
        let a = rng.gen_range(-2.0..2.0);
        let b = rng.gen_range(-2.0..2.0);
        let c = rng.gen_range(-2.0..2.0);

        // O ponto x onde queremos calcular a derivada
        let x = rng.gen_range(-1.0..1.0);

        // Input: coeficientes do polinômio e o ponto x
        let input = Array1::from(vec![a, b, c, x]);

        // Target: a derivada no ponto x
        // A derivada de ax^2 + bx + c é 2ax + b
        let derivative = 2.0 * a * x + b;
        let target = Array1::from(vec![derivative]);

        inputs.push(input);
        targets.push(target);
    }

    (inputs, targets)
}

fn main() {
    let model_path = "derivative_model.json";
    let mut network: NeuralNetwork;

    // Tenta carregar o modelo existente
    if Path::new(model_path).exists() {
        println!("Carregando modelo existente...");
        network = NeuralNetwork::load(model_path).expect("Erro ao carregar o modelo");
        println!("Modelo carregado com sucesso.");
    } else {
        println!("Criando novo modelo...");
        network = NeuralNetwork::new();
        network.add_layer(DenseLayer::new(4, 32, ActivationType::ReLU));
        network.add_layer(DenseLayer::new(32, 32, ActivationType::ReLU));
        network.add_layer(DenseLayer::new(32, 16, ActivationType::ReLU));
        network.add_layer(DenseLayer::new(16, 1, ActivationType::Linear));
    }

    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

    let num_epochs = 1000;
    let batch_size = 64;
    let num_samples = 10000;

    let mut best_val_loss = f64::INFINITY;
    let mut epochs_without_improvement = 0;

    for epoch in 0..num_epochs {
        let (inputs, targets) = generate_derivative_data(num_samples);

        // Shuffle the data
        let mut indices: Vec<usize> = (0..num_samples).collect();
        indices.shuffle(&mut rand::thread_rng());

        let mut total_loss = 0.0;
        for batch in 0..(num_samples / batch_size) {
            let start = batch * batch_size;
            let end = (batch + 1) * batch_size;

            let batch_indices = &indices[start..end];
            let batch_inputs: Vec<_> = batch_indices
                .iter()
                .map(|&i| normalize_input(&inputs[i]))
                .collect();
            let batch_targets: Vec<_> = batch_indices.iter().map(|&i| targets[i].clone()).collect();

            let mut batch_loss = 0.0;
            for (input, target) in batch_inputs.iter().zip(batch_targets.iter()) {
                let output = network.forward(input);
                let error = &output - target;
                let loss = error.mapv(|e| e.powi(2)).sum();
                batch_loss += loss;

                network.backward(&error, &mut optimizer);
            }

            total_loss += batch_loss;
        }

        // Learning rate decay
        if epoch % 1000 == 0 {
            optimizer.learning_rate *= 0.9;
        }

        // Validation
        if epoch % 100 == 0 {
            let (val_inputs, val_targets) = generate_derivative_data(1000);
            let mut val_loss = 0.0;
            for (input, target) in val_inputs.iter().zip(val_targets.iter()) {
                let normalized_input = normalize_input(input);
                let output = network.forward(&normalized_input);
                let error = &output - target;
                val_loss += error.mapv(|e| e.powi(2)).sum();
            }
            val_loss /= 1000.0;

            println!(
                "Epoch {}: Train Loss = {:.4}, Val Loss = {:.4}",
                epoch,
                total_loss / num_samples as f64,
                val_loss
            );

            // Early stopping
            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
                if epochs_without_improvement >= 10 {
                    println!("Early stopping triggered.");
                    break;
                }
            }
        }
    }

    // Teste final
    println!("\nResultados finais:");
    let (test_inputs, test_targets) = generate_derivative_data(10);
    for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
        let output = network.forward(input);
        println!(
                "Input: a={:.2}, b={:.2}, c={:.2}, x={:.2}, Predicted Derivative: {:.4}, Actual Derivative: {:.4}",
                input[0], input[1], input[2], input[3], output[0], target[0]
            );
    }

    // Salvar o modelo
    println!("\nSalvando o modelo...");
    network.save(model_path).expect("Erro ao salvar o modelo");
    println!("Modelo salvo em '{}'", model_path);
}
