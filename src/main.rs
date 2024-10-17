use rocket::fs::{relative, FileServer};
use rocket::http::Status;
use rocket::response::status::Custom;
use rocket::serde::json::Json;
use rocket::{get, launch, post, routes, State};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

// Módulos da sua aplicação
mod activations;
mod layers;
mod loss;
mod model;
mod optimizer;
mod visualization;

use activations::ActivationType;
use layers::DenseLayer;
use loss::MeanSquaredError;
use model::NeuralNetwork;
use optimizer::{Optimizer, SGD};
use visualization::TrainingStats;

// Estrutura para estado compartilhado
struct AppState {
    network: Arc<Mutex<Option<NeuralNetwork>>>,
    stats: Arc<Mutex<TrainingStats>>,
}

// Configuração das camadas enviada pelo usuário
#[derive(Serialize, Deserialize, Clone)]
struct LayerConfig {
    input_size: usize,
    output_size: usize,
    activation: String,
}

// Requisição para treino
#[derive(Serialize, Deserialize, Clone)]
struct TrainRequest {
    epochs: usize,
    batch_size: usize,
    learning_rate: f32,
    l2_reg: f32,
    optimizer: String,
    layers: Vec<LayerConfig>,
    x_train: Vec<Vec<f32>>,
    y_train: Vec<Vec<f32>>,
}

// Resposta do treino
#[derive(Serialize)]
struct TrainResponse {
    message: String,
}

#[derive(Serialize, Deserialize)]
struct TestRequest {
    x_test: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct TestResponse {
    predictions: Vec<Vec<f32>>,
}

#[post("/test", data = "<test_request>")]
fn test_model(
    test_request: Json<TestRequest>,
    state: &State<AppState>,
) -> Result<Json<TestResponse>, Custom<String>> {
    let test_data = test_request.into_inner();
    let x_test = ndarray::Array2::from_shape_vec(
        (test_data.x_test.len(), test_data.x_test[0].len()),
        test_data.x_test.into_iter().flatten().collect(),
    )
    .map_err(|e| Custom(Status::BadRequest, e.to_string()))?;

    let mut network = state.network.lock().unwrap();
    if let Some(ref mut net) = *network {
        let predictions = net.forward(&x_test);
        let predictions_vec = predictions.outer_iter().map(|row| row.to_vec()).collect();
        Ok(Json(TestResponse {
            predictions: predictions_vec,
        }))
    } else {
        Err(Custom(
            Status::BadRequest,
            "Nenhum modelo disponível para teste".to_string(),
        ))
    }
}

// Rota POST para iniciar o treino
#[post("/train", data = "<train_request>")]
fn train(train_request: Json<TrainRequest>, state: &State<AppState>) -> Json<TrainResponse> {
    let train_data = train_request.into_inner();
    let x_train = ndarray::Array2::from_shape_vec(
        (train_data.x_train.len(), train_data.x_train[0].len()),
        train_data.x_train.into_iter().flatten().collect(),
    )
    .expect("Erro ao converter x_train");

    let y_train = ndarray::Array2::from_shape_vec(
        (train_data.y_train.len(), train_data.y_train[0].len()),
        train_data.y_train.into_iter().flatten().collect(),
    )
    .expect("Erro ao converter y_train");

    // Construção dinâmica do modelo
    let mut network = NeuralNetwork::new();
    for layer in train_data.layers {
        let activation = match layer.activation.as_str() {
            "ReLU" => ActivationType::ReLU,
            "Sigmoid" => ActivationType::Sigmoid,
            "Tanh" => ActivationType::Tanh, // Agora Tanh está disponível
            _ => ActivationType::ReLU,      // Valor padrão
        };
        network.add_layer(DenseLayer::new(
            layer.input_size,
            layer.output_size,
            activation,
        ));
    }

    // Escolha do otimizador
    let mut optimizer: Box<dyn Optimizer + Send> = match train_data.optimizer.as_str() {
        "SGD" => Box::new(SGD::new(train_data.learning_rate, train_data.l2_reg)),
        // Adicione outros otimizadores aqui
        _ => Box::new(SGD::new(train_data.learning_rate, train_data.l2_reg)), // Valor padrão
    };

    let loss_fn = MeanSquaredError;

    // Clonando estado compartilhado
    let stats_clone = Arc::clone(&state.stats);
    let network_arc = Arc::clone(&state.network);

    // Reiniciar estatísticas
    {
        let mut stats_lock = stats_clone.lock().unwrap();
        stats_lock.reset(); // Agora o método reset() existe
    }

    // Treinamento em thread separada
    std::thread::spawn(move || {
        network.train(
            &x_train,
            &y_train,
            train_data.epochs,
            train_data.batch_size,
            &loss_fn,
            &mut *optimizer,
            stats_clone,
        );

        let mut network_lock = network_arc.lock().unwrap();
        *network_lock = Some(network); // Armazena o modelo treinado
    });

    Json(TrainResponse {
        message: "Treinamento iniciado".to_string(),
    })
}

// Rota GET para pegar estatísticas
#[get("/stats")]
fn get_stats(state: &State<AppState>) -> Json<TrainingStats> {
    let stats = state.stats.lock().unwrap(); // Trava o mutex
    Json((*stats).clone()) // Clona o valor interno de TrainingStats
}

#[post("/save_model", data = "<filename>")]
fn save_model(
    filename: String,
    state: &State<AppState>,
) -> Result<Json<TrainResponse>, Custom<String>> {
    let network = state.network.lock().unwrap();
    if let Some(ref net) = *network {
        if let Err(e) = net.save(&filename) {
            return Err(Custom(Status::InternalServerError, e.to_string()));
        }
        Ok(Json(TrainResponse {
            message: format!("Modelo salvo como '{}'", filename),
        }))
    } else {
        Err(Custom(
            Status::BadRequest,
            "Nenhum modelo treinado disponível".to_string(),
        ))
    }
}

#[post("/load_model", data = "<filename>")]
fn load_model(
    filename: String,
    state: &State<AppState>,
) -> Result<Json<TrainResponse>, Custom<String>> {
    let mut network = state.network.lock().unwrap();
    match NeuralNetwork::load(&filename) {
        Ok(net) => {
            *network = Some(net);
            Ok(Json(TrainResponse {
                message: format!("Modelo carregado de '{}'", filename),
            }))
        }
        Err(e) => Err(Custom(Status::InternalServerError, e.to_string())),
    }
}

// Função de inicialização do Rocket
#[launch]
fn rocket() -> _ {
    let stats = Arc::new(Mutex::new(TrainingStats::new()));
    let network = Arc::new(Mutex::new(None));

    rocket::build().manage(AppState { stats, network }).mount(
        "/api",
        routes![train, get_stats, save_model, load_model, test_model],
    )
}
