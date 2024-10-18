# Guia Completo para Utilização da Biblioteca de Redes Neurais Bran em Rust

Bem-vindo ao guia completo sobre como utilizar a biblioteca **Bran** para criar redes neurais em Rust. Vamos explorar detalhadamente todos os métodos e funcionalidades da biblioteca, capacitando você a desenvolver modelos de aprendizado profundo de forma eficiente.

## Índice

1. [Introdução à Biblioteca Bran](#introdução-à-biblioteca-bran)
2. [Instalação e Configuração](#instalação-e-configuração)
3. [Principais Componentes da Biblioteca](#principais-componentes-da-biblioteca)
   - [Funções de Ativação](#funções-de-ativação)
   - [Camadas Densas](#camadas-densas)
   - [Funções de Perda](#funções-de-perda)
   - [Otimizadores](#otimizadores)
   - [Rede Neural](#rede-neural)
4. [Como Criar uma Rede Neural](#como-criar-uma-rede-neural)
   - [Construindo o Modelo](#construindo-o-modelo)
   - [Treinando o Modelo](#treinando-o-modelo)
   - [Salvando e Carregando Modelos](#salvando-e-carregando-modelos)
5. [Exemplo Prático: Classificação Binária](#exemplo-prático-classificação-binária)
6. [Testes e Visualização](#testes-e-visualização)
7. [Conclusão](#conclusão)

---

## Introdução à Biblioteca Bran

**Bran** é uma biblioteca em Rust projetada para facilitar a criação e o treinamento de redes neurais. Ela oferece uma estrutura modular que permite a construção de modelos personalizados, suportando diversas funções de ativação, camadas densas, funções de perda e otimizadores.

## Instalação e Configuração

Para utilizar a biblioteca Bran, você precisa adicioná-la como dependência em seu projeto Rust. Como o código fonte está disponível, você pode cloná-lo ou incorporá-lo diretamente em seu projeto.

```toml
[dependencies]
bran = { path = "caminho/para/o/bran" }
```

Certifique-se de que todas as dependências necessárias também estejam incluídas, como `ndarray`, `rand`, `serde`, entre outras.

## Principais Componentes da Biblioteca

### Funções de Ativação

As funções de ativação são componentes essenciais em redes neurais, introduzindo não-linearidades que permitem ao modelo aprender relações complexas nos dados.

#### Trait `Activation`

Define os métodos padrão para funções de ativação:

- `activate(&self, x: f32) -> f32`: Aplica a função de ativação a um único valor.
- `derivative(&self, x: f32) -> f32`: Calcula a derivada da função de ativação para um único valor.
- `activate_array(&self, x: &Array2<f32>) -> Array2<f32>`: Aplica a função de ativação a cada elemento de uma matriz.
- `derivative_array(&self, x: &Array2<f32>) -> Array2<f32>`: Calcula a derivada para cada elemento de uma matriz.

#### Implementações Disponíveis

- **ReLU (Rectified Linear Unit)**: `ActivationType::ReLU`
  - `f(x) = max(0, x)`
- **Sigmoid**: `ActivationType::Sigmoid`
  - `f(x) = 1 / (1 + e^{-x})`
- **Tanh (Tangente Hiperbólica)**: `ActivationType::Tanh`
  - `f(x) = tanh(x)`
- **Linear**: `ActivationType::Linear`
  - `f(x) = x`

### Camadas Densas

As camadas densas (totalmente conectadas) são fundamentais em arquiteturas de redes neurais.

#### Estrutura `DenseLayer`

Representa uma camada densa com os seguintes atributos:

- `weights: Array2<f32>`: Matriz de pesos.
- `biases: Array1<f32>`: Vetor de vieses.
- `activation_type: ActivationType`: Tipo de função de ativação.
- Métodos essenciais:
  - `new(input_size, output_size, activation_type)`: Cria uma nova camada.
  - `forward(&mut self, input: &Array2<f32>) -> Array2<f32>`: Executa a passagem forward.
  - `backward(&mut self, output_error: &Array2<f32>, optimizer: &mut dyn Optimizer) -> Array2<f32>`: Executa a passagem backward e atualiza os pesos.

### Funções de Perda

As funções de perda quantificam a diferença entre as predições do modelo e os valores reais.

#### Trait `Loss`

Define os métodos:

- `loss(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> f32`: Calcula a perda total.
- `derivative(&self, predicted: &Array2<f32>, target: &Array2<f32>) -> Array2<f32>`: Calcula o gradiente da perda.

#### Implementações Disponíveis

- **Mean Squared Error (MSE)**: `MeanSquaredError`
  - Usada em problemas de regressão.
- **Cross-Entropy Loss**: `CrossEntropyLoss`
  - Usada em problemas de classificação, especialmente com probabilidades.

### Otimizadores

Otimizadores ajustam os pesos e vieses do modelo durante o treinamento.

#### Trait `Optimizer`

Define o método:

- `update(...)`: Atualiza os pesos e vieses com base nos gradientes.

#### Implementações Disponíveis

- **Stochastic Gradient Descent (SGD)**: `SGD`
  - Simples e eficaz para muitos problemas.
- **Adam (Adaptive Moment Estimation)**: `Adam`
  - Combina o melhor de dois outros otimizadores: AdaGrad e RMSProp.

### Rede Neural

A classe principal que coordena as camadas, funções de perda e otimizadores.

#### Estrutura `NeuralNetwork`

- Contém um vetor de camadas: `layers: Vec<DenseLayer>`.
- Métodos essenciais:
  - `new()`: Cria uma nova rede neural vazia.
  - `add_layer(&mut self, layer: DenseLayer)`: Adiciona uma camada à rede.
  - `forward(&mut self, input: &Array2<f32>) -> Array2<f32>`: Executa a passagem forward em todas as camadas.
  - `backward(&mut self, output_error: &Array2<f32>, optimizer: &mut dyn Optimizer)`: Executa a passagem backward.
  - `train(...)`: Método para treinar a rede neural.

## Como Criar uma Rede Neural

### Construindo o Modelo

1. **Instancie a Rede Neural**

```rust
use bran::prelude::*;

let mut nn = NeuralNetwork::new();
```

2. **Adicione Camadas**

```rust
nn.add_layer(DenseLayer::new(entrada, unidades, ActivationType::ReLU));
nn.add_layer(DenseLayer::new(unidades, saída, ActivationType::Sigmoid));
```

3. **Configure a Função de Perda e o Otimizador**

```rust
let loss_fn = MeanSquaredError;
let mut optimizer = SGD::new(learning_rate, l2_reg);
```

### Treinando o Modelo

Utilize o método `train` da rede neural.

```rust
nn.train(
    Arc::new(Mutex::new(nn)),
    &x_train,
    &y_train,
    epochs,
    batch_size,
    Arc::new(loss_fn),
    Arc::new(Mutex::new(optimizer)),
    Arc::new(Mutex::new(TrainingStats::new())),
);
```

Parâmetros:

- `x_train`: Dados de entrada.
- `y_train`: Rótulos ou valores alvo.
- `epochs`: Número de épocas.
- `batch_size`: Tamanho do lote.
- `TrainingStats`: Para armazenar estatísticas do treinamento.

### Salvando e Carregando Modelos

#### Salvando

```rust
nn.save("modelo.bran").expect("Falha ao salvar o modelo");
```

#### Carregando

```rust
let nn = NeuralNetwork::load("modelo.bran").expect("Falha ao carregar o modelo");
```

## Exemplo Prático: Classificação Binária

Vamos construir e treinar uma rede neural para resolver um problema de classificação binária.

### Preparação dos Dados

```rust
use ndarray::array;

let x_train = array![[0., 0.], [0., 1.], [1., 0.], [1., 1.]];
let y_train = array![[0.], [1.], [1.], [0.]];
```

### Construção do Modelo

```rust
let mut nn = NeuralNetwork::new();
nn.add_layer(DenseLayer::new(2, 4, ActivationType::ReLU));
nn.add_layer(DenseLayer::new(4, 1, ActivationType::Sigmoid));
```

### Configuração da Função de Perda e Otimizador

```rust
let loss_fn = CrossEntropyLoss;
let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8, 0.0);
```

### Treinamento

```rust
use std::sync::{Arc, Mutex};
use bran::visualization::TrainingStats;

let stats = Arc::new(Mutex::new(TrainingStats::new()));

nn.train(
    Arc::new(Mutex::new(nn)),
    &x_train,
    &y_train,
    1000,
    4,
    Arc::new(loss_fn),
    Arc::new(Mutex::new(optimizer)),
    stats.clone(),
);
```

### Avaliação do Modelo

```rust
let output = nn.forward(&x_train);
println!("Saídas previstas: {:?}", output);
```

## Testes e Visualização

### Testes Unitários

A biblioteca Bran inclui testes unitários para garantir a integridade das funcionalidades.

```rust
cargo test
```

### Visualização das Estatísticas de Treinamento

Utilize a estrutura `TrainingStats` para acessar os dados de perda e acurácia ao longo das épocas.

```rust
let stats = stats.lock().unwrap();
println!("Perdas: {:?}", stats.losses);
println!("Acurácias: {:?}", stats.acuracies);
```

Você pode exportar esses dados para visualização em ferramentas como o Python Matplotlib.

## Conclusão

A biblioteca Bran oferece uma estrutura robusta para a construção e treinamento de redes neurais em Rust. Com suporte para diversas funções de ativação, otimizadores e funções de perda, ela permite a criação de modelos personalizados para atender às necessidades específicas de diferentes problemas.

Este guia apresentou uma visão abrangente de como utilizar a Bran, desde a configuração inicial até a implementação prática de um modelo de classificação binária. Esperamos que este documento tenha maximizado seu aprendizado e capacitado você a explorar todo o potencial da biblioteca.
