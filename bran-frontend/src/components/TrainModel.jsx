import React, { useState } from "react";
import {
  Container,
  TextField,
  Button,
  Typography,
  Grid,
  Select,
  MenuItem,
  InputLabel,
  FormControl,
  Paper,
  IconButton,
} from "@mui/material";
import { AddCircleOutline, RemoveCircleOutline } from "@mui/icons-material";
import axios from "axios";

function TrainModel() {
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.01);
  const [l2Reg, setL2Reg] = useState(0.0);
  const [optimizer, setOptimizer] = useState("SGD");
  const [layers, setLayers] = useState([
    { input_size: 784, output_size: 128, activation: "ReLU" },
  ]);
  const [xTrain, setXTrain] = useState("");
  const [yTrain, setYTrain] = useState("");
  const [message, setMessage] = useState("");

  const handleAddLayer = () => {
    setLayers([
      ...layers,
      { input_size: 0, output_size: 0, activation: "ReLU" },
    ]);
  };

  const handleRemoveLayer = (index) => {
    const newLayers = layers.filter((_, i) => i !== index);
    setLayers(newLayers);
  };

  const handleLayerChange = (index, field, value) => {
    const newLayers = [...layers];
    newLayers[index][field] =
      field === "input_size" || field === "output_size"
        ? parseInt(value)
        : value;
    setLayers(newLayers);
  };

  const handleTrain = async () => {
    try {
      const xTrainData = JSON.parse(xTrain);
      const yTrainData = JSON.parse(yTrain);

      const response = await axios.post("/api/train", {
        epochs: parseInt(epochs, 10),
        batch_size: parseInt(batchSize, 10),
        learning_rate: parseFloat(learningRate),
        l2_reg: parseFloat(l2Reg),
        optimizer,
        layers,
        x_train: xTrainData,
        y_train: yTrainData,
      });
      setMessage(response.data.message);
    } catch (error) {
      console.error(error);
      setMessage("Erro ao iniciar o treinamento");
    }
  };

  return (
    <Container>
      <Typography variant="h4" gutterBottom>
        Configurar e Treinar Modelo
      </Typography>
      <Paper
        style={{ padding: 24, backgroundColor: "#1e1e1e", borderRadius: 8 }}
      >
        <Grid container spacing={3}>
          {/* Parâmetros de treinamento */}
          <Grid item xs={12}>
            <Typography variant="h6">Parâmetros de Treinamento</Typography>
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Épocas"
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(e.target.value)}
              variant="filled" // Usar estilo filled para o tema escuro
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Tamanho do Batch"
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(e.target.value)}
              variant="filled"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Taxa de Aprendizado"
              type="number"
              value={learningRate}
              onChange={(e) => setLearningRate(e.target.value)}
              variant="filled"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Regularização L2"
              type="number"
              value={l2Reg}
              onChange={(e) => setL2Reg(e.target.value)}
              variant="filled"
            />
          </Grid>
          <Grid item xs={12}>
            <FormControl fullWidth variant="filled">
              <InputLabel>Otimizador</InputLabel>
              <Select
                value={optimizer}
                onChange={(e) => setOptimizer(e.target.value)}
              >
                <MenuItem value="SGD">SGD</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          {/* Configuração das camadas */}
          <Grid item xs={12}>
            <Typography variant="h6">Configuração das Camadas</Typography>
            {layers.map((layer, index) => (
              <Grid
                container
                spacing={2}
                key={index}
                style={{ marginBottom: 8 }}
              >
                <Grid item xs={3}>
                  <TextField
                    fullWidth
                    label="Neurônios de Entrada"
                    type="number"
                    value={layer.input_size}
                    onChange={(e) =>
                      handleLayerChange(index, "input_size", e.target.value)
                    }
                    variant="filled"
                  />
                </Grid>
                <Grid item xs={3}>
                  <TextField
                    fullWidth
                    label="Neurônios de Saída"
                    type="number"
                    value={layer.output_size}
                    onChange={(e) =>
                      handleLayerChange(index, "output_size", e.target.value)
                    }
                    variant="filled"
                  />
                </Grid>
                <Grid item xs={3}>
                  <FormControl fullWidth variant="filled">
                    <InputLabel>Ativação</InputLabel>
                    <Select
                      value={layer.activation}
                      onChange={(e) =>
                        handleLayerChange(index, "activation", e.target.value)
                      }
                    >
                      <MenuItem value="ReLU">ReLU</MenuItem>
                      <MenuItem value="Sigmoid">Sigmoid</MenuItem>
                      <MenuItem value="Tanh">Tanh</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={3}>
                  <IconButton
                    color="secondary"
                    onClick={() => handleRemoveLayer(index)}
                  >
                    <RemoveCircleOutline />
                  </IconButton>
                </Grid>
              </Grid>
            ))}
            <Button
              variant="contained"
              color="primary"
              onClick={handleAddLayer}
              startIcon={<AddCircleOutline />}
              style={{ marginTop: 16 }}
            >
              Adicionar Camada
            </Button>
          </Grid>

          {/* Dados de treinamento */}
          <Grid item xs={12}>
            <Typography variant="h6">Dados de Treinamento</Typography>
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="X Train (JSON)"
              multiline
              minRows={4}
              value={xTrain}
              onChange={(e) => setXTrain(e.target.value)}
              helperText="Insira os dados de treinamento em formato JSON"
              variant="filled"
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Y Train (JSON)"
              multiline
              minRows={4}
              value={yTrain}
              onChange={(e) => setYTrain(e.target.value)}
              helperText="Insira as labels de treinamento em formato JSON"
              variant="filled"
            />
          </Grid>

          {/* Botão para iniciar o treinamento */}
          <Grid item xs={12}>
            <Button
              variant="contained"
              color="primary"
              onClick={handleTrain}
              fullWidth
            >
              Iniciar Treinamento
            </Button>
          </Grid>

          {/* Mensagem de status */}
          {message && (
            <Grid item xs={12}>
              <Typography variant="body1" style={{ marginTop: 16 }}>
                {message}
              </Typography>
            </Grid>
          )}
        </Grid>
      </Paper>
    </Container>
  );
}

export default TrainModel;
