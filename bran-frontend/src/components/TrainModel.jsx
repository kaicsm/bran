// src/components/TrainModel.jsx
import React, { useState } from 'react';
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
} from '@mui/material';
import axios from 'axios';

function TrainModel() {
  const [epochs, setEpochs] = useState(10);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.01);
  const [l2Reg, setL2Reg] = useState(0.0);
  const [optimizer, setOptimizer] = useState('SGD');
  const [layers, setLayers] = useState([
    { input_size: 784, output_size: 128, activation: 'ReLU' },
    { input_size: 128, output_size: 10, activation: 'Softmax' },
  ]);
  const [xTrain, setXTrain] = useState('');
  const [yTrain, setYTrain] = useState('');
  const [message, setMessage] = useState('');

  const handleTrain = async () => {
    try {
      const xTrainData = JSON.parse(xTrain);
      const yTrainData = JSON.parse(yTrain);

      const response = await axios.post('/api/train', {
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        l2_reg: l2Reg,
        optimizer,
        layers,
        x_train: xTrainData,
        y_train: yTrainData,
      });
      setMessage(response.data.message);
    } catch (error) {
      console.error(error);
      setMessage('Erro ao iniciar o treinamento');
    }
  };

  return (
    <Container>
      <Typography variant="h4" gutterBottom>
        Configurar e Treinar Modelo
      </Typography>
      <Paper style={{ padding: 16 }}>
        <Grid container spacing={2}>
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
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Tamanho do Batch"
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(e.target.value)}
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Taxa de Aprendizado"
              type="number"
              value={learningRate}
              onChange={(e) => setLearningRate(e.target.value)}
            />
          </Grid>
          <Grid item xs={6}>
            <TextField
              fullWidth
              label="Regularização L2"
              type="number"
              value={l2Reg}
              onChange={(e) => setL2Reg(e.target.value)}
            />
          </Grid>
          <Grid item xs={12}>
            <FormControl fullWidth>
              <InputLabel>Otimizador</InputLabel>
              <Select
                value={optimizer}
                onChange={(e) => setOptimizer(e.target.value)}
              >
                <MenuItem value="SGD">SGD</MenuItem>
                {/* Adicione outros otimizadores se necessário */}
              </Select>
            </FormControl>
          </Grid>

          {/* Configuração das camadas */}
          <Grid item xs={12}>
            <Typography variant="h6">Configuração das Camadas</Typography>
            {/* Aqui você pode adicionar uma interface para adicionar/remover camadas dinamicamente */}
            {/* Para simplicidade, estamos usando camadas fixas */}
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
            />
          </Grid>

          {/* Botão para iniciar o treinamento */}
          <Grid item xs={12}>
            <Button variant="contained" color="primary" onClick={handleTrain}>
              Iniciar Treinamento
            </Button>
          </Grid>

          {/* Mensagem de status */}
          {message && (
            <Grid item xs={12}>
              <Typography variant="body1">{message}</Typography>
            </Grid>
          )}
        </Grid>
      </Paper>
    </Container>
  );
}

export default TrainModel;
