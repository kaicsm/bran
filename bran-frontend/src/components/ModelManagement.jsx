// src/components/ModelManagement.jsx
import React, { useState } from 'react';
import {
  Container,
  TextField,
  Button,
  Typography,
  Grid,
  Paper,
} from '@mui/material';
import axios from 'axios';

function ModelManagement() {
  const [filename, setFilename] = useState('');
  const [message, setMessage] = useState('');

  const handleSave = async () => {
    try {
      const response = await axios.post('/api/save_model', filename, {
        headers: { 'Content-Type': 'text/plain' },
      });
      setMessage(response.data.message);
    } catch (error) {
      console.error(error);
      setMessage('Erro ao salvar o modelo');
    }
  };

  const handleLoad = async () => {
    try {
      const response = await axios.post('/api/load_model', filename, {
        headers: { 'Content-Type': 'text/plain' },
      });
      setMessage(response.data.message);
    } catch (error) {
      console.error(error);
      setMessage('Erro ao carregar o modelo');
    }
  };

  return (
    <Container>
      <Typography variant="h4" gutterBottom>
        Gerenciar Modelo
      </Typography>
      <Paper style={{ padding: 16 }}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Nome do Arquivo"
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              helperText="Insira o nome do arquivo para salvar/carregar o modelo"
            />
          </Grid>

          {/* Botões para salvar e carregar */}
          <Grid item xs={6}>
            <Button variant="contained" color="primary" onClick={handleSave}>
              Salvar Modelo
            </Button>
          </Grid>
          <Grid item xs={6}>
            <Button variant="contained" color="secondary" onClick={handleLoad}>
              Carregar Modelo
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

export default ModelManagement;
