// src/components/TestModel.jsx
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

function TestModel() {
  const [xTest, setXTest] = useState('');
  const [predictions, setPredictions] = useState(null);

  const handleTest = async () => {
    try {
      const xTestData = JSON.parse(xTest);

      const response = await axios.post('/api/test', {
        x_test: xTestData,
      });
      setPredictions(response.data.predictions);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <Container>
      <Typography variant="h4" gutterBottom>
        Testar Modelo
      </Typography>
      <Paper style={{ padding: 16 }}>
        <Grid container spacing={2}>
          {/* Dados de teste */}
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="X Test (JSON)"
              multiline
              minRows={4}
              value={xTest}
              onChange={(e) => setXTest(e.target.value)}
              helperText="Insira os dados de teste em formato JSON"
            />
          </Grid>

          {/* Botão para testar o modelo */}
          <Grid item xs={12}>
            <Button variant="contained" color="primary" onClick={handleTest}>
              Testar Modelo
            </Button>
          </Grid>

          {/* Exibição das previsões */}
          {predictions && (
            <Grid item xs={12}>
              <Typography variant="h6">Previsões:</Typography>
              <pre>{JSON.stringify(predictions, null, 2)}</pre>
            </Grid>
          )}
        </Grid>
      </Paper>
    </Container>
  );
}

export default TestModel;
