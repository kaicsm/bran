// src/components/TrainingStats.jsx
import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Paper,
  Grid,
} from '@mui/material';
import axios from 'axios';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

function TrainingStats() {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await axios.get('/api/stats');
        setStats(response.data);
      } catch (error) {
        console.error(error);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 5000); // Atualiza a cada 5 segundos

    return () => clearInterval(interval);
  }, []);

  if (!stats) {
    return (
      <Container>
        <Typography variant="h6">Carregando estatísticas...</Typography>
      </Container>
    );
  }

  const data = stats.loss_history.map((loss, index) => ({
    epoch: index + 1,
    loss,
  }));

  return (
    <Container>
      <Typography variant="h4" gutterBottom>
        Estatísticas de Treinamento
      </Typography>
      <Paper style={{ padding: 16 }}>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" label={{ value: 'Época', position: 'insideBottomRight', offset: -5 }} />
                <YAxis label={{ value: 'Perda', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="loss" stroke="#8884d8" activeDot={{ r: 8 }} name="Perda" />
              </LineChart>
            </ResponsiveContainer>
          </Grid>
        </Grid>
      </Paper>
    </Container>
  );
}

export default TrainingStats;
