// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button } from '@mui/material';

import TrainModel from './components/TrainModel';
import TrainingStats from './components/TrainingStats';
import TestModel from './components/TestModel';
import ModelManagement from './components/ModelManagement';

function App() {
  return (
    <Router>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" style={{ flexGrow: 1 }}>
            Neural Network Trainer
          </Typography>
          <Button color="inherit" component={Link} to="/train">
            Treinar Modelo
          </Button>
          <Button color="inherit" component={Link} to="/stats">
            Estatísticas
          </Button>
          <Button color="inherit" component={Link} to="/test">
            Testar Modelo
          </Button>
          <Button color="inherit" component={Link} to="/manage">
            Gerenciar Modelo
          </Button>
        </Toolbar>
      </AppBar>
      <Routes>
        <Route path="/train" element={<TrainModel />} />
        <Route path="/stats" element={<TrainingStats />} />
        <Route path="/test" element={<TestModel />} />
        <Route path="/manage" element={<ModelManagement />} />
        <Route path="/" element={<TrainModel />} />
      </Routes>
    </Router>
  );
}

export default App;
