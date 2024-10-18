// bran/src/visualization.rs

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct TrainingStats {
    pub epochs: Vec<f32>,
    pub losses: Vec<f32>,
}

impl TrainingStats {
    pub fn new() -> Self {
        TrainingStats {
            epochs: Vec::new(),
            losses: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.epochs.clear();
        self.losses.clear();
    }

    pub fn log_epoch(&mut self, epoch: f32, loss: f32) {
        self.epochs.push(epoch);
        self.losses.push(loss);
    }
}
