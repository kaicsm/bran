use plotlib::style::LineStyle;
use plotlib::view::ContinuousView;
use plotlib::{page::Page, repr::Plot};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

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

pub fn visualize_stats(stats: Arc<Mutex<TrainingStats>>) -> Result<(), Box<dyn std::error::Error>> {
    let stats = stats.lock().unwrap();

    let data: Vec<(f64, f64)> = stats
        .epochs
        .iter()
        .zip(stats.losses.iter())
        .map(|(&x, &y)| (x as f64, y as f64))
        .collect();

    let line = Plot::new(data).line_style(LineStyle::new().colour("#ff0000")); // Usando cor em formato hexadecimal

    let v = ContinuousView::new()
        .add(line)
        .x_label("Épocas")
        .y_label("Perda");

    Page::single(&v).save("training_stats.svg")?;

    println!("Gráfico salvo como 'training_stats.svg'.");

    Ok(())
}
