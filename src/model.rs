use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};

#[derive(Debug, Clone)]
pub struct QNet {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl QNet {
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(in_dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = linear(hidden_dim, hidden_dim, vb.pp("fc2"))?;
        let fc3 = linear(hidden_dim, out_dim, vb.pp("fc3"))?;
        Ok(Self { fc1, fc2, fc3 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.fc2.forward(&xs)?;
        let xs = xs.relu()?;
        self.fc3.forward(&xs)
    }
}
