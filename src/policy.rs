use crate::batch::Batch;
use rand::Rng;

pub trait Policy {
    type Observation;
    type Action;

    fn forward(&mut self, obs: &[Self::Observation]) -> Vec<Self::Action>;
    fn learn(&mut self, batch: &Batch<Self::Observation, Self::Action>);
}

pub struct RandomPolicy;

impl RandomPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl Policy for RandomPolicy {
    type Observation = Vec<f64>;
    type Action = f64;

    fn forward(&mut self, obs: &[Self::Observation]) -> Vec<Self::Action> {
        let mut rng = rand::thread_rng(); // Renamed from thread_rng
        let mut actions = Vec::with_capacity(obs.len());
        for _ in 0..obs.len() {
            if rng.gen_bool(0.5) {
                actions.push(1.0);
            } else {
                actions.push(0.0);
            }
        }
        actions
    }

    fn learn(&mut self, _batch: &Batch<Self::Observation, Self::Action>) {
        // Random policy does not learn
    }
}
