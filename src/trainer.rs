use crate::collector::Collector;
use crate::env::Environment;
use crate::policy::Policy;
use std::fmt::Debug;

use crate::venv::VectorEnv;

pub struct Trainer<V: VectorEnv, P: Policy> {
    collector: Collector<V, P>,
    max_epochs: usize,
    step_per_epoch: usize,
    batch_size: usize,
}

impl<V, P> Trainer<V, P>
where
    V: VectorEnv,
    P: Policy<Observation = V::Observation, Action = V::Action>,
    V::Observation: Clone + Debug,
    V::Action: Clone + Debug,
{
    pub fn new(
        collector: Collector<V, P>,
        max_epochs: usize,
        step_per_epoch: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            collector,
            max_epochs,
            step_per_epoch,
            batch_size,
        }
    }

    pub fn train(&mut self) -> Result<(), String> {
        println!("Collecting initial data...");
        self.collector.collect(10);

        println!("Starting Training...");
        for epoch in 1..=self.max_epochs {
            let mut total_reward = 0.0;
            // Collect experience
            let episodes = self.collector.collect(self.step_per_epoch);
            for r in episodes {
                total_reward += r;
            }
            let avg_reward = total_reward / self.step_per_epoch as f64;

            // Train
            self.collector.train_step(self.batch_size);

            println!("Epoch {}: Avg Reward: {:.2}", epoch, avg_reward);
        }
        Ok(())
    }
}
