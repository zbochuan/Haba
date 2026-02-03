mod batch;
mod buffer;
mod cartpole;
mod collector;
mod dqn; // Connect DQN
mod env;
mod model; // Connect Model
mod policy;
mod trainer;
mod venv; // Import venv module

use crate::buffer::ReplayBuffer;
use crate::cartpole::CartPole;
use crate::collector::Collector;
use crate::dqn::DQNPolicy; // Import DQNPolicy
use crate::trainer::Trainer;
use crate::venv::{DummyVectorEnv, VectorEnv};

fn main() -> Result<(), String> {
    // 1. Initialize the World and Agent
    let env = CartPole::new(1000);
    let venv = DummyVectorEnv::new(vec![env]);

    // 2. Define Policy
    let mut policy = DQNPolicy::new(4, 64, 2, 0.99, 0.1).unwrap();

    // 3. Initialize Collector
    // Buffer size: capacity for raw transitions.
    let buffer = ReplayBuffer::new(10000);
    let collector = Collector::new(venv, policy, Some(buffer));

    // 4. Initialize Trainer
    let mut trainer = Trainer::new(collector, 1000, 10, 32); // 10 epochs, 10 episodes each

    // 5. Run Training
    println!("Training DQN on CartPole...");
    trainer.train()?;

    Ok(())
}
