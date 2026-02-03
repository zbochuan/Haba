use Haba::buffer::ReplayBuffer;
use Haba::cartpole::CartPole;
use Haba::collector::Collector;
use Haba::dqn::DQNPolicy;
use Haba::trainer::Trainer;
use Haba::venv::DummyVectorEnv;

fn main() -> Result<(), String> {
    // 1. Initialize the World and Agent
    let env = CartPole::new(1000);

    let venv = DummyVectorEnv::new(vec![env]);

    // 2. Define Policy
    let policy = DQNPolicy::new(4, 64, 2, 0.99, 0.1).unwrap();

    // 3. Initialize Collector
    // Buffer size: capacity for raw transitions.
    let buffer = ReplayBuffer::new(10000);
    let collector = Collector::new(venv, policy, Some(buffer));

    // 4. Initialize Trainer
    let mut trainer = Trainer::new(collector, 200, 1000, 64);

    // 5. Run Training
    println!("Training DQN on CartPole...");
    trainer.train()?;

    Ok(())
}
