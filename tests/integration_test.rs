use Haba::buffer::ReplayBuffer;
use Haba::cartpole::CartPole;
use Haba::collector::Collector;
use Haba::dqn::DQNPolicy;
use Haba::trainer::Trainer;
use Haba::venv::DummyVectorEnv;

#[test]
fn test_integration_cartpole_dqn() {
    // 1. Env
    // Use fewer steps for fast testing
    let env = CartPole::new(500);
    let venv = DummyVectorEnv::new(vec![env]);

    // 2. Policy
    // Epsilon 0.5 to force some exploration and learning updates
    let policy = DQNPolicy::new(4, 64, 2, 0.99, 0.5).expect("Failed to create DQN Policy");

    // 3. Buffer
    let buffer = ReplayBuffer::new(1000);

    // 4. Collector
    let collector = Collector::new(venv, policy, Some(buffer));

    // 5. Trainer
    // 2 epochs, 2 episodes per epoch, batch size 16
    let mut trainer = Trainer::new(collector, 2, 2, 16);

    // 6. Train
    let result = trainer.train();

    // 7. Verify
    assert!(result.is_ok(), "Training failed: {:?}", result.err());
}
