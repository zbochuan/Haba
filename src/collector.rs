use crate::buffer::ReplayBuffer;
use crate::policy::Policy;
use crate::venv::VectorEnv;
use std::fmt::Debug;

pub struct Collector<V: VectorEnv, P: Policy> {
    env: V,
    policy: P,
    buffer: Option<ReplayBuffer<V::Observation, V::Action>>,
    current_obs: Vec<V::Observation>,
    episode_returns: Vec<f64>,
}

impl<V, P> Collector<V, P>
where
    V: VectorEnv,
    P: Policy<Observation = V::Observation, Action = V::Action>,
    V::Observation: Clone + Debug,
    V::Action: Clone + Debug,
{
    pub fn new(
        mut env: V,
        policy: P,
        buffer: Option<ReplayBuffer<V::Observation, V::Action>>,
    ) -> Self {
        // Initial reset to get first observations
        let current_obs = env.reset().expect("Failed to reset env");
        let len = env.len();

        Collector {
            env,
            policy,
            buffer,
            current_obs,
            episode_returns: vec![0.0; len],
        }
    }

    pub fn collect(&mut self, n_steps: usize) -> Vec<f64> {
        let mut steps_collected = 0;
        let mut completed_rewards = Vec::new();

        while steps_collected < n_steps {
            // 1. Select Actions (Batch)
            let actions = self.policy.forward(&self.current_obs);

            // 2. Step Environment (Batch)
            // Tianshou steps all envs.
            let steps = self.env.step(&actions).expect("Failed to step env");

            // 3. Add to Buffer
            if let Some(buf) = &mut self.buffer {
                for (i, step) in steps.iter().enumerate() {
                    // buffer.add needs to handle batch logic or we add one by one.
                    // ReplayBuffer is not generic over batch yet, it takes single items.
                    // But we have a loop here.
                    // Note: step.obs is the NEXT observation.
                    // self.current_obs[i] is the CURRENT observation.

                    buf.add(
                        self.current_obs[i].clone(),
                        actions[i].clone(),
                        step.reward,
                        step.done,
                        step.obs.clone(),
                    );
                }
            }

            // 4. Update current observations and track rewards
            // For environments that are done, step.obs is the reset definition?
            // In DummyVectorEnv, we implemented auto-reset returning the new start state in step.obs.
            // So we can just update current_obs.
            for (i, step) in steps.iter().enumerate() {
                self.episode_returns[i] += step.reward;
                if step.done {
                    completed_rewards.push(self.episode_returns[i]);
                    self.episode_returns[i] = 0.0;
                }
                self.current_obs[i] = step.obs.clone();
            }

            steps_collected += self.env.len(); // We collected N transitions
        }
        completed_rewards
    }

    pub fn get_buffer_len(&self) -> Option<usize> {
        self.buffer.as_ref().map(|b| b.len())
    }

    pub fn train_step(&mut self, batch_size: usize) {
        if let Some(buf) = &self.buffer {
            if buf.len() >= batch_size {
                let batch = buf.sample(batch_size);
                self.policy.learn(&batch);
            }
        }
    }
}
