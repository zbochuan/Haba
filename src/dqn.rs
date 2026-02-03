use crate::batch::Batch;
use crate::model::QNet;
use crate::policy::Policy;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::Rng;

pub struct DQNPolicy {
    // Model
    q_net: QNet,
    target_q_net: QNet,
    varmap: VarMap,
    target_varmap: VarMap,
    optimizer: AdamW,
    device: Device,

    // Hyperparameters
    gamma: f64,
    epsilon: f64,
    target_update_freq: usize,
    update_count: usize,
}

impl DQNPolicy {
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        gamma: f64,
        epsilon: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F64, &device);

        let q_net = QNet::new(in_dim, hidden_dim, out_dim, vb.clone())?;

        // Target net with separate vars
        let target_varmap = VarMap::new();
        let target_vb = VarBuilder::from_varmap(&target_varmap, DType::F64, &device);
        let target_q_net = QNet::new(in_dim, hidden_dim, out_dim, target_vb)?;

        // Optimizer
        let params = ParamsAdamW {
            lr: 1e-3,
            ..Default::default()
        };
        let optimizer = AdamW::new(varmap.all_vars(), params)?;

        let mut policy = Self {
            q_net,
            target_q_net,
            varmap,
            target_varmap,
            optimizer,
            device,
            gamma,
            epsilon,
            target_update_freq: 100,
            update_count: 0,
        };

        // Initial sync
        policy.sync_target()?;

        Ok(policy)
    }

    fn sync_target(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let src_data = self.varmap.data();
        let target_data = self.target_varmap.data();

        let src_lock = src_data.lock().unwrap();
        let mut target_lock = target_data.lock().unwrap();

        for (name, src_var) in src_lock.iter() {
            if let Some(target_var) = target_lock.get_mut(name) {
                target_var.set(&src_var.as_tensor())?;
            }
        }
        Ok(())
    }
}

impl Policy for DQNPolicy {
    type Observation = Vec<f64>;
    type Action = f64;

    fn forward(&mut self, obs: &[Self::Observation]) -> Vec<Self::Action> {
        let mut rng = rand::thread_rng();
        let batch_size = obs.len();

        // 1. Get Greedy Actions from Model
        // Flatten obs
        let obs_flat: Vec<f64> = obs.iter().flatten().cloned().collect();
        let obs_tensor = Tensor::from_vec(obs_flat, (batch_size, 4), &self.device).unwrap();
        let q_values = self.q_net.forward(&obs_tensor).unwrap();
        let greedy_actions: Vec<u32> = q_values.argmax(1).unwrap().to_vec1().unwrap();

        // 2. Select final actions (epsilon-greedy)
        let mut final_actions = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            if rng.gen_bool(self.epsilon) {
                // Random action
                if rng.gen_bool(0.5) {
                    final_actions.push(1.0);
                } else {
                    final_actions.push(0.0);
                }
            } else {
                final_actions.push(greedy_actions[i] as f64);
            }
        }
        final_actions
    }

    fn learn(&mut self, batch: &Batch<Self::Observation, Self::Action>) {
        if self.update_count % self.target_update_freq == 0 {
            self.sync_target().unwrap();
        }

        // 1. Prepare Tensors
        let b_size = batch.len();
        let obs_flat: Vec<f64> = batch.obs.iter().flatten().cloned().collect();
        let next_obs_flat: Vec<f64> = batch.obs_next.iter().flatten().cloned().collect();
        // Actions to u32 indices
        let acts_idx: Vec<u32> = batch.act.iter().map(|&a| a as u32).collect();
        let rews: Vec<f64> = batch.rew.clone();
        let dones: Vec<f64> = batch
            .done
            .iter()
            .map(|&d| if d { 1.0 } else { 0.0 })
            .collect();

        // Assuming obs dim is 4
        let obs = Tensor::from_vec(obs_flat, (b_size, 4), &self.device).unwrap();
        let next_obs = Tensor::from_vec(next_obs_flat, (b_size, 4), &self.device).unwrap();
        let action_idx = Tensor::from_vec(acts_idx, (b_size, 1), &self.device).unwrap(); // (B, 1) integers
        let reward = Tensor::from_vec(rews, (b_size, 1), &self.device).unwrap();
        let done = Tensor::from_vec(dones, (b_size, 1), &self.device).unwrap();

        // 2. Compute Target Q
        // Q_target = r + gamma * max(Q_target(s', a'))
        // Use target_q_net for stability
        let next_q_values = self.target_q_net.forward(&next_obs).unwrap().detach();
        let max_next_q = next_q_values.max(1).unwrap().reshape((b_size, 1)).unwrap();
        let target_q = (reward + (1.0 - done).unwrap() * self.gamma * max_next_q)
            .unwrap()
            .detach();

        // 3. Compute Current Q
        let q_values = self.q_net.forward(&obs).unwrap(); // (B, 2)
        // Gather Q values for the taken actions
        let current_q = q_values.gather(&action_idx, 1).unwrap(); // (B, 1)

        // 4. Loss
        let loss = (current_q - target_q)
            .unwrap()
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap();

        // 5. Optimize
        self.optimizer.backward_step(&loss).unwrap();

        self.update_count += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::Batch;

    #[test]
    fn test_dqn_forward() {
        // 1. Setup
        let mut policy = DQNPolicy::new(4, 16, 2, 0.99, 0.0).unwrap(); // Epsilon 0.0 for deterministic greedy

        // 2. Create Dummy Batch Observations
        let obs1 = vec![0.0, 0.0, 0.0, 0.0];
        let obs2 = vec![1.0, 1.0, 1.0, 1.0];
        let obs_batch = vec![obs1, obs2];

        // 3. Inference
        let actions = policy.forward(&obs_batch);

        // 4. Verification
        assert_eq!(actions.len(), 2);
        // Actions should be 0.0 or 1.0
        assert!(actions[0] == 0.0 || actions[0] == 1.0);
        assert!(actions[1] == 0.0 || actions[1] == 1.0);
    }

    #[test]
    fn test_dqn_learn() {
        // 1. Setup
        let mut policy = DQNPolicy::new(4, 16, 2, 0.99, 0.1).unwrap();

        // 2. Create Dummy Batch
        let obs = vec![vec![0.0; 4], vec![1.0; 4]];
        let next_obs = vec![vec![0.0; 4], vec![1.0; 4]];
        let act = vec![0.0, 1.0];
        let rew = vec![1.0, 0.0];
        let done = vec![false, true];

        let batch = Batch::new(obs, act, rew, done, next_obs);

        // 3. Learn
        // Just verify it doesn't panic
        policy.learn(&batch);

        // Verify update count increased
        assert_eq!(policy.update_count, 1);
    }
}
