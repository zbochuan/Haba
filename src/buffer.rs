use crate::batch::Batch;
use rand::seq::SliceRandom;

#[derive(Debug)]
pub struct ReplayBuffer<O, A> {
    obs: Vec<O>,
    act: Vec<A>,
    rew: Vec<f64>,
    done: Vec<bool>,
    obs_next: Vec<O>,

    capacity: usize,
    index: usize, // Current write position
    size: usize,  // Current number of elements
}

impl<O: Clone, A: Clone> ReplayBuffer<O, A> {
    pub fn new(capacity: usize) -> Self {
        Self {
            obs: Vec::with_capacity(capacity),
            act: Vec::with_capacity(capacity),
            rew: Vec::with_capacity(capacity),
            done: Vec::with_capacity(capacity),
            obs_next: Vec::with_capacity(capacity),
            capacity,
            index: 0,
            size: 0,
        }
    }

    pub fn add(&mut self, obs: O, act: A, rew: f64, done: bool, obs_next: O) {
        if self.size < self.capacity {
            // Append
            self.obs.push(obs);
            self.act.push(act);
            self.rew.push(rew);
            self.done.push(done);
            self.obs_next.push(obs_next);
            self.size += 1;
        } else {
            // Overwrite
            self.obs[self.index] = obs;
            self.act[self.index] = act;
            self.rew[self.index] = rew;
            self.done[self.index] = done;
            self.obs_next[self.index] = obs_next;
        }

        self.index = (self.index + 1) % self.capacity;
    }

    pub fn sample(&self, batch_size: usize) -> Batch<O, A> {
        let mut rng = rand::thread_rng();
        let indices: Vec<usize> = (0..self.size).collect();
        let sampled_indices: Vec<usize> = indices
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect();

        let mut b_obs = Vec::with_capacity(batch_size);
        let mut b_act = Vec::with_capacity(batch_size);
        let mut b_rew = Vec::with_capacity(batch_size);
        let mut b_done = Vec::with_capacity(batch_size);
        let mut b_obs_next = Vec::with_capacity(batch_size);

        for &idx in &sampled_indices {
            b_obs.push(self.obs[idx].clone());
            b_act.push(self.act[idx].clone());
            b_rew.push(self.rew[idx]);
            b_done.push(self.done[idx]);
            b_obs_next.push(self.obs_next[idx].clone());
        }

        Batch::new(b_obs, b_act, b_rew, b_done, b_obs_next)
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}
