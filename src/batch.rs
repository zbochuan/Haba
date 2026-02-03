#[derive(Debug, Clone)]
pub struct Batch<O, A> {
    pub obs: Vec<O>,
    pub act: Vec<A>,
    pub rew: Vec<f64>,
    pub done: Vec<bool>,
    pub obs_next: Vec<O>,
}

impl<O, A> Batch<O, A> {
    pub fn new(obs: Vec<O>, act: Vec<A>, rew: Vec<f64>, done: Vec<bool>, obs_next: Vec<O>) -> Self {
        Self {
            obs,
            act,
            rew,
            done,
            obs_next,
        }
    }

    pub fn len(&self) -> usize {
        self.obs.len()
    }
}
