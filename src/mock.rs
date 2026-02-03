use crate::env::{EnvResult, Environment, Step};

#[derive(Debug, Clone)]
pub struct MockEnv {
    obs: f64,
    count: usize,
    max_steps: usize,
}

impl MockEnv {
    pub fn new(max_steps: usize) -> Self {
        Self {
            obs: 0.0,
            count: 0,
            max_steps,
        }
    }
}

impl Environment for MockEnv {
    type Observation = f64;
    type Action = ();

    fn step(&mut self, _action: Self::Action) -> EnvResult<Step<Self::Observation>> {
        self.count += 1;
        self.obs += 1.0;
        let done = self.count >= self.max_steps;

        Ok(Step {
            obs: self.obs,
            reward: 1.0,
            done,
            info: None,
        })
    }

    fn reset(&mut self) -> EnvResult<Self::Observation> {
        self.obs = 0.0;
        self.count = 0;
        Ok(self.obs)
    }
}
