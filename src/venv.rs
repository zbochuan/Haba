use crate::env::{EnvResult, Environment, Step};
use std::error::Error;
use std::fmt::Debug;

pub trait VectorEnv {
    type Observation: Clone + Debug;
    type Action: Clone + Debug;

    fn step(
        &mut self,
        actions: &[Self::Action],
    ) -> Result<Vec<Step<Self::Observation>>, Box<dyn Error>>;
    fn reset(&mut self) -> Result<Vec<Self::Observation>, Box<dyn Error>>;
    fn len(&self) -> usize;
}

pub struct DummyVectorEnv<E: Environment> {
    envs: Vec<E>,
}

impl<E: Environment> DummyVectorEnv<E> {
    pub fn new(envs: Vec<E>) -> Self {
        Self { envs }
    }
}

impl<E: Environment> VectorEnv for DummyVectorEnv<E>
where
    E::Observation: Clone + Debug,
    E::Action: Clone + Debug,
{
    type Observation = E::Observation;
    type Action = E::Action;

    fn step(
        &mut self,
        actions: &[Self::Action],
    ) -> Result<Vec<Step<Self::Observation>>, Box<dyn Error>> {
        if actions.len() != self.envs.len() {
            return Err("Action count must match env count".into());
        }

        let mut next_steps = Vec::with_capacity(self.envs.len());

        for (i, env) in self.envs.iter_mut().enumerate() {
            let action = actions[i].clone();
            let mut step = env.step(action)?;

            if step.done {
                // Auto-reset
                let obs = env.reset()?;
                // We typically want to return the 'last' observation in info or somewhere,
                // but for now, let's just update the next_obs to be the reset one
                // so the agent can continue acting, but mark it as done.
                // Tianshou actually returns the 'next_obs' as the reset observation if done is true,
                // and puts the terminal observation in 'info'.
                // Our Step struct has `obs` (next_obs).
                // Let's replace `step.obs` with the reset observation for continuity,
                // but keep `step.done = true`.
                // The true terminal observation is lost if we overwrite it,
                // but `step.obs` WAS the terminal observation before this override.
                // Ideally we check `info`.
                // For simplicity: simple auto-reset logic.
                step.obs = obs;
            }
            next_steps.push(step);
        }

        Ok(next_steps)
    }

    fn reset(&mut self) -> Result<Vec<Self::Observation>, Box<dyn Error>> {
        let mut obs_vec = Vec::with_capacity(self.envs.len());
        for env in self.envs.iter_mut() {
            obs_vec.push(env.reset()?);
        }
        Ok(obs_vec)
    }

    fn len(&self) -> usize {
        self.envs.len()
    }
}
