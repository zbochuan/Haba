use std::collections::HashMap;

//Handle illegal inputs
pub type EnvResult<T> = Result<T, String>;

//Step struct reture the sequence of observations
#[derive(Debug)]
pub struct Step<O> {
    pub obs: O,
    pub done: bool,
    pub reward: f64,
    pub info: Option<HashMap<String, String>>,
}

pub trait Environment {
    type Observation;
    type Action;

    fn reset(&mut self) -> EnvResult<Self::Observation>;
    fn step(&mut self, action: Self::Action) -> EnvResult<Step<Self::Observation>>;
}
