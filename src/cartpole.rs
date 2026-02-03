use crate::env::{EnvResult, Environment, Step};

pub struct CartPoleState {
    pub x: f64,
    pub x_dot: f64,
    pub theta: f64,
    pub theta_dot: f64,
}

pub struct CartPole {
    state: CartPoleState,
    max_steps: usize,
    current_step: usize,
}

impl CartPole {
    pub fn new(max_steps: usize) -> Self {
        CartPole {
            state: CartPoleState {
                x: 0.0,
                x_dot: 0.0,
                theta: 0.0,
                theta_dot: 0.0,
            },
            max_steps,
            current_step: 0,
        }
    }
}

impl Environment for CartPole {
    type Observation = Vec<f64>;
    type Action = f64;

    fn reset(&mut self) -> EnvResult<Self::Observation> {
        self.state.x = 0.0;
        self.state.x_dot = 0.0;
        self.state.theta = 0.0;
        self.state.theta_dot = 0.0;
        self.current_step = 0;
        Ok(vec![
            self.state.x,
            self.state.x_dot,
            self.state.theta,
            self.state.theta_dot,
        ])
    }

    fn step(&mut self, action: Self::Action) -> EnvResult<Step<Self::Observation>> {
        self.current_step += 1;

        // Constants for Physics (Standard CartPole)
        const GRAVITY: f64 = 9.8;
        const MASSCART: f64 = 1.0;
        const MASSPOLE: f64 = 0.1;
        const TOTAL_MASS: f64 = MASSCART + MASSPOLE;
        const LENGTH: f64 = 0.5; // actually half the pole's length
        const POLEMASS_LENGTH: f64 = MASSPOLE * LENGTH;
        const FORCE_MAG: f64 = 10.0;
        const TAU: f64 = 0.02; // seconds between state updates

        let force = if action == 1.0 { FORCE_MAG } else { -FORCE_MAG };

        let cos_theta = self.state.theta.cos();
        let sin_theta = self.state.theta.sin();

        // Equations of Motion
        let temp =
            (force + POLEMASS_LENGTH * self.state.theta_dot.powi(2) * sin_theta) / TOTAL_MASS;
        let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
            / (LENGTH * (4.0 / 3.0 - MASSPOLE * cos_theta.powi(2) / TOTAL_MASS));
        let x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

        // Euler Integration
        self.state.x += TAU * self.state.x_dot;
        self.state.x_dot += TAU * x_acc;
        self.state.theta += TAU * self.state.theta_dot;
        self.state.theta_dot += TAU * theta_acc;

        // Done condition: Pole falls over (> 12 degrees) or cart goes out of bounds (> 2.4 units)
        // 12 degrees in radians is approx 0.209
        let done = self.state.x.abs() > 2.4
            || self.state.theta.abs() > 0.209
            || self.current_step >= self.max_steps;

        Ok(Step {
            obs: vec![
                self.state.x,
                self.state.x_dot,
                self.state.theta,
                self.state.theta_dot,
            ],
            reward: 1.0, // Survive one more frame = +1 point
            done,
            info: None,
        })
    }
}
