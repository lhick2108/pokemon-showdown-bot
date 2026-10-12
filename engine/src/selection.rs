//! Simultaneous-move selection for MCTS: Exp3 and regret matching.
//!
//! Decoupled UCB1 is the default and lives in `mcts.rs` untouched. These two
//! alternatives follow Lisy, Kovarik, Lanctot, Bosansky (2013, "Convergence of
//! Monte Carlo Tree Search in Simultaneous Move Games", NeurIPS) and Tak,
//! Lanctot, Winands (2014, "Monte Carlo Tree Search Variants for Simultaneous
//! Move Games", IEEE CIG). Rewards are in [0, 1] from the view of the side that
//! owns the stats (side two feeds `1 - r`).
//!
//! * Exp3: sigma_i = (1-g) * softmax(eta * G)_i + g/K, eta = g/K,
//!   G_i += r / sigma_i for the chosen i.
//! * RM:   sigma_i = (1-g) * R_i^+ / sum R^+ + g/K (uniform if sum R^+ = 0),
//!   R_j += xhat_j - r with xhat_a = r / sigma_a and 0 elsewhere.
//!
//! Both sample the action from sigma and accumulate the average strategy
//! S_i += sigma_i per visit; the normalised S is the quantity that converges
//! to a Nash equilibrium, and is what the root reports as its strategy.

use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionMode {
    Ucb1,
    Exp3,
    Rm,
}

impl SelectionMode {
    pub fn parse(name: &str) -> Option<SelectionMode> {
        match name.to_ascii_lowercase().as_str() {
            "ucb1" => Some(SelectionMode::Ucb1),
            "exp3" => Some(SelectionMode::Exp3),
            "rm" => Some(SelectionMode::Rm),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SelectionConfig {
    pub mode: SelectionMode,
    pub gamma: f64,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        SelectionConfig {
            mode: SelectionMode::Ucb1,
            gamma: 0.1,
        }
    }
}

/// Exp3 mixed strategy from cumulative importance-weighted rewards `g`.
pub fn exp3_sigma(g: &[f64], gamma: f64, out: &mut [f64]) {
    let k = g.len() as f64;
    let eta = gamma / k;
    let max = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for (o, &gi) in out.iter_mut().zip(g) {
        *o = (eta * (gi - max)).exp();
        sum += *o;
    }
    for o in out.iter_mut() {
        *o = (1.0 - gamma) * (*o / sum) + gamma / k;
    }
}

/// Regret-matching mixed strategy (with uniform exploration) from regrets `r`.
pub fn rm_sigma(r: &[f64], gamma: f64, out: &mut [f64]) {
    let k = r.len() as f64;
    let pos_sum: f64 = r.iter().map(|x| x.max(0.0)).sum();
    for (o, &ri) in out.iter_mut().zip(r) {
        let base = if pos_sum > 0.0 {
            ri.max(0.0) / pos_sum
        } else {
            1.0 / k
        };
        *o = (1.0 - gamma) * base + gamma / k;
    }
}

/// Per-node, per-side state for Exp3 / RM.
#[derive(Debug)]
pub struct SideStats {
    mode: SelectionMode,
    gamma: f64,
    /// cumulative importance-weighted reward (Exp3) or regret (RM)
    acc: Vec<f64>,
    /// the strategy used at the most recent selection
    sigma: Vec<f64>,
    /// cumulative strategy S
    avg: Vec<f64>,
}

impl SideStats {
    pub fn new(cfg: &SelectionConfig, k: usize) -> SideStats {
        SideStats {
            mode: cfg.mode,
            gamma: cfg.gamma,
            acc: vec![0.0; k],
            sigma: vec![1.0 / k as f64; k],
            avg: vec![0.0; k],
        }
    }

    /// Compute sigma, add it to the average strategy, and sample an action.
    pub fn select(&mut self, rng: &mut impl Rng) -> usize {
        match self.mode {
            SelectionMode::Exp3 => exp3_sigma(&self.acc, self.gamma, &mut self.sigma),
            SelectionMode::Rm => rm_sigma(&self.acc, self.gamma, &mut self.sigma),
            SelectionMode::Ucb1 => unreachable!("SideStats is not used for ucb1"),
        }
        for (a, s) in self.avg.iter_mut().zip(&self.sigma) {
            *a += *s;
        }
        let mut threshold: f64 = rng.random_range(0.0..1.0);
        for (i, s) in self.sigma.iter().enumerate() {
            threshold -= *s;
            if threshold < 0.0 {
                return i;
            }
        }
        self.sigma.len() - 1
    }

    /// Update after action `chosen` (selected with the current `sigma`) earned `reward`.
    pub fn update(&mut self, chosen: usize, reward: f64) {
        let est = reward / self.sigma[chosen];
        match self.mode {
            SelectionMode::Exp3 => self.acc[chosen] += est,
            SelectionMode::Rm => {
                for (j, r) in self.acc.iter_mut().enumerate() {
                    *r += if j == chosen { est } else { 0.0 } - reward;
                }
            }
            SelectionMode::Ucb1 => {}
        }
    }

    pub fn avg_strategy(&self) -> &[f64] {
        &self.avg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn cfg(mode: SelectionMode, gamma: f64) -> SelectionConfig {
        SelectionConfig { mode, gamma }
    }

    #[test]
    fn sigma_sums_to_one_with_gamma_floor() {
        let mut out = [0.0; 4];
        for g in [[0.0; 4], [5.0, -3.0, 1e4, 0.2], [-1e6, -1e6, -1e6, 0.0]] {
            exp3_sigma(&g, 0.1, &mut out);
            assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(out.iter().all(|&s| s >= 0.1 / 4.0 - 1e-12 && s.is_finite()));
            rm_sigma(&g, 0.1, &mut out);
            assert!((out.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(out.iter().all(|&s| s >= 0.1 / 4.0 - 1e-12));
        }
    }

    #[test]
    fn rm_uniform_when_no_positive_regret_and_proportional_otherwise() {
        let mut out = [0.0; 3];
        rm_sigma(&[-1.0, -2.0, 0.0], 0.1, &mut out);
        for s in out {
            assert!((s - 1.0 / 3.0).abs() < 1e-12);
        }
        rm_sigma(&[3.0, 1.0, -5.0], 0.0, &mut out);
        assert!((out[0] - 0.75).abs() < 1e-12 && (out[1] - 0.25).abs() < 1e-12 && out[2] == 0.0);
    }

    #[test]
    fn update_rules() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut e = SideStats::new(&cfg(SelectionMode::Exp3, 0.1), 2);
        let i = e.select(&mut rng);
        let s = e.sigma[i];
        e.update(i, 0.6);
        assert!((e.acc[i] - 0.6 / s).abs() < 1e-12 && e.acc[1 - i] == 0.0);

        let mut m = SideStats::new(&cfg(SelectionMode::Rm, 0.1), 3);
        let i = m.select(&mut rng);
        let s = m.sigma[i];
        m.update(i, 0.5);
        for j in 0..3 {
            let want = if j == i { 0.5 / s - 0.5 } else { -0.5 };
            assert!((m.acc[j] - want).abs() < 1e-12);
        }
        assert!((m.avg_strategy().iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    /// Self-play on a zero-sum matrix game (row player payoff in [0,1]).
    fn self_play(mode: SelectionMode, payoff: &[&[f64]], iters: usize) -> (Vec<f64>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(7);
        let c = cfg(mode, 0.1);
        let mut a = SideStats::new(&c, payoff.len());
        let mut b = SideStats::new(&c, payoff[0].len());
        for _ in 0..iters {
            let i = a.select(&mut rng);
            let j = b.select(&mut rng);
            let r = payoff[i][j];
            a.update(i, r);
            b.update(j, 1.0 - r);
        }
        let norm = |v: &[f64]| {
            let t: f64 = v.iter().sum();
            v.iter().map(|x| x / t).collect::<Vec<_>>()
        };
        (norm(a.avg_strategy()), norm(b.avg_strategy()))
    }

    #[test]
    fn rps_average_strategy_near_uniform() {
        // rock-paper-scissors, win = 1, draw = 0.5, loss = 0
        let rps: [&[f64]; 3] = [&[0.5, 0.0, 1.0], &[1.0, 0.5, 0.0], &[0.0, 1.0, 0.5]];
        for mode in [SelectionMode::Exp3, SelectionMode::Rm] {
            let (p, q) = self_play(mode, &rps, 300_000);
            for x in p.iter().chain(q.iter()) {
                assert!((x - 1.0 / 3.0).abs() < 0.06, "{:?}: {:?} {:?}", mode, p, q);
            }
        }
    }

    #[test]
    fn biased_2x2_equilibrium() {
        // row payoffs [[0.9, 0.1], [0.2, 0.8]]: equilibrium row p(0) = 3/7
        let g: [&[f64]; 2] = [&[0.9, 0.1], &[0.2, 0.8]];
        for mode in [SelectionMode::Exp3, SelectionMode::Rm] {
            let (p, _) = self_play(mode, &g, 300_000);
            assert!((p[0] - 3.0 / 7.0).abs() < 0.07, "{:?}: {:?}", mode, p);
        }
    }
}
