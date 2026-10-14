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
pub struct SelectionConfig<'a> {
    pub mode: SelectionMode,
    pub gamma: f64,
    /// Policy priors (PUCT selection, `prior.rs`). `None` (the default) is the stock search.
    /// Only meaningful with `SelectionMode::Ucb1`.
    pub prior: Option<crate::prior::PriorConfig<'a>>,
}

impl Default for SelectionConfig<'_> {
    fn default() -> Self {
        SelectionConfig {
            mode: SelectionMode::Ucb1,
            gamma: 0.1,
            prior: None,
        }
    }
}

/// Largest number of options per side the inline arrays hold (4 moves + 4 tera
/// moves + 5 switches = 13 at most).
pub const MAX_OPTIONS: usize = 16;

/// Exp3 weights are renormalised when one exceeds this, and floored (relative to
/// the new maximum) so that an arm never underflows to a permanent zero.
const EXP3_RENORM_AT: f32 = 1e20;
const EXP3_RENORM_SCALE: f32 = 1e-20;

/// Per-node, per-side state for Exp3 / RM. Fixed-size inline arrays (no heap
/// allocation of its own); `NodeStats` boxes both sides once per node.
///
/// * Exp3 keeps `acc[i] = exp(eta * (G_i - shift))` incrementally (only the
///   updated arm changes), so `select` needs one multiply per arm and no `exp`.
/// * RM keeps `acc[i] = R_i` and `pos_sum = sum R^+`, refreshed in `update`.
#[derive(Debug)]
pub struct SideStats {
    mode: SelectionMode,
    k: usize,
    gamma: f32,
    eta: f32,
    pos_sum: f32,
    acc: [f32; MAX_OPTIONS],
    /// the strategy used at the most recent selection
    sigma: [f32; MAX_OPTIONS],
    /// cumulative strategy S (f64: millions of small additions)
    avg: [f64; MAX_OPTIONS],
}

impl SideStats {
    pub fn new(cfg: &SelectionConfig, k: usize) -> SideStats {
        assert!(
            k > 0 && k <= MAX_OPTIONS,
            "selection supports at most {} options per side (got {})", MAX_OPTIONS, k
        );
        let mut acc = [0.0; MAX_OPTIONS];
        if cfg.mode == SelectionMode::Exp3 {
            acc[..k].fill(1.0);
        }
        SideStats {
            mode: cfg.mode,
            k,
            gamma: cfg.gamma as f32,
            eta: (cfg.gamma / k as f64) as f32,
            pos_sum: 0.0,
            acc,
            sigma: [1.0 / k as f32; MAX_OPTIONS],
            avg: [0.0; MAX_OPTIONS],
        }
    }

    /// Compute sigma, add it to the average strategy, and sample an action.
    pub fn select(&mut self, rng: &mut impl Rng) -> usize {
        let k = self.k;
        let explore = self.gamma / k as f32;
        match self.mode {
            SelectionMode::Exp3 => {
                let sum: f32 = self.acc[..k].iter().sum();
                let scale = (1.0 - self.gamma) / sum;
                for (s, &w) in self.sigma[..k].iter_mut().zip(&self.acc[..k]) {
                    *s = w * scale + explore;
                }
            }
            SelectionMode::Rm => {
                if self.pos_sum > 0.0 {
                    let scale = (1.0 - self.gamma) / self.pos_sum;
                    for (s, &r) in self.sigma[..k].iter_mut().zip(&self.acc[..k]) {
                        *s = r.max(0.0) * scale + explore;
                    }
                } else {
                    self.sigma[..k].fill((1.0 - self.gamma) / k as f32 + explore);
                }
            }
            SelectionMode::Ucb1 => unreachable!("SideStats is not used for ucb1"),
        }
        let mut threshold: f32 = rng.random::<f32>();
        let mut pick = k - 1;
        let mut found = false;
        for i in 0..k {
            let s = self.sigma[i];
            self.avg[i] += s as f64;
            if !found {
                if threshold < s {
                    pick = i;
                    found = true;
                } else {
                    threshold -= s;
                }
            }
        }
        pick
    }

    /// Update after action `chosen` (selected with the current `sigma`) earned `reward`.
    pub fn update(&mut self, chosen: usize, reward: f32) {
        let est = reward / self.sigma[chosen];
        match self.mode {
            SelectionMode::Exp3 => {
                let w = self.acc[chosen] * (self.eta * est).exp();
                self.acc[chosen] = w;
                if w > EXP3_RENORM_AT {
                    for x in self.acc[..self.k].iter_mut() {
                        *x = (*x * EXP3_RENORM_SCALE).max(EXP3_RENORM_SCALE);
                    }
                }
            }
            SelectionMode::Rm => {
                self.acc[chosen] += est;
                let mut pos = 0.0;
                for r in self.acc[..self.k].iter_mut() {
                    *r -= reward;
                    pos += r.max(0.0);
                }
                self.pos_sum = pos;
            }
            SelectionMode::Ucb1 => {}
        }
    }

    pub fn avg_strategy(&self) -> &[f64] {
        &self.avg[..self.k]
    }
}

/// Both sides' state for one node, allocated once at first selection.
#[derive(Debug)]
pub struct NodeStats {
    pub s1: SideStats,
    pub s2: SideStats,
}

impl NodeStats {
    pub fn new(cfg: &SelectionConfig, k1: usize, k2: usize) -> NodeStats {
        NodeStats {
            s1: SideStats::new(cfg, k1),
            s2: SideStats::new(cfg, k2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn cfg(mode: SelectionMode, gamma: f64) -> SelectionConfig<'static> {
        SelectionConfig { mode, gamma, prior: None }
    }

    #[test]
    fn sigma_sums_to_one_with_gamma_floor() {
        for mode in [SelectionMode::Exp3, SelectionMode::Rm] {
            let mut rng = StdRng::seed_from_u64(3);
            let mut st = SideStats::new(&cfg(mode, 0.1), 4);
            for n in 0..5000 {
                let i = st.select(&mut rng);
                let sum: f32 = st.sigma[..4].iter().sum();
                assert!((sum - 1.0).abs() < 1e-4, "{mode:?} {n}: {:?}", st.sigma);
                assert!(st.sigma[..4].iter().all(|&s| s >= 0.1 / 4.0 - 1e-6 && s.is_finite()));
                st.update(i, if i == 2 { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn exp3_weights_stay_finite_under_extreme_gaps() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut st = SideStats::new(&cfg(SelectionMode::Exp3, 0.1), 3);
        for _ in 0..2_000_000 {
            let i = st.select(&mut rng);
            st.update(i, if i == 0 { 1.0 } else { 0.0 });
        }
        assert!(st.acc[..3].iter().all(|w| w.is_finite() && *w > 0.0));
        assert!(st.sigma[0] > 0.85, "{:?}", st.sigma);
    }

    #[test]
    fn rm_uniform_when_no_positive_regret_and_proportional_otherwise() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut st = SideStats::new(&cfg(SelectionMode::Rm, 0.1), 3);
        st.select(&mut rng);
        for s in &st.sigma[..3] {
            assert!((s - 1.0 / 3.0).abs() < 1e-6);
        }
        let mut st = SideStats::new(&cfg(SelectionMode::Rm, 0.0), 3);
        st.acc[..3].copy_from_slice(&[3.0, 1.0, -5.0]);
        st.pos_sum = 4.0;
        st.select(&mut rng);
        assert!((st.sigma[0] - 0.75).abs() < 1e-6 && (st.sigma[1] - 0.25).abs() < 1e-6);
        assert_eq!(st.sigma[2], 0.0);
    }

    #[test]
    fn update_rules() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut e = SideStats::new(&cfg(SelectionMode::Exp3, 0.1), 2);
        let i = e.select(&mut rng);
        let s = e.sigma[i];
        e.update(i, 0.6);
        let eta = 0.1f32 / 2.0;
        assert!((e.acc[i] - (eta * 0.6 / s).exp()).abs() < 1e-5 && e.acc[1 - i] == 1.0);

        let mut m = SideStats::new(&cfg(SelectionMode::Rm, 0.1), 3);
        let i = m.select(&mut rng);
        let s = m.sigma[i];
        m.update(i, 0.5);
        let mut pos = 0.0;
        for j in 0..3 {
            let want = if j == i { 0.5 / s - 0.5 } else { -0.5 };
            assert!((m.acc[j] - want).abs() < 1e-5);
            pos += want.max(0.0);
        }
        assert!((m.pos_sum - pos).abs() < 1e-5);
        assert!((m.avg_strategy().iter().sum::<f64>() - 1.0).abs() < 1e-6);
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
            a.update(i, r as f32);
            b.update(j, 1.0 - r as f32);
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
