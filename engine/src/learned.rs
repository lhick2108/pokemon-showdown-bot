//! Learned leaf evaluators for MCTS (linear and 1-hidden-layer ReLU MLP).
//!
//! An `Evaluator` maps `features(state, SideOne)` to a single logit
//! z = logit(P(side one wins)). The MCTS leaf score is then
//!
//! * relative (default): `sigmoid(z_leaf - z_root)` (same shape as the stock
//!   `sigmoid(0.0125 * (evaluate(leaf) - evaluate(root)))`, but z is already in
//!   log-odds so no 0.0125 scale), or
//! * absolute: `sigmoid(z_leaf)` (AlphaZero style).
//!
//! Terminal states keep their exact 0 / 1 value, as in stock.
//!
//! Weights file format (little endian, see `README.md`):
//! ```text
//! magic    4 bytes  "PEVN"
//! format   u32      1
//! feat_ver u32      FEATURE_VERSION used to train
//! kind     u32      1 = linear, 2 = mlp
//! n_in     u32      FEATURE_LEN
//! n_hidden u32      0 for linear
//! linear:  w[n_in] f32, b f32
//! mlp:     W1[n_hidden][n_in] f32 (row-major, PyTorch nn.Linear weight layout),
//!          b1[n_hidden], w2[n_hidden], b2
//! ```

use crate::engine::evaluate::evaluate;
use crate::features::{features_into, FEATURE_LEN, FEATURE_VERSION};
use crate::state::{SideReference, State};
use std::convert::TryInto;

pub const MAGIC: &[u8; 4] = b"PEVN";
pub const FORMAT_VERSION: u32 = 1;
const KIND_LINEAR: u32 = 1;
const KIND_MLP: u32 = 2;

#[derive(Debug, Clone)]
enum Model {
    Linear {
        w: Vec<f32>,
        b: f32,
    },
    Mlp {
        hidden: usize,
        /// transposed first layer: `w1t[i * hidden + j]` = weight from input i to hidden j
        w1t: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: f32,
    },
}

#[derive(Debug, Clone)]
pub struct Evaluator {
    model: Model,
    /// If true the leaf score is `sigmoid(z_leaf - z_root)`, else `sigmoid(z_leaf)`.
    pub relative_to_root: bool,
}

/// Per-worker scratch buffers so evaluation does not allocate.
pub struct Scratch {
    feats: Vec<f32>,
    hidden: Vec<f32>,
}

impl Scratch {
    pub fn new() -> Scratch {
        Scratch {
            feats: vec![0.0; FEATURE_LEN],
            hidden: Vec::new(),
        }
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Scratch::new()
    }
}

#[inline]
pub fn logistic(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Evaluator {
    /// value = w . features + b
    pub fn linear(w: Vec<f32>, b: f32) -> Result<Evaluator, String> {
        if w.len() != FEATURE_LEN {
            return Err(format!(
                "linear weights have length {}, expected FEATURE_LEN = {}",
                w.len(),
                FEATURE_LEN
            ));
        }
        Ok(Evaluator {
            model: Model::Linear { w, b },
            relative_to_root: true,
        })
    }

    /// value = w2 . relu(W1 x + b1) + b2, with `w1` row-major `[hidden][FEATURE_LEN]`.
    pub fn mlp(
        hidden: usize,
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: f32,
    ) -> Result<Evaluator, String> {
        if hidden == 0 {
            return Err("hidden size must be > 0".to_string());
        }
        if w1.len() != hidden * FEATURE_LEN {
            return Err(format!(
                "w1 has length {}, expected hidden * FEATURE_LEN = {}",
                w1.len(),
                hidden * FEATURE_LEN
            ));
        }
        if b1.len() != hidden || w2.len() != hidden {
            return Err(format!(
                "b1 / w2 must have length hidden = {} (got {} / {})",
                hidden,
                b1.len(),
                w2.len()
            ));
        }
        let mut w1t = vec![0.0f32; w1.len()];
        for j in 0..hidden {
            for i in 0..FEATURE_LEN {
                w1t[i * hidden + j] = w1[j * FEATURE_LEN + i];
            }
        }
        Ok(Evaluator {
            model: Model::Mlp {
                hidden,
                w1t,
                b1,
                w2,
                b2,
            },
            relative_to_root: true,
        })
    }

    pub fn with_relative_to_root(mut self, relative: bool) -> Evaluator {
        self.relative_to_root = relative;
        self
    }

    pub fn kind_name(&self) -> &'static str {
        match self.model {
            Model::Linear { .. } => "linear",
            Model::Mlp { .. } => "mlp",
        }
    }

    /// Hidden layer size (0 for linear).
    pub fn hidden_size(&self) -> usize {
        match &self.model {
            Model::Linear { .. } => 0,
            Model::Mlp { hidden, .. } => *hidden,
        }
    }

    /// Logit from an already computed feature vector (length `FEATURE_LEN`).
    /// Inputs that are exactly 0.0 are skipped (the encoding is sparse).
    pub fn logit_from_features(&self, x: &[f32], hidden_buf: &mut Vec<f32>) -> f32 {
        match &self.model {
            Model::Linear { w, b } => {
                let mut z = *b;
                for (xi, wi) in x.iter().zip(w.iter()) {
                    if *xi != 0.0 {
                        z += xi * wi;
                    }
                }
                z
            }
            Model::Mlp {
                hidden,
                w1t,
                b1,
                w2,
                b2,
            } => {
                let h = *hidden;
                hidden_buf.clear();
                hidden_buf.extend_from_slice(b1);
                for (i, xi) in x.iter().enumerate() {
                    if *xi != 0.0 {
                        let row = &w1t[i * h..(i + 1) * h];
                        for (hj, wj) in hidden_buf.iter_mut().zip(row.iter()) {
                            *hj += xi * wj;
                        }
                    }
                }
                let mut z = *b2;
                for (hj, wj) in hidden_buf.iter().zip(w2.iter()) {
                    z += hj.max(0.0) * wj;
                }
                z
            }
        }
    }

    /// z = logit of P(side one wins) at `state`.
    pub fn logit(&self, state: &State, scratch: &mut Scratch) -> f32 {
        features_into(state, SideReference::SideOne, &mut scratch.feats);
        self.logit_from_features(&scratch.feats, &mut scratch.hidden)
    }

    /// Leaf score in [0, 1] from the leaf and root logits.
    #[inline]
    pub fn score(&self, leaf_logit: f32, root_logit: f32) -> f32 {
        if self.relative_to_root {
            logistic(leaf_logit - root_logit)
        } else {
            logistic(leaf_logit)
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        let push = |out: &mut Vec<u8>, v: u32| out.extend_from_slice(&v.to_le_bytes());
        push(&mut out, FORMAT_VERSION);
        push(&mut out, FEATURE_VERSION);
        let pushf = |out: &mut Vec<u8>, xs: &[f32]| {
            for x in xs {
                out.extend_from_slice(&x.to_le_bytes());
            }
        };
        match &self.model {
            Model::Linear { w, b } => {
                push(&mut out, KIND_LINEAR);
                push(&mut out, FEATURE_LEN as u32);
                push(&mut out, 0);
                pushf(&mut out, w);
                pushf(&mut out, &[*b]);
            }
            Model::Mlp {
                hidden,
                w1t,
                b1,
                w2,
                b2,
            } => {
                push(&mut out, KIND_MLP);
                push(&mut out, FEATURE_LEN as u32);
                push(&mut out, *hidden as u32);
                // back to row-major [hidden][n_in]
                for j in 0..*hidden {
                    for i in 0..FEATURE_LEN {
                        pushf(&mut out, &[w1t[i * hidden + j]]);
                    }
                }
                pushf(&mut out, b1);
                pushf(&mut out, w2);
                pushf(&mut out, &[*b2]);
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Evaluator, String> {
        if bytes.len() < 24 || &bytes[0..4] != MAGIC {
            return Err("not a poke-engine evaluator file (bad magic)".to_string());
        }
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let format = u32_at(4);
        if format != FORMAT_VERSION {
            return Err(format!("unsupported weights format version {format}"));
        }
        let feat_ver = u32_at(8);
        if feat_ver != FEATURE_VERSION {
            return Err(format!(
                "weights were trained with FEATURE_VERSION {feat_ver}, this engine has {FEATURE_VERSION}"
            ));
        }
        let kind = u32_at(12);
        let n_in = u32_at(16) as usize;
        let hidden = u32_at(20) as usize;
        if n_in != FEATURE_LEN {
            return Err(format!(
                "weights expect {n_in} inputs, this engine has FEATURE_LEN = {FEATURE_LEN}"
            ));
        }
        let body = &bytes[24..];
        let floats_needed = match kind {
            KIND_LINEAR => n_in + 1,
            KIND_MLP => hidden * n_in + hidden + hidden + 1,
            _ => return Err(format!("unknown model kind {kind}")),
        };
        if body.len() != floats_needed * 4 {
            return Err(format!(
                "weights body is {} bytes, expected {}",
                body.len(),
                floats_needed * 4
            ));
        }
        let floats: Vec<f32> = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if kind == KIND_LINEAR {
            Evaluator::linear(floats[..n_in].to_vec(), floats[n_in])
        } else {
            let a = hidden * n_in;
            Evaluator::mlp(
                hidden,
                floats[..a].to_vec(),
                floats[a..a + hidden].to_vec(),
                floats[a + hidden..a + 2 * hidden].to_vec(),
                floats[a + 2 * hidden],
            )
        }
    }

    pub fn load(path: &str) -> Result<Evaluator, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        Evaluator::from_bytes(&bytes)
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, self.to_bytes()).map_err(|e| format!("cannot write {path}: {e}"))
    }
}

/// Leaf evaluation strategy used by the MCTS loops: stock `evaluate()` or a
/// learned `Evaluator`. Each search worker owns one (it carries scratch buffers).
pub struct LeafEval<'a> {
    learned: Option<&'a Evaluator>,
    scratch: Scratch,
}

impl<'a> LeafEval<'a> {
    pub fn new(learned: Option<&'a Evaluator>) -> LeafEval<'a> {
        LeafEval {
            learned,
            scratch: Scratch::new(),
        }
    }

    /// Raw value of a non-terminal position: stock evaluation points, or the
    /// learned logit.
    #[inline]
    pub fn value(&mut self, state: &State) -> f32 {
        match self.learned {
            None => evaluate(state),
            Some(ev) => ev.logit(state, &mut self.scratch),
        }
    }

    /// Maps (leaf value, root value) to an MCTS score in [0, 1].
    #[inline]
    pub fn score(&self, leaf: f32, root: f32) -> f32 {
        match self.learned {
            // identical to the stock `sigmoid(eval - root_eval)`
            None => 1.0 / (1.0 + (-0.0125 * (leaf - root)).exp()),
            Some(ev) => ev.score(leaf, root),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{feature_names, features};

    fn idx(name: &str) -> usize {
        feature_names().iter().position(|n| n == name).unwrap()
    }

    fn state_with_hp(own_hp: i16, opp_hp: i16) -> State {
        let mut s = State::default();
        s.side_one.get_active().hp = own_hp;
        s.side_two.get_active().hp = opp_hp;
        s
    }

    #[test]
    fn linear_expected_output() {
        // z = 2 * own.p0.hp - 3 * opp.p0.hp + 0.5
        let mut w = vec![0.0; FEATURE_LEN];
        w[idx("own.p0.hp")] = 2.0;
        w[idx("opp.p0.hp")] = -3.0;
        let ev = Evaluator::linear(w, 0.5).unwrap();
        let mut sc = Scratch::new();
        let z = ev.logit(&state_with_hp(100, 40), &mut sc);
        assert!((z - (2.0 * 1.0 - 3.0 * 0.4 + 0.5)).abs() < 1e-5, "z = {z}");
        let z2 = ev.logit(&state_with_hp(50, 100), &mut sc);
        assert!((z2 - (2.0 * 0.5 - 3.0 + 0.5)).abs() < 1e-5, "z2 = {z2}");
    }

    #[test]
    fn mlp_expected_output() {
        // hidden0 = relu(own.hp - opp.hp), hidden1 = relu(-own.hp + opp.hp) , hidden2 = relu(-1)
        // z = 4 * h0 + 1 * h1 - 5 * h2 + 0.25
        let hidden = 3;
        let mut w1 = vec![0.0; hidden * FEATURE_LEN];
        let (own, opp) = (idx("own.p0.hp"), idx("opp.p0.hp"));
        w1[0 * FEATURE_LEN + own] = 1.0;
        w1[0 * FEATURE_LEN + opp] = -1.0;
        w1[1 * FEATURE_LEN + own] = -1.0;
        w1[1 * FEATURE_LEN + opp] = 1.0;
        let b1 = vec![0.0, 0.0, -1.0];
        let w2 = vec![4.0, 1.0, -5.0];
        let ev = Evaluator::mlp(hidden, w1, b1, w2, 0.25).unwrap();
        let mut sc = Scratch::new();
        // own 100%, opp 40%: h0 = 0.6, h1 = relu(-0.6) = 0, h2 = relu(-1) = 0
        let z = ev.logit(&state_with_hp(100, 40), &mut sc);
        assert!((z - (4.0 * 0.6 + 0.25)).abs() < 1e-5, "z = {z}");
        // own 20%, opp 100%: h0 = 0, h1 = 0.8
        let z = ev.logit(&state_with_hp(20, 100), &mut sc);
        assert!((z - (0.8 + 0.25)).abs() < 1e-5, "z = {z}");
    }

    #[test]
    fn mlp_matches_dense_reference_on_random_weights() {
        // deterministic pseudo-random weights, compare against a dense computation
        let hidden = 7;
        let mut seed = 12345u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let w1: Vec<f32> = (0..hidden * FEATURE_LEN).map(|_| next()).collect();
        let b1: Vec<f32> = (0..hidden).map(|_| next()).collect();
        let w2: Vec<f32> = (0..hidden).map(|_| next()).collect();
        let ev = Evaluator::mlp(hidden, w1.clone(), b1.clone(), w2.clone(), 0.1).unwrap();
        let mut s = state_with_hp(60, 30);
        s.side_one.attack_boost = 2;
        s.weather.weather_type = crate::engine::state::Weather::SUN;
        s.weather.turns_remaining = 4;
        let x = features(&s, SideReference::SideOne);
        let mut expected = 0.1f32;
        for j in 0..hidden {
            let mut a = b1[j];
            for i in 0..FEATURE_LEN {
                a += w1[j * FEATURE_LEN + i] * x[i];
            }
            expected += a.max(0.0) * w2[j];
        }
        let mut buf = Vec::new();
        let z = ev.logit_from_features(&x, &mut buf);
        assert!((z - expected).abs() < 1e-3, "z = {z}, expected = {expected}");
    }

    #[test]
    fn score_mapping() {
        let ev = Evaluator::linear(vec![0.0; FEATURE_LEN], 0.0).unwrap();
        assert!((ev.score(1.5, 1.5) - 0.5).abs() < 1e-6);
        assert!((ev.score(2.0, 0.0) - logistic(2.0)).abs() < 1e-6);
        let abs = ev.clone().with_relative_to_root(false);
        assert!((abs.score(2.0, 7.0) - logistic(2.0)).abs() < 1e-6);
        // stock path is unchanged
        let le = LeafEval::new(None);
        let stock = 1.0 / (1.0 + (-0.0125f32 * (30.0 - 10.0)).exp());
        assert_eq!(le.score(30.0, 10.0), stock);
    }

    #[test]
    fn bytes_roundtrip_and_validation() {
        let mut w = vec![0.0; FEATURE_LEN];
        w[3] = 1.25;
        let lin = Evaluator::linear(w, -2.0).unwrap();
        let back = Evaluator::from_bytes(&lin.to_bytes()).unwrap();
        let mut b = Vec::new();
        let x = {
            let mut x = vec![0.0; FEATURE_LEN];
            x[3] = 2.0;
            x
        };
        assert_eq!(back.logit_from_features(&x, &mut b), 0.5);

        let hidden = 2;
        let mut w1 = vec![0.0; hidden * FEATURE_LEN];
        w1[FEATURE_LEN + 5] = 3.0;
        let mlp = Evaluator::mlp(hidden, w1, vec![0.0, 1.0], vec![1.0, 2.0], 0.5).unwrap();
        let bytes = mlp.to_bytes();
        let back = Evaluator::from_bytes(&bytes).unwrap();
        let mut x = vec![0.0; FEATURE_LEN];
        x[5] = 1.0;
        // h = [0, 3 + 1] -> 0 + 2*4 + 0.5
        assert_eq!(back.logit_from_features(&x, &mut b), 8.5);
        assert_eq!(back.hidden_size(), 2);

        assert!(Evaluator::from_bytes(&bytes[..bytes.len() - 4]).is_err());
        let mut bad = bytes.clone();
        bad[8] = 99; // feature version
        assert!(Evaluator::from_bytes(&bad).is_err());
    }
}
