//! Timing harness for the learned leaf evaluator.
//! usage: cargo run --release --example bench_net -- <states.txt> <model.pev> [iters_per_state] [reps]
use poke_engine::engine::evaluate::evaluate;
use poke_engine::features::{features_into, FEATURE_LEN};
use poke_engine::learned::{Evaluator, Scratch};
use poke_engine::mcts::perform_mcts_with;
use poke_engine::state::{SideReference, State};
use std::hint::black_box;
use std::time::{Duration, Instant};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&a[1]).unwrap();
    let ev = if a[2].starts_with("zero") {
        // net compute with w2 = 0 and stock_coef = 0.0125: identical search tree to stock,
        // so the it/s ratio isolates the pure evaluation overhead
        let (hs, ver) = match a[2][4..].split_once('v') { Some((h, v)) => (h, v.parse::<u32>().unwrap()), None => (&a[2][4..], 1) };
        let h: usize = hs.parse().unwrap();
        let n_in = poke_engine::features::feature_len(ver).unwrap();
        if poke_engine::features_v4::has_identity(ver) {
            // embedding dim 8 for each of species / item / ability / move (random-ish weights)
            use poke_engine::features_v4::vocab_size;
            let dims = [8usize; 4]; // zero<H>v4 | zero<H>v5
            let tables: [Vec<f32>; 4] = std::array::from_fn(|t| {
                (0..vocab_size(t) * dims[t]).map(|i| ((i * 104729) % 1000) as f32 / 1000.0 - 0.5).collect()
            });
            let emb = poke_engine::learned::Embed::new(dims, tables).unwrap();
            let n_total = n_in + emb.input_len();
            let w1: Vec<f32> = (0..h * n_total).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
            Evaluator::mlp_embed_v(ver, h, w1, vec![0.1; h], vec![0.0; h], 0.0, emb)
                .unwrap()
                .with_stock_coef(0.0125)
                .unwrap()
        } else {
            let w1: Vec<f32> = (0..h * n_in).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
            Evaluator::mlp_v(ver, h, w1, vec![0.1; h], vec![0.0; h], 0.0)
                .unwrap()
                .with_stock_coef(0.0125)
                .unwrap()
        }
    } else {
        Evaluator::from_bytes(&std::fs::read(&a[2]).unwrap()).unwrap()
    };
    let iters: u32 = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(20000);
    let reps: usize = a.get(4).map(|s| s.parse().unwrap()).unwrap_or(3);
    let states: Vec<State> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(State::deserialize)
        .collect();
    println!(
        "{} states, model v{} hidden={} residual={}",
        states.len(),
        ev.feature_version(),
        ev.hidden_size(),
        ev.is_residual()
    );

    {
        let mut sc2 = Scratch::new();
        let mut md = 0.0f32;
        for s in &states {
            let mut sc = Scratch::new(); // a Scratch caches per-team stats: one per state
            md = md.max((ev.logit(s, &mut sc) - ev.logit_reference(s, &mut sc2)).abs());
        }
        println!("max |fast - reference| over states: {md:e}");
    }
    // micro: per-call cost
    let n = 200;
    let mut buf = vec![0.0f32; FEATURE_LEN];
    let mut scratch = Scratch::new();
    let mut h = Vec::new();
    let micro = |name: &str, f: &mut dyn FnMut(&State) -> f32| {
        let mut best = f64::MAX;
        for _ in 0..7 {
            let t = Instant::now();
            let mut acc = 0.0;
            // same state repeatedly (as inside one MCTS search) so the per-Scratch cache hits
            for s in &states {
                for _ in 0..n {
                    acc += f(s);
                }
            }
            black_box(acc);
            best = best.min(t.elapsed().as_nanos() as f64 / (n * states.len()) as f64);
        }
        println!("  {name:<14} {best:8.1} ns/call (min of 7)");
    };
    micro("stock evaluate", &mut |s| evaluate(black_box(s)));
    let mut buf2 = buf.clone();
    micro("features_into", &mut |s| {
        features_into(black_box(s), SideReference::SideOne, &mut buf2);
        buf2[0]
    });
    features_into(&states[0], SideReference::SideOne, &mut buf);
    micro("mlp(feats)", &mut |_| {
        ev.logit_from_features(black_box(&buf), &mut h)
    });
    micro("ev.logit_ref", &mut |s| ev.logit_reference(black_box(s), &mut scratch));
    micro("ev.logit", &mut |s| ev.logit(black_box(s), &mut scratch));
    {
        use poke_engine::features_v2::{dynamic_blocks, static_key, Sink};
        struct Null(f32);
        impl Sink for Null {
            fn one(&mut self, i: usize) {
                self.0 += i as f32;
            }
            fn val(&mut self, i: usize, v: f32) {
                self.0 += i as f32 * v;
            }
        }
        micro("v2 dynamic", &mut |s| {
            let mut n = Null(0.0);
            dynamic_blocks(&mut n, black_box(s), SideReference::SideOne);
            n.0
        });
        micro("v3 matchup", &mut |s| {
            let mut n = Null(0.0);
            poke_engine::features_v3::matchup_blocks(&mut n, black_box(s), SideReference::SideOne);
            n.0
        });
        micro("v2 keys x12", &mut |s| {
            let mut a = 0u64;
            for side in [&s.side_one, &s.side_two] {
                for p in side.pokemon.pkmn.iter() {
                    let k = static_key(black_box(p));
                    a = a.wrapping_add(k[0] ^ k[1]);
                }
            }
            a as f32
        });
    }

    // MCTS it/s: per state, interleave stock / net `reps` times and keep the fastest of each
    let mut tot = [0f64; 2];
    let mut its_tot = [0u64; 2];
    for s in &states {
        let mut best = [f64::MAX; 2];
        let mut its = [0u64; 2];
        for _ in 0..reps {
            for k in 0..2 {
                let mut st = s.clone();
                let (o1, o2) = st.root_get_all_options();
                let t = Instant::now();
                let res = perform_mcts_with(
                    &mut st,
                    o1,
                    o2,
                    Duration::ZERO,
                    iters,
                    if k == 1 { Some(&ev) } else { None },
                );
                let dt = t.elapsed().as_secs_f64();
                if dt < best[k] {
                    best[k] = dt;
                    its[k] = res.iteration_count as u64;
                }
            }
        }
        for k in 0..2 {
            tot[k] += best[k];
            its_tot[k] += its[k];
        }
    }
    let best = [its_tot[0] as f64 / tot[0], its_tot[1] as f64 / tot[1]];
    println!(
        "best stock {:.0}  net {:.0}  ratio {:.3}",
        best[0],
        best[1],
        best[1] / best[0]
    );
}
