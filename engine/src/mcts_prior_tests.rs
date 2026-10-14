//! Tests of PUCT selection with policy priors (child module of `mcts`).

use super::*;
use crate::features::tests::constructed;
use crate::prior::PriorConfig;

fn mn(visits: u32, total: f32, prior: f32) -> MoveNode {
    MoveNode {
        move_choice: MoveChoice::None,
        total_score: total,
        visits,
        prior,
    }
}

fn parent(visits: u32, total: f32) -> Node {
    let mut n = Node::new();
    n.times_visited = visits;
    n.total_score = total;
    n
}

#[test]
fn ucb1_unvisited_first_and_unchanged_without_priors() {
    let p = parent(10, 5.0);
    // a side without priors (-1) is plain UCB1: first unvisited option wins
    let opts = vec![mn(5, 4.0, -1.0), mn(0, 0.0, -1.0), mn(5, 1.0, -1.0)];
    let pc = PriorConfig::new(1.0);
    assert_eq!(p.maximize_ucb_for_side(&opts), 1);
    assert_eq!(p.maximize_puct_for_side(&opts, &pc, true), 1);
}

#[test]
fn puct_follows_prior_then_value() {
    let pc = PriorConfig::new(1.0);
    // fresh node: the highest prior is tried first, not the first index
    let p = parent(0, 0.0);
    let opts = vec![mn(0, 0.0, 0.1), mn(0, 0.0, 0.7), mn(0, 0.0, 0.2)];
    assert_eq!(p.maximize_puct_for_side(&opts, &pc, true), 1);
    // with many visits a clearly better option beats a high-prior bad one
    let p = parent(1000, 500.0);
    let opts = vec![mn(500, 150.0, 0.8), mn(100, 90.0, 0.1), mn(400, 260.0, 0.1)];
    assert_eq!(p.maximize_puct_for_side(&opts, &pc, true), 1);
    // side two reads the parent mean as 1 - mean for the unvisited default
    let p = parent(100, 90.0); // side one's mean 0.9 -> side two's 0.1
    let opts = vec![mn(50, 15.0, 0.01), mn(0, 0.0, 0.01)];
    assert_eq!(p.maximize_puct_for_side(&opts, &pc, false), 0);
    assert_eq!(p.maximize_puct_for_side(&opts, &pc, true), 1);
}

fn run(prior: Option<PriorConfig>, iters: u32) -> MctsResult {
    let mut state = constructed();
    let (s1, s2) = state.root_get_all_options();
    let sel = SelectionConfig { prior, ..SelectionConfig::default() };
    perform_mcts_joint(&mut state, s1, s2, Duration::from_secs(5), iters, None, &sel, false)
}

#[test]
fn default_visits_every_root_option_and_prior_concentrates() {
    let state = constructed();
    let (o1, o2) = state.root_get_all_options();
    assert!(o1.len() >= 3 && o2.len() >= 2, "{} {}", o1.len(), o2.len());

    let plain = run(None, 400);
    assert!(plain.s1.iter().all(|m| m.visits > 0), "stock UCB1 must try every option");
    assert!(plain.s2.iter().all(|m| m.visits > 0));
    assert!(plain.depth_max >= 1 && plain.depth_mean >= 1.0);

    let k1 = plain.s1.len();
    let mut p1 = vec![0.0f32; k1];
    p1[2] = 1.0;
    let mut pc = PriorConfig::new(2.0);
    pc.root_s1 = Some(&p1);
    let with = run(Some(pc), 400);
    let best = (0..k1).max_by_key(|&i| with.s1[i].visits).unwrap();
    assert_eq!(best, 2, "visits {:?}", with.s1.iter().map(|m| m.visits).collect::<Vec<_>>());
    assert!(with.s1[2].visits > plain.s1[2].visits);
    // side two has no prior here: still plain UCB1, every option visited
    assert!(with.s2.iter().all(|m| m.visits > 0));
}

#[test]
fn peaked_priors_search_deeper() {
    let state = constructed();
    let (o1, o2) = state.root_get_all_options();
    let plain = run(None, 4000);
    let mut a = vec![0.0f32; o1.len()];
    a[0] = 1.0;
    let mut b = vec![0.0f32; o2.len()];
    b[0] = 1.0;
    let mut pc = PriorConfig::new(1.0);
    pc.root_s1 = Some(&a);
    pc.root_s2 = Some(&b);
    let with = run(Some(pc), 4000);
    assert!(with.s1[0].visits > plain.s1[0].visits);
    assert!(with.depth_mean >= plain.depth_mean, "{} vs {}", with.depth_mean, plain.depth_mean);
}
