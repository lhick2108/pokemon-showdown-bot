"""Fair-compute preflight and progress output (no server needed)."""

from __future__ import annotations

from types import SimpleNamespace

from showdown_bot import compute, league


def test_preflight_warns_and_strict_refuses():
    info, warns, refuse = compute.preflight(workers=8, demand=8, cores=16)
    assert "8 pool workers, 16 cores" in info[0] and not warns and not refuse
    _, warns, refuse = compute.preflight(workers=16, demand=8, cores=16)
    assert any("fewer than" in w for w in warns) and not refuse  # warn only without --strict
    _, warns, refuse = compute.preflight(workers=20, demand=8, cores=16)
    assert any("20 workers > 16 cores" in w for w in warns)
    assert compute.preflight(workers=15, demand=1, cores=16, strict=True)[2]
    assert not compute.preflight(workers=14, demand=1, cores=16, strict=True)[2]


def test_preflight_warns_when_demand_exceeds_workers():
    _, warns, refuse = compute.preflight(workers=4, demand=24 * 2 * 8, cores=32)
    assert any("jobs queue" in w for w in warns) and not refuse


def test_progress_line_format_and_flush(capsys):
    line = compute.progress_line(37, 300, "net", 19, 18, now=0)
    assert line.endswith("37/300 games  net 19-18 (51.4%)") and line.startswith("[")
    printer = compute.ProgressPrinter(300)
    assert printer.update(0, "net", 0, 0) is None
    assert printer.update(5, "net", 3, 2, paired="paired 2 pairs 60.0% [10.0%, 90.0%]")
    assert printer.update(5, "net", 3, 2) is None  # unchanged: silent
    out = capsys.readouterr().out
    assert out.count("games") == 1 and "paired 2 pairs" in out


def test_paired_text():
    assert compute.paired_text([(0, 1.0)]) is None  # incomplete pair
    text = compute.paired_text([(0, 1.0), (0, 0.0), (1, 1.0), (1, 1.0)])
    assert text.startswith("paired 2 pairs 75.0%")


def test_league_preflight_counts_world_demand(tmp_path):
    specs = league.parse_bots("a=search:worlds=8,b=search:worlds=8,r=random")
    cfg = league.LeagueConfig(out=tmp_path, concurrency=3, parallel_pairs=1,
                              search_workers=4, strict_compute=True)
    rep, warns, refuse = league.league_preflight(specs, cfg, cpu_count=6)
    assert rep["peak_world_jobs"] == 2 * 8 * 3  # busiest pair: a vs b
    assert any("jobs queue" in w for w in warns)
    assert not refuse  # 4 workers on 6 cores leaves exactly the reserve of 2: allowed
    cfg.search_workers = 5
    assert league.league_preflight(specs, cfg, cpu_count=6)[2]


def test_league_markdown_has_compute_section(tmp_path):
    comp = {"preflight": {"line": "compute preflight: 4 pool workers"},
            "preflight_warnings": ["WARNING: x"],
            "bots": {"net": {"iterations": 1000, "search_ms": 500, "iterations_per_s": 2000.0,
                             "decisions": 3, "escalations": 1, "fallbacks": 0, "world_errors": 0}}}
    res = {"n_games": 0, "anchor": None, "prior_weight": 1, "bootstrap": 0, "bootstrap_seed": 0,
           "ratings": [], "head_to_head": [], "compute": comp}
    md = league.format_markdown(res)
    assert "## Compute" in md and "| net | 2,000 | 1,000 | 3 | 1 | 0 | 0 |" in md


def test_league_progress_tally(capsys):
    tracker = league.Tracker()
    pa = SimpleNamespace(n_won_battles=3, n_tied_battles=0)
    pb = SimpleNamespace(n_won_battles=2, n_tied_battles=0)
    tracker.live[("a", "b")] = (pa, pb)
    printer = compute.ProgressPrinter(30)
    league.emit_progress(tracker, printer)
    assert "5/30 games  a v b 3-2" in capsys.readouterr().out
    league.emit_progress(tracker, printer)  # nothing new
    assert capsys.readouterr().out == ""
