"""League spec engine=<tag>: a search worker pool on a pinned engine wheel (showdown_bot.engines)."""

import pytest

from showdown_bot import engines, league
from showdown_bot.players import _worker
from showdown_bot.players.search import get_pool, reset_pool

TAG = "sdb.28"
HAVE_WHEEL = any((engines.REPO / "engine" / "wheels").glob(f"poke_engine-0.0.48+{TAG}-*.whl"))


def test_engine_spec_key_parses():
    spec = league.parse_spec(f"x=best:worlds=16:engine={TAG}")
    assert spec.param("engine") == TAG
    assert "engine" not in league.search_config_fields(spec)


def test_unknown_engine_tag_is_rejected():
    with pytest.raises(ValueError):
        engines.engine_dir("sdb.nope")


@pytest.mark.skipif(not HAVE_WHEEL, reason=f"no {TAG} wheel in engine/wheels")
def test_pinned_pool_runs_that_engine():
    pool = get_pool(1, engine=TAG)
    try:
        assert get_pool((1, TAG)) is pool
        assert pool.submit(_worker.engine_version).result(timeout=120) == f"0.0.48+{TAG}"
    finally:
        reset_pool((1, TAG), kill=True)
