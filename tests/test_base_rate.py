"""The base-rate measurement: the split it searches and the expected false accepts."""

import numpy as np

from svs import base_rate
from svs.base_rate import expected_rows, pick
from svs.calibrate import POOL_TIERS
from svs.trials import Utterance


def utts() -> list[Utterance]:
    # 50 speakers x 2 videos x 30 clips; clip durations 3-19 s.
    return [
        Utterance(f"id{s}/v{v}/{c}", f"id{s}", f"v{v}", "mf"[s % 2], 3.0 + c % 17)
        for s in range(50)
        for v in range(2)
        for c in range(30)
    ]


def test_pick_keeps_query_speakers_out_of_the_pool(monkeypatch):
    monkeypatch.setattr(base_rate, "N_QUERIES", 100)
    monkeypatch.setattr(base_rate, "POOL_TIERS", (500, 1_000))
    rows = utts()
    queries, targets, pool = pick(rows, np.random.default_rng(0))
    query_speakers = {rows[q].speaker for q in queries}
    assert not query_speakers & {rows[p].speaker for p in pool}
    assert len(pool) == 1_000 and len(set(pool)) == 1_000
    for q, t in zip(queries, targets, strict=True):
        # The added profile is the same speaker from another session (SVS guide §8.1).
        assert rows[t].speaker == rows[q].speaker and rows[t].session != rows[q].session
        assert 3.0 <= rows[q].duration < 10.0


def test_expected_false_accepts_grow_with_the_pool_and_the_tier_answers_it():
    def row(threshold, false_accepts, recall):
        return {
            "threshold": threshold,
            "false_accepts": false_accepts,
            "n_nontarget": 1_000_000,
            "recall": recall,
        }

    # STRONG rows only (first of each tier); FAR 1e-4 at the pool-1,000 threshold.
    table = {
        ("pcm16k", "3-10s"): {
            pool: [row(str(5 + i), 100_000 // pool, 0.9 - 0.1 * i)]
            for i, pool in enumerate(POOL_TIERS)
        }
    }
    rows = expected_rows(table)
    assert [r["fixed_fa_per_query"] for r in rows] == ["0.10", "1.00", "5.00", "10.00"]
    assert [r["tier_fa_per_query"] for r in rows] == ["0.100", "0.100", "0.100", "0.100"]
    assert [r["tier_frr_%"] for r in rows] == ["10.00", "20.00", "30.00", "40.00"]
