"""Identify: the decision comes from float32 vectors, never from stage 1, and follows
the 1:N rules of SVS guide §4.6."""

from datetime import UTC, datetime

import numpy as np
import pytest
from qdrant_client import QdrantClient

from svs.asnorm import as_norm, cohort_stats
from svs.embed import EMBEDDING_DIM
from svs.retrieve import (
    MATCH,
    REVIEW_REQUIRED,
    Candidate,
    FilterRequired,
    IdentifyFilter,
    brute_force,
    decide,
    identify,
    recall_at_k,
    recall_stage,
    two_stage,
)
from svs.store import ProfileStore

N = 50
THRESHOLDS = {"STRONG": 5.0, "PROBABLE": 3.0, "WEAK": 1.0}
# A pool of up to 30 profiles has lower thresholds than one of up to 50,000.
SMALL_POOL = {"STRONG": 1.0, "PROBABLE": 0.5, "WEAK": 0.0}


def rows(thresholds: dict, pool: int) -> list[dict]:
    return [{"band": b, "threshold": str(t), "pool": str(pool)} for b, t in thresholds.items()]


TABLE = {("pcm16k", "10-30s"): {50_000: rows(THRESHOLDS, 50_000)}}
TIERED = {("pcm16k", "10-30s"): {30: rows(SMALL_POOL, 30), 50_000: rows(THRESHOLDS, 50_000)}}
SINCE = datetime(2026, 1, 1, tzinfo=UTC)
NORTH = IdentifyFilter(source=("cc_north",), last_seen_after=SINCE)
BOTH = IdentifyFilter(source=("cc_north", "cc_south"), last_seen_after=SINCE)


def units(n: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


VECTORS = units(N, 0)
COHORT = units(400, 1)
MU, SD = cohort_stats(VECTORS, COHORT)


def payload(i: int) -> dict:
    return {
        "status": "CONFIRMED",
        "source": "cc_north" if i % 2 else "cc_south",  # spk_7 is in cc_north
        "last_seen": "2026-08-19T09:14:22Z",
        "evidence_seconds": 184.5,
        "cohort_mu": float(MU[i]),
        "cohort_sd": float(SD[i]),
    }


class WrongScores(ProfileStore):
    """Stage 1 with its order reversed and every score made up, as a bad index could."""

    def search(self, *args, **kwargs):
        hits = super().search(*args, **kwargs)[::-1]
        for i, hit in enumerate(hits):
            hit.score = 1.0 - i / len(hits)
        return hits


def make(cls=ProfileStore) -> ProfileStore:
    store = cls(QdrantClient(":memory:"), "v1")
    store.create()
    store.upsert_many("v1", [f"spk_{i}" for i in range(N)], VECTORS, [payload(i) for i in range(N)])
    return store


def query_near(i: int) -> np.ndarray:
    q = VECTORS[i] + 0.3 * units(1, 2)[0]
    return (q / np.linalg.norm(q)).astype(np.float32)


def test_stage_one_asks_for_k_times_oversampling():
    hits = recall_stage(make(), "v1", query_near(7), None, k=10, oversampling=3.0)
    assert len(hits) == 30


def test_the_decision_ignores_stage_one_scores():
    q = query_near(7)
    out = two_stage(make(WrongScores), "v1", q, 20.0, "pcm16k", COHORT, TABLE, N, k=10)
    exact = VECTORS.astype(np.float64) @ q.astype(np.float64)
    assert out.candidates[0].profile_id == "spk_7"
    assert out.candidates[0].raw_score == pytest.approx(exact[7], abs=1e-6)
    # The k best exact cosines among the 30 stage 1 candidates, best first.
    assert len(out.rescored) == 10


def test_the_normalised_score_is_as_norm_of_the_exact_cosine():
    q = query_near(7)
    out = two_stage(make(), "v1", q, 20.0, "pcm16k", COHORT, TABLE, N, k=10)
    mu_q, sd_q = cohort_stats(q[None, :], COHORT)
    top = out.candidates[0]
    want = as_norm(np.array([top.raw_score]), mu_q[0], sd_q[0], MU[7], SD[7])[0]
    assert top.normalised_score == pytest.approx(want, rel=1e-6)
    scores = [c.normalised_score for c in out.candidates]
    assert scores == sorted(scores, reverse=True)
    assert top.band == next(b for b, t in THRESHOLDS.items() if want >= t)


def test_under_three_seconds_is_refused():
    with pytest.raises(ValueError, match="3.0"):
        two_stage(make(), "v1", query_near(7), 2.9, "pcm16k", COHORT, TABLE, N, k=10)
    with pytest.raises(ValueError, match="3.0"):
        identify(make(), "v1", query_near(7), 2.9, "pcm16k", COHORT, TABLE, NORTH)


def test_margin_to_next_is_the_gap_to_the_next_normalised_score():
    out = two_stage(make(), "v1", query_near(7), 20.0, "pcm16k", COHORT, TABLE, N, top_k=5, k=10)
    scores = [c.normalised_score for c in out.candidates]
    margins = [c.margin_to_next for c in out.candidates]
    assert margins[:4] == pytest.approx(np.diff(scores[::-1])[::-1].tolist())
    # The 5th candidate's next is the 6th rescored profile, which is not returned.
    assert margins[4] is not None and margins[4] >= 0
    # With every rescored profile returned, the last one has no next.
    out = two_stage(make(), "v1", query_near(7), 20.0, "pcm16k", COHORT, TABLE, N, top_k=10, k=10)
    assert out.candidates[-1].margin_to_next is None


def test_candidates_carry_the_profile_evidence():
    out = two_stage(make(), "v1", query_near(7), 20.0, "pcm16k", COHORT, TABLE, N, k=10)
    top = out.candidates[0]
    assert (top.evidence_seconds, top.last_seen) == (184.5, "2026-08-19T09:14:22Z")


def test_identify_refuses_a_missing_or_empty_filter():
    with pytest.raises(FilterRequired):
        identify(make(), "v1", query_near(7), 20.0, "pcm16k", COHORT, TABLE, None)
    with pytest.raises(FilterRequired):
        IdentifyFilter(source=(), last_seen_after=SINCE)
    with pytest.raises(FilterRequired, match="time zone"):
        IdentifyFilter(source=("cc_north",), last_seen_after=datetime(2026, 1, 1))
    assert FilterRequired.code == "E_SV_FILTER_REQUIRED"


def test_the_bands_use_the_thresholds_of_the_filtered_pool():
    store = make()
    small = identify(store, "v1", query_near(7), 20.0, "pcm16k", COHORT, TIERED, NORTH)
    large = identify(store, "v1", query_near(7), 20.0, "pcm16k", COHORT, TIERED, BOTH)
    assert (small.candidate_pool, small.pool_tier) == (25, 30)
    assert (large.candidate_pool, large.pool_tier) == (50, 50_000)
    assert {c.profile_id for c in small.candidates} <= {f"spk_{i}" for i in range(1, N, 2)}
    # One profile, one score, two bands: the larger pool needs a higher score.
    top_small = small.candidates[0]
    top_large = next(c for c in large.candidates if c.profile_id == top_small.profile_id)
    assert top_small.normalised_score == pytest.approx(top_large.normalised_score)
    assert [b for b, t in SMALL_POOL.items() if top_small.normalised_score >= t][
        0
    ] == top_small.band
    assert [b for b, t in THRESHOLDS.items() if top_large.normalised_score >= t][
        0
    ] == top_large.band


def test_a_pool_above_max_auto_pool_is_never_assigned(monkeypatch):
    store = make()
    out = identify(store, "v1", query_near(7), 20.0, "pcm16k", COHORT, TIERED, NORTH)
    assert out.candidates[0].band == "STRONG"
    # The same search with a lower limit: candidates come back, the assignment does not.
    monkeypatch.setattr("svs.retrieve.MAX_AUTO_POOL", 24)
    out = identify(store, "v1", query_near(7), 20.0, "pcm16k", COHORT, TIERED, NORTH)
    assert out.candidates[0].band == "STRONG"
    assert (out.decision, out.decision_profile_id) == (REVIEW_REQUIRED, None)


def test_a_filter_that_keeps_nobody_searches_nothing():
    nobody = IdentifyFilter(source=("cc_west",), last_seen_after=SINCE)
    out = identify(make(), "v1", query_near(7), 20.0, "pcm16k", COHORT, TABLE, nobody)
    assert (out.candidate_pool, out.candidates, out.decision) == (0, [], "NO_MATCH")


def cand(band: str) -> Candidate:
    return Candidate("p", 0.5, 4.0, band, 0.1, 120.0, "2026-08-19T09:14:22Z")


@pytest.mark.parametrize(
    ("bands", "pool", "decision"),
    [
        (["STRONG", "WEAK"], 1_000, MATCH),
        (["PROBABLE", "NO_MATCH"], 1_000, MATCH),  # assigned, flagged by its band (§4.5)
        (["STRONG", "WEAK"], 50_001, REVIEW_REQUIRED),  # §4.6, regardless of score
        (["STRONG", "PROBABLE"], 1_000, REVIEW_REQUIRED),  # two assignable profiles
        (["STRONG", "STRONG"], 1_000, REVIEW_REQUIRED),
        (["WEAK", "WEAK"], 1_000, "NO_MATCH"),  # shown to a person, new profile (§6.1)
        ([], 1_000, "NO_MATCH"),
    ],
)
def test_decide(bands, pool, decision):
    assert decide([cand(b) for b in bands], pool) == decision


def test_brute_force_keeps_the_filter_and_the_order():
    q = query_near(7)
    mask = np.ones(N, bool)
    mask[7] = False
    top = brute_force(VECTORS, mask, q, k=5)
    scores = VECTORS @ q
    assert 7 not in top
    assert list(top) == list(np.argsort(-np.where(mask, scores, -np.inf))[:5])
    small = np.zeros(N, bool)
    small[:3] = True
    assert sorted(brute_force(VECTORS, small, q, k=5)) == [0, 1, 2]


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c", "d", "f"], ["a", "b", "c", "d", "e"]) == 0.8
    with pytest.raises(ValueError):
        recall_at_k(["a"], [])
