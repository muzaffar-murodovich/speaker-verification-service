"""Two-stage identify: the decision comes from float32 vectors, never from stage 1."""

import numpy as np
import pytest
from qdrant_client import QdrantClient

from svs.asnorm import as_norm, cohort_stats
from svs.embed import EMBEDDING_DIM
from svs.retrieve import brute_force, identify, recall_at_k, recall_stage
from svs.store import ProfileStore

N = 50
THRESHOLDS = {"STRONG": 5.0, "PROBABLE": 3.0, "WEAK": 1.0}
TABLE = {("pcm16k", "10-30s"): [{"band": b, "threshold": str(t)} for b, t in THRESHOLDS.items()]}


def units(n: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


VECTORS = units(N, 0)
COHORT = units(400, 1)
MU, SD = cohort_stats(VECTORS, COHORT)


def payload(i: int) -> dict:
    return {
        "status": "CONFIRMED",
        "source": "cc_north",
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
    out = identify(make(WrongScores), "v1", q, 20.0, "pcm16k", COHORT, TABLE, k=10)
    exact = VECTORS.astype(np.float64) @ q.astype(np.float64)
    assert out.candidates[0].profile_id == "spk_7"
    assert out.candidates[0].raw_score == pytest.approx(exact[7], abs=1e-6)
    # The k best exact cosines among the 30 stage 1 candidates, best first.
    assert len(out.rescored) == 10


def test_the_normalised_score_is_as_norm_of_the_exact_cosine():
    q = query_near(7)
    out = identify(make(), "v1", q, 20.0, "pcm16k", COHORT, TABLE, k=10)
    mu_q, sd_q = cohort_stats(q[None, :], COHORT)
    top = out.candidates[0]
    want = as_norm(np.array([top.raw_score]), mu_q[0], sd_q[0], MU[7], SD[7])[0]
    assert top.normalised_score == pytest.approx(want, rel=1e-6)
    scores = [c.normalised_score for c in out.candidates]
    assert scores == sorted(scores, reverse=True)
    assert top.band == next(b for b, t in THRESHOLDS.items() if want >= t)


def test_under_three_seconds_is_refused():
    with pytest.raises(ValueError, match="3.0"):
        identify(make(), "v1", query_near(7), 2.9, "pcm16k", COHORT, TABLE, k=10)


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
