"""Identify: two-stage retrieval (handbook D13) under the 1:N rules (handbook D14).

Stage 1 asks Qdrant for k × oversampling candidates. Its distances may come from int8
vectors, so they only choose candidates. Stage 2 fetches the candidates' float32
vectors, computes the cosine again, keeps the best k, applies AS-Norm and the band. Only
the stage 2 score ever meets a threshold (§7.4): no Qdrant score reaches a Candidate.

SVS guide §4.6 adds three rules, all used together: every call carries a filter; the
bands come from the thresholds of the pool size the filter leaves; and a pool above
CFG_SV_MAX_AUTO_POOL gets REVIEW_REQUIRED whatever the scores.
"""

import math
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from qdrant_client import models

from svs.asnorm import as_norm, cohort_stats
from svs.bands import NO_MATCH, band, tier_rows
from svs.calibrate import MAX_AUTO_POOL
from svs.store import ProfileStore

RECALL_K = 200  # handbook D13 recall@200; SVS guide §3.4 recall_stage_k
OVERSAMPLING = 3.0  # CFG_SV_OVERSAMPLING
# CFG_SV_HNSW_EF_SEARCH_UI: identify is the interactive query (SVS guide §7.4).
EF_SEARCH = 256
TOP_K = 20  # the §3.4 request example
MATCH, REVIEW_REQUIRED = "MATCH", "REVIEW_REQUIRED"
# Bands that assign an identity (SVS guide §4.5); WEAK is only shown to a person.
ASSIGNING = ("STRONG", "PROBABLE")


class FilterRequired(ValueError):
    code = "E_SV_FILTER_REQUIRED"


@dataclass(frozen=True)
class IdentifyFilter:
    """The filter every identify call carries: the §3.4 fields, source and dates required.

    A source and a date window are what cut the pool by orders of magnitude (§4.6), so
    they have no default; asking for every source is allowed, but has to be said.
    """

    source: tuple[str, ...]
    last_seen_after: datetime
    status: tuple[str, ...] = ("CONFIRMED", "PROVISIONAL")
    min_evidence_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.source or not self.status:
            raise FilterRequired("the filter needs at least one source and one status")
        if self.last_seen_after.tzinfo is None:
            raise FilterRequired("last_seen_after needs a time zone")

    def to_qdrant(self) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(key="source", match=models.MatchAny(any=list(self.source))),
                models.FieldCondition(key="status", match=models.MatchAny(any=list(self.status))),
                models.FieldCondition(
                    key="last_seen", range=models.DatetimeRange(gte=self.last_seen_after)
                ),
                models.FieldCondition(
                    key="evidence_seconds", range=models.Range(gte=self.min_evidence_seconds)
                ),
            ]
        )


@dataclass(frozen=True)
class Candidate:
    profile_id: str
    raw_score: float  # float32 cosine, computed in stage 2
    normalised_score: float
    band: str
    # Gap to the next candidate's normalised score; None if nothing came after it (§3.4).
    margin_to_next: float | None
    # The profile's supporting evidence, for the person who checks the result (§4.7).
    evidence_seconds: float
    last_seen: str


@dataclass(frozen=True)
class Identified:
    candidates: list[Candidate]  # best normalised score first, at most top_k
    rescored: list[str]  # point ids of the k best exact cosines, best first
    recall_ms: float
    exact_ms: float


def recall_stage(
    store: ProfileStore,
    model_version: str,
    query: NDArray[np.float32],
    query_filter: models.Filter | None,
    k: int = RECALL_K,
    oversampling: float = OVERSAMPLING,
    ef: int = EF_SEARCH,
) -> list[models.ScoredPoint]:
    """k × oversampling candidates from the index; no vectors, no rescore by Qdrant."""
    params = models.SearchParams(
        hnsw_ef=ef, quantization=models.QuantizationSearchParams(rescore=False)
    )
    return store.search(model_version, query, math.ceil(k * oversampling), query_filter, params)


def two_stage(
    store: ProfileStore,
    model_version: str,
    query: NDArray[np.float32],
    speech_seconds: float,
    condition: str,
    cohort: NDArray,
    table: dict,
    pool: int,
    query_filter: models.Filter | None = None,
    top_k: int = TOP_K,
    k: int = RECALL_K,
    oversampling: float = OVERSAMPLING,
    ef: int = EF_SEARCH,
) -> Identified:
    """Candidates for a query embedding, each with its normalised score and band.

    Each profile's top-n cohort mean and spread are read from its payload (cohort_mu,
    cohort_sd), cached when the profile was written (SVS guide §4.3). The query's are
    computed here against the same cohort. The bands are those of a pool of this size.
    Under 3.0 s of speech band() raises.
    """
    start = time.perf_counter()
    hits = recall_stage(store, model_version, query, query_filter, k, oversampling, ef)
    mid = time.perf_counter()

    if not hits:  # the filter keeps no profile
        return Identified([], [], (mid - start) * 1e3, 0.0)
    ids = [str(hit.id) for hit in hits]
    q = np.asarray(query, np.float64)
    q = q / np.linalg.norm(q)
    raw = store.vectors(ids).astype(np.float64) @ q
    best = np.argsort(-raw, kind="stable")[:k]
    mu_q, sd_q = cohort_stats(q[None, :], cohort)
    mu_p = np.array([hits[i].payload["cohort_mu"] for i in best])
    sd_p = np.array([hits[i].payload["cohort_sd"] for i in best])
    norm = as_norm(raw[best], mu_q[0], sd_q[0], mu_p, sd_p)
    order = np.argsort(-norm, kind="stable")
    candidates = []
    for rank, j in enumerate(order[:top_k]):
        i = best[j]
        name, _ = band(float(norm[j]), speech_seconds, condition, table, pool)
        margin = float(norm[j] - norm[order[rank + 1]]) if rank + 1 < len(order) else None
        payload = hits[i].payload
        candidates.append(
            Candidate(
                payload["profile_id"],
                float(raw[i]),
                float(norm[j]),
                name,
                margin,
                payload["evidence_seconds"],
                payload["last_seen"],
            )
        )
    end = time.perf_counter()
    return Identified(candidates, [ids[i] for i in best], (mid - start) * 1e3, (end - mid) * 1e3)


@dataclass(frozen=True)
class Identification:
    candidates: list[Candidate]
    decision: str  # MATCH, REVIEW_REQUIRED or NO_MATCH
    decision_profile_id: str | None  # set only for MATCH
    candidate_pool: int  # profiles the filter keeps
    pool_tier: int  # the pool size whose thresholds gave the bands
    rescored: int  # best exact cosines kept for AS-Norm, at most RECALL_K
    count_ms: float
    recall_ms: float
    exact_ms: float


def decide(candidates: list[Candidate], pool: int) -> str:
    """MATCH, REVIEW_REQUIRED or NO_MATCH for the ranked candidates of one query.

    Above CFG_SV_MAX_AUTO_POOL nothing is assigned automatically (SVS guide §4.6). A top
    candidate below PROBABLE is no match: WEAK is shown to a person and the speaker
    becomes a new profile (§4.5, §6.1). Two candidates at PROBABLE or above are two
    identities the service could assign, likely one duplicated profile (§3.4, §5.6): a
    person decides.
    """
    if pool > MAX_AUTO_POOL:
        return REVIEW_REQUIRED
    if not candidates or candidates[0].band not in ASSIGNING:
        return NO_MATCH
    if len(candidates) > 1 and candidates[1].band in ASSIGNING:
        return REVIEW_REQUIRED
    return MATCH


def identify(
    store: ProfileStore,
    model_version: str,
    query: NDArray[np.float32],
    speech_seconds: float,
    condition: str,
    cohort: NDArray,
    table: dict,
    query_filter: IdentifyFilter,
    top_k: int = TOP_K,
) -> Identification:
    """1:N identify: count the filtered pool, rank in two stages, then decide.

    The pool is counted before the search so the bands use the thresholds of that pool
    size (§4.6), and the response names the tier used.
    """
    if not isinstance(query_filter, IdentifyFilter):
        raise FilterRequired("identify needs an IdentifyFilter; an unfiltered search is refused")
    start = time.perf_counter()
    qdrant_filter = query_filter.to_qdrant()
    pool = store.count(model_version, qdrant_filter)
    count_ms = (time.perf_counter() - start) * 1e3
    tier = int(tier_rows(speech_seconds, condition, table, pool)[0]["pool"])
    ranked = two_stage(
        store,
        model_version,
        query,
        speech_seconds,
        condition,
        cohort,
        table,
        pool,
        qdrant_filter,
        top_k,
    )
    decision = decide(ranked.candidates, pool)
    profile_id = ranked.candidates[0].profile_id if decision == MATCH else None
    return Identification(
        ranked.candidates,
        decision,
        profile_id,
        pool,
        tier,
        len(ranked.rescored),
        count_ms,
        ranked.recall_ms,
        ranked.exact_ms,
    )


def brute_force(
    vectors: NDArray[np.float32], mask: NDArray[np.bool_], query: NDArray, k: int = RECALL_K
) -> NDArray[np.int64]:
    """Row indices of the k highest exact cosines among the rows the filter keeps.

    The ground truth for recall@k (SVS guide Appendix D). Rows are L2-normalised.
    """
    rows = np.flatnonzero(mask)
    scores = vectors[rows] @ np.asarray(query, np.float32)
    if len(rows) > k:
        top = np.argpartition(-scores, k)[:k]
    else:
        top = np.arange(len(rows))
    return rows[top[np.argsort(-scores[top], kind="stable")]]


def recall_at_k(found: list, truth: list) -> float:
    """Share of the exact top k that the two-stage search also returned.

    With fewer than k profiles in the filtered pool, truth is the whole pool.
    """
    if not truth:
        raise ValueError("empty ground truth")
    return len(set(found) & set(truth)) / len(truth)
