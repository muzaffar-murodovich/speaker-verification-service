"""Two-stage identify: approximate recall, then an exact decision (handbook D13, SVS guide §7.5).

Stage 1 asks Qdrant for k × oversampling candidates. Its distances may come from int8
vectors, so they only choose candidates. Stage 2 fetches the candidates' float32
vectors, computes the cosine again, keeps the best k, applies AS-Norm and the band. Only
the stage 2 score ever meets a threshold (§7.4): no Qdrant score reaches a Candidate.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qdrant_client import models

from svs.asnorm import as_norm, cohort_stats
from svs.bands import band
from svs.store import ProfileStore

RECALL_K = 200  # handbook D13 recall@200; SVS guide §3.4 recall_stage_k
OVERSAMPLING = 3.0  # CFG_SV_OVERSAMPLING
# CFG_SV_HNSW_EF_SEARCH_UI: identify is the interactive query (SVS guide §7.4).
EF_SEARCH = 256
TOP_K = 20  # the §3.4 request example


@dataclass(frozen=True)
class Candidate:
    profile_id: str
    raw_score: float  # float32 cosine, computed in stage 2
    normalised_score: float
    band: str


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


def identify(
    store: ProfileStore,
    model_version: str,
    query: NDArray[np.float32],
    speech_seconds: float,
    condition: str,
    cohort: NDArray,
    table: dict,
    query_filter: models.Filter | None = None,
    top_k: int = TOP_K,
    k: int = RECALL_K,
    oversampling: float = OVERSAMPLING,
    ef: int = EF_SEARCH,
) -> Identified:
    """Candidates for a query embedding, each with its normalised score and band.

    Each profile's top-n cohort mean and spread are read from its payload (cohort_mu,
    cohort_sd), cached when the profile was written (SVS guide §4.3). The query's are
    computed here against the same cohort. Under 3.0 s of speech band() raises.
    """
    start = time.perf_counter()
    hits = recall_stage(store, model_version, query, query_filter, k, oversampling, ef)
    mid = time.perf_counter()

    ids = [str(hit.id) for hit in hits]
    q = np.asarray(query, np.float64)
    q = q / np.linalg.norm(q)
    raw = store.vectors(ids).astype(np.float64) @ q
    best = np.argsort(-raw, kind="stable")[:k]
    mu_q, sd_q = cohort_stats(q[None, :], cohort)
    mu_p = np.array([hits[i].payload["cohort_mu"] for i in best])
    sd_p = np.array([hits[i].payload["cohort_sd"] for i in best])
    norm = as_norm(raw[best], mu_q[0], sd_q[0], mu_p, sd_p)
    candidates = []
    for j in np.argsort(-norm, kind="stable")[:top_k]:
        i = best[j]
        name, _ = band(float(norm[j]), speech_seconds, condition, table)
        candidates.append(
            Candidate(hits[i].payload["profile_id"], float(raw[i]), float(norm[j]), name)
        )
    end = time.perf_counter()
    return Identified(candidates, [ids[i] for i in best], (mid - start) * 1e3, (end - mid) * 1e3)


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
