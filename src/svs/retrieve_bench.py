"""Recall@200 and stage latency of two-stage identify against brute force (handbook D13).

The pool is every usable VoxCeleb1 clip embedding except the queries: 151,174 vectors
(M3 needs ≥ 100,000). VoxCeleb has no status, source or dates, so the payload is drawn
at random with a fixed seed. The filters then keep a known share of the pool: all, 10 %,
1 % and 0.1 %, the selectivity bands of SVS guide Appendix D.

Two collections hold the same vectors: float32 only, and int8 for the search graph with
the float32 originals kept. Each query runs through svs.retrieve.two_stage and through
brute force with the same filter; the exact stage after brute force decides the same
way, so the two decisions can be compared (Appendix D's last criterion). Bands use the
pool-50,000 thresholds for every filter, so only retrieval differs between filters.
"""

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from svs.asnorm import as_norm, cohort_stats
from svs.bands import NO_MATCH, read_table
from svs.bands import band as band_of
from svs.calibrate import MAX_AUTO_POOL
from svs.embed import MODEL_VERSION
from svs.retrieve import RECALL_K, brute_force, recall_at_k, two_stage
from svs.store import BENCH_PREFIX, ProfileStore, point_id

SOURCES = [f"cc_{i:02d}" for i in range(10)]
END = datetime(2026, 10, 28, tzinfo=UTC)
DAYS = 365
N_QUERIES = 1000
WARM_UP = 20


def load_pool(emb_dir: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """utt_ids, L2-normalised embeddings and net seconds of every clip with an embedding."""
    ids, vecs, secs = [], [], []
    for path in sorted(Path(emb_dir).glob("*.npz")):
        d = np.load(path)
        keep = ~np.isnan(d["embeddings"]).any(axis=1)
        ids += [str(u) for u in d["utt_ids"][keep]]
        vecs.append(d["embeddings"][keep])
        secs.append(d["net_seconds"][keep])
    vecs = np.concatenate(vecs).astype(np.float32)
    return ids, vecs / np.linalg.norm(vecs, axis=1, keepdims=True), np.concatenate(secs)


def random_payload(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Filter fields for n profiles: uniform source and day, 80 % CONFIRMED, log-uniform
    evidence from 20 s to 1 h. Synthetic: only the shares a filter keeps matter here."""
    rng = np.random.default_rng(seed)
    return {
        "source": rng.choice(SOURCES, n),
        "status": np.where(rng.random(n) < 0.8, "CONFIRMED", "PROVISIONAL"),
        "age_days": rng.uniform(0, DAYS, n),
        "evidence_seconds": np.exp(rng.uniform(np.log(20), np.log(3600), n)),
    }


def filters() -> dict[str, tuple[models.Filter | None, callable]]:
    """Name -> (Qdrant filter, the same filter as a numpy mask over the payload)."""

    def source_and_days(days: float):
        since = END - timedelta(days=days)
        qdrant = models.Filter(
            must=[
                models.FieldCondition(key="source", match=models.MatchValue(value="cc_00")),
                models.FieldCondition(key="last_seen", range=models.DatetimeRange(gte=since)),
            ]
        )
        return qdrant, lambda p: (p["source"] == "cc_00") & (p["age_days"] <= days)

    only_source = models.Filter(
        must=[models.FieldCondition(key="source", match=models.MatchValue(value="cc_00"))]
    )
    return {
        "all": (None, lambda p: np.ones(len(p["source"]), bool)),
        "10%": (only_source, lambda p: p["source"] == "cc_00"),
        "1%": source_and_days(DAYS / 10),
        "0.1%": source_and_days(DAYS / 100),
    }


def build(
    client: QdrantClient, name: str, quantized: bool, ids, vecs, payload, mu, sd
) -> ProfileStore:
    """(Re)create one benchmark collection and wait until Qdrant has indexed it."""
    store = ProfileStore(client, MODEL_VERSION, f"{BENCH_PREFIX}{name}_")
    if client.collection_exists(store.collection):
        client.delete_collection(store.collection)
    store.create(quantized=quantized)
    payloads = [
        {
            "source": str(payload["source"][i]),
            "status": str(payload["status"][i]),
            "last_seen": (END - timedelta(days=float(payload["age_days"][i]))).isoformat(),
            "evidence_seconds": float(payload["evidence_seconds"][i]),
            "cohort_mu": float(mu[i]),
            "cohort_sd": float(sd[i]),
        }
        for i in range(len(ids))
    ]
    start = time.perf_counter()
    store.upsert_many(MODEL_VERSION, ids, vecs, payloads)
    while client.get_collection(store.collection).status != models.CollectionStatus.GREEN:
        time.sleep(1)
    info = client.get_collection(store.collection)
    print(
        f"{store.collection}: {info.points_count} points, {info.indexed_vectors_count} "
        f"indexed, {time.perf_counter() - start:.0f} s to upload and index"
    )
    return store


def exact_decision(raw, mu_q, sd_q, mu, sd, seconds, condition, table):
    """Top profile row and band from exact cosines, as the exact stage decides."""
    norm = as_norm(raw, mu_q, sd_q, mu, sd)
    j = int(np.argmax(norm))
    return j, band_of(float(norm[j]), seconds, condition, table, MAX_AUTO_POOL)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("emb_dir")
    parser.add_argument("cohort")
    parser.add_argument("--url", default="http://localhost:6333")
    parser.add_argument("--queries", type=int, default=N_QUERIES)
    args = parser.parse_args()

    ids, vecs, secs = load_pool(args.emb_dir)
    rng = np.random.default_rng(0)
    is_query = np.zeros(len(ids), bool)
    is_query[rng.choice(len(ids), args.queries, replace=False)] = True
    q_vecs, q_secs = vecs[is_query], secs[is_query]
    ids = [u for u, q in zip(ids, is_query, strict=True) if not q]
    vecs = vecs[~is_query]
    print(f"pool {len(ids):,} clips, {args.queries:,} queries held out")

    cohort = np.load(args.cohort)["embeddings"]
    mu, sd = cohort_stats(vecs, cohort)
    payload = random_payload(len(ids))
    table = read_table()
    pids = [point_id(u) for u in ids]

    client = QdrantClient(url=args.url, timeout=60)
    stores = {
        "float32": build(client, "f32", False, ids, vecs, payload, mu, sd),
        "int8": build(client, "int8", True, ids, vecs, payload, mu, sd),
    }
    q_mu, q_sd = cohort_stats(q_vecs, cohort)

    for fname, (qfilter, mask_of) in filters().items():
        mask = mask_of(payload)
        truths, decisions = [], []
        for q in range(args.queries):
            rows = brute_force(vecs, mask, q_vecs[q], RECALL_K)
            truths.append([pids[r] for r in rows])
            j, b = exact_decision(
                vecs[rows] @ q_vecs[q],
                q_mu[q],
                q_sd[q],
                mu[rows],
                sd[rows],
                float(q_secs[q]),
                "pcm16k",
                table,
            )
            decisions.append((ids[rows[j]], b))
        for sname, store in stores.items():
            for ef in (128, 256):

                def run(q, store=store, ef=ef, qfilter=qfilter):
                    return two_stage(
                        store,
                        MODEL_VERSION,
                        q_vecs[q],
                        float(q_secs[q]),
                        "pcm16k",
                        cohort,
                        table,
                        MAX_AUTO_POOL,
                        qfilter,
                        ef=ef,
                    )

                # Warm-up: the first queries after a switch are slower and are not timed.
                for q in range(min(WARM_UP, args.queries)):
                    run(q)
                recalls, rec_ms, ex_ms, same = [], [], [], 0
                for q in range(args.queries):
                    out = run(q)
                    recalls.append(recall_at_k(out.rescored, truths[q]))
                    rec_ms.append(out.recall_ms)
                    ex_ms.append(out.exact_ms)
                    top = out.candidates[0]
                    same += (top.profile_id, top.band) == decisions[q]
                r = np.array(recalls)
                rec_p = np.percentile(rec_ms, [50, 95])
                ex_p = np.percentile(ex_ms, [50, 95])
                print(
                    f"filter {fname:>4} pool {mask.sum():>7,}  {sname:>7} ef {ef}  "
                    f"recall@{RECALL_K} mean {r.mean():.4f} min {r.min():.4f} "
                    f"below 0.99 {int((r < 0.99).sum())}  "
                    f"same decision {same}/{args.queries}  "
                    f"recall ms p50 {rec_p[0]:.1f} p95 {rec_p[1]:.1f}  "
                    f"exact ms p50 {ex_p[0]:.1f} p95 {ex_p[1]:.1f}"
                )
        in_band = sum(b != NO_MATCH for _, b in decisions)
        print(
            f"filter {fname:>4}: brute-force top candidate WEAK or higher {in_band}/{args.queries}"
        )


if __name__ == "__main__":
    main()
