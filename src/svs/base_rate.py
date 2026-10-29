"""False accepts and decisions as the candidate pool grows (handbook D14, SVS guide §4.6).

One identify query is compared with every profile in the pool, so with a fixed threshold
it gets about N x FAR false accepts. Two tables:
- expected: from the operating-point table alone, per condition and bucket: false accepts
  per query at N for the pool-1,000 STRONG threshold, and the STRONG threshold and FRR of
  each pool tier (the threshold change that answers it, M3);
- measured: 1:N searches on clean VoxCeleb1 clips with the fixed pool-1,000 thresholds
  and with the pool-size rule, counting false accepts and the decisions of
  svs.retrieve.decide.

Speakers are split at random (seed 0): a fifth give the queries, the rest give the pool,
so the pool never holds the query's speaker unless it is added on purpose. Pools of
1,000 to 100,000 clips are prefixes of one shuffled order. A query is a 3-10 s clip and
every pool clip has >= 3 s, so each pair is in the query's calibration bucket (the
shorter clip decides it). Each query is searched twice: unknown (the pool alone) and
known (one clip of its speaker from another session added, standing in for its profile).
Clips stand in for profiles as in D13 (LOG choice 35): about 120 per pool speaker.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from svs.asnorm import as_norm, cohort_stats
from svs.bands import NO_MATCH, band, pool_tier, read_table
from svs.calibrate import MAX_AUTO_POOL, POOL_TIERS
from svs.duplicates import merge_sessions, read_video_pairs
from svs.embed_cache import read_cache
from svs.retrieve import ASSIGNING, MATCH, REVIEW_REQUIRED, Candidate, decide
from svs.trials import BUCKETS
from svs.voxceleb import read_genders, utterances

CONDITION = "pcm16k"
BUCKET = "3-10s"
N_QUERIES = 2_000
QUERY_SHARE = 0.2
# The fixed policy: the pool-1,000 thresholds whatever the pool, as if 1:1-style
# thresholds were reused for a growing 1:N pool (SVS guide pitfall 4).
POLICIES = {"fixed-1000": lambda pool: POOL_TIERS[0], "pool-rule": lambda pool: pool}
BATCH = 100  # queries scored at once: 100 x 100,000 float64 is 80 MB


def far_of(row: dict) -> float:
    """The FAR measured at a row's threshold, not the target FAR it was chosen for."""
    return int(row["false_accepts"]) / int(row["n_nontarget"])


def expected_rows(table: dict) -> list[dict]:
    """Per condition and bucket: N x FAR at the pool-1,000 STRONG threshold, and each
    tier's own STRONG threshold with its FRR."""
    out = []
    for (condition, bucket), by_tier in sorted(table.items()):
        fixed = by_tier[POOL_TIERS[0]][0]  # STRONG row of the smallest tier
        for pool in POOL_TIERS:
            strong = by_tier[pool][0]
            out.append(
                {
                    "condition": condition,
                    "bucket": bucket,
                    "pool": pool,
                    "fixed_threshold": fixed["threshold"],
                    "fixed_fa_per_query": f"{pool * far_of(fixed):.2f}",
                    "fixed_frr_%": f"{100 * (1 - float(fixed['recall'])):.2f}",
                    "tier_threshold": strong["threshold"],
                    "tier_fa_per_query": f"{pool * far_of(strong):.3f}",
                    "tier_frr_%": f"{100 * (1 - float(strong['recall'])):.2f}",
                    "tier_false_accepts": strong["false_accepts"],
                }
            )
    out.sort(key=lambda r: (r["condition"], BUCKETS.index(r["bucket"]), r["pool"]))
    return out


def pick(utts, rng):
    """Query rows, their target rows (same speaker, another session) and the pool rows."""
    speakers = sorted({u.speaker for u in utts})
    n_query = round(QUERY_SHARE * len(speakers))
    query_speakers = set(rng.choice(speakers, n_query, replace=False).tolist())
    by_speaker: dict[str, list[int]] = {}
    for i, u in enumerate(utts):
        by_speaker.setdefault(u.speaker, []).append(i)
    candidates = [
        i
        for i, u in enumerate(utts)
        if u.speaker in query_speakers
        and u.duration < 10.0
        and any(utts[j].session != u.session for j in by_speaker[u.speaker])
    ]
    queries = rng.choice(candidates, N_QUERIES, replace=False)
    targets = np.array(
        [
            rng.choice(
                [j for j in by_speaker[utts[q].speaker] if utts[j].session != utts[q].session]
            )
            for q in queries
        ]
    )
    pool = np.array([i for i, u in enumerate(utts) if u.speaker not in query_speakers])
    pool = rng.permutation(pool)[: max(POOL_TIERS)]
    return queries, targets, pool


def top_candidates(scores: list[float], seconds: float, table: dict, tier: int) -> list[Candidate]:
    """The two best scores as candidates with their bands; only band and score matter."""
    return [
        Candidate("", 0.0, s, band(s, seconds, CONDITION, table, tier)[0], None, 0.0, "")
        for s in scores
    ]


def measure(utts, x, mu, sd, queries, targets, pool, table) -> list[dict]:
    by_tier = table[CONDITION, BUCKET]
    tier = {(p, n): pool_tier(f(n), by_tier) for p, f in POLICIES.items() for n in POOL_TIERS}
    genders = np.array([u.gender for u in utts])
    counts = {key: Counter() for key in tier}
    speakers = np.array([u.speaker for u in utts])
    # Speaker pairs behind the STRONG false accepts: many pairs, or one look-alike pair?
    pairs = {key: set() for key in tier}
    for start in range(0, len(queries), BATCH):
        qs, ts = queries[start : start + BATCH], targets[start : start + BATCH]
        impostor = as_norm(x[qs] @ x[pool].T, mu[qs, None], sd[qs, None], mu[pool], sd[pool])
        target = as_norm(np.einsum("ij,ij->i", x[qs], x[ts]), mu[qs], sd[qs], mu[ts], sd[ts])
        for k, q in enumerate(qs):
            seconds, t = utts[q].duration, float(target[k])
            same_gender = np.cumsum(genders[pool] == genders[q])
            for (policy, n), used in tier.items():
                c = counts[policy, n]
                imp = impostor[k, :n]
                strong, probable = (float(r["threshold"]) for r in by_tier[used][:2])
                c["fa_strong"] += int((imp >= strong).sum())
                pairs[policy, n].update(
                    (utts[q].speaker, p) for p in speakers[pool[:n][imp >= strong]]
                )
                c["fa_probable"] += int((imp >= probable).sum())
                c["same_gender"] += int(same_gender[n - 1])
                c["target_strong"] += band(t, seconds, CONDITION, table, used)[0] == "STRONG"
                # decide sees at most MAX_AUTO_POOL, so the 100,000 row shows what the
                # bands alone would assign; the service returns REVIEW_REQUIRED there.
                seen = min(n, MAX_AUTO_POOL)
                top2 = np.sort(np.partition(imp, -2)[-2:])[::-1].tolist()
                cands = top_candidates(top2, seconds, table, used)
                decision = decide(cands, seen)
                if decision == MATCH:  # by the band that assigned it
                    decision += "_" + cands[0].band
                c["unknown_" + decision] += 1
                known = sorted([*top2, t], reverse=True)[:2]
                cands = top_candidates(known, seconds, table, used)
                decision = decide(cands, seen)
                if decision == MATCH:
                    decision += ("_right_" if known[0] == t else "_wrong_") + cands[0].band
                c["known_" + decision] += 1
    rows = []
    for (policy, n), c in counts.items():
        strong_row = by_tier[tier[policy, n]][0]
        q = len(queries)
        rows.append(
            {
                "policy": policy,
                "pool": n,
                "tier": tier[policy, n],
                "strong_threshold": strong_row["threshold"],
                "fa_strong_per_query": f"{c['fa_strong'] / q:.3f}",
                "expected_n_x_far": f"{n * far_of(strong_row):.3f}",
                "expected_same_gender": f"{c['same_gender'] / q * far_of(strong_row):.3f}",
                "fa_strong_speaker_pairs": len(pairs[policy, n]),
                "fa_probable_per_query": f"{c['fa_probable'] / q:.3f}",
                "frr_strong_%": f"{100 * (1 - c['target_strong'] / q):.2f}",
                **{
                    f"known_{d}_%": f"{100 * c['known_' + d] / q:.2f}"
                    for d in (
                        *(f"MATCH_{r}_{b}" for r in ("right", "wrong") for b in ASSIGNING),
                        REVIEW_REQUIRED,
                        NO_MATCH,
                    )
                },
                **{
                    f"unknown_{d}_%": f"{100 * c['unknown_' + d] / q:.2f}"
                    for d in (*(f"MATCH_{b}" for b in ASSIGNING), REVIEW_REQUIRED, NO_MATCH)
                },
                "n_queries": q,
            }
        )
    return rows


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clip_cache")
    parser.add_argument("meta_csv")
    parser.add_argument("shared_videos")
    parser.add_argument("cohort_npz")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    table = read_table()
    out = Path(args.out_dir)
    write(out / "expected.tsv", expected_rows(table))

    sessions = merge_sessions(read_video_pairs(args.shared_videos))
    utt_ids, seconds, embeddings = read_cache(args.clip_cache)
    utts = utterances(utt_ids, seconds, read_genders(args.meta_csv), sessions)
    index = {u: k for k, u in enumerate(utt_ids.tolist())}
    x = embeddings[[index[u.utt_id] for u in utts]].astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    queries, targets, pool = pick(utts, np.random.default_rng(0))
    print(f"{len(utts):,} clips; {len(queries):,} queries; pool up to {len(pool):,} clips")
    mu, sd = cohort_stats(x, np.load(args.cohort_npz)["embeddings"])
    rows = measure(utts, x, mu, sd, queries, targets, pool, table)
    for row in rows:
        print("\t".join(f"{k}={v}" for k, v in row.items()))
    write(out / f"voxceleb1-{CONDITION}.tsv", rows)


if __name__ == "__main__":
    main()
