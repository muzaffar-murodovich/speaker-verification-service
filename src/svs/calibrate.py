"""Operating points: the STRONG, PROBABLE and WEAK thresholds of each bucket, at a fixed FAR.

SVS guide §8.4: STRONG at the FAR where N_expected x FAR <= 0.1, PROBABLE at 10 x that FAR,
WEAK (the floor) for a review workload people will do. N is the candidate pool size, so
each bucket gets one set of thresholds per pool tier (§4.6); svs.bands picks the tier.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from svs.asnorm import as_norm, cohort_stats
from svs.embed_cache import read_cache
from svs.metrics import error_rates, read_bucket_scores
from svs.trial_list import SOURCES
from svs.trials import BUCKETS, nontarget_pairs
from svs.voxceleb import read_genders, utterances

MAX_AUTO_POOL = 50_000  # CFG_SV_MAX_AUTO_POOL: the largest pool the service assigns from (§4.6).
# Handbook D14: pools from 1 k to 100 k. Above MAX_AUTO_POOL the bands are still given, but
# no automatic decision (svs.retrieve).
POOL_TIERS = (1_000, 10_000, MAX_AUTO_POOL, 100_000)
# Per bucket: about 10 false accepts at STRONG for pool 50,000, so the threshold rests on more
# than one trial; pool 100,000 STRONG rests on about 5 (LOG, D14).
N_NONTARGET = 5_000_000
# Pairs scored at once: two 250,000 x 192 float64 gathers are 0.8 GB (the home PC has 7 GB).
BATCH = 250_000


def band_fars(pool: int) -> dict[str, float]:
    """FAR of each band for a pool of this size, floor first. Pool 50,000: 4e-5, 2e-5, 2e-6."""
    strong = 0.1 / pool
    probable = 10 * strong
    # The floor adds one impostor candidate per query for a person to review (LOG choice 21).
    return {"WEAK": probable + 1 / pool, "PROBABLE": probable, "STRONG": strong}


def threshold_at_far(target: NDArray, nontarget: NDArray, far: float) -> tuple[float, int, float]:
    """Lowest threshold whose false-accept rate is <= far; its false accepts and FRR.

    The threshold is chosen from the non-targets alone; the targets only give the FRR.
    Below one allowed false accept the rate cannot be measured, so it raises.
    """
    if far * len(nontarget) < 1:
        raise ValueError(
            f"FAR {far:g} needs >= {1 / far:.0f} non-target trials, got {len(nontarget)}"
        )
    thresholds, miss, false_alarm = error_rates(target, nontarget)
    i = np.argmax(false_alarm <= far)  # false_alarm falls as the threshold rises
    return float(thresholds[i]), round(false_alarm[i] * len(nontarget)), float(miss[i])


def precision(recall: float, far: float, pool: int) -> float:
    """Share of accepts that are right, for one query against a pool holding its speaker.

    The true profile passes with probability recall; each of the other pool members with
    probability far. If the speaker may be missing from the pool, precision is lower.
    """
    return recall / (recall + pool * far)


def pair_scores(
    embeddings: NDArray, mu: NDArray, sd: NDArray, i: NDArray, j: NDArray, batch: int = BATCH
) -> NDArray[np.float64]:
    """AS-Norm score of each pair (i[k], j[k]) of rows, BATCH pairs at a time."""
    parts = []
    for k in range(0, len(i), batch):
        e, f = i[k : k + batch], j[k : k + batch]
        raw = np.einsum("ij,ij->i", embeddings[e], embeddings[f])
        parts.append(as_norm(raw, mu[e], sd[e], mu[f], sd[f]))
    return np.concatenate(parts) if parts else np.empty(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_cache")
    parser.add_argument("video_cache")
    parser.add_argument("meta_csv")
    parser.add_argument("cohort_npz")
    parser.add_argument("asnorm_scores", help="AS-Norm score file: its target trials give the FRR")
    parser.add_argument("out_tsv")
    parser.add_argument("--condition", default="pcm16k", help="written in each row")
    args = parser.parse_args()
    genders = read_genders(args.meta_csv)
    cohort = np.load(args.cohort_npz)["embeddings"]
    targets = {bucket: t for bucket, (t, _) in read_bucket_scores(args.asnorm_scores).items()}
    rows = []
    for source, cache in (("clip", args.clip_cache), ("video", args.video_cache)):
        utt_ids, seconds, embeddings = read_cache(cache)
        utts = utterances(utt_ids, seconds, genders)
        index = {u: k for k, u in enumerate(utt_ids.tolist())}
        x = embeddings[[index[u.utt_id] for u in utts]].astype(np.float64)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        mu, sd = cohort_stats(x, cohort)
        speakers = np.array([u.speaker for u in utts])
        for bucket in (b for b in BUCKETS if SOURCES[b] == source):
            i, j = nontarget_pairs(utts, N_NONTARGET, bucket)
            nontarget = pair_scores(x, mu, sd, i, j)
            for pool, band, far in (
                (p, b, f) for p in POOL_TIERS for b, f in band_fars(p).items()
            ):
                threshold, false_accepts, frr = threshold_at_far(targets[bucket], nontarget, far)
                # Trustworthy if the false accepts come from many speaker pairs, not one.
                accepted = nontarget >= threshold
                sides = speakers[i[accepted]], speakers[j[accepted]]
                pairs = {tuple(sorted(p)) for p in zip(*sides, strict=True)}
                measured_far = false_accepts / len(nontarget)
                rows.append(
                    {
                        "condition": args.condition,
                        "bucket": bucket,
                        "pool": pool,
                        "band": band,
                        "far": f"{far:g}",
                        "threshold": f"{threshold:.4f}",
                        "false_accepts": false_accepts,
                        "speaker_pairs": len(pairs),
                        "n_nontarget": len(nontarget),
                        "frr_%": f"{100 * frr:.2f}",
                        "n_target": len(targets[bucket]),
                        "recall": f"{1 - frr:.4f}",
                        "precision": f"{precision(1 - frr, measured_far, pool):.3f}",
                    }
                )
                print("\t".join(map(str, rows[-1].values())))
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_tsv, "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} operating points to {args.out_tsv}")
