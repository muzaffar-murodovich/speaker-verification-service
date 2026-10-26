"""The operating-point table: one row per condition, bucket and band (handbook M2).

Joins each condition's AS-Norm EER on the trial list with its thresholds, recall and
precision from svs.calibrate, and names the model and cohort behind the scores: a new
model or cohort is a new table (SVS guide §8.4 step 6).
"""

import argparse
import csv
from pathlib import Path

from svs.calibrate import POOL_SIZE, precision
from svs.cohort import COHORT_VERSION
from svs.embed import MODEL_VERSION
from svs.metrics import eer, read_bucket_scores

CLEAN = "pcm16k"
BAND_ORDER = ("STRONG", "PROBABLE", "WEAK")  # highest threshold first


def cohort_of(condition: str) -> str:
    """The cohort file a condition is normalised with (svs.cohort_condition)."""
    return COHORT_VERSION if condition == CLEAN else f"{COHORT_VERSION}-{condition}"


def band_precision(row: dict, above: dict | None) -> float:
    """Precision of the scores in this band only: above its threshold, below the next one.

    Recall and false accepts of the band are those at its threshold minus those at the
    band above it; STRONG has none above.
    """
    if above is None:
        return float(row["precision"])
    d_recall = float(row["recall"]) - float(above["recall"])
    d_far = (int(row["false_accepts"]) - int(above["false_accepts"])) / int(row["n_nontarget"])
    return precision(d_recall, d_far)


def table_rows(condition: str, scores_tsv: str, points_tsv: str) -> list[dict]:
    """The rows of one condition, STRONG, PROBABLE, WEAK within each bucket."""
    eers = {
        bucket: (eer(t, n)[0], len(t), len(n))
        for bucket, (t, n) in read_bucket_scores(scores_tsv).items()
    }
    with open(points_tsv, newline="") as f:
        points = list(csv.DictReader(f, delimiter="\t"))
    if {p["condition"] for p in points} != {condition}:
        raise ValueError(f"{points_tsv} is not the table of {condition}")
    rows = []
    for bucket, (rate, n_target, n_nontarget) in eers.items():
        by_band = {p["band"]: p for p in points if p["bucket"] == bucket}
        above = None
        for band in BAND_ORDER:
            p = dict(by_band[band], precision=by_band[band][f"precision_pool{POOL_SIZE}"])
            rows.append(
                {
                    "model": MODEL_VERSION,
                    "cohort": cohort_of(condition),
                    "condition": condition,
                    "bucket": bucket,
                    "band": band,
                    "threshold": p["threshold"],
                    "far": p["far"],
                    "false_accepts": p["false_accepts"],
                    "speaker_pairs": p["speaker_pairs"],
                    "n_nontarget": p["n_nontarget"],
                    "n_target": p["n_target"],
                    "recall": p["recall"],
                    f"precision_pool{POOL_SIZE}": p["precision"],
                    f"band_precision_pool{POOL_SIZE}": f"{band_precision(p, above):.3f}",
                    "eer_%": f"{100 * rate:.3f}",
                    "eer_n_target": n_target,
                    "eer_n_nontarget": n_nontarget,
                }
            )
            above = p
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("conditions", nargs="+", metavar="CONDITION=ASNORM_SCORES,POINTS")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = []
    for item in args.conditions:
        condition, paths = item.split("=", 1)
        rows += table_rows(condition, *paths.split(","))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")
