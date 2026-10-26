"""EER per condition and bucket, and what the clean 16 kHz threshold does under each one.

Handbook D7: measure the damage per codec, sample rate and duration bucket. Every
condition scores the same trial list, so a trial missing from a condition's file was
refused there (under 3.0 s of speech after degradation).
"""

import argparse
import csv
from collections import Counter

import numpy as np

from svs.metrics import GENDERS, eer
from svs.trials import BUCKETS

CLEAN = "pcm16k"


def read_scores(path: str) -> dict[tuple[str, str], tuple[str, str, bool, float]]:
    """(enrol, test) -> (bucket, gender, is_target, score) for every scored trial."""
    with open(path, newline="") as f:
        return {
            (r["enrol"], r["test"]): (
                r["bucket"],
                r["gender"],
                r["target"] == "1",
                float(r["score"]),
            )
            for r in csv.DictReader(f, delimiter="\t")
        }


def groups(by_gender: bool) -> list[str]:
    return [f"{b} {g}" for b in BUCKETS for g in GENDERS] if by_gender else list(BUCKETS)


def group_of(row: tuple[str, str, bool, float], by_gender: bool) -> str:
    bucket, gender, _, _ = row
    return f"{bucket} {gender}" if by_gender else bucket


def split(scores, keys, by_gender: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Target and non-target scores of each group, over the given trials only."""
    out = {group: ([], []) for group in groups(by_gender)}
    for key in keys:
        _, _, is_target, score = scores[key]
        out[group_of(scores[key], by_gender)][0 if is_target else 1].append(score)
    return {group: (np.array(t), np.array(n)) for group, (t, n) in out.items()}


def damage_rows(
    files: dict[str, str], common_only: bool = False, by_gender: bool = False
) -> list[dict]:
    """One row per condition and group: EER, its threshold, counts, and the clean
    threshold's error rates under that condition.

    files maps condition -> score file and must hold CLEAN. With common_only, every
    condition is scored on the trials that no condition refused, so the rows differ
    only in the audio.
    """
    if CLEAN not in files:
        raise ValueError(f"The {CLEAN} score file is needed: its thresholds are the reference")
    scores = {condition: read_scores(path) for condition, path in files.items()}
    clean = scores[CLEAN]
    if any(key not in clean for s in scores.values() for key in s):
        raise ValueError("A condition scored a trial that the clean file does not have")
    common = set(clean).intersection(*scores.values()) if common_only else None

    clean_split = split(clean, common if common_only else clean, by_gender)
    clean_threshold = {group: eer(*pair)[1] for group, pair in clean_split.items()}
    clean_count = Counter(group_of(row, by_gender) for row in clean.values())

    rows = []
    for condition, s in scores.items():
        count = Counter(group_of(row, by_gender) for row in s.values())
        for group, (target, nontarget) in split(s, common if common_only else s, by_gender).items():
            rate, threshold = eer(target, nontarget)
            t0 = clean_threshold[group]
            rows.append(
                {
                    "condition": condition,
                    "group": group,
                    "eer": rate,
                    "threshold": threshold,
                    "n_target": len(target),
                    "n_nontarget": len(nontarget),
                    # Refused under this condition, whatever common_only drops on top.
                    "refused": clean_count[group] - count[group],
                    # Accept when score >= threshold, as in svs.metrics.error_rates.
                    "frr_at_clean": float(np.mean(target < t0)),
                    "far_at_clean": float(np.mean(nontarget >= t0)),
                }
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", metavar="CONDITION=SCORES_TSV")
    parser.add_argument("--common", action="store_true", help="only trials no condition refused")
    args = parser.parse_args()
    files = dict(item.split("=", 1) for item in args.files)
    for by_gender in (False, True):
        group = "bucket gender" if by_gender else "bucket"
        print(
            f"condition\t{group}\tEER_%\tthreshold\tn_target\tn_nontarget\trefused"
            "\tFRR_%_at_clean_threshold\tFAR_%_at_clean_threshold"
        )
        for r in damage_rows(files, args.common, by_gender):
            print(
                f"{r['condition']}\t{r['group']}\t{100 * r['eer']:.3f}\t{r['threshold']:.4f}"
                f"\t{r['n_target']}\t{r['n_nontarget']}\t{r['refused']}"
                f"\t{100 * r['frr_at_clean']:.3f}\t{100 * r['far_at_clean']:.3f}"
            )
