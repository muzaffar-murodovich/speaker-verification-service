"""Verification metrics: equal error rate (EER) per duration bucket."""

import argparse
import csv

import numpy as np
from numpy.typing import NDArray

from svs.trials import BUCKETS


def error_rates(
    target: NDArray, nontarget: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Thresholds, and at each one the miss and false-alarm rates.

    A trial is accepted when its score >= threshold. A miss is a rejected target trial;
    a false alarm is an accepted non-target trial. The thresholds are every distinct score.
    """
    if len(target) == 0 or len(nontarget) == 0:
        raise ValueError("EER needs both target and non-target trials")
    target, nontarget = np.sort(target), np.sort(nontarget)
    thresholds = np.unique(np.concatenate([target, nontarget]))
    # searchsorted(..., "left") counts the scores strictly below each threshold.
    miss = np.searchsorted(target, thresholds, "left") / len(target)
    false_alarm = 1 - np.searchsorted(nontarget, thresholds, "left") / len(nontarget)
    return thresholds, miss, false_alarm


def eer(target: NDArray, nontarget: NDArray) -> tuple[float, float]:
    """Equal error rate and its threshold.

    Miss rate only grows and false-alarm rate only falls as the threshold rises, so they
    cross once. At the threshold where they are closest, the EER is their mean.
    """
    thresholds, miss, false_alarm = error_rates(target, nontarget)
    i = np.argmin(np.abs(miss - false_alarm))
    return float((miss[i] + false_alarm[i]) / 2), float(thresholds[i])


def read_bucket_scores(path: str) -> dict[str, tuple[NDArray, NDArray]]:
    """Target and non-target scores of each bucket of a score file, in BUCKETS order."""
    scores = {bucket: {"1": [], "0": []} for bucket in BUCKETS}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            scores[r["bucket"]][r["target"]].append(float(r["score"]))
    return {b: (np.array(s["1"]), np.array(s["0"])) for b, s in scores.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores_tsv")
    args = parser.parse_args()
    print("bucket\tEER_%\tthreshold\tn_target\tn_nontarget")
    for bucket, (target, nontarget) in read_bucket_scores(args.scores_tsv).items():
        rate, threshold = eer(target, nontarget)
        print(f"{bucket}\t{100 * rate:.3f}\t{threshold:.4f}\t{len(target)}\t{len(nontarget)}")
