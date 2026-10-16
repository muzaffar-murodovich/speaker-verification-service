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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores_tsv")
    args = parser.parse_args()
    with open(args.scores_tsv, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    print("bucket\tEER_%\tthreshold\tn_target\tn_nontarget")
    for bucket in BUCKETS:
        scores = {"1": [], "0": []}
        for r in rows:
            if r["bucket"] == bucket:
                scores[r["target"]].append(float(r["score"]))
        rate, threshold = eer(np.array(scores["1"]), np.array(scores["0"]))
        n_target, n_nontarget = len(scores["1"]), len(scores["0"])
        print(f"{bucket}\t{100 * rate:.3f}\t{threshold:.4f}\t{n_target}\t{n_nontarget}")
