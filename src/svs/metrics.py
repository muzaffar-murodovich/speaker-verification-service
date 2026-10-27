"""Verification metrics: equal error rate (EER) and minDCF per duration bucket."""

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


# minDCF prior and costs. Docs are silent on the values (handbook M2: state them; SVS
# guide §8.2 gives none). p_target 0.01 with equal costs is the default of WeSpeaker's
# wespeaker/bin/compute_metrics.py, with the same normalisation, so our numbers are
# computed the way the model's own toolkit computes them.
P_TARGET = 0.01
C_MISS = 1.0
C_FA = 1.0


def min_dcf(
    target: NDArray,
    nontarget: NDArray,
    p_target: float = P_TARGET,
    c_miss: float = C_MISS,
    c_fa: float = C_FA,
) -> tuple[float, float]:
    """Normalised minimum detection cost and its threshold.

    DCF = c_miss * p_target * miss + c_fa * (1 - p_target) * false_alarm, at every
    threshold. It is divided by the cost of the better of the two systems that need no
    scores (accept all, reject all), so 1.0 means the scores are worth nothing.
    Rejecting every trial (threshold above every score) is a candidate too.
    """
    thresholds, miss, false_alarm = error_rates(target, nontarget)
    thresholds = np.append(thresholds, np.inf)
    miss, false_alarm = np.append(miss, 1.0), np.append(false_alarm, 0.0)
    dcf = c_miss * p_target * miss + c_fa * (1 - p_target) * false_alarm
    i = np.argmin(dcf)
    return float(dcf[i] / min(c_miss * p_target, c_fa * (1 - p_target))), float(thresholds[i])


GENDERS = ("f", "m")


def read_bucket_scores(path: str, by_gender: bool = False) -> dict[str, tuple[NDArray, NDArray]]:
    """Target and non-target scores of each bucket of a score file, in BUCKETS order.

    With by_gender the groups are "<bucket> <gender>" (SVS guide §8.3: reported for
    fairness, not for tuning).
    """
    keys = [f"{b} {g}" for b in BUCKETS for g in GENDERS] if by_gender else list(BUCKETS)
    scores = {key: {"1": [], "0": []} for key in keys}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = f"{r['bucket']} {r['gender']}" if by_gender else r["bucket"]
            scores[key][r["target"]].append(float(r["score"]))
    return {key: (np.array(s["1"]), np.array(s["0"])) for key, s in scores.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores_tsv")
    args = parser.parse_args()
    for by_gender in (False, True):
        group = "bucket gender" if by_gender else "bucket"
        print(f"{group}\tEER_%\tthreshold\tminDCF_p{P_TARGET}\tn_target\tn_nontarget")
        for key, (target, nontarget) in read_bucket_scores(args.scores_tsv, by_gender).items():
            rate, threshold = eer(target, nontarget)
            cost = min_dcf(target, nontarget)[0]
            print(
                f"{key}\t{100 * rate:.3f}\t{threshold:.4f}\t{cost:.4f}"
                f"\t{len(target)}\t{len(nontarget)}"
            )
