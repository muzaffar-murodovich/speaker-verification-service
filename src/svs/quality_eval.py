"""Does the quality score predict errors, and what does the usability gate refuse?

Handbook D10 and week 1 "if ahead": EER on the worst-quality decile against the best.
A trial is as good as its worse side, so its quality is the minimum of the two. Deciles
are taken within each duration bucket: short clips differ in quality and in EER, and a
decile across buckets would measure duration instead of quality.
"""

import argparse
import csv

import numpy as np

from svs.metrics import P_TARGET, eer, min_dcf
from svs.quality import unusable_reason
from svs.quality_cache import read_quality
from svs.trials import BUCKETS

DECILE = 0.1


def read_trials(scores_tsv: str, quality_tsv: str) -> dict[str, dict[str, np.ndarray]]:
    """Per bucket: score, target flag, trial quality and refusal reason of each trial."""
    units = read_quality(quality_tsv)
    reason = {
        key: unusable_reason(u["net_seconds"], u["quality"], u["turn_consistency"])
        for key, u in units.items()
    }
    columns = {b: {"score": [], "target": [], "quality": [], "reason": []} for b in BUCKETS}
    with open(scores_tsv, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            enrol, test = (r["source"], r["enrol"]), (r["source"], r["test"])
            c = columns[r["bucket"]]
            c["score"].append(float(r["score"]))
            c["target"].append(r["target"] == "1")
            c["quality"].append(min(units[enrol]["quality"], units[test]["quality"]))
            c["reason"].append(reason[enrol] or reason[test] or "")
    return {b: {k: np.array(v) for k, v in c.items()} for b, c in columns.items()}


def metrics(score: np.ndarray, target: np.ndarray) -> dict:
    return {
        "eer": eer(score[target], score[~target])[0],
        "min_dcf": min_dcf(score[target], score[~target])[0],
        "n_target": int(target.sum()),
        "n_nontarget": int((~target).sum()),
    }


def decile_rows(trials: dict[str, np.ndarray]) -> list[dict]:
    """Worst and best tenth of one bucket's trials by quality, and all of them.

    Many trials share the top quality (1.0: every input at or past its best), often more
    than a tenth. Equal qualities are put in a seeded random order, so the best tenth is
    a fair sample of them; file order would pick by trial type (the list is not mixed).
    """
    tie_break = np.random.default_rng(0).permutation(len(trials["quality"]))
    order = np.lexsort((tie_break, trials["quality"]))
    n = round(DECILE * len(order))
    rows = []
    for name, index in (("worst", order[:n]), ("best", order[-n:]), ("all", order)):
        q = trials["quality"][index]
        rows.append(
            {
                "decile": name,
                "quality_min": float(q.min()),
                "quality_max": float(q.max()),
                "share_at_top": float(np.mean(q == trials["quality"].max())),
                **metrics(trials["score"][index], trials["target"][index]),
            }
        )
    return rows


def gate_row(trials: dict[str, np.ndarray]) -> dict:
    """What the gate refuses in one bucket, and the error rates of what it lets through."""
    passed = trials["reason"] == ""
    refused = {code: int((trials["reason"] == code).sum()) for code in set(trials["reason"])}
    refused.pop("", None)
    before = metrics(trials["score"], trials["target"])
    after = metrics(trials["score"][passed], trials["target"][passed])
    return {
        "refused": refused,
        "refused_target": int((~passed & trials["target"]).sum()),
        "eer_before": before["eer"],
        "eer_after": after["eer"],
        "min_dcf_before": before["min_dcf"],
        "min_dcf_after": after["min_dcf"],
        "n_before": len(passed),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("conditions", nargs="+", metavar="CONDITION=SCORES_TSV,QUALITY_TSV")
    args = parser.parse_args()
    data = {}
    for item in args.conditions:
        condition, paths = item.split("=", 1)
        data[condition] = read_trials(*paths.split(","))

    print(
        f"condition\tbucket\tdecile\tquality_range\tshare_at_top\tEER_%\tminDCF_p{P_TARGET}"
        "\tn_target\tn_nontarget"
    )
    for condition, buckets in data.items():
        for bucket, trials in buckets.items():
            for r in decile_rows(trials):
                print(
                    f"{condition}\t{bucket}\t{r['decile']}"
                    f"\t{r['quality_min']:.3f}-{r['quality_max']:.3f}\t{r['share_at_top']:.3f}"
                    f"\t{100 * r['eer']:.3f}"
                    f"\t{r['min_dcf']:.4f}\t{r['n_target']}\t{r['n_nontarget']}"
                )
    print(
        "\ncondition\tbucket\ttrials\trefused (by reason)\trefused_target"
        "\tEER_%_before\tEER_%_after\tminDCF_before\tminDCF_after"
    )
    for condition, buckets in data.items():
        for bucket, trials in buckets.items():
            g = gate_row(trials)
            reasons = ", ".join(f"{k} {v}" for k, v in sorted(g["refused"].items())) or "none"
            print(
                f"{condition}\t{bucket}\t{g['n_before']}\t{reasons}\t{g['refused_target']}"
                f"\t{100 * g['eer_before']:.3f}\t{100 * g['eer_after']:.3f}"
                f"\t{g['min_dcf_before']:.4f}\t{g['min_dcf_after']:.4f}"
            )
