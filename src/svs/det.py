"""DET plot: miss rate against false-alarm rate at every threshold, per bucket or condition."""

import argparse
from pathlib import Path

from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.stats import norm

from svs.metrics import eer, error_rates, read_bucket_scores
from svs.trials import BUCKETS

# Axis ticks as rates. 0.001 % is one false alarm in 100,000 non-target trials.
TICKS = [0.00001, 0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]


def det_points(target: NDArray, nontarget: NDArray) -> tuple[NDArray, NDArray]:
    """False-alarm and miss rates on the normal-deviate scale of a DET plot.

    The scale is the inverse of the normal CDF, so two normal score distributions
    give a straight line. A rate of 0 or 1 lies at infinity on it and is left out.
    """
    _, miss, false_alarm = error_rates(target, nontarget)
    keep = (miss > 0) & (miss < 1) & (false_alarm > 0) & (false_alarm < 1)
    return norm.ppf(false_alarm[keep]), norm.ppf(miss[keep])


def draw_det(
    ax, curves: dict[str, tuple[NDArray, NDArray]], legend_title: str, fontsize: int = 8
) -> None:
    """One DET line per (target, non-target) pair; the legend gives EER and trial counts."""
    for label, (target, nontarget) in curves.items():
        x, y = det_points(target, nontarget)
        rate, _ = eer(target, nontarget)
        ax.plot(x, y, label=f"{label}: EER {100 * rate:.3f} % ({len(target)} / {len(nontarget)})")
    ticks = norm.ppf(TICKS)
    labels = [f"{100 * t:g}" for t in TICKS]
    limits = ticks[[0, -1]]
    ax.plot(limits, limits, color="grey", linewidth=0.5, linestyle="--")  # miss = false alarm
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("False-alarm rate (%)")
    ax.set_ylabel("Miss rate (%)")
    ax.grid(True, linewidth=0.3)
    ax.legend(title=legend_title, loc="upper right", fontsize=fontsize, title_fontsize=fontsize)


def save(fig: Figure, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_det(curves: dict[str, tuple[NDArray, NDArray]], path: str | Path) -> None:
    """One plot, one line per bucket."""
    fig = Figure(figsize=(6, 6))  # Not pyplot: no window and no global state.
    draw_det(fig.subplots(), curves, "bucket: EER (target / non-target)")
    save(fig, path)


def plot_det_conditions(files: dict[str, str], path: str | Path) -> None:
    """One plot per bucket, one line per condition (handbook D9: DET per condition)."""
    scores = {condition: read_bucket_scores(f) for condition, f in files.items()}
    fig = Figure(figsize=(12, 12))
    for ax, bucket in zip(fig.subplots(2, 2).flat, BUCKETS, strict=True):
        curves = {c: s[bucket] for c, s in scores.items()}
        draw_det(ax, curves, "condition: EER (target / non-target)", fontsize=7)
        ax.set_title(bucket)
    save(fig, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", nargs="+", help="one score file, or CONDITION=SCORES_TSV pairs")
    parser.add_argument("out_png")
    args = parser.parse_args()
    if any("=" in item for item in args.scores):
        plot_det_conditions(dict(item.split("=", 1) for item in args.scores), args.out_png)
    else:
        plot_det(read_bucket_scores(*args.scores), args.out_png)
    print(f"Wrote {args.out_png}")
