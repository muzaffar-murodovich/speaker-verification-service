"""DET plot: miss rate against false-alarm rate at every threshold, one line per bucket."""

import argparse
from pathlib import Path

from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.stats import norm

from svs.metrics import eer, error_rates, read_bucket_scores

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


def plot_det(curves: dict[str, tuple[NDArray, NDArray]], path: str | Path) -> None:
    """One DET line per (target, non-target) pair; the legend gives EER and trial counts."""
    fig = Figure(figsize=(6, 6))  # Not pyplot: no window and no global state.
    ax = fig.subplots()
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
    ax.legend(
        title="bucket: EER (target / non-target)", loc="upper right", fontsize=8, title_fontsize=8
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores_tsv")
    parser.add_argument("out_png")
    args = parser.parse_args()
    plot_det(read_bucket_scores(args.scores_tsv), args.out_png)
    print(f"Wrote {args.out_png}")
