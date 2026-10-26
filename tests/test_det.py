"""DET plot."""

import numpy as np
from scipy.stats import norm

from svs.det import det_points, plot_det, plot_det_conditions
from svs.trials import BUCKETS


def test_det_points_are_the_error_rates_without_zero_or_one():
    # Same example as the EER test. Thresholds 0.1, 0.2 and 0.9 give a rate of 0 or 1.
    target = np.array([0.1, 0.6, 0.7, 0.9])
    nontarget = np.array([0.2, 0.3, 0.4, 0.8])
    x, y = det_points(target, nontarget)
    np.testing.assert_allclose(norm.cdf(x), [0.75, 0.5, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(norm.cdf(y), [0.25, 0.25, 0.25, 0.5, 0.75])


def test_plot_det_writes_a_png(tmp_path):
    rng = np.random.default_rng(0)
    curves = {"3-10s": (rng.normal(1, 1, 200), rng.normal(-1, 1, 200))}
    plot_det(curves, tmp_path / "plots" / "det.png")
    assert (tmp_path / "plots" / "det.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_det_conditions_writes_one_png(tmp_path):
    path = tmp_path / "scores.tsv"
    rows = ["bucket\tgender\ttarget\tscore"]
    rows += [
        f"{b}\tf\t{t}\t{s}" for b in BUCKETS for t, s in [(1, 0.9), (1, 0.3), (0, 0.5), (0, 0.1)]
    ]
    path.write_text("\n".join(rows) + "\n")
    plot_det_conditions({"pcm16k": str(path), "g729": str(path)}, tmp_path / "det.png")
    assert (tmp_path / "det.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
