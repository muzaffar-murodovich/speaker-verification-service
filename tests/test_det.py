"""DET plot."""

import numpy as np
from scipy.stats import norm

from svs.det import det_points, plot_det


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
