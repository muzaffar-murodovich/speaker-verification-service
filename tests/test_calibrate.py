"""Operating points: thresholds from a fixed FAR, and the batched AS-Norm pair scores."""

import numpy as np
import pytest

from svs.asnorm import as_norm
from svs.calibrate import POOL_SIZE, PROBABLE_FAR, STRONG_FAR, pair_scores, threshold_at_far

NONTARGET = np.arange(1, 11) / 10  # 0.1, 0.2, ..., 1.0
TARGET = np.array([0.85, 0.95])


def test_strong_far_follows_the_pool_size_rule():
    # SVS guide §8.4: N_expected x FAR <= 0.1 for STRONG, PROBABLE at 10 x that FAR.
    assert POOL_SIZE * STRONG_FAR == pytest.approx(0.1)
    assert PROBABLE_FAR == pytest.approx(10 * STRONG_FAR)


def test_threshold_at_far_allows_at_most_far_false_accepts():
    # FAR 0.2 allows 2 of 10: at 0.85 only 0.9 and 1.0 pass; at 0.8 three would.
    assert threshold_at_far(TARGET, NONTARGET, 0.2) == (0.85, 2, 0.0)
    # FAR 0.1 allows 1: only 1.0 passes 0.95, and the 0.85 target is rejected.
    assert threshold_at_far(TARGET, NONTARGET, 0.1) == (0.95, 1, 0.5)


def test_threshold_at_far_refuses_a_far_below_one_trial():
    with pytest.raises(ValueError, match="needs >= 20 non-target"):
        threshold_at_far(TARGET, NONTARGET, 0.05)


def test_pair_scores_in_batches_equal_one_pass():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 4))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    mu, sd = rng.uniform(0, 0.3, 6), rng.uniform(0.05, 0.2, 6)
    i, j = np.array([0, 1, 2, 3, 0, 4, 1]), np.array([5, 4, 3, 5, 2, 5, 3])
    whole = as_norm(np.sum(x[i] * x[j], axis=1), mu[i], sd[i], mu[j], sd[j])
    np.testing.assert_allclose(pair_scores(x, mu, sd, i, j, batch=3), whole)
