"""AS-Norm: top-n cohort statistics and the symmetric z-score."""

import numpy as np
import pytest

from svs import asnorm
from svs.asnorm import as_norm, cohort_stats, top_n_stats

# Audio A and B each scored against the same six impostors (the worked example).
A = [0.55, 0.20, 0.60, 0.10, 0.50, 0.30]
B = [0.15, 0.05, 0.25, 0.10, 0.20, 0.00]


def test_top_n_stats_use_only_the_highest_scores():
    mu, sd = top_n_stats(np.array([A, B]), n=3)
    # Top three of A are 0.50, 0.55, 0.60; all six would give a mean of 0.375.
    np.testing.assert_allclose(mu, [0.55, 0.20])
    np.testing.assert_allclose(sd, [np.std([0.50, 0.55, 0.60])] * 2)


def test_same_raw_score_means_less_for_a_voice_close_to_impostors():
    mu, sd = top_n_stats(np.array([A, B]), n=3)
    z = (0.50 - mu) / sd
    np.testing.assert_allclose(z, [-1.2247, 7.3485], atol=1e-4)


def test_as_norm_is_symmetric_and_averages_the_two_z_scores():
    raw = np.array([0.50])
    mu_e, sd_e, mu_f, sd_f = np.array([0.55]), np.array([0.05]), np.array([0.20]), np.array([0.10])
    ef = as_norm(raw, mu_e, sd_e, mu_f, sd_f)
    np.testing.assert_allclose(ef, as_norm(raw, mu_f, sd_f, mu_e, sd_e))
    np.testing.assert_allclose(ef, 0.5 * (-0.05 / (0.05 + 1e-6) + 0.30 / (0.10 + 1e-6)))


def test_zero_spread_does_not_divide_by_zero():
    assert np.isfinite(
        as_norm(np.array([0.5]), np.array([0.2]), np.array([0.0]), np.array([0.2]), np.array([0.0]))
    ).all()


def test_top_n_larger_than_the_cohort_is_refused():
    with pytest.raises(ValueError, match="only 6"):
        top_n_stats(np.array([A]), n=7)


def test_cohort_stats_match_scoring_one_row_at_a_time(monkeypatch):
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(10, 4))
    cohort = rng.normal(size=(20, 4)) * 3  # not unit length: cohort_stats normalises
    monkeypatch.setattr(asnorm, "BATCH", 3)  # 10 rows in batches of 3, 3, 3, 1
    mu, sd = cohort_stats(embeddings, cohort, n=5)

    unit = cohort / np.linalg.norm(cohort, axis=1, keepdims=True)
    for i, e in enumerate(embeddings):
        top = np.sort(unit @ (e / np.linalg.norm(e)))[-5:]
        assert mu[i] == pytest.approx(top.mean())
        assert sd[i] == pytest.approx(top.std())


def test_nan_embedding_gets_nan_stats():
    embeddings = np.array([[1.0, 0.0], [np.nan, np.nan]])
    mu, sd = cohort_stats(embeddings, np.array([[1.0, 0.0], [0.0, 1.0]]), n=1)
    assert np.isfinite(mu[0]) and np.isnan(mu[1]) and np.isnan(sd[1])
