"""Profile centroid: log1p weight with a cap, exact weighted mean, order independence."""

import math

import numpy as np
import pytest

from svs.embed import EMBEDDING_DIM
from svs.profile import MAX_OCC_WEIGHT, Occurrence, Profile, occurrence_weight


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def occ(seed: int, seconds: float, quality: float) -> Occurrence:
    return Occurrence(f"rec{seed}", unit(seed), seconds, quality)


def test_weight_grows_with_log1p_of_seconds():
    assert occurrence_weight(0.8, 30) == pytest.approx(0.8 * math.log1p(30))


def test_weight_is_capped():
    # 0.9 * log1p(600) = 5.76: without the cap this call would count almost twice.
    assert occurrence_weight(0.9, 600) == MAX_OCC_WEIGHT


def test_centroid_is_the_normalised_weighted_mean():
    occs = [occ(1, 30, 0.8), occ(2, 12, 0.6), occ(3, 600, 0.9)]
    profile = Profile("spk_1")
    for o in occs:
        profile.accept(o)
    weights = [occurrence_weight(o.quality, o.speech_seconds) for o in occs]
    weighted = [w * o.embedding.astype(np.float64) for w, o in zip(weights, occs, strict=True)]
    mean = sum(weighted) / sum(weights)
    assert profile.sum_w == pytest.approx(sum(weights))
    np.testing.assert_allclose(profile.centroid, mean / np.linalg.norm(mean), atol=1e-6)
    assert np.linalg.norm(profile.centroid) == pytest.approx(1.0, abs=1e-6)


def test_centroid_does_not_depend_on_arrival_order():
    occs = [occ(seed, 10 + 20 * seed, 0.5 + 0.05 * seed) for seed in range(8)]
    forward, backward = Profile("spk_f"), Profile("spk_b")
    for o in occs:
        forward.accept(o)
    for o in reversed(occs):
        backward.accept(o)
    np.testing.assert_allclose(forward.centroid, backward.centroid, atol=1e-6)


def test_an_empty_profile_has_no_centroid():
    with pytest.raises(ValueError, match="no accepted occurrence"):
        _ = Profile("spk_empty").centroid
