"""Profile centroid (log1p weight with a cap, exact weighted mean, order independence)
and the evidence rule for PROVISIONAL -> CONFIRMED, one test per condition."""

import math

import numpy as np
import pytest

from svs.embed import EMBEDDING_DIM
from svs.profile import (
    CONFIRMED,
    MAX_OCC_WEIGHT,
    PROMOTE_MIN_COHESION,
    PROVISIONAL,
    EnrolRuleUnmet,
    Occurrence,
    Profile,
    UnusableOccurrence,
    UpdateBandTooLow,
    occurrence_weight,
)


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


def near(seed: int) -> np.ndarray:
    """One speaker's embedding: a fixed direction plus a little noise (cosine about 0.96)."""
    v = unit(0) + 0.3 * unit(100 + seed)
    return (v / np.linalg.norm(v)).astype(np.float32)


def profile_of(*occs: Occurrence) -> Profile:
    profile = Profile("spk_1")
    for o in occs:
        profile.accept(o)
    return profile


def test_cohesion_is_the_mean_cosine_to_the_centroid():
    profile = profile_of(occ(1, 30, 0.8), occ(2, 12, 0.6), occ(3, 600, 0.9))
    c = profile.centroid.astype(np.float64)
    expected = np.mean([o.embedding @ c for o in profile.occurrences])
    assert profile.cohesion == pytest.approx(expected)


def test_promotion_when_every_condition_holds():
    profile = profile_of(
        Occurrence("rec_a", near(1), 70, 0.8), Occurrence("rec_b", near(2), 60, 0.7)
    )
    assert profile.status == PROVISIONAL
    assert profile.promotion_missing() == []
    profile.promote()
    assert profile.status == CONFIRMED


def test_one_recording_is_refused_however_long():
    # 300 s of clean, consistent speech, but all from one recording.
    profile = profile_of(
        Occurrence("rec_a", near(1), 150, 0.9), Occurrence("rec_a", near(2), 150, 0.9)
    )
    assert profile.sessions == 1
    assert profile.promotion_missing() == ["sessions"]
    with pytest.raises(EnrolRuleUnmet) as err:
        profile.promote()
    assert err.value.code == "E_SV_ENROL_RULE_UNMET"
    assert profile.status == PROVISIONAL


def test_too_little_speech_is_refused():
    profile = profile_of(
        Occurrence("rec_a", near(1), 61.9, 0.8), Occurrence("rec_b", near(2), 50, 0.8)
    )
    assert profile.promotion_missing() == ["evidence_seconds"]


def test_low_mean_quality_is_refused():
    profile = profile_of(
        Occurrence("rec_a", near(1), 70, 0.7), Occurrence("rec_b", near(2), 60, 0.45)
    )
    assert profile.promotion_missing() == ["mean_quality"]


def test_an_incoherent_profile_is_refused():
    # Three unrelated directions: each is about 1/sqrt(3) = 0.58 from their mean.
    profile = profile_of(*(Occurrence(f"rec_{s}", unit(s), 50, 0.8) for s in (1, 2, 3)))
    assert profile.cohesion < PROMOTE_MIN_COHESION
    assert profile.promotion_missing() == ["cohesion"]


def test_only_a_strong_match_moves_the_centroid():
    profile = profile_of(Occurrence("rec_a", near(1), 70, 0.8))
    before = profile.centroid.copy()
    for band in ("PROBABLE", "WEAK", "NO_MATCH"):
        with pytest.raises(UpdateBandTooLow) as err:
            profile.update(Occurrence("rec_b", unit(9), 60, 0.8), band)
        assert err.value.code == "E_SV_UPDATE_BAND_TOO_LOW"
    np.testing.assert_array_equal(profile.centroid, before)
    assert profile.sessions == 1
    profile.update(Occurrence("rec_b", near(2), 60, 0.8), "STRONG")
    assert profile.sessions == 2


@pytest.mark.parametrize(
    ("seconds", "quality", "code"),
    [(2.9, 0.8, "E_SV_INSUFFICIENT_SPEECH"), (60, 0.44, "E_SV_LOW_QUALITY")],
)
def test_unusable_speech_never_enters(seconds, quality, code):
    profile = Profile("spk_1")
    with pytest.raises(UnusableOccurrence) as err:
        profile.accept(Occurrence("rec_a", near(1), seconds, quality))
    assert err.value.code == code
    assert profile.occurrences == [] and profile.sum_w == 0.0
