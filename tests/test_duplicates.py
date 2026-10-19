"""Finding videos that hold the same audio under two video ids."""

import numpy as np

from svs.duplicates import candidate_pairs, confirm_video_pairs, shared_audio

rng = np.random.default_rng(0)
SPEECH = rng.standard_normal(16_000)


def test_the_same_audio_scores_one_even_when_shifted_or_quieter():
    shifted = np.concatenate([np.zeros(800), SPEECH[:-800]])  # cut 50 ms later
    assert shared_audio(SPEECH, SPEECH) > 0.999
    assert shared_audio(SPEECH, 0.3 * SPEECH) > 0.999
    assert shared_audio(SPEECH, shifted) > 0.9


def test_different_audio_scores_near_zero():
    assert shared_audio(SPEECH, rng.standard_normal(16_000)) < 0.05


def test_candidates_are_cross_video_and_above_the_cosine_floor():
    ids = np.array(["s/v1/1", "s/v1/2", "s/v2/1", "s/v3/1"])
    e = np.array([[1, 0], [1, 0], [0.96, 0.28], [0, 1]], np.float32)
    # v1/1 and v1/2 are identical but share a video; v3/1 is at right angles to all.
    assert [(a, b) for a, b, _ in candidate_pairs(ids, e, 0.9)] == [
        ("s/v1/1", "s/v2/1"),
        ("s/v1/2", "s/v2/1"),
    ]


def test_candidates_come_highest_cosine_first():
    ids = np.array(["s/v1/1", "s/v2/1", "s/v3/1"])
    e = np.array([[1, 0], [0.9, 0.436], [0.99, 0.141]], np.float32)
    # Cosines: v1-v2 0.90, v1-v3 0.99, v2-v3 0.95; index order would put v1-v2 first.
    assert [(a, b) for a, b, _ in candidate_pairs(ids, e, 0.85)] == [
        ("s/v1/1", "s/v3/1"),
        ("s/v2/1", "s/v3/1"),
        ("s/v1/1", "s/v2/1"),
    ]


def test_a_nan_embedding_is_never_a_candidate():
    ids = np.array(["s/v1/1", "s/v2/1"])
    e = np.array([[1, 0], [np.nan, np.nan]], np.float32)
    assert candidate_pairs(ids, e, 0.0) == []


def test_one_confirmed_clip_pair_per_video_pair():
    candidates = [("s/v1/1", "s/v2/1", 0.99), ("s/v1/2", "s/v2/2", 0.98), ("s/v1/1", "s/v3/1", 0.9)]
    checked = []

    def shared(a, b):
        checked.append((a, b))
        return 0.9 if "v2" in b else 0.05

    confirmed = confirm_video_pairs(candidates, shared, 0.2)
    assert confirmed == [("s/v1/1", "s/v2/1", 0.99, 0.9)]
    # v1-v2 was confirmed by its first pair, so its second pair is never read.
    assert checked == [("s/v1/1", "s/v2/1"), ("s/v1/1", "s/v3/1")]
