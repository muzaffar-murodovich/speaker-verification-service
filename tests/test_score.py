"""Cosine scoring of trials."""

import numpy as np
import pytest

from svs.score import asnorm_scores, cosine_scores, score_condition, score_trials, write_scores


def test_cosine_of_known_pairs():
    # Same direction, at right angles, opposite; the last pair differs only in length.
    enrol = np.array([[1, 0], [1, 0], [1, 0], [3, 4]], np.float32)
    test = np.array([[1, 0], [0, 1], [-1, 0], [6, 8]], np.float32)
    np.testing.assert_allclose(cosine_scores(enrol, test), [1.0, 0.0, -1.0, 1.0])


def test_nan_embedding_is_refused():
    enrol = np.array([[1, 0], [np.nan, np.nan]], np.float32)
    test = np.array([[1, 0], [1, 0]], np.float32)
    with pytest.raises(ValueError, match="1 trials have a NaN"):
        cosine_scores(enrol, test)


def test_source_picks_the_cache():
    # The same id holds different vectors in the two caches; only the source decides.
    caches = {
        "clip": {"a": np.array([1, 0], np.float32), "b": np.array([1, 0], np.float32)},
        "video": {"a": np.array([1, 0], np.float32), "b": np.array([0, 1], np.float32)},
    }
    rows = [
        {"enrol": "a", "test": "b", "source": "clip"},
        {"enrol": "a", "test": "b", "source": "video"},
    ]
    np.testing.assert_allclose(score_trials(rows, caches), [1.0, 0.0])


def test_id_missing_from_its_cache_is_refused():
    caches = {"clip": {"a": np.array([1, 0], np.float32)}, "video": {}}
    with pytest.raises(KeyError):
        score_trials([{"enrol": "a", "test": "a", "source": "video"}], caches)


def test_written_scores_keep_rows_and_every_digit(tmp_path):
    rows = [{"enrol": "a", "test": "b", "target": "1"}, {"enrol": "c", "test": "d", "target": "0"}]
    scores = np.array([0.1234567890123, -0.5])
    write_scores(tmp_path / "s.tsv", rows, scores)
    lines = (tmp_path / "s.tsv").read_text().splitlines()
    assert lines == ["enrol\ttest\ttarget\tscore", "a\tb\t1\t0.1234567890123", "c\td\t0\t-0.5"]


NAN = np.array([np.nan, np.nan], np.float32)
CACHES = {
    "clip": {"a": np.array([1, 0], np.float32), "b": np.array([0, 1], np.float32), "c": NAN},
    "video": {},
}
ROWS = [
    {"enrol": "a", "test": "b", "source": "clip"},
    {"enrol": "a", "test": "c", "source": "clip"},
    {"enrol": "c", "test": "b", "source": "clip"},
]


def test_degraded_trials_below_the_minimum_are_counted_not_scored():
    kept, scores, refused = score_condition(ROWS, CACHES, "g729")
    assert refused == 2
    assert [(r["enrol"], r["test"]) for r in kept] == [("a", "b")]
    np.testing.assert_allclose(scores, [0.0])


def test_every_scored_trial_carries_its_condition():
    kept, _, _ = score_condition(ROWS[:1], CACHES, "g711a")
    assert kept == [{"enrol": "a", "test": "b", "source": "clip", "condition": "g711a"}]


def test_a_refused_clean_trial_is_a_bug():
    with pytest.raises(ValueError, match="2 clean trials"):
        score_condition(ROWS, CACHES, "pcm16k")


def test_asnorm_uses_each_audio_s_own_cohort_statistics():
    # Two audio with the same raw score against a third; "a" sits near the cohort.
    caches = {
        "clip": {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0]), "c": np.array([0.6, 0.8])}
    }
    cohort = np.array([[1.0, 0.1], [1.0, -0.1], [0.9, 0.0]])
    rows = [
        {"enrol": "a", "test": "c", "source": "clip"},
        {"enrol": "b", "test": "c", "source": "clip"},
    ]
    raw = score_trials(rows, caches)
    got = asnorm_scores(rows, caches, raw, cohort, n=3)

    unit = cohort / np.linalg.norm(cohort, axis=1, keepdims=True)

    def z(x, side):
        top = np.sort(unit @ caches["clip"][side])[-3:]
        return (x - top.mean()) / (top.std() + 1e-6)

    want = [0.5 * (z(raw[0], "a") + z(raw[0], "c")), 0.5 * (z(raw[1], "b") + z(raw[1], "c"))]
    np.testing.assert_allclose(got, want)
    assert got[0] < got[1]  # a is close to the impostors, so its score counts for less
