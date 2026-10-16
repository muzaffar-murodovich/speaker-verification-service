"""Cosine scoring of trials."""

import numpy as np
import pytest

from svs.score import cosine_scores, score_trials, write_scores


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
