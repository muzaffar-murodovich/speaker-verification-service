"""Cosine scoring of trials."""

import numpy as np
import pytest

from svs.score import cosine_scores


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
