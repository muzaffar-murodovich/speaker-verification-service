"""Segment guards: invalid ranges must never reach speech pooling."""

import pytest

from svs.extract import validate_segments


def test_accepts_adjacent_segments_gaps_and_audio_boundaries():
    validate_segments([(0, 100), (100, 200), (300, 1000)], duration_ms=1000)


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [(0,)],
        [(0, 100, 200)],
        [(-1, 100)],
        [(100, 100)],
        [(200, 100)],
        [(0, 1001)],
        [(0, float("inf"))],
        [(float("nan"), 100)],
        [(0, 100), (99, 200)],
        [(300, 400), (0, 100)],
    ],
)
def test_rejects_invalid_segments(segments):
    with pytest.raises(ValueError):
        validate_segments(segments, duration_ms=1000)


@pytest.mark.parametrize("duration_ms", [0, -1, float("inf"), float("nan")])
def test_rejects_invalid_audio_duration(duration_ms):
    with pytest.raises(ValueError):
        validate_segments([(0, 100)], duration_ms=duration_ms)
