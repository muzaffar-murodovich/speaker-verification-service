"""Speech extraction helpers. Segment boundaries use milliseconds (SVS guide §3.2)."""

import math
from collections.abc import Sequence


def validate_segments(segments: Sequence[Sequence[float]], duration_ms: float) -> None:
    """Require nonempty, ordered, non-overlapping segments inside the audio.

    Intervals are half-open: [start, end), so adjacent segments are allowed.
    Reject overlaps instead of counting the same speech twice.
    """
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("Audio duration must be finite and positive")
    if not segments:
        raise ValueError("At least one segment is required")

    previous_end = 0.0
    for segment in segments:
        if len(segment) != 2:
            raise ValueError("Each segment must contain start and end in milliseconds")
        start, end = segment
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Segment boundaries must be finite")
        if not 0 <= start < end <= duration_ms:
            raise ValueError("Segments must have positive length and stay inside the audio")
        if start < previous_end:
            raise ValueError("Segments must be ordered and must not overlap")
        previous_end = end
