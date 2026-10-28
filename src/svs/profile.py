"""Speaker profile: a quality-weighted centroid kept as sum_vec / sum_w (SVS guide §5.4).

The profile stores the weighted sum of its accepted embeddings and the total weight, not
the centroid itself. The centroid is derived from the sum, so it does not depend on the
order the occurrences arrived in.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from svs.embed import EMBEDDING_DIM

MAX_OCC_WEIGHT = 3.0  # CFG_SV_MAX_OCC_WEIGHT


def occurrence_weight(quality: float, speech_seconds: float) -> float:
    """quality × log1p(seconds), capped at MAX_OCC_WEIGHT.

    log1p, not seconds: a 600 s call holds more evidence than a 30 s one, but not 20 times
    more. The cap stops one long, clean call from becoming the whole profile.
    """
    return min(quality * math.log1p(speech_seconds), MAX_OCC_WEIGHT)


@dataclass(frozen=True)
class Occurrence:
    """One speaker's speech in one recording: its embedding, net speech and quality."""

    recording_id: str
    embedding: NDArray[np.float32]
    speech_seconds: float
    quality: float


@dataclass
class Profile:
    profile_id: str
    sum_vec: NDArray[np.float64] = field(default_factory=lambda: np.zeros(EMBEDDING_DIM))
    sum_w: float = 0.0

    def accept(self, occ: Occurrence) -> None:
        if occ.embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(f"expected a {EMBEDDING_DIM}-dim vector, got {occ.embedding.shape}")
        w = occurrence_weight(occ.quality, occ.speech_seconds)
        self.sum_vec += w * occ.embedding.astype(np.float64)
        self.sum_w += w

    @property
    def centroid(self) -> NDArray[np.float32]:
        """The weighted mean direction, L2-normalised like every embedding (§4.1)."""
        norm = np.linalg.norm(self.sum_vec)
        if norm == 0:
            raise ValueError(f"profile {self.profile_id} has no accepted occurrence")
        return (self.sum_vec / norm).astype(np.float32)
