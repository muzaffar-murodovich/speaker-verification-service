"""Speaker profile: a quality-weighted centroid kept as sum_vec / sum_w (SVS guide §5.4).

The profile stores the weighted sum of its accepted embeddings and the total weight, not
the centroid itself. The centroid is derived from the sum, so it does not depend on the
order the occurrences arrived in.

A new profile is PROVISIONAL. It becomes CONFIRMED only when the evidence rule holds
(SVS guide §5.2, handbook D12).
"""

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from svs.embed import EMBEDDING_DIM

MAX_OCC_WEIGHT = 3.0  # CFG_SV_MAX_OCC_WEIGHT
# The evidence rule for PROVISIONAL -> CONFIRMED (SVS guide §5.2, handbook M3).
PROMOTE_MIN_SECONDS = 120.0  # CFG_SV_PROMOTE_MIN_SECONDS
PROMOTE_MIN_SESSIONS = 2  # CFG_SV_PROMOTE_MIN_SESSIONS
PROMOTE_MIN_QUALITY = 0.6  # CFG_SV_PROMOTE_MIN_QUALITY
PROMOTE_MIN_COHESION = 0.65  # CFG_SV_PROMOTE_MIN_COHESION

PROVISIONAL = "PROVISIONAL"
CONFIRMED = "CONFIRMED"


class EnrolRuleUnmet(ValueError):
    code = "E_SV_ENROL_RULE_UNMET"


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
    status: str = PROVISIONAL
    # Kept for the evidence rule: sessions and cohesion need each occurrence, not the sum.
    occurrences: list[Occurrence] = field(default_factory=list)

    def accept(self, occ: Occurrence) -> None:
        if occ.embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(f"expected a {EMBEDDING_DIM}-dim vector, got {occ.embedding.shape}")
        w = occurrence_weight(occ.quality, occ.speech_seconds)
        self.sum_vec += w * occ.embedding.astype(np.float64)
        self.sum_w += w
        self.occurrences.append(occ)

    @property
    def evidence_seconds(self) -> float:
        return sum(o.speech_seconds for o in self.occurrences)

    @property
    def sessions(self) -> int:
        """Distinct recordings. Two occurrences from one recording are one session."""
        return len({o.recording_id for o in self.occurrences})

    @property
    def mean_quality(self) -> float:
        return float(np.mean([o.quality for o in self.occurrences]))

    @property
    def cohesion(self) -> float:
        """Mean cosine of the accepted occurrences to the centroid (SVS guide §5.2)."""
        c = self.centroid.astype(np.float64)
        return float(np.mean([o.embedding.astype(np.float64) @ c for o in self.occurrences]))

    def promotion_missing(self) -> list[str]:
        """The evidence-rule conditions that do not hold yet; empty when all hold."""
        missing = []
        if self.evidence_seconds < PROMOTE_MIN_SECONDS:
            missing.append("evidence_seconds")
        # The one most often dropped: a centroid from one recording learns that recording's
        # line, codec and room, and then fails the same person on another line (§5.2).
        if self.sessions < PROMOTE_MIN_SESSIONS:
            missing.append("sessions")
        if self.mean_quality < PROMOTE_MIN_QUALITY:
            missing.append("mean_quality")
        if self.cohesion < PROMOTE_MIN_COHESION:
            missing.append("cohesion")
        return missing

    def promote(self) -> None:
        """PROVISIONAL -> CONFIRMED, or EnrolRuleUnmet naming what is missing."""
        missing = self.promotion_missing()
        if missing:
            raise EnrolRuleUnmet(f"profile {self.profile_id} lacks {missing}")
        self.status = CONFIRMED

    @property
    def centroid(self) -> NDArray[np.float32]:
        """The weighted mean direction, L2-normalised like every embedding (§4.1)."""
        norm = np.linalg.norm(self.sum_vec)
        if norm == 0:
            raise ValueError(f"profile {self.profile_id} has no accepted occurrence")
        return (self.sum_vec / norm).astype(np.float32)
