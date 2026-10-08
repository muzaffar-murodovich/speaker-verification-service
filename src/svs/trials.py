"""Trial-list building. Target trials must be cross-session (SVS guide §8.1)."""

import math
import random
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations

MIN_SPEECH_SECONDS = 3.0  # CFG_SV_MIN_SPEECH_SECONDS: below this the service refuses to score.
BUCKET_EDGES = (10.0, 30.0, 120.0)
BUCKETS = ("3-10s", "10-30s", "30-120s", ">120s")  # SVS guide §4.4, §8.3


def duration_bucket(seconds: float) -> str:
    """Bucket of net speech seconds. Intervals are half-open: 10.0 s belongs to 10-30s."""
    if not math.isfinite(seconds) or seconds < MIN_SPEECH_SECONDS:
        raise ValueError(f"Speech duration must be finite and >= {MIN_SPEECH_SECONDS} s")
    return BUCKETS[bisect_right(BUCKET_EDGES, seconds)]


@dataclass(frozen=True)
class Utterance:
    """One recording. session is the recording source; for VoxCeleb it is the video id."""

    utt_id: str
    speaker: str
    session: str
    gender: str


@dataclass(frozen=True)
class Trial:
    enrol: Utterance
    test: Utterance

    def __post_init__(self) -> None:
        # The guard: the same speaker from the same session would measure channel
        # similarity, not speaker similarity (handbook pitfall #1).
        if self.is_target and self.enrol.session == self.test.session:
            raise ValueError(f"Same-session target trial: {self.enrol.utt_id}, {self.test.utt_id}")

    @property
    def is_target(self) -> bool:
        return self.enrol.speaker == self.test.speaker


def target_trials(utterances: Iterable[Utterance]) -> Iterator[Trial]:
    """Yield every cross-session pair of the same speaker, in a deterministic order."""
    by_speaker: dict[str, list[Utterance]] = defaultdict(list)
    for utt in utterances:
        by_speaker[utt.speaker].append(utt)
    for speaker in sorted(by_speaker):
        utts = sorted(by_speaker[speaker], key=lambda u: u.utt_id)
        for enrol, test in combinations(utts, 2):
            if enrol.session != test.session:
                yield Trial(enrol, test)


def nontarget_trials(utterances: Iterable[Utterance], n: int, seed: int = 0) -> list[Trial]:
    """Sample n distinct pairs of different speakers of the same gender (SVS guide §8.1).

    Cross-gender pairs are easy to reject and would make the EER look better than it is.
    """
    utts = sorted(utterances, key=lambda u: u.utt_id)
    by_gender: dict[str, list[Utterance]] = defaultdict(list)
    for utt in utts:
        by_gender[utt.gender].append(utt)
    rng = random.Random(seed)
    seen: set[tuple[str, str]] = set()
    trials: list[Trial] = []
    for _ in range(100 * n):
        if len(trials) == n:
            break
        enrol = rng.choice(utts)
        test = rng.choice(by_gender[enrol.gender])
        if enrol.speaker == test.speaker:
            continue
        enrol, test = sorted((enrol, test), key=lambda u: u.utt_id)
        if (enrol.utt_id, test.utt_id) in seen:
            continue
        seen.add((enrol.utt_id, test.utt_id))
        trials.append(Trial(enrol, test))
    if len(trials) < n:
        raise ValueError(f"Found only {len(trials)} distinct non-target pairs, need {n}")
    return trials
