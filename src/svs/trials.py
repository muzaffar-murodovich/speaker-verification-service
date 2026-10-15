"""Trial-list building. Target trials must be cross-session (SVS guide §8.1)."""

import csv
import math
import random
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

MIN_SPEECH_SECONDS = 3.0  # CFG_SV_MIN_SPEECH_SECONDS: below this the service refuses to score.
BUCKET_EDGES = (10.0, 30.0, 120.0)
BUCKETS = ("3-10s", "10-30s", "30-120s", ">120s")  # SVS guide §4.4, §8.3
BUCKET_STARTS = dict(zip(BUCKETS, (MIN_SPEECH_SECONDS, *BUCKET_EDGES), strict=True))


def duration_bucket(seconds: float) -> str:
    """Bucket of net speech seconds. Intervals are half-open: 10.0 s belongs to 10-30s."""
    if not math.isfinite(seconds) or seconds < MIN_SPEECH_SECONDS:
        raise ValueError(f"Speech duration must be finite and >= {MIN_SPEECH_SECONDS} s")
    return BUCKETS[bisect_right(BUCKET_EDGES, seconds)]


@dataclass(frozen=True)
class Utterance:
    """One recording. session is the recording source; for VoxCeleb it is the video id.

    duration is net speech seconds after VAD, the quantity the buckets refer to.
    """

    utt_id: str
    speaker: str
    session: str
    gender: str
    duration: float


@dataclass(frozen=True)
class Trial:
    enrol: Utterance
    test: Utterance

    def __post_init__(self) -> None:
        # The guard: the same speaker from the same session would measure channel
        # similarity, not speaker similarity (handbook pitfall #1).
        if self.is_target and self.enrol.session == self.test.session:
            raise ValueError(f"Same-session target trial: {self.enrol.utt_id}, {self.test.utt_id}")
        for utt in (self.enrol, self.test):
            if utt.duration < MIN_SPEECH_SECONDS:
                raise ValueError(f"Clip below {MIN_SPEECH_SECONDS} s: {utt.utt_id}")

    @property
    def is_target(self) -> bool:
        return self.enrol.speaker == self.test.speaker

    @property
    def bucket(self) -> str:
        """The weaker side decides: a trial is as short as its shorter clip."""
        return duration_bucket(min(self.enrol.duration, self.test.duration))


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


def sample_target_trials(
    utterances: Iterable[Utterance], n: int, bucket: str, seed: int = 0
) -> list[Trial]:
    """Up to n target trials of one bucket, drawn uniformly from all of them.

    Reservoir sampling: every pair has the same chance, not only the first speakers in
    sorted order, and the 10 million clip pairs of 3-10s are never held in memory.
    If fewer than n exist, all of them are returned.
    """
    # A clip shorter than the bucket's start would put every pair it is in below it.
    utts = [u for u in utterances if u.duration >= BUCKET_STARTS[bucket]]
    rng = random.Random(seed)
    kept: list[Trial] = []
    matching = (t for t in target_trials(utts) if t.bucket == bucket)
    for i, trial in enumerate(matching):
        if i < n:
            kept.append(trial)
        elif (j := rng.randrange(i + 1)) < n:
            kept[j] = trial
    return kept


def nontarget_trials(
    utterances: Iterable[Utterance], n: int, seed: int = 0, bucket: str | None = None
) -> list[Trial]:
    """Sample n distinct pairs of different speakers of the same gender (SVS guide §8.1).

    Cross-gender pairs are easy to reject and would make the EER look better than it is.
    With a bucket, only pairs whose shorter clip falls in that bucket are kept.
    """
    utts = sorted(utterances, key=lambda u: u.utt_id)
    if bucket is not None:
        utts = [u for u in utts if u.duration >= BUCKET_STARTS[bucket]]
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
        trial = Trial(*sorted((enrol, test), key=lambda u: u.utt_id))
        if bucket is not None and trial.bucket != bucket:
            continue
        pair = (trial.enrol.utt_id, trial.test.utt_id)
        if pair in seen:
            continue
        seen.add(pair)
        trials.append(trial)
    if len(trials) < n:
        raise ValueError(f"Found only {len(trials)} distinct non-target pairs, need {n}")
    return trials


def write_trials(path: str | Path, rows: Iterable[tuple[str, Trial]]) -> int:
    """Write (source, trial) rows as a tab-separated list with a header; return the count.

    source records where the audio came from, e.g. "clip" or "video" (SVS guide §8.1).
    """
    count = 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["enrol", "test", "target", "bucket", "source"])
        for source, t in rows:
            writer.writerow([t.enrol.utt_id, t.test.utt_id, int(t.is_target), t.bucket, source])
            count += 1
    return count
