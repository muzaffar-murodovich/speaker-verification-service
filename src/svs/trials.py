"""Trial-list building. Target trials must be cross-session (SVS guide §8.1)."""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Utterance:
    """One recording. session is the recording source; for VoxCeleb it is the video id."""

    utt_id: str
    speaker: str
    session: str


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
