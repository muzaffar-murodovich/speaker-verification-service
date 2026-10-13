"""VoxCeleb1 catalogue: wav/<speaker>/<video>/<clip>.wav plus the tab-separated vox1_meta.csv."""

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from svs.trials import MIN_SPEECH_SECONDS, Utterance


@dataclass(frozen=True)
class Clip:
    """One wav file. Duration is not known here; it comes from VAD at embedding time."""

    utt_id: str  # <speaker>/<video>/<clip>: clip names repeat across videos.
    speaker: str
    session: str  # The YouTube video id.
    gender: str


def read_genders(meta_csv: str | Path) -> dict[str, str]:
    with open(meta_csv, newline="") as f:
        rows = csv.DictReader(f, delimiter="\t")
        return {row["VoxCeleb1 ID"]: row["Gender"] for row in rows}


def list_clips(wav_dir: str | Path, meta_csv: str | Path) -> list[Clip]:
    """Every wav file under wav_dir, sorted by utt_id. Reads paths only, not audio."""
    wav_dir = Path(wav_dir)
    genders = read_genders(meta_csv)
    clips = []
    for path in sorted(wav_dir.glob("*/*/*.wav")):
        speaker, session, name = path.relative_to(wav_dir).with_suffix("").parts
        if genders.get(speaker) not in ("m", "f"):
            raise ValueError(f"No gender in metadata for speaker {speaker}")
        clips.append(Clip(f"{speaker}/{session}/{name}", speaker, session, genders[speaker]))
    if not clips:
        raise ValueError(f"No wav files under {wav_dir}")
    return clips


def utterances(
    utt_ids: Iterable[str], net_seconds: Iterable[float], genders: dict[str, str]
) -> list[Utterance]:
    """Clips with at least MIN_SPEECH_SECONDS of net speech; shorter clips are left out.

    Speaker and session come from the utt_id (<speaker>/<video>[/<clip>]), so no audio
    or wav tree is needed. A joined video has no clip part.
    """
    utts = []
    for utt_id, seconds in zip(utt_ids, net_seconds, strict=True):
        if seconds < MIN_SPEECH_SECONDS:
            continue
        speaker, session = utt_id.split("/")[:2]
        utts.append(Utterance(utt_id, speaker, session, genders[speaker], float(seconds)))
    return utts
