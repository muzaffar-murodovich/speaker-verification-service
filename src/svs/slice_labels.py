"""Reference speaker segments from the labelling tool's JSON export (D16).

Each exported item has an id (the audio file's name without extension) and segments
with start, end (seconds) and speaker_id. Speaker ids are local to one recording:
"speaker-left-channel" in two calls is two different people. So the reference says who
spoke when inside a recording, and nothing about the same person across recordings.

In a stereo recording, the left-channel speaker is channel 0 and the right is channel 1.

source is the folder of the item's original file name (a region or a collection, e.g.
"Qashqadaryo"): the only hint of where a call came from. The file name itself is not
kept: some hold a phone number.
"""

import json
from collections import Counter
from pathlib import Path

import soundfile as sf

from svs.slice_run import segments_text

CHANNEL_OF = {"speaker-left-channel": 0, "speaker-right-channel": 1}


def merged(segments: list[tuple[float, float]], length_ms: float) -> list[tuple[float, float]]:
    """Sorted, overlaps joined, clipped to the audio: what svs.extract accepts."""
    out: list[list[float]] = []
    for s, e in sorted(segments):
        s, e = max(s, 0.0), min(e, length_ms)
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def read_export(path: Path, files: list[dict], root: Path) -> dict:
    """{audio: {"channels", "source", "speakers": {speaker_id: [(start_ms, end_ms)]},
    "genders", "langs"}}.

    genders is the annotator's gender label of each speaker, the most common over its
    segments: a label, not a measurement (there is no gender classifier here). langs
    counts the annotator's language label (UZ, RU, ...) over the speaker's segments.

    files is slice_run's files.tsv; an exported item without a readable file is left out.
    The length comes from the file under root: files.tsv rounds seconds to 1 ms, and a
    segment clipped to a rounded-up length ends outside the audio.
    """
    by_stem = {Path(f["audio"]).stem: f for f in files if not f["error"]}
    reference = {}
    for item in json.loads(Path(path).read_text())["data"]:
        f = by_stem.get(item["id"])
        if f is None:
            continue
        info = sf.info(root / f["audio"])
        length_ms = info.frames * 1000 / info.samplerate
        speakers: dict[str, list[tuple[float, float]]] = {}
        genders: dict[str, Counter] = {}
        langs: dict[str, Counter] = {}
        for seg in item["segments"]:
            span = (float(seg["start"]) * 1000, float(seg["end"]) * 1000)
            speakers.setdefault(seg["speaker_id"], []).append(span)
            labels = seg.get("metadata") or {}
            genders.setdefault(seg["speaker_id"], Counter())[labels.get("gender") or ""] += 1
            langs.setdefault(seg["speaker_id"], Counter())[labels.get("lang") or ""] += 1
        name = item.get("audio", {}).get("filename") or ""
        reference[f["audio"]] = {
            "channels": int(f["channels"]),
            "source": name.rsplit("/", 1)[0] if "/" in name else "",
            "speakers": {k: merged(v, length_ms) for k, v in speakers.items()},
            "genders": {k: v.most_common(1)[0][0] for k, v in genders.items()},
            "langs": langs,
        }
    return reference


def item_genders(reference: dict) -> dict[str, str]:
    """The annotator's gender label per reference_manifest item_id."""
    return {
        f"{audio}#ref{k}": ref["genders"][speaker_id]
        for audio, ref in reference.items()
        for k, speaker_id in enumerate(ref["speakers"])
    }


def item_labels(reference: dict) -> dict[str, dict]:
    """Per reference_manifest item_id: its recording's source and its language counts."""
    return {
        f"{audio}#ref{k}": {"source": ref["source"], "langs": ref["langs"][speaker_id]}
        for audio, ref in reference.items()
        for k, speaker_id in enumerate(ref["speakers"])
    }


def reference_manifest(reference: dict) -> list[dict]:
    """An item per reference speaker; in a stereo file, on that speaker's channel."""
    rows = []
    for audio, ref in reference.items():
        for k, (speaker_id, segments) in enumerate(ref["speakers"].items()):
            if not segments:
                continue
            stereo = ref["channels"] > 1
            rows.append(
                {
                    "item_id": f"{audio}#ref{k}",
                    "audio": audio,
                    "channel": CHANNEL_OF[speaker_id] if stereo else "",
                    "segments": segments_text(segments),
                    "speaker": "",
                }
            )
    return rows
