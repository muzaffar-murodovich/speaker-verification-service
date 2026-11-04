"""A minimal diarizer for mono recordings: who spoke when (SVS guide §1.1, D16).

SVS is not a diarizer (§0.3); the pipeline's diarizer calls it. The slice has mono
recordings with two or more voices in one channel, and embed_item needs one speaker's
segments, so this module stands in for the diarizer, in the shape §1.1 gives it:

1. VAD (svs.extract.speech_frames) marks each 20 ms frame speech or not.
2. Speech frames with gaps of at most GAP_FRAMES between them form a region.
3. Each region is cut into equal windows of at most WINDOW_FRAMES. A window with fewer
   than MIN_WINDOW_FRAMES speech frames is left out: too short for an embedding.
4. Each window's speech gets one embedding. These are transient, never stored (§7.2).
5. Ward clustering puts the windows into n_speakers groups. The embeddings are unit
   length, so their Euclidean distance is a function of the cosine. Average linkage
   (cosine) split one odd window or two off as a "speaker" in half of the files.
6. For embedding (the manifest), neighbouring windows in the same group form one turn,
   gaps included: VAD drops the gaps again when the turn is embedded. For DER, only the
   windows count: the gaps between them are not speech.

The number of speakers is given, not estimated: two, the agent and the caller.
Overlapping speech goes to one speaker, or to neither when VAD drops it.

der() scores a result against reference segments as diarization error rate: missed
speech, false alarm and speaker confusion over the reference speech time, on 10 ms
frames, with no collar. Speakers are paired so that the overlap is largest.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly

from svs.embed import Embedder
from svs.extract import FRAME_SAMPLES, SAMPLE_RATE, load_audio, speech_frames
from svs.slice_labels import read_export, reference_manifest
from svs.slice_run import (
    MANIFEST_FIELDS,
    OUT,
    default_manifest,
    read_tsv,
    segments_text,
    write_tsv,
)

FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE  # 20
GAP_FRAMES = 15  # 0.3 s of silence still inside one region
WINDOW_FRAMES = 75  # 1.5 s windows
MIN_WINDOW_FRAMES = 25  # 0.5 s of speech for a window embedding
N_SPEAKERS = 2
DER_STEP_MS = 10
DER_FIELDS = ["audio", "ref_speakers", "ref_seconds", "missed", "false_alarm", "confusion", "der"]


def speech_regions(flags: NDArray[np.bool_], gap: int = GAP_FRAMES) -> list[tuple[int, int]]:
    """Frame ranges [start, end) of speech, bridging silences of at most gap frames."""
    regions: list[list[int]] = []
    for i in np.flatnonzero(flags):
        if regions and i - regions[-1][1] <= gap:
            regions[-1][1] = i + 1
        else:
            regions.append([i, i + 1])
    return [(a, b) for a, b in regions]


def windows(regions: list[tuple[int, int]], size: int = WINDOW_FRAMES) -> list[tuple[int, int]]:
    """Each region cut into the fewest equal windows of at most size frames."""
    out = []
    for a, b in regions:
        n = math.ceil((b - a) / size)
        edges = [a + round(k * (b - a) / n) for k in range(n + 1)]
        out += list(zip(edges, edges[1:], strict=False))
    return out


def cluster(embeddings: NDArray, n_speakers: int = N_SPEAKERS) -> NDArray[np.int_]:
    """A group per window, numbered by first appearance: the first window is speaker 0."""
    if len(embeddings) < 2:
        return np.zeros(len(embeddings), int)
    tree = linkage(embeddings, method="ward")
    groups = fcluster(tree, n_speakers, criterion="maxclust")
    order = {g: k for k, g in enumerate(dict.fromkeys(groups))}
    return np.array([order[g] for g in groups])


Labelled = list[tuple[float, float, int]]  # (start_ms, end_ms, speaker) per window, in order


def turns(labelled: Labelled) -> dict[int, list[tuple[float, float]]]:
    """Neighbouring windows of one speaker -> one turn, gaps included, per speaker."""
    return by_speaker(labelled, bridge=True)


def speech_of(labelled: Labelled) -> dict[int, list[tuple[float, float]]]:
    """Each speaker's windows, touching ones joined; the gaps between them left out."""
    return by_speaker(labelled, bridge=False)


def by_speaker(labelled: Labelled, bridge: bool) -> dict[int, list[tuple[float, float]]]:
    out: dict[int, list[list[float]]] = {}
    previous = None
    for start, end, label in labelled:
        last = out.get(label, [[None, None]])[-1]
        if label == previous and (bridge or last[1] == start):
            last[1] = end
        else:
            out.setdefault(label, []).append([start, end])
        previous = label
    return {int(k): [(s, e) for s, e in v] for k, v in sorted(out.items())}


def diarize(embedder, audio: NDArray, rate: int, n_speakers: int = N_SPEAKERS) -> Labelled:
    """Mono audio -> its speech windows with a speaker each; ends clipped to the audio."""
    divisor = math.gcd(rate, SAMPLE_RATE)
    if rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)
    flags = speech_frames(audio)
    kept, vectors = [], []
    for a, b in windows(speech_regions(flags)):
        speech = np.flatnonzero(flags[a:b]) + a
        if len(speech) < MIN_WINDOW_FRAMES:
            continue
        frames = [audio[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES] for i in speech]
        samples = np.concatenate(frames)
        kept.append((a, b))
        vectors.append(embedder.embed(samples.astype(np.float32)))
    if not kept:
        return []
    length_ms = len(audio) * 1000 / SAMPLE_RATE
    labels = cluster(np.stack(vectors), n_speakers)
    return [
        (a * FRAME_MS, min(b * FRAME_MS, length_ms), int(k))
        for (a, b), k in zip(kept, labels, strict=True)
    ]


def frames_of(segments: list[tuple[float, float]], n: int, step: int = DER_STEP_MS) -> NDArray:
    mask = np.zeros(n, bool)
    for s, e in segments:
        mask[round(s / step) : round(e / step)] = True
    return mask


def der(reference: dict, hypothesis: dict, step: int = DER_STEP_MS) -> dict:
    """Missed, false alarm and confusion in seconds; der is their sum over reference speech."""
    ends = [e for segs in (*reference.values(), *hypothesis.values()) for _, e in segs]
    n = round(max(ends, default=0) / step) + 1
    ref = np.array([frames_of(v, n, step) for v in reference.values()]).reshape(-1, n)
    hyp = np.array([frames_of(v, n, step) for v in hypothesis.values()]).reshape(-1, n)
    overlap = ref.astype(int) @ hyp.T.astype(int)
    rows, cols = linear_sum_assignment(-overlap) if overlap.size else ([], [])
    n_ref, n_hyp = ref.sum(0), hyp.sum(0)
    correct = np.zeros(n, int)
    for r, h in zip(rows, cols, strict=True):
        correct += ref[r] & hyp[h]
    seconds = step / 1000
    result = {
        "ref_seconds": n_ref.sum() * seconds,
        "missed": np.maximum(n_ref - n_hyp, 0).sum() * seconds,
        "false_alarm": np.maximum(n_hyp - n_ref, 0).sum() * seconds,
        "confusion": (np.minimum(n_ref, n_hyp) - correct).sum() * seconds,
    }
    errors = result["missed"] + result["false_alarm"] + result["confusion"]
    result["der"] = errors / result["ref_seconds"] if result["ref_seconds"] else float("nan")
    return result


def diarized_manifest(embedder, root: Path, files: list[dict]) -> tuple[list[dict], dict]:
    """Mono files diarized, an item per speaker; stereo files an item per channel."""
    rows, found = [], {}
    for item in default_manifest(files):
        if str(item["channel"]):
            rows.append(item)
            continue
        audio, rate = load_audio(root / item["audio"])
        found[item["audio"]] = diarize(embedder, audio, rate)
        for k, segments in turns(found[item["audio"]]).items():
            rows.append(
                {
                    "item_id": f"{item['audio']}#spk{k}",
                    "audio": item["audio"],
                    "channel": "",
                    "segments": segments_text(segments),
                    "speaker": "",
                }
            )
        if len(found) % 10 == 0:
            print(f"diarized {len(found)} mono files", flush=True)
    return rows, found


def der_rows(found: dict, reference: dict) -> list[dict]:
    rows = []
    for audio, hypothesis in found.items():
        ref = reference[audio]["speakers"]
        r = der(ref, speech_of(hypothesis))
        rows.append({"audio": audio, "ref_speakers": len(ref), **r})
    return rows


def der_summary(rows: list[dict]) -> list[str]:
    """Pooled DER over all files, then by the reference speaker count (2 or more)."""
    lines = []
    for name, part in (
        ("all", rows),
        ("2 speakers", [r for r in rows if r["ref_speakers"] == 2]),
        ("3+ speakers", [r for r in rows if r["ref_speakers"] > 2]),
    ):
        ref = sum(r["ref_seconds"] for r in part)
        if not ref:
            continue
        shares = [sum(r[k] for r in part) / ref for k in ("missed", "false_alarm", "confusion")]
        lines.append(
            f"{name}: {len(part)} files, {ref / 3600:.2f} h reference speech, DER {sum(shares):.1%}"
            f" (missed {shares[0]:.1%}, false alarm {shares[1]:.1%}, confusion {shares[2]:.1%})"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Diarize the mono recordings of a slice")
    parser.add_argument("root", type=Path, help="the folder of recordings")
    parser.add_argument("--out", type=Path, default=OUT, help="holds files.tsv (slice-inspect)")
    parser.add_argument("--export", type=Path, help="the labelling tool's JSON export")
    args = parser.parse_args()
    files = read_tsv(args.out / "files.tsv")
    rows, found = diarized_manifest(Embedder(), args.root, files)
    write_tsv(args.out / "manifest-diarized.tsv", MANIFEST_FIELDS, rows)
    print(f"{len(found)} mono files diarized, {len(rows)} items")
    print(f"-> {args.out}/manifest-diarized.tsv")
    if args.export is None:
        return
    reference = read_export(args.export, files, args.root)
    write_tsv(args.out / "manifest-reference.tsv", MANIFEST_FIELDS, reference_manifest(reference))
    scored = der_rows(found, reference)
    formatted = [
        {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in r.items()} for r in scored
    ]
    write_tsv(args.out / "der.tsv", DER_FIELDS, formatted)
    print("\n".join(der_summary(scored)))
    print(f"-> {args.out}/manifest-reference.tsv, {args.out}/der.tsv")


if __name__ == "__main__":
    main()
