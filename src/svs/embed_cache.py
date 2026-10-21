"""Embed every VoxCeleb1 clip (or video) once and cache the result, one .npz per speaker."""

import argparse
import time
from itertools import groupby
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from svs.degrade import CONDITIONS, degrade
from svs.embed import EMBEDDING_DIM, Embedder
from svs.extract import SAMPLE_RATE, load_audio, speech_from_samples
from svs.trials import MIN_SPEECH_SECONDS
from svs.voxceleb import list_clips

# Model memory grows with length: 1.67 GB at 300 s, 8.13 GB at 1718 s (Mac, 2026-10-13).
# 300 s keeps the >120s bucket and fits the 7 GB home PC. A plain choice, not tuned.
MAX_SPEECH_SECONDS = 300.0


def embed_clips(
    embedder: Embedder, paths: list[Path], condition: str = "pcm16k"
) -> tuple[float, NDArray[np.float32]]:
    """Pooled net speech of clips from one session, and one embedding for all of it.

    VAD runs per clip, so the gaps between clips never count as speech. Speech beyond
    MAX_SPEECH_SECONDS is cut off; the seconds returned are what the model saw.
    The service refuses to score less than MIN_SPEECH_SECONDS of speech, so no embedding
    is computed for it. NaN makes any accidental use of the row visible as a NaN score.
    Each clip is degraded to the condition before VAD, as the service sees channel audio.
    The clip's utt_id picks its noise, so a clip gets the same noise alone and in a video.
    """
    speech = []
    for path in paths:
        audio, rate = load_audio(path)
        if rate != SAMPLE_RATE:
            raise ValueError(f"{path} is {rate} Hz; the degradation harness expects 16 kHz")
        utt_id = "/".join(path.with_suffix("").parts[-3:])
        audio, rate = degrade(audio, condition, noise_key=utt_id)
        speech.append(speech_from_samples(audio, rate, [(0, len(audio) * 1000 / rate)]).samples)
    samples = np.concatenate(speech)[: int(MAX_SPEECH_SECONDS * SAMPLE_RATE)]
    seconds = len(samples) / SAMPLE_RATE
    if seconds < MIN_SPEECH_SECONDS:
        return seconds, np.full(EMBEDDING_DIM, np.nan, np.float32)
    return seconds, embedder.embed(samples)


def embed_all(
    embedder: Embedder,
    wav_dir: str | Path,
    meta_csv: str | Path,
    out_dir: str | Path,
    join_videos: bool = False,
    condition: str = "pcm16k",
) -> int:
    """Write <out_dir>/<speaker>.npz for each speaker without a file; return how many.

    With join_videos, all clips of a video become one unit with id <speaker>/<video>.
    A video is one session, so joining never mixes sessions.

    A speaker's file is written only after all its clips are embedded, so an interrupted
    run never leaves a partial file, and a rerun continues with the missing speakers.
    """
    wav_dir, out_dir = Path(wav_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    # list_clips sorts by utt_id, which starts with the speaker, so groups are contiguous.
    for speaker, group in groupby(list_clips(wav_dir, meta_csv), key=lambda c: c.speaker):
        path = out_dir / f"{speaker}.npz"
        if path.exists():
            continue
        clips = list(group)
        if join_videos:
            videos = groupby(clips, key=lambda c: c.session)
            units = [(f"{speaker}/{video}", list(unit)) for video, unit in videos]
        else:
            units = [(c.utt_id, [c]) for c in clips]
        results = [
            embed_clips(embedder, [wav_dir / f"{c.utt_id}.wav" for c in unit], condition)
            for _, unit in units
        ]
        # Write under a name read_cache ignores, then rename: a stop mid-write (a run
        # killed on a time limit) leaves only a .part file, never a broken id*.npz.
        part = path.with_suffix(".npz.part")
        with open(part, "wb") as f:
            np.savez(
                f,
                utt_ids=np.array([unit_id for unit_id, _ in units]),
                net_seconds=np.array([seconds for seconds, _ in results]),
                embeddings=np.stack([embedding for _, embedding in results]),
            )
        part.replace(path)
        written += 1
        print(f"{speaker}: {len(units)} {'videos' if join_videos else 'clips'}", flush=True)
    return written


def read_cache(
    cache_dir: str | Path,
) -> tuple[NDArray[np.str_], NDArray[np.float64], NDArray[np.float32]]:
    """utt_ids, net speech seconds and embeddings of every speaker file, in speaker order."""
    files = sorted(Path(cache_dir).glob("*.npz"))
    if not files:
        raise ValueError(f"No speaker files in {cache_dir}")
    parts = [np.load(f) for f in files]
    keys = ("utt_ids", "net_seconds", "embeddings")
    return tuple(np.concatenate([p[key] for p in parts]) for key in keys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_dir")
    parser.add_argument("meta_csv")
    parser.add_argument("out_dir")
    parser.add_argument("--join-videos", action="store_true", help="one embedding per video")
    parser.add_argument("--condition", choices=CONDITIONS, default="pcm16k")
    args = parser.parse_args()
    start = time.perf_counter()
    written = embed_all(
        Embedder(), args.wav_dir, args.meta_csv, args.out_dir, args.join_videos, args.condition
    )
    print(f"Wrote {written} speaker files in {time.perf_counter() - start:.0f} s")
