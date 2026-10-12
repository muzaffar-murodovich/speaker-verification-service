"""Embed every VoxCeleb1 clip once and cache the result, one .npz file per speaker."""

import sys
import time
from itertools import groupby
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from svs.embed import EMBEDDING_DIM, Embedder
from svs.extract import extract_speech
from svs.trials import MIN_SPEECH_SECONDS
from svs.voxceleb import list_clips


def embed_clip(embedder: Embedder, path: str | Path) -> tuple[float, NDArray[np.float32]]:
    """Net speech seconds for the whole file, and its embedding (NaN below the 3 s minimum).

    The service refuses to score less than MIN_SPEECH_SECONDS of speech, so no embedding
    is computed for it. NaN makes any accidental use of the row visible as a NaN score.
    """
    info = sf.info(str(path))
    speech = extract_speech(path, [(0, info.frames * 1000 / info.samplerate)])
    if speech.net_speech_seconds < MIN_SPEECH_SECONDS:
        return speech.net_speech_seconds, np.full(EMBEDDING_DIM, np.nan, np.float32)
    return speech.net_speech_seconds, embedder.embed(speech.samples)


def embed_all(
    embedder: Embedder, wav_dir: str | Path, meta_csv: str | Path, out_dir: str | Path
) -> int:
    """Write <out_dir>/<speaker>.npz for each speaker without a file; return how many.

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
        results = [embed_clip(embedder, wav_dir / f"{c.utt_id}.wav") for c in clips]
        np.savez(
            path,
            utt_ids=np.array([c.utt_id for c in clips]),
            net_seconds=np.array([seconds for seconds, _ in results]),
            embeddings=np.stack([embedding for _, embedding in results]),
        )
        written += 1
        print(f"{speaker}: {len(clips)} clips", flush=True)
    return written


if __name__ == "__main__":
    wav_dir, meta_csv, out_dir = sys.argv[1:]
    start = time.perf_counter()
    written = embed_all(Embedder(), wav_dir, meta_csv, out_dir)
    print(f"Wrote {written} speaker files in {time.perf_counter() - start:.0f} s")
