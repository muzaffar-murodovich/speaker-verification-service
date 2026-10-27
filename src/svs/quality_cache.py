"""Quality of every VoxCeleb1 clip and video under one condition, in one TSV.

The audio goes through the same steps as svs.embed_cache (degrade, then VAD per clip),
so a quality row describes exactly the speech its embedding was made from. A video is
the clips of one session joined; its turns are its clips, and its turn consistency is
the mean cosine between their cached embeddings. A clip is one turn.
"""

import argparse
import csv
import time
from itertools import groupby
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from svs.degrade import CONDITIONS, degrade
from svs.embed_cache import read_cache
from svs.extract import SAMPLE_RATE, ExtractedSpeech, load_audio, speech_from_samples
from svs.quality import QUALITY_VERSION, quality, quality_detail, turn_consistency
from svs.voxceleb import list_clips

COLUMNS = (
    "unit",
    "source",
    "net_seconds",
    "snr_db",
    "speech_ratio",
    "clipping_ratio",
    "bandwidth_hz",
    "turn_consistency",
    "quality",
)


def row(unit: str, source: str, speech: ExtractedSpeech, detail: dict[str, float]) -> dict:
    return {
        "unit": unit,
        "source": source,
        "net_seconds": f"{speech.net_speech_seconds:.2f}",
        **{key: f"{value:.6g}" for key, value in detail.items()},
        "quality": f"{quality(**detail):.4f}",
    }


def speaker_rows(
    wav_dir: Path, utt_ids: list[str], condition: str, clip_embeddings: dict[str, np.ndarray]
) -> list[dict]:
    """Rows of one speaker's clips, then of its videos."""
    rows, clips = [], {}
    for utt_id in utt_ids:
        audio, rate = load_audio(wav_dir / f"{utt_id}.wav")
        if rate != SAMPLE_RATE:
            raise ValueError(f"{utt_id} is {rate} Hz; the degradation harness expects 16 kHz")
        audio, rate = degrade(audio, condition, noise_key=utt_id)
        speech = speech_from_samples(audio, rate, [(0, len(audio) * 1000 / rate)])
        clips[utt_id] = (audio, speech)
        rows.append(row(utt_id, "clip", speech, quality_detail(audio, rate, speech)))
    for video, group in groupby(utt_ids, key=lambda u: u.rsplit("/", 1)[0]):
        members = list(group)
        audio = np.concatenate([clips[u][0] for u in members])
        speech = ExtractedSpeech(
            np.concatenate([clips[u][1].samples for u in members]),
            sum(clips[u][1].selected_seconds for u in members),
        )
        # Clips under 3.0 s have a NaN embedding (svs.embed_cache) and are not turns.
        turns = [clip_embeddings[u] for u in members if np.isfinite(clip_embeddings[u]).all()]
        detail = quality_detail(audio, rate, speech, turn_consistency(turns))
        rows.append(row(video, "video", speech, detail))
    return rows


def _speaker_job(args):
    return speaker_rows(*args)


def quality_table(
    wav_dir: str | Path,
    meta_csv: str | Path,
    clip_cache: str | Path,
    condition: str,
    out_tsv: str | Path,
    workers: int,
) -> int:
    """Write the quality TSV of one condition; return its row count."""
    wav_dir = Path(wav_dir)
    utt_ids, _, embeddings = read_cache(clip_cache)
    clip_embeddings = dict(zip(utt_ids.tolist(), embeddings, strict=True))
    # list_clips sorts by utt_id, so a speaker's clips, and a video's, are contiguous.
    speakers = groupby(list_clips(wav_dir, meta_csv), key=lambda c: c.speaker)
    jobs = []
    for _, group in speakers:
        ids = [c.utt_id for c in group]
        jobs.append((wav_dir, ids, condition, {u: clip_embeddings[u] for u in ids}))
    with Pool(workers) as pool:
        rows = [r for rows in pool.imap(_speaker_job, jobs, chunksize=4) for r in rows]
    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", newline="") as f:
        f.write(f"# {QUALITY_VERSION} condition={condition}\n")
        writer = csv.DictWriter(f, COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_quality(path: str | Path) -> dict[tuple[str, str], dict[str, float]]:
    """(source, unit) -> numeric columns of a quality TSV."""
    with open(path, newline="") as f:
        lines = (line for line in f if not line.startswith("#"))
        return {
            (r["source"], r["unit"]): {k: float(v) for k, v in r.items() if k in COLUMNS[2:]}
            for r in csv.DictReader(lines, delimiter="\t")
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_dir")
    parser.add_argument("meta_csv")
    parser.add_argument("clip_cache", help="the condition's clip embeddings, for turn consistency")
    parser.add_argument("out_tsv")
    parser.add_argument("--condition", choices=CONDITIONS, default="pcm16k")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    start = time.perf_counter()
    n = quality_table(
        args.wav_dir, args.meta_csv, args.clip_cache, args.condition, args.out_tsv, args.workers
    )
    print(f"Wrote {n} rows to {args.out_tsv} in {time.perf_counter() - start:.0f} s")
