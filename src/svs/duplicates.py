"""Find VoxCeleb1 videos of one speaker that hold the same audio under two video ids.

Some interviews were uploaded twice. The same-session guard compares video ids, so a
target trial across two such videos passes it although both sides are one session.
"""

import argparse
import csv
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import correlate

# Samples of cross-video clip pairs of one speaker found shared audio in 4.1 % at
# cosine 0.90-0.95 and in 0 of 500 at 0.85-0.90. The full run at 0.85 still confirmed
# 58 video pairs through the 0.85-0.90 band only (LOG.md 2026-10-19), so some pairs
# below 0.85 are likely missed. Kept: those 58 touch at most 0.07 % of a bucket's
# targets, and 0.80 would cost about 1.5 h more.
MIN_COSINE = 0.85
# 500 random cross-video pairs, none of them shared audio, reached at most 0.131.
MIN_SHARED = 0.2


def video_of(utt_id: str) -> str:
    """<speaker>/<video> of a <speaker>/<video>/<clip> id."""
    return utt_id.rsplit("/", 1)[0]


def shared_audio(x: NDArray, y: NDArray) -> float:
    """Normalised cross-correlation at the best lag: 1 for the same audio, near 0 otherwise.

    Every lag is tried, so a copy cut a little earlier or later still scores high.
    abs() also catches a copy with flipped polarity.
    """
    peak = np.abs(correlate(x, y, method="fft")).max()
    return float(peak / (np.linalg.norm(x) * np.linalg.norm(y)))


def candidate_pairs(
    utt_ids: NDArray[np.str_], embeddings: NDArray[np.float32], min_cosine: float
) -> list[tuple[str, str, float]]:
    """Clip pairs of one speaker from two different videos with cosine >= min_cosine.

    Highest cosine first, so the likeliest copy of a video pair is read first. A clip
    with a NaN embedding (under 3 s of speech) has a NaN cosine, which never passes.
    """
    embeddings = embeddings.astype(np.float64)
    i, j = np.triu_indices(len(utt_ids), 1)
    cosine = np.einsum("ij,ij->i", embeddings[i], embeddings[j])
    videos = np.array([video_of(u) for u in utt_ids])
    keep = (videos[i] != videos[j]) & (cosine >= min_cosine)
    order = np.argsort(-cosine[keep], kind="stable")
    i, j, cosine = i[keep][order], j[keep][order], cosine[keep][order]
    return [(utt_ids[a], utt_ids[b], float(c)) for a, b, c in zip(i, j, cosine, strict=True)]


def confirm_video_pairs(
    candidates: Iterable[tuple[str, str, float]],
    shared: Callable[[str, str], float],
    min_shared: float,
) -> list[tuple[str, str, float, float]]:
    """One confirmed clip pair (clip_a, clip_b, cosine, shared) per video pair.

    One clip of shared audio makes two videos one session, so once a video pair is
    confirmed its other clip pairs are not checked.
    """
    confirmed, done = [], set()
    for a, b, cosine in candidates:
        videos = frozenset((video_of(a), video_of(b)))
        if videos in done:
            continue
        score = shared(a, b)
        if score >= min_shared:
            confirmed.append((a, b, cosine, score))
            done.add(videos)
    return confirmed


def merge_sessions(video_pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Session of each <speaker>/<video> in a pair: the first video id of its group.

    Videos linked by a chain of pairs (A-B, B-C) form one group, so A and C share a
    session although no pair names them together. Taking the first id in sorted order
    makes the result independent of the pair order.
    """
    parent: dict[str, str] = {}

    def root(video: str) -> str:
        while parent.setdefault(video, video) != video:
            video = parent[video]
        return video

    for a, b in video_pairs:
        ra, rb = root(a), root(b)
        parent[max(ra, rb)] = min(ra, rb)
    return {video: root(video).split("/")[1] for video in parent}


def read_video_pairs(path: str | Path) -> list[tuple[str, str]]:
    """(video_a, video_b) of every row of a file written by this module."""
    with open(path, newline="") as f:
        return [(r["video_a"], r["video_b"]) for r in csv.DictReader(f, delimiter="\t")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_dir")
    parser.add_argument("clip_cache")
    parser.add_argument("out_tsv")
    args = parser.parse_args()

    def shared(a: str, b: str) -> float:
        x, _ = sf.read(f"{args.wav_dir}/{a}.wav")
        y, _ = sf.read(f"{args.wav_dir}/{b}.wav")
        return shared_audio(x, y)

    start, n_checked, rows = time.perf_counter(), 0, []
    for path in sorted(Path(args.clip_cache).glob("*.npz")):
        cache = np.load(path)
        candidates = candidate_pairs(cache["utt_ids"], cache["embeddings"], MIN_COSINE)
        n_checked += len(candidates)
        rows += confirm_video_pairs(candidates, shared, MIN_SHARED)
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_tsv, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["video_a", "video_b", "clip_a", "clip_b", "cosine", "shared"])
        for a, b, cosine, score in rows:
            writer.writerow([video_of(a), video_of(b), a, b, f"{cosine:.6f}", f"{score:.6f}"])
    print(f"{n_checked} candidate clip pairs at cosine >= {MIN_COSINE}")
    print(f"Wrote {len(rows)} video pairs to {args.out_tsv}")
    print(f"{time.perf_counter() - start:.0f} s")
