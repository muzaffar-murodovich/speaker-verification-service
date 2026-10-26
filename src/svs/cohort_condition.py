"""The versioned cohort, embedded under a degraded condition.

SVS guide §4.3: the cohort should match the audio it normalises, so each condition gets
the same cohort members degraded the same way as the trials. The member list is the
committed one (cohorts/); only the audio changes.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from svs.cohort import wav_path
from svs.degrade import CONDITIONS
from svs.embed import Embedder
from svs.embed_cache import embed_clips
from svs.trials import MIN_SPEECH_SECONDS


def read_members(list_tsv: str | Path) -> list[str]:
    with open(list_tsv, newline="") as f:
        return [r["member"] for r in csv.DictReader(f, delimiter="\t")]


def embed_condition(
    embedder: Embedder, wav_dir: Path, members: list[str], condition: str
) -> tuple[list[str], NDArray[np.float64], NDArray[np.float32]]:
    """Members with at least MIN_SPEECH_SECONDS of speech after degradation, in order.

    A member that falls below it is left out, not replaced: the service never scores
    such audio, and a replacement would make the cohort differ from its version.
    """
    kept, seconds, embeddings = [], [], []
    for member in members:
        net, embedding = embed_clips(embedder, [wav_path(wav_dir, member)], condition)
        if net >= MIN_SPEECH_SECONDS:
            kept.append(member)
            seconds.append(net)
            embeddings.append(embedding)
    if not kept:
        raise ValueError(f"No cohort member keeps {MIN_SPEECH_SECONDS} s under {condition}")
    return kept, np.array(seconds), np.stack(embeddings)


def save(path: Path, members: list[str], seconds: NDArray, embeddings: NDArray) -> None:
    """Write under .part, fsync, rename: a stop or a power cut leaves no broken .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".npz.part")
    with open(part, "wb") as f:
        np.savez(f, members=np.array(members), net_seconds=seconds, embeddings=embeddings)
        f.flush()
        os.fsync(f.fileno())
    part.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("list_tsv", help="committed cohort list, cohorts/<version>.tsv")
    parser.add_argument("wav_dir", type=Path)
    parser.add_argument("condition", choices=CONDITIONS)
    parser.add_argument("out_npz", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    members = read_members(args.list_tsv)
    kept, seconds, embeddings = embed_condition(Embedder(), args.wav_dir, members, args.condition)
    save(args.out_npz, kept, seconds, embeddings)
    print(f"{args.condition}: {len(kept)} of {len(members)} members kept, ", end="")
    print(f"{len(members) - len(kept)} under {MIN_SPEECH_SECONDS} s left out")
    print(f"{time.perf_counter() - start:.0f} s")
