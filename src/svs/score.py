"""Score trials: cosine similarity between the enrol and test embeddings."""

import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from svs.embed_cache import read_cache


def cosine_scores(enrol: NDArray[np.float32], test: NDArray[np.float32]) -> NDArray[np.float64]:
    """Cosine similarity of each enrol row with the test row at the same index.

    The embedder already returns unit-length rows, so this equals the dot product.
    Dividing by the norms anyway keeps the score a true cosine for any input.
    A NaN embedding marks speech below the 3 s minimum; scoring it is a bug, so it raises.
    """
    enrol, test = np.asarray(enrol, np.float64), np.asarray(test, np.float64)
    scores = np.einsum("ij,ij->i", enrol, test)
    scores /= np.linalg.norm(enrol, axis=1) * np.linalg.norm(test, axis=1)
    if np.isnan(scores).any():
        raise ValueError(f"{np.isnan(scores).sum()} trials have a NaN embedding")
    return scores


def load_embeddings(cache_dir: str | Path) -> dict[str, NDArray[np.float32]]:
    """Embedding of every utt_id in an embedding cache."""
    utt_ids, _, embeddings = read_cache(cache_dir)
    return dict(zip(utt_ids.tolist(), embeddings, strict=True))


def score_trials(
    rows: list[dict[str, str]], caches: dict[str, dict[str, NDArray[np.float32]]]
) -> NDArray[np.float64]:
    """Cosine score of each trial row; its source column picks the cache for both sides.

    A clip id and its video id never share a cache, so a lookup in the wrong cache
    raises KeyError instead of scoring other audio.
    """
    enrol = np.stack([caches[r["source"]][r["enrol"]] for r in rows])
    test = np.stack([caches[r["source"]][r["test"]] for r in rows])
    return cosine_scores(enrol, test)


def write_scores(path: str | Path, rows: list[dict[str, str]], scores: NDArray) -> None:
    """The trial rows with a score column added, in the same order."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow([*rows[0], "score"])
        for row, score in zip(rows, scores, strict=True):
            # repr keeps every digit, so no rounding can create ties between scores.
            writer.writerow([*row.values(), repr(float(score))])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trials_tsv")
    parser.add_argument("clip_cache")
    parser.add_argument("video_cache")
    parser.add_argument("out_tsv")
    args = parser.parse_args()
    with open(args.trials_tsv, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    caches = {"clip": load_embeddings(args.clip_cache), "video": load_embeddings(args.video_cache)}
    scores = score_trials(rows, caches)
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    write_scores(args.out_tsv, rows, scores)
    print(f"Wrote {len(scores)} scores to {args.out_tsv}")
