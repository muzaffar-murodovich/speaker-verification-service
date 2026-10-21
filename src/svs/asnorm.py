"""Adaptive symmetric score normalisation (AS-Norm) against a fixed impostor cohort."""

import numpy as np
from numpy.typing import NDArray

# SVS guide §4.3 and CFG_SV_COHORT_TOP_N; the handbook names top-n but gives no value.
TOP_N = 300
# Rows scored against the cohort at once: 4096 x 5000 float64 scores is 164 MB.
BATCH = 4096


def top_n_stats(cohort_scores: NDArray, n: int = TOP_N) -> tuple[NDArray, NDArray]:
    """Mean and standard deviation of the n highest cohort scores of each row.

    Only the top n count: they come from the impostors closest to this voice, which
    are the ones that cause false accepts. Low scores from far voices would pull the
    mean down and make an ordinary score look strong.
    """
    cohort_scores = np.asarray(cohort_scores, np.float64)
    if n > cohort_scores.shape[1]:
        raise ValueError(f"top n = {n} but the cohort has only {cohort_scores.shape[1]} scores")
    top = np.partition(cohort_scores, -n, axis=1)[:, -n:]
    return top.mean(axis=1), top.std(axis=1)


def cohort_stats(embeddings: NDArray, cohort: NDArray, n: int = TOP_N) -> tuple[NDArray, NDArray]:
    """Top-n cohort mean and standard deviation of each embedding, computed once.

    Each audio takes part in many trials, so its statistics are computed here once and
    looked up per trial (handbook D8, SVS guide §4.3: cache the profile-side statistics).
    A NaN embedding (under 3 s of speech) gets NaN statistics.
    """
    embeddings = np.asarray(embeddings, np.float64)
    cohort = np.asarray(cohort, np.float64)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cohort = cohort / np.linalg.norm(cohort, axis=1, keepdims=True)
    parts = [
        top_n_stats(embeddings[i : i + BATCH] @ cohort.T, n)
        for i in range(0, len(embeddings), BATCH)
    ]
    return np.concatenate([mu for mu, _ in parts]), np.concatenate([sd for _, sd in parts])


def as_norm(
    raw: NDArray, mu_e: NDArray, sd_e: NDArray, mu_f: NDArray, sd_f: NDArray
) -> NDArray[np.float64]:
    """Mean of the two z-scores of the raw cosine, one per side of the trial.

    The 1e-6 keeps a zero spread from dividing by zero, as in the SVS guide §4.3.
    """
    z_e = (raw - mu_e) / (sd_e + 1e-6)
    z_f = (raw - mu_f) / (sd_f + 1e-6)
    return 0.5 * (z_e + z_f)
