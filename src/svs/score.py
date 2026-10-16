"""Score trials: cosine similarity between the enrol and test embeddings."""

import numpy as np
from numpy.typing import NDArray


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
