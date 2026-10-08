import numpy as np
import pytest

from svs.embed import EMBEDDING_DIM, MODEL_PATH, Embedder, compute_fbank

needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `make model` first")


def noise(seconds: float) -> np.ndarray:
    return np.random.default_rng(0).normal(0, 0.1, int(seconds * 16000)).astype(np.float32)


def test_fbank_has_80_bins_and_10_ms_frames():
    # 1 s at 16 kHz: 1 + (16000 - 400) // 160 = 98 frames of 25 ms every 10 ms.
    assert compute_fbank(noise(1.0)).shape == (98, 80)


def test_fbank_is_mean_normalised():
    assert np.allclose(compute_fbank(noise(1.0)).mean(axis=0), 0, atol=1e-4)


def test_fbank_rejects_audio_shorter_than_one_frame():
    with pytest.raises(ValueError, match="shorter than one"):
        compute_fbank(noise(0.02))


@needs_model
def test_embedding_is_192_dim_and_l2_normalised():
    embedding = Embedder().embed(noise(2.0))
    assert embedding.shape == (EMBEDDING_DIM,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)
