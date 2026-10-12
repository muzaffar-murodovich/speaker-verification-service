"""Embedding cache: short clips get no embedding, and a rerun resumes instead of redoing work."""

import numpy as np
import pytest
import soundfile as sf

from svs.embed import EMBEDDING_DIM
from svs.embed_cache import embed_all

META_HEADER = "VoxCeleb1 ID\tVGGFace1 ID\tGender\tNationality\tSet\n"


class FakeEmbedder:
    """Stands in for the ONNX model and records how many clips it was given."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls = 0
        self.fail_after = fail_after

    def embed(self, samples):
        if self.calls == self.fail_after:
            raise RuntimeError("interrupted")
        self.calls += 1
        return np.ones(EMBEDDING_DIM, np.float32) / np.sqrt(EMBEDDING_DIM)


def make_tree(root, clips):
    """WebRTC VAD keeps loud noise as speech and drops digital silence."""
    for rel, kind in clips.items():
        path = root / "wav" / f"{rel}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        noise = np.random.default_rng(0).normal(0, 0.1, 4 * 16000)
        sf.write(path, noise if kind == "speech" else 0 * noise, 16000, subtype="PCM_16")
    speakers = sorted({rel.split("/")[0] for rel in clips})
    meta = root / "vox1_meta.csv"
    meta.write_text(META_HEADER + "".join(f"{s}\tName\tm\tX\tdev\n" for s in speakers))
    return root / "wav", meta


def test_clip_below_minimum_keeps_its_seconds_but_gets_no_embedding(tmp_path):
    wav, meta = make_tree(tmp_path, {"id1/v1/00001": "speech", "id1/v1/00002": "silence"})
    embedder = FakeEmbedder()
    embed_all(embedder, wav, meta, tmp_path / "out")

    cache = np.load(tmp_path / "out" / "id1.npz")
    assert list(cache["utt_ids"]) == ["id1/v1/00001", "id1/v1/00002"]
    np.testing.assert_allclose(cache["net_seconds"], [4.0, 0.0])
    assert np.isfinite(cache["embeddings"][0]).all()
    assert np.isnan(cache["embeddings"][1]).all()
    assert embedder.calls == 1


def test_rerun_skips_speakers_that_already_have_a_file(tmp_path):
    wav, meta = make_tree(tmp_path, {"id1/v1/00001": "speech", "id2/v1/00001": "speech"})
    out = tmp_path / "out"
    out.mkdir()
    (out / "id1.npz").write_bytes(b"done earlier")
    embedder = FakeEmbedder()

    assert embed_all(embedder, wav, meta, out) == 1
    assert (out / "id1.npz").read_bytes() == b"done earlier"
    assert embedder.calls == 1


def test_interrupted_speaker_leaves_no_file(tmp_path):
    wav, meta = make_tree(tmp_path, {"id1/v1/00001": "speech", "id1/v1/00002": "speech"})
    with pytest.raises(RuntimeError):
        embed_all(FakeEmbedder(fail_after=1), wav, meta, tmp_path / "out")
    assert not (tmp_path / "out" / "id1.npz").exists()
