"""Embedding cache: short clips get no embedding, and a rerun resumes instead of redoing work."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from svs import embed_cache
from svs.embed import EMBEDDING_DIM
from svs.embed_cache import embed_all, read_cache

META_HEADER = "VoxCeleb1 ID\tVGGFace1 ID\tGender\tNationality\tSet\n"


class FakeEmbedder:
    """Stands in for the ONNX model and records how many clips it was given."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls = 0
        self.fail_after = fail_after
        self.lengths: list[int] = []
        self.powers: list[float] = []

    def embed(self, samples):
        if self.calls == self.fail_after:
            raise RuntimeError("interrupted")
        self.calls += 1
        self.lengths.append(len(samples))
        self.powers.append(float(np.mean(samples.astype(np.float64) ** 2)))
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


def test_stop_while_writing_leaves_no_file_that_a_rerun_would_skip(tmp_path, monkeypatch):
    def half_write(file, **arrays):
        if hasattr(file, "write"):
            file.write(b"half")
        else:
            Path(file).write_bytes(b"half")
        raise RuntimeError("stopped mid-write")

    wav, meta = make_tree(tmp_path, {"id1/v1/00001": "speech"})
    monkeypatch.setattr(embed_cache.np, "savez", half_write)
    with pytest.raises(RuntimeError):
        embed_all(FakeEmbedder(), wav, meta, tmp_path / "out")
    assert not (tmp_path / "out" / "id1.npz").exists()


def test_data_reaches_the_disk_before_the_file_gets_its_final_name(tmp_path, monkeypatch):
    synced_before_rename = []

    def fake_fsync(fd):
        synced_before_rename.append(not (tmp_path / "out" / "id1.npz").exists())

    wav, meta = make_tree(tmp_path, {"id1/v1/00001": "speech"})
    monkeypatch.setattr(embed_cache.os, "fsync", fake_fsync)
    embed_all(FakeEmbedder(), wav, meta, tmp_path / "out")
    assert synced_before_rename == [True]
    assert (tmp_path / "out" / "id1.npz").exists()


def test_read_cache_joins_all_speaker_files_in_order(tmp_path):
    wav, meta = make_tree(
        tmp_path, {"id2/v1/00001": "speech", "id1/v1/00001": "speech", "id1/v1/00002": "silence"}
    )
    embed_all(FakeEmbedder(), wav, meta, tmp_path / "out")

    utt_ids, seconds, embeddings = read_cache(tmp_path / "out")
    assert list(utt_ids) == ["id1/v1/00001", "id1/v1/00002", "id2/v1/00001"]
    np.testing.assert_allclose(seconds, [4.0, 0.0, 4.0])
    assert embeddings.shape == (3, EMBEDDING_DIM)


def test_join_videos_gives_one_embedding_per_video_from_all_its_speech(tmp_path):
    clips = {"id1/v1/00001": "speech", "id1/v1/00002": "speech", "id1/v2/00001": "silence"}
    wav, meta = make_tree(tmp_path, clips)
    embedder = FakeEmbedder()
    embed_all(embedder, wav, meta, tmp_path / "out", join_videos=True)

    cache = np.load(tmp_path / "out" / "id1.npz")
    assert list(cache["utt_ids"]) == ["id1/v1", "id1/v2"]
    np.testing.assert_allclose(cache["net_seconds"], [8.0, 0.0])
    assert embedder.lengths == [8 * 16000]
    assert np.isnan(cache["embeddings"][1]).all()


def test_speech_beyond_the_maximum_is_cut_and_not_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_cache, "MAX_SPEECH_SECONDS", 5.0)
    wav, meta = make_tree(tmp_path, {"id1/v1/00001": "speech", "id1/v1/00002": "speech"})
    embedder = FakeEmbedder()
    embed_all(embedder, wav, meta, tmp_path / "out", join_videos=True)

    assert embedder.lengths == [5 * 16000]
    np.testing.assert_allclose(np.load(tmp_path / "out" / "id1.npz")["net_seconds"], [5.0])


def test_condition_degrades_the_audio_before_it_reaches_the_model(tmp_path):
    wav, meta = make_tree(tmp_path, {"id1/v1/00001": "speech"})
    clean, narrow = FakeEmbedder(), FakeEmbedder()
    embed_all(clean, wav, meta, tmp_path / "clean")
    embed_all(narrow, wav, meta, tmp_path / "narrow", condition="pcm8k")
    # White noise spreads evenly up to 8 kHz; the 8 kHz channel keeps only the half below 4 kHz.
    assert narrow.powers[0] == pytest.approx(clean.powers[0] / 2, rel=0.1)


def test_rejects_audio_that_is_not_16khz(tmp_path):
    path = tmp_path / "clip.wav"
    sf.write(path, np.zeros(8000), 8000)
    with pytest.raises(ValueError, match="16 kHz"):
        embed_cache.embed_clips(FakeEmbedder(), [path])
