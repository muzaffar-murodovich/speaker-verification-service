"""The cohort under a condition: same members, degraded audio, short ones left out."""

import numpy as np
import pytest
import soundfile as sf

from svs.cohort import wav_path
from svs.cohort_condition import embed_condition, read_members, save
from svs.embed import EMBEDDING_DIM

MEMBERS = ["dev/aac/id1/v/1.m4a", "dev/aac/id2/v/1.m4a", "dev/aac/id3/v/1.m4a"]


class FakeEmbedder:
    def __init__(self) -> None:
        self.powers: list[float] = []

    def embed(self, samples):
        self.powers.append(float(np.mean(samples.astype(np.float64) ** 2)))
        return np.ones(EMBEDDING_DIM, np.float32) / np.sqrt(EMBEDDING_DIM)


def write_members(root, silent=()):
    noise = np.random.default_rng(0).normal(0, 0.1, 4 * 16000)  # VAD keeps loud noise
    for member in MEMBERS:
        path = wav_path(root, member)
        path.parent.mkdir(parents=True)
        sf.write(path, 0 * noise if member in silent else noise, 16000, subtype="PCM_16")


def test_member_under_3s_is_left_out_and_not_replaced(tmp_path):
    write_members(tmp_path, silent={MEMBERS[1]})
    kept, seconds, embeddings = embed_condition(FakeEmbedder(), tmp_path, MEMBERS, "pcm8k")
    assert kept == [MEMBERS[0], MEMBERS[2]]
    assert len(seconds) == len(embeddings) == 2


def test_condition_degrades_the_cohort_audio(tmp_path):
    write_members(tmp_path)
    clean, narrow = FakeEmbedder(), FakeEmbedder()
    embed_condition(clean, tmp_path, MEMBERS, "pcm16k")
    embed_condition(narrow, tmp_path, MEMBERS, "pcm8k")
    # White noise spreads evenly up to 8 kHz; 8 kHz audio keeps only the half below 4 kHz.
    assert narrow.powers[0] == pytest.approx(clean.powers[0] / 2, rel=0.1)


def test_no_member_left_is_an_error(tmp_path):
    write_members(tmp_path, silent=set(MEMBERS))
    with pytest.raises(ValueError, match="No cohort member"):
        embed_condition(FakeEmbedder(), tmp_path, MEMBERS, "pcm8k")


def test_saved_file_reads_back_and_leaves_no_part_file(tmp_path):
    out = tmp_path / "cohort-g729.npz"
    save(out, MEMBERS[:2], np.array([4.0, 5.0]), np.zeros((2, EMBEDDING_DIM), np.float32))
    with np.load(out) as z:
        assert list(z["members"]) == MEMBERS[:2]
        assert z["embeddings"].shape == (2, EMBEDDING_DIM)
    assert list(tmp_path.iterdir()) == [out]


def test_members_come_from_the_committed_list():
    members = read_members("cohorts/voxceleb2-dev-v1.tsv")
    assert len(members) == 5000
    assert members[0] == "dev/aac/id08731/oV3yhiKgpe4/00119.m4a"
