"""Cohort disjointness: a cohort speaker who is also a trial speaker is refused."""

import subprocess

import numpy as np
import pytest
import soundfile as sf

from svs.cohort import (
    assert_disjoint,
    choose_members,
    crc_key,
    embed_cohort,
    save_wav,
    vox1_names,
    vox2_dev_names,
    wav_path,
)
from svs.embed import EMBEDDING_DIM

VOX1_HEADER = "VoxCeleb1 ID\tVGGFace1 ID\tGender\tNationality\tSet\n"
VOX2_HEADER = "﻿VoxCeleb2 ID ,VGGFace2 ID ,Gender ,Set \n"
IDENTITY_HEADER = "Class_ID, Name, Sample_Num, Flag, Gender\n"


def test_same_person_spelled_differently_is_caught():
    # Real pair: VoxCeleb1 id10123 and VoxCeleb2 id01333 are one person.
    with pytest.raises(ValueError, match="id01333 Candace_Cameron-Bure = id10123"):
        assert_disjoint({"id01333": "Candace_Cameron-Bure"}, {"id10123": "Candace_Cameron_Bure"})


def test_different_people_pass():
    assert_disjoint({"id00012": "Aaron_Ashmore"}, {"id10001": "A.J._Buckley"})


def test_names_come_from_the_files_and_only_dev_is_kept(tmp_path):
    vox1 = tmp_path / "vox1_meta.csv"
    vox1.write_text(VOX1_HEADER + "id10123\tCandace_Cameron_Bure\tf\tUSA\tdev\n")
    vox2 = tmp_path / "vox2_meta.csv"
    vox2.write_text(VOX2_HEADER + "id00012 ,n000012 ,m ,dev \nid01333 ,n001333 ,f ,test \n")
    identity = tmp_path / "identity_meta.csv"
    identity.write_text(
        IDENTITY_HEADER
        + 'n000012, "Aaron_Ashmore", 1, 1, m\nn001333, "Candace_Cameron-Bure", 1, 1, f\n'
    )

    assert vox1_names(vox1) == {"id10123": "Candace_Cameron_Bure"}
    assert vox2_dev_names(vox2, identity) == {"id00012": "Aaron_Ashmore"}


def test_dev_speaker_without_a_name_is_refused(tmp_path):
    vox2 = tmp_path / "vox2_meta.csv"
    vox2.write_text(VOX2_HEADER + "id00012 ,n000012 ,m ,dev \n")
    identity = tmp_path / "identity_meta.csv"
    identity.write_text(IDENTITY_HEADER)
    with pytest.raises(ValueError, match="No name for id00012"):
        vox2_dev_names(vox2, identity)


def test_one_clip_per_speaker_in_a_fixed_order():
    names = [
        "dev/aac/id1/v1/00001.m4a",
        "dev/aac/id1/v2/00001.m4a",
        "dev/aac/id2/v1/00001.m4a",
        "dev/aac/id3/v1/00001.m4a",  # not a cohort speaker
        "dev/aac/id1/",  # directory entry
    ]
    chosen = choose_members(names, {"id1", "id2"}, count=2)
    assert sorted(m.split("/")[2] for m in chosen) == ["id1", "id2"]
    assert chosen == choose_members(reversed(names), {"id1", "id2"}, count=2)
    first_id1 = min(names[:2], key=crc_key)
    assert first_id1 in chosen


def test_too_few_speakers_is_refused():
    with pytest.raises(ValueError, match="3 speakers asked, 1 have clips"):
        choose_members(["dev/aac/id1/v1/00001.m4a"], {"id1"}, count=3)


def test_m4a_is_saved_as_16khz_mono_wav(tmp_path):
    tone = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)
    sf.write(tmp_path / "in.wav", stereo, 44100)
    m4a = tmp_path / "in.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-i", tmp_path / "in.wav", m4a], check=True)

    out = wav_path(tmp_path / "wav", "dev/aac/id1/v1/00001.m4a")
    save_wav(m4a.read_bytes(), out)
    info = sf.info(out)
    assert out == tmp_path / "wav" / "id1" / "v1" / "00001.wav"
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert info.duration == pytest.approx(1.0, abs=0.05)
    assert not out.with_suffix(".wav.part").exists()


class FakeEmbedder:
    def embed(self, samples):
        return np.ones(EMBEDDING_DIM, np.float32) / np.sqrt(EMBEDDING_DIM)


def test_clip_under_3s_of_speech_gives_its_place_to_the_next_speaker(tmp_path):
    noise = np.random.default_rng(0).normal(0, 0.1, 4 * 16000)  # VAD keeps loud noise
    members = ["dev/aac/id1/v/1.m4a", "dev/aac/id2/v/1.m4a", "dev/aac/id3/v/1.m4a"]
    for member, audio in zip(members, [noise, 0 * noise, noise], strict=True):
        path = wav_path(tmp_path, member)
        path.parent.mkdir(parents=True)
        sf.write(path, audio, 16000, subtype="PCM_16")

    kept, seconds, embeddings = embed_cohort(FakeEmbedder(), tmp_path, members, size=2)
    assert kept == [members[0], members[2]]
    np.testing.assert_allclose(seconds, [4.0, 4.0])
    assert embeddings.shape == (2, EMBEDDING_DIM)
    with pytest.raises(ValueError, match="Only 2 of 3"):
        embed_cohort(FakeEmbedder(), tmp_path, members, size=3)
