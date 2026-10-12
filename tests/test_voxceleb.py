"""VoxCeleb1 catalogue: paths become clips, and every speaker must have a known gender."""

import pytest

from svs.voxceleb import Clip, list_clips

META_HEADER = "VoxCeleb1 ID\tVGGFace1 ID\tGender\tNationality\tSet\n"


def make_tree(root, files, meta_rows):
    """Empty wav files are enough: the catalogue never reads audio."""
    for rel in files:
        path = root / "wav" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    meta = root / "vox1_meta.csv"
    meta.write_text(META_HEADER + "".join(f"{s}\tName\t{g}\tX\tdev\n" for s, g in meta_rows))
    return root / "wav", meta


def test_list_clips_reads_speaker_video_and_gender_from_the_tree(tmp_path):
    files = ["id2/v9/00001.wav", "id1/v2/00001.wav", "id1/v1/00001.wav", "id1/v1/notes.txt"]
    wav, meta = make_tree(tmp_path, files, [("id1", "m"), ("id2", "f")])
    assert list_clips(wav, meta) == [
        Clip("id1/v1/00001", speaker="id1", session="v1", gender="m"),
        Clip("id1/v2/00001", speaker="id1", session="v2", gender="m"),
        Clip("id2/v9/00001", speaker="id2", session="v9", gender="f"),
    ]


def test_list_clips_rejects_a_speaker_missing_from_metadata(tmp_path):
    wav, meta = make_tree(tmp_path, ["id1/v1/00001.wav", "id3/v1/00001.wav"], [("id1", "m")])
    with pytest.raises(ValueError, match="id3"):
        list_clips(wav, meta)


def test_list_clips_rejects_an_empty_tree(tmp_path):
    wav, meta = make_tree(tmp_path, [], [("id1", "m")])
    wav.mkdir()
    with pytest.raises(ValueError, match="No wav"):
        list_clips(wav, meta)
