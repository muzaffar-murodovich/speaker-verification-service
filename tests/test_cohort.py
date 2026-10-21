"""Cohort disjointness: a cohort speaker who is also a trial speaker is refused."""

import pytest

from svs.cohort import assert_disjoint, vox1_names, vox2_dev_names

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
