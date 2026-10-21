"""Impostor cohort for AS-Norm: VoxCeleb2 dev speakers, proven disjoint from VoxCeleb1."""

import csv
import re
from pathlib import Path


def name_key(name: str) -> str:
    """Name in one spelling: the two datasets write Candace_Cameron_Bure and
    Candace_Cameron-Bure for the same person, so case, _, - and . do not count."""
    return re.sub(r"[\s_.\-]+", " ", name.lower()).strip()


def vox1_names(vox1_meta: str | Path) -> dict[str, str]:
    """VoxCeleb1 id -> name. The VGGFace1 ID column holds the name."""
    with open(vox1_meta, newline="") as f:
        return {
            row["VoxCeleb1 ID"]: row["VGGFace1 ID"] for row in csv.DictReader(f, delimiter="\t")
        }


def vox2_dev_names(vox2_meta: str | Path, identity_meta: str | Path) -> dict[str, str]:
    """VoxCeleb2 dev id -> name, through the VGGFace2 id; vox2_meta has no names.

    Only the dev set: it has 5,994 speakers against 120 in test, and test holds two
    VoxCeleb1 speakers. Both files pad their cells with spaces, which are stripped.
    """
    with open(identity_meta, newline="", encoding="utf-8-sig") as f:
        rows = csv.reader(f, skipinitialspace=True)
        next(rows)
        names = {row[0].strip(): row[1].strip() for row in rows}
    with open(vox2_meta, newline="", encoding="utf-8-sig") as f:
        rows = [[cell.strip() for cell in row] for row in csv.reader(f)]
    dev = {}
    for vox2_id, vggface2_id, _gender, split in rows[1:]:
        if split != "dev":
            continue
        if vggface2_id not in names:
            raise ValueError(
                f"No name for {vox2_id} ({vggface2_id}); disjointness cannot be checked"
            )
        dev[vox2_id] = names[vggface2_id]
    return dev


def assert_disjoint(cohort: dict[str, str], trials: dict[str, str]) -> None:
    """Refuse a cohort that shares a person with the trial speakers (SVS guide §4.3, §8.1).

    Ids cannot show this: the datasets number speakers separately. Names can.
    """
    trial_keys = {name_key(name): speaker for speaker, name in trials.items()}
    shared = [
        f"{speaker} {name} = {trial_keys[name_key(name)]}"
        for speaker, name in sorted(cohort.items())
        if name_key(name) in trial_keys
    ]
    if shared:
        raise ValueError(f"{len(shared)} cohort speakers are trial speakers: {', '.join(shared)}")
