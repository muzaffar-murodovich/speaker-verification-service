"""Impostor cohort for AS-Norm: VoxCeleb2 dev speakers, proven disjoint from VoxCeleb1."""

import argparse
import csv
import re
import subprocess
import tempfile
import time
import zlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from svs.embed import Embedder
from svs.embed_cache import embed_clips
from svs.extract import SAMPLE_RATE
from svs.split_zip import SplitZip, hf_token, http_fetch
from svs.trials import MIN_SPEECH_SECONDS

# The cohort is versioned with the model (SVS guide §4.3): a new list is a new version.
COHORT_VERSION = "voxceleb2-dev-v1"
# Handbook D8: 2,000-5,000 is plenty at this scale. The SVS guide's default is 20,000 for
# production; the handbook is the brief here. One clip per speaker, so 5,000 speakers.
COHORT_SIZE = 5000
# Speakers chosen beyond the size, for clips that turn out to have under 3 s of speech.
SPARE = 100
VOX2_URL = (
    "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/"
    "b8a8825c87582831cc217b6ff84a36558ced84d6/vox2"
)
VOX2_PARTS = [f"{VOX2_URL}/vox2_dev_aac_parta{c}" for c in "abcdefgh"]
VOX2_PART_SIZES = [10_737_418_240] * 7 + [2_315_355_528]  # HF API, at the revision above


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


def crc_key(text: str) -> tuple[int, str]:
    """Order by CRC32, as for the noise pick: the same on every machine, unlike hash()."""
    return zlib.crc32(text.encode()), text


def choose_members(names: Iterable[str], speakers: set[str], count: int) -> list[str]:
    """One clip per speaker for the first `count` speakers, both picked in CRC32 order.

    Names look like dev/aac/<speaker>/<video>/<clip>.m4a. One clip per speaker spreads
    the top-n impostors over many voices instead of a few voices with many clips.
    """
    first: dict[str, str] = {}
    for name in names:
        parts = name.split("/")
        if len(parts) != 5 or not name.endswith(".m4a") or parts[2] not in speakers:
            continue
        if parts[2] not in first or crc_key(name) < crc_key(first[parts[2]]):
            first[parts[2]] = name
    if len(first) < count:
        raise ValueError(f"{count} speakers asked, {len(first)} have clips")
    return [first[s] for s in sorted(first, key=crc_key)[:count]]


def wav_path(wav_dir: Path, member: str) -> Path:
    """dev/aac/<speaker>/<video>/<clip>.m4a -> wav_dir/<speaker>/<video>/<clip>.wav."""
    return wav_dir.joinpath(*member.split("/")[2:]).with_suffix(".wav")


def save_wav(m4a: bytes, path: Path) -> None:
    """Decode AAC to 16 kHz mono 16-bit wav, the form the VoxCeleb1 pipeline reads.

    ffmpeg reads the m4a from a file: the index of an mp4 may sit at its end, which a
    pipe cannot seek to. The wav is renamed into place only when complete.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".m4a") as source:
        source.write(m4a)
        source.flush()
        command = ["ffmpeg", "-v", "error", "-i", source.name, "-ac", "1"]
        command += ["-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"]
        wav = subprocess.run(command, capture_output=True, check=True).stdout
    part = path.with_suffix(".wav.part")
    part.write_bytes(wav)
    part.replace(path)


def fetch_wavs(archive: SplitZip, infos: dict, members: list[str], wav_dir: Path) -> None:
    """Download and decode the members that are not on disk yet, 8 at a time.

    Each request waits about 1.5 s for the server; 8 in parallel took 0.23 s per clip.
    """

    def one(member: str) -> None:
        path = wav_path(wav_dir, member)
        if not path.exists():
            save_wav(archive.read_member(infos[member]), path)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, members))


def embed_cohort(
    embedder: Embedder, wav_dir: Path, members: list[str], size: int
) -> tuple[list[str], NDArray[np.float64], NDArray[np.float32]]:
    """The first `size` members, in order, with at least MIN_SPEECH_SECONDS of speech.

    A shorter clip would give an embedding the service itself refuses to score.
    """
    kept, seconds, embeddings = [], [], []
    for member in members:
        net, embedding = embed_clips(embedder, [wav_path(wav_dir, member)])
        if net < MIN_SPEECH_SECONDS:
            continue
        kept.append(member)
        seconds.append(net)
        embeddings.append(embedding)
        if len(kept) == size:
            return kept, np.array(seconds), np.stack(embeddings)
    raise ValueError(f"Only {len(kept)} of {len(members)} clips have {MIN_SPEECH_SECONDS} s")


def write_list(path: Path, members: list[str], names: dict[str, str], infos: dict) -> None:
    """The versioned cohort: which clips, from whom, and their CRC32 in the archive."""
    with open(path, "w", newline="") as f:
        out = csv.writer(f, delimiter="\t", lineterminator="\n")
        out.writerow(["speaker", "name", "member", "crc32"])
        for member in members:
            speaker = member.split("/")[2]
            out.writerow([speaker, names[speaker], member, f"{infos[member].CRC:08x}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vox1_meta")
    parser.add_argument("vox2_meta")
    parser.add_argument("identity_meta")
    parser.add_argument("wav_dir", type=Path)
    parser.add_argument("list_out", type=Path)
    parser.add_argument("embeddings_out", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()

    names = vox2_dev_names(args.vox2_meta, args.identity_meta)
    assert_disjoint(names, vox1_names(args.vox1_meta))
    archive = SplitZip(VOX2_PART_SIZES, http_fetch(VOX2_PARTS, hf_token()))
    infos = {info.filename: info for info in archive.members()}
    members = choose_members(infos, set(names), COHORT_SIZE + SPARE)
    fetch_wavs(archive, infos, members, args.wav_dir)
    kept, seconds, embeddings = embed_cohort(Embedder(), args.wav_dir, members, COHORT_SIZE)

    args.list_out.parent.mkdir(parents=True, exist_ok=True)
    write_list(args.list_out, kept, names, infos)
    np.savez(
        args.embeddings_out, members=np.array(kept), net_seconds=seconds, embeddings=embeddings
    )
    skipped = members.index(kept[-1]) + 1 - len(kept)
    print(f"{COHORT_VERSION}: {len(kept)} speakers, {skipped} clips under 3 s skipped")
    print(f"{time.perf_counter() - start:.0f} s")
