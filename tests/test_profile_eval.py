"""Evidence rule on cached speaker videos: the gate before the profile, and the intruder."""

import numpy as np

from svs.embed import EMBEDDING_DIM
from svs.profile_eval import evaluate, speaker_videos

QUALITY_HEADER = (
    "# appendix-c-v1 condition=pcm16k\n"
    "unit\tsource\tnet_seconds\tsnr_db\tspeech_ratio\tclipping_ratio\tbandwidth_hz"
    "\tturn_consistency\tquality\n"
)


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def near(base: int, seed: int) -> np.ndarray:
    v = unit(base) + 0.3 * unit(100 + seed)
    return (v / np.linalg.norm(v)).astype(np.float32)


def write_speaker(video_dir, spk, embeddings, seconds):
    utts = [f"{spk}/v{i}" for i in range(len(embeddings))]
    np.savez(video_dir / f"{spk}.npz", utt_ids=utts, net_seconds=seconds, embeddings=embeddings)
    return utts


def test_unusable_videos_never_enter_and_the_intruder_is_counted(tmp_path):
    nan = np.full(EMBEDDING_DIM, np.nan, np.float32)
    a = write_speaker(tmp_path, "id1", [near(1, 1), near(1, 2), near(1, 3), nan], [70, 60, 50, 2])
    # id2: two unrelated directions, cosine 1/sqrt(2) = 0.71 to their mean.
    b = write_speaker(tmp_path, "id2", [unit(4), unit(5)], [80, 70])
    quality = tmp_path / "quality.tsv"
    rows = [(u, s, 1.0) for u, s in zip(a, [70, 60, 50, 2], strict=True)]
    rows[2] = (a[2], 50, 0.3)  # a second voice in the video: turn consistency below 0.55
    rows += [(u, s, 1.0) for u, s in zip(b, [80, 70], strict=True)]
    quality.write_text(
        QUALITY_HEADER
        + "".join(f"{u}\tvideo\t{s}\t20\t0.9\t0\t8000\t{c}\t0.8\n" for u, s, c in rows)
    )
    speakers, refused = speaker_videos(tmp_path, quality)
    assert [o.recording_id for o in speakers["id1"]] == ["id1/v0", "id1/v1"]
    assert refused == {"E_SV_INSUFFICIENT_SPEECH": 1, "E_SV_TURN_INCONSISTENT": 1}
    r = evaluate(speakers)
    assert r["speakers"] == 2 and r["missing"] == {"none": 2}
    assert (r["pairs"], r["low_pair"]) == (2, 0)
    # id1 + id2's first video: cohesion 0.74, passes (one wrong video in three).
    # id2 + id1's first video: three unrelated directions, cohesion 0.57, caught.
    assert r["low_intruded"] == 1
