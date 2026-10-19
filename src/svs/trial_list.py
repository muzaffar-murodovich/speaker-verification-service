"""The VoxCeleb1 trial list: short buckets from single clips, long buckets from whole videos."""

import argparse
from collections import Counter
from pathlib import Path

from svs.duplicates import merge_sessions, read_video_pairs
from svs.embed_cache import read_cache
from svs.trials import (
    BUCKETS,
    Trial,
    Utterance,
    nontarget_trials,
    sample_target_trials,
    write_trials,
)
from svs.voxceleb import read_genders, utterances

# Single clips reach 120 s only once, so the long buckets use clips joined per video.
# A trial never mixes the two: a clip lies inside its video, so they share audio.
SOURCES = {"3-10s": "clip", "10-30s": "clip", "30-120s": "video", ">120s": "video"}
N_TARGET = 50_000  # Per bucket; all of them where fewer exist (>120s has 3,186).
N_NONTARGET = 100_000  # Per bucket, so each bucket has >= 100 000 trials (SVS guide §8.1).


def build_trial_list(
    pools: dict[str, list[Utterance]], n_target: int, n_nontarget: int, seed: int = 0
) -> list[tuple[str, Trial]]:
    """(source, trial) rows for every bucket, each bucket drawn from its own source pool."""
    rows = []
    for bucket in BUCKETS:
        source = SOURCES[bucket]
        utts = pools[source]
        rows += [(source, t) for t in sample_target_trials(utts, n_target, bucket, seed)]
        rows += [(source, t) for t in nontarget_trials(utts, n_nontarget, seed, bucket)]
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_cache")
    parser.add_argument("video_cache")
    parser.add_argument("meta_csv")
    parser.add_argument("shared_videos", help="video pairs from make duplicates")
    parser.add_argument("out_tsv")
    args = parser.parse_args()
    genders = read_genders(args.meta_csv)
    # Videos that hold the same audio become one session, so the guard keeps them apart.
    sessions = merge_sessions(read_video_pairs(args.shared_videos))
    pools = {}
    for source, cache in (("clip", args.clip_cache), ("video", args.video_cache)):
        utt_ids, seconds, _ = read_cache(cache)
        pools[source] = utterances(utt_ids, seconds, genders, sessions)
    rows = build_trial_list(pools, N_TARGET, N_NONTARGET)
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {write_trials(args.out_tsv, rows)} trials to {args.out_tsv}")
    counts = Counter((t.bucket, t.is_target) for _, t in rows)
    for bucket in BUCKETS:
        print(f"{bucket}: {counts[bucket, True]} target, {counts[bucket, False]} non-target")
