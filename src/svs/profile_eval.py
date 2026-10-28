"""The evidence rule on VoxCeleb1 speakers, per condition.

Each video is one recording, so one video is one occurrence. Two questions:
1. A profile from all of a speaker's usable videos: does it pass the evidence rule, and
   which condition fails it when it does not?
2. A profile from a speaker's first two usable videos, with and without one video of the
   next speaker added: how often is cohesion below the floor? This shows how much one
   wrong occurrence moves cohesion.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from svs.profile import PROMOTE_MIN_COHESION, Occurrence, Profile
from svs.quality import unusable_reason
from svs.quality_cache import read_quality


def speaker_videos(
    video_dir: str | Path, quality_tsv: str | Path
) -> tuple[dict[str, list[Occurrence]], Counter]:
    """speaker -> usable video occurrences in utt_id order, and refusals by reason.

    No embedding (NaN, under 3.0 s) counts as E_SV_INSUFFICIENT_SPEECH, as the gate would.
    """
    quality = read_quality(quality_tsv)
    speakers, refused = {}, Counter()
    for path in sorted(Path(video_dir).glob("*.npz")):
        d = np.load(path)
        occs = []
        for utt, emb in sorted(zip(d["utt_ids"], d["embeddings"], strict=True)):
            q = quality[("video", str(utt))]
            reason = unusable_reason(q["net_seconds"], q["quality"], q["turn_consistency"])
            if reason is None and np.isnan(emb).any():
                reason = "E_SV_INSUFFICIENT_SPEECH"
            if reason:
                refused[reason] += 1
            else:
                occs.append(Occurrence(str(utt), emb, q["net_seconds"], q["quality"]))
        speakers[path.stem] = occs
    return speakers, refused


def profile_of(profile_id: str, occs: list[Occurrence]) -> Profile:
    profile = Profile(profile_id)
    for o in occs:
        profile.accept(o)
    return profile


def evaluate(speakers: dict[str, list[Occurrence]]) -> dict:
    whole, missing, cohesion = 0, Counter(), []
    pairs, low_pair, low_intruded = 0, 0, 0
    ids = sorted(spk for spk, occs in speakers.items() if occs)
    for i, spk in enumerate(ids):
        profile = profile_of(spk, speakers[spk])
        whole += 1
        cohesion.append(profile.cohesion)
        missing.update(profile.promotion_missing() or ["none"])
        if len(speakers[spk]) < 2:
            continue
        # The intruder is the next speaker's first video: fixed, so a rerun gives the same.
        intruder = speakers[ids[(i + 1) % len(ids)]][0]
        pairs += 1
        low_pair += profile_of(spk, speakers[spk][:2]).cohesion < PROMOTE_MIN_COHESION
        intruded = profile_of(spk, [*speakers[spk][:2], intruder])
        low_intruded += intruded.cohesion < PROMOTE_MIN_COHESION
    return {
        "speakers": whole,
        "missing": missing,
        "cohesion": np.array(cohesion),
        "pairs": pairs,
        "low_pair": low_pair,
        "low_intruded": low_intruded,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("conditions", nargs="+", metavar="CONDITION=VIDEO_DIR,QUALITY_TSV")
    args = parser.parse_args()
    print(
        "condition\tspeakers\tvideos_refused (by reason)\tpromoted\tmissing (by condition)"
        f"\tcohesion_min\tp1\tp5\tmedian\ttwo_videos\tbelow_{PROMOTE_MIN_COHESION}_own"
        f"\tbelow_{PROMOTE_MIN_COHESION}_with_intruder"
    )
    for item in args.conditions:
        condition, paths = item.split("=", 1)
        speakers, refused = speaker_videos(*paths.split(","))
        r = evaluate(speakers)
        coh = r["cohesion"]
        refused_txt = ", ".join(f"{k} {v}" for k, v in sorted(refused.items())) or "none"
        missing = {k: v for k, v in r["missing"].items() if k != "none"}
        missing_txt = ", ".join(f"{k} {v}" for k, v in sorted(missing.items())) or "none"
        print(
            f"{condition}\t{r['speakers']}\t{refused_txt}\t{r['missing']['none']}\t{missing_txt}"
            f"\t{coh.min():.4f}\t{np.percentile(coh, 1):.4f}\t{np.percentile(coh, 5):.4f}"
            f"\t{np.median(coh):.4f}\t{r['pairs']}\t{r['low_pair']}\t{r['low_intruded']}"
        )
