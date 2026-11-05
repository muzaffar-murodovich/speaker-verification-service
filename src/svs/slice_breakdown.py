"""Breakdowns on the internal slice (D17), and three scorings that test why it differs.

Trials come from what is known about who is who, never from a score alone:

- target: two occurrences (one speaker in one recording) the listener put in one person
  (listen-answers.tsv, same_person = yes, same group), in two recordings. These were
  found by score (slice_link), so they are biased towards high scores: a share of them
  at STRONG is an upper bound, and no EER is computed from them.
- same_recording: the two labelled speakers of one call, two different people who share
  the line (and in stereo, crosstalk).
- cross_gender: two recordings, different annotator gender labels: different people.

Pairs of one gender in two recordings that the listener did not confirm are left out:
some of them may be one person.

Non-target pairs were not chosen by score, so the share at STRONG is a fair estimate of
the false accept rate at the STRONG threshold. The VoxCeleb table sets that threshold for
FAR 1e-4 at pool tier 1,000 (the slice's 216-occurrence pool).

Three scorings, each a hypothesis for the gap between VoxCeleb and the slice:

- verify: as /v1/verify with no condition block (D16): the most conservative condition
  of the bucket, normalised with that condition's VoxCeleb2 cohort.
- pcm8k: the condition the audio is closest to (8 kHz telephone), with its cohort and
  thresholds. Tests "the wrong condition was picked".
- slice_cohort: AS-Norm against the slice itself, every occurrence outside the two
  recordings of the pair (top SLICE_TOP_N), with pcm8k thresholds. Tests "the cohort
  does not look like this audio". It is not a production setting: the cohort holds
  the evaluation audio's own speakers, which a fixed cohort must not (handbook D8).

Each breakdown axis splits the pairs: duration bucket of the shorter side, file format
(mp3 / wav), sample rate (8 kHz / wider), layout (stereo channel / mono), and source
(same folder / different folders). Language and session gap have no split: 99.7% of
labelled segments are Uzbek, and only 8 of 117 recordings carry a date.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from svs.api import COHORT_DIR, resolve_condition
from svs.asnorm import as_norm, cohort_stats, top_n_stats
from svs.bands import OPTABLE, band, read_table
from svs.optable import cohort_of
from svs.slice_labels import item_genders, item_labels, read_export
from svs.slice_link import recording_of
from svs.slice_run import OUT, read_tsv, write_tsv
from svs.trials import duration_bucket

KINDS = ("target", "same_recording", "cross_gender")
SCORINGS = ("verify", "pcm8k", "slice_cohort")
AXES = ("all", "bucket", "format", "rate", "layout", "source")
SLICE_TOP_N = 50  # of about 210 slice occurrences outside a pair's two recordings
POOL = 216
FIELDS = ["scoring", "kind", "axis", "value", "n", "strong", "probable_up", "strong_rate"]
ABOVE_PROBABLE = {"STRONG", "PROBABLE"}


def people(answers: list[dict]) -> dict[str, int]:
    """item_id -> person: the 'yes' clips of a listening group are one person."""
    return {r["item_id"]: int(r["group"]) for r in answers if r["same_person"] == "yes"}


def kind_of(a: str, b: str, person: dict[str, int], gender: dict[str, str]) -> str | None:
    if recording_of(a) == recording_of(b):
        return "same_recording"
    if a in person and person.get(a) == person.get(b):
        return "target"
    if gender.get(a) and gender.get(b) and gender[a] != gender[b]:
        return "cross_gender"
    return None


def facets(a: dict, b: dict) -> dict[str, str]:
    """The breakdown value of each axis for a pair of item descriptions."""

    def both(key: str, name=lambda v: v) -> str:
        x, y = name(a[key]), name(b[key])
        return x if x == y else "mixed"

    return {
        "all": "all",
        "bucket": duration_bucket(min(a["seconds"], b["seconds"])),
        "format": both("format"),
        "rate": both("rate", lambda r: "8k" if r == 8000 else "wide"),
        "layout": both("layout"),
        "source": "same" if a["source"] == b["source"] else "different",
    }


def slice_cohort_stats(raw: np.ndarray, recordings: list[str], owner: list, i: int, j: int, n: int):
    """Top-n mean and sd of i and j against the slice outside their two recordings and
    outside the confirmed people of i and j (a cohort must not hold the trial's speakers)."""
    people_ij = {owner[i], owner[j]} - {None}
    keep = np.array(
        [
            r not in (recordings[i], recordings[j]) and o not in people_ij
            for r, o in zip(recordings, owner, strict=True)
        ]
    )
    mu, sd = top_n_stats(raw[[i, j]][:, keep], n)
    return mu, sd


def score_all(ids, vectors, seconds, recordings, owner, kinds, table, cohorts) -> dict[str, list]:
    """scoring -> [(i, j, band)] for every pair with a kind."""
    raw = vectors @ vectors.T
    stats = {c: cohort_stats(vectors, cohorts[c]) for c in cohorts}
    out = {s: [] for s in SCORINGS}
    for (i, j), _ in kinds.items():
        shorter = min(seconds[i], seconds[j])
        verify_condition, _ = resolve_condition(None, shorter, table)
        for scoring in SCORINGS:
            if scoring == "slice_cohort":
                mu, sd = slice_cohort_stats(raw, recordings, owner, i, j, SLICE_TOP_N)
                condition = "pcm8k"
            else:
                condition = verify_condition if scoring == "verify" else "pcm8k"
                m, s = stats[condition]
                mu, sd = m[[i, j]], s[[i, j]]
            score = float(as_norm(raw[i, j], mu[0], sd[0], mu[1], sd[1]))
            name, _ = band(score, shorter, condition, table, POOL)
            out[scoring].append((i, j, name))
    return out


def breakdown(scored: dict[str, list], kinds: dict, info: list[dict]) -> list[dict]:
    counts: dict[tuple, Counter] = {}
    for scoring, rows in scored.items():
        for i, j, name in rows:
            for axis, value in facets(info[i], info[j]).items():
                c = counts.setdefault((scoring, kinds[(i, j)], axis, value), Counter())
                c["n"] += 1
                c["strong"] += name == "STRONG"
                c["probable_up"] += name in ABOVE_PROBABLE
    rows = []
    for (scoring, kind, axis, value), c in sorted(
        counts.items(),
        key=lambda kv: (
            (SCORINGS.index(kv[0][0]), KINDS.index(kv[0][1]), AXES.index(kv[0][2])) + (kv[0][3],)
        ),
    ):
        rows.append(
            {
                "scoring": scoring,
                "kind": kind,
                "axis": axis,
                "value": value,
                "n": c["n"],
                "strong": c["strong"],
                "probable_up": c["probable_up"],
                "strong_rate": f"{c['strong'] / c['n']:.3f}",
            }
        )
    return rows


def facts(reference: dict, results: list[dict], labels: dict) -> list[str]:
    """Language labels, segment lengths and refusals: the D16 items with no breakdown."""
    langs = Counter()
    for v in labels.values():
        langs.update(v["langs"])
    short = total = 0
    for ref in reference.values():
        for segments in ref["speakers"].values():
            for s, e in segments:
                total += 1
                short += (e - s) < 1000
    reasons = Counter(r["reason"] for r in results if r["usable"] != "True")
    return [
        "language labels per segment: " + ", ".join(f"{k} {v}" for k, v in langs.most_common()),
        f"labelled turns (overlaps joined) under 1 s: {short} of {total} ({short / total:.0%})",
        "refused occurrences: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="D17 breakdowns on the internal slice")
    parser.add_argument("root", type=Path, help="the folder of recordings")
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=OUT / "reference")
    parser.add_argument("--answers", type=Path, default=OUT / "listen-answers.tsv")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    all_results = read_tsv(args.run / "results.tsv")
    results = [r for r in all_results if r["usable"] == "True"]
    npz = np.load(args.run / "embeddings.npz")
    embeddings = dict(zip(npz["item_ids"], npz["embeddings"], strict=True))
    files = {f["audio"]: f for f in read_tsv(args.out / "files.tsv")}
    reference = read_export(args.export, list(files.values()), args.root)
    gender, labels = item_genders(reference), item_labels(reference)
    person = people(read_tsv(args.answers))

    ids = [r["item_id"] for r in results]
    vectors = np.stack([embeddings[i] for i in ids]).astype(np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    seconds = [float(r["net_speech_seconds"]) for r in results]
    recordings = [recording_of(i) for i in ids]
    info = []
    for item, s in zip(ids, seconds, strict=True):
        f = files[recording_of(item)]
        info.append(
            {
                "seconds": s,
                "format": f["format"].lower(),
                "rate": int(f["sample_rate"]),
                "layout": "stereo" if int(f["channels"]) > 1 else "mono",
                "source": labels[item]["source"],
            }
        )
    kinds = {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            k = kind_of(ids[i], ids[j], person, gender)
            if k is not None:
                kinds[(i, j)] = k
    table = read_table(OPTABLE)
    conditions = {c for c, _ in table}
    cohorts = {c: np.load(COHORT_DIR / f"{cohort_of(c)}.npz")["embeddings"] for c in conditions}
    owner = [person.get(i) for i in ids]
    scored = score_all(ids, vectors, seconds, recordings, owner, kinds, table, cohorts)
    rows = breakdown(scored, kinds, info)
    write_tsv(args.out / "breakdown.tsv", FIELDS, rows)
    print(f"{len(ids)} usable occurrences; pairs: {dict(Counter(kinds.values()))}")
    for r in rows:
        if r["axis"] == "all":
            print(
                f"{r['scoring']:>12} {r['kind']:>14}: n {r['n']}, STRONG {r['strong']} "
                f"({float(r['strong_rate']):.1%}), PROBABLE or above {r['probable_up']}"
            )
    print("\n".join(facts(reference, all_results, labels)))
    print(f"-> {args.out}/breakdown.tsv")


if __name__ == "__main__":
    main()
