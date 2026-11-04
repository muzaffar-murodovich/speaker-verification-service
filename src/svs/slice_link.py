"""Candidate same-person links between recordings of the slice (D16, for D17 trials).

The export's speaker ids are local to one recording, so nothing says whether one person
speaks in two recordings. This scores every pair of usable occurrences (one speaker in
one recording) as /v1/verify scores them, and links the pairs from different recordings
that score above every pair inside one recording. Linked occurrences form groups:
candidates for one person each, to be confirmed by listening. A score is evidence, not
a label.

As in verify (svs.api): the bucket is the shorter side's, AS-Norm uses the cohort of the
condition, and with no condition known (the slice has none) the condition is the most
conservative of the bucket. The pool is the number of occurrences each one is compared
with, so the pool tier is the one a search of that size would use (D14).

Pairs inside one recording are two different people by the labels. On the slice, 39 of
109 reached STRONG (19 of 30 in stereo calls): the VoxCeleb thresholds do not hold on
this audio, and STRONG links joined 212 of 217 occurrences into one group. So the link
threshold is the highest score of a pair inside one recording: above every known pair of
two different people. Those pairs share a line, and may share crosstalk, so they likely
score higher than two different people in two recordings; still no guarantee.

A group is a connected component of links: A-B and B-C put A and C together even when
A-C is weak. min_score in groups.tsv shows how weak the weakest pair inside is.

listen writes, for each occurrence in a group, up to CLIP_SECONDS of its labelled
segments (longest first, as recorded, no VAD) to listen/<group>/, listen.tsv with an
empty same_person column, and listen.html: the same list as a local page with a player
and Ha / Yoʻq / Bilmayman per clip, which downloads listen.tsv filled in. The page stays
local next to the clips: T2 audio goes to no other service.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.sparse.csgraph import connected_components

from svs.api import COHORT_DIR, resolve_condition
from svs.asnorm import as_norm, cohort_stats
from svs.bands import OPTABLE, band, read_table
from svs.extract import load_audio
from svs.optable import cohort_of
from svs.slice_labels import item_genders, read_export
from svs.slice_run import OUT, parse_segments, read_tsv, write_tsv

PAIR_FIELDS = ["left", "right", "same_recording", "seconds", "condition", "raw", "score", "band"]
GROUP_FIELDS = ["group", "item_id", "audio", "gender", "size", "min_score", "genders"]
LISTEN_FIELDS = ["group", "item_id", "clip", "seconds", "gender", "same_person"]
CLIP_SECONDS = 10.0
PAGE = Path(__file__).with_name("listen_page.html")


def recording_of(item_id: str) -> str:
    """ "a1.mp3#ref0" -> "a1.mp3"; a stereo item "a1.wav#ch0" -> "a1.wav"."""
    return item_id.split("#")[0]


def score_pairs(items: list[dict], embeddings: dict, table: dict, cohorts: dict) -> list[dict]:
    """Every pair of items once, with its verify score and band against a pool of n - 1."""
    ids = [r["item_id"] for r in items]
    seconds = np.array([float(r["net_speech_seconds"]) for r in items])
    vectors = np.stack([embeddings[i] for i in ids]).astype(np.float64)
    raw = vectors @ vectors.T
    stats = {}  # condition -> (mu, sd) of every item, computed once
    pool = len(ids) - 1
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            shorter = min(seconds[i], seconds[j])
            condition, _ = resolve_condition(None, shorter, table)
            if condition not in stats:
                stats[condition] = cohort_stats(vectors, cohorts[condition])
            mu, sd = stats[condition]
            score = float(as_norm(raw[i, j], mu[i], sd[i], mu[j], sd[j]))
            name, _ = band(score, shorter, condition, table, pool)
            pairs.append(
                {
                    "left": ids[i],
                    "right": ids[j],
                    "same_recording": recording_of(ids[i]) == recording_of(ids[j]),
                    "seconds": f"{shorter:.3f}",
                    "condition": condition,
                    "raw": f"{raw[i, j]:.4f}",
                    "score": f"{score:.4f}",
                    "band": name,
                }
            )
    return pairs


def link_threshold(pairs: list[dict]) -> float:
    """The highest score of a pair inside one recording: two different people."""
    return max(float(p["score"]) for p in pairs if p["same_recording"])


def groups(
    ids: list[str], pairs: list[dict], genders: dict[str, str], threshold: float
) -> list[dict]:
    """Connected components of links above threshold across recordings; singles left out."""
    index = {item: k for k, item in enumerate(ids)}
    links = np.zeros((len(ids), len(ids)), bool)
    score = np.full((len(ids), len(ids)), np.nan)
    for p in pairs:
        i, j = index[p["left"]], index[p["right"]]
        score[i, j] = score[j, i] = float(p["score"])
        links[i, j] = links[j, i] = float(p["score"]) > threshold and not p["same_recording"]
    _, label = connected_components(links, directed=False)
    members = [np.flatnonzero(label == g) for g in range(label.max() + 1)]
    members = sorted((m for m in members if len(m) > 1), key=lambda m: (-len(m), m[0]))
    rows = []
    for g, m in enumerate(members):
        inside = score[np.ix_(m, m)][np.triu_indices(len(m), 1)]
        labels = Counter(genders.get(ids[k], "") for k in m)
        for k in m:
            rows.append(
                {
                    "group": g,
                    "item_id": ids[k],
                    "audio": recording_of(ids[k]),
                    "gender": genders.get(ids[k], ""),
                    "size": len(m),
                    "min_score": f"{np.nanmin(inside):.4f}",
                    "genders": " ".join(f"{v}{n}" for v, n in sorted(labels.items())),
                }
            )
    return rows


def clip(root: Path, row: dict, seconds: float = CLIP_SECONDS) -> tuple[np.ndarray, int]:
    """Up to seconds of an item's segments, longest first, in recording order."""
    channel = int(row["channel"]) if str(row["channel"]) else None
    audio, rate = load_audio(root / row["audio"], channel)
    chosen, total = [], 0.0
    for s, e in sorted(parse_segments(row["segments"]), key=lambda x: x[0] - x[1]):
        take = min(e - s, (seconds - total) * 1000)
        if take <= 0:
            break
        chosen.append((s, s + take))
        total += take / 1000
    parts = [
        audio[math.ceil(s * rate / 1000) : math.ceil(e * rate / 1000)] for s, e in sorted(chosen)
    ]
    return np.concatenate(parts), rate


def listen(root: Path, out: Path, manifest: dict[str, dict], group_rows: list[dict]) -> list[dict]:
    rows = []
    for r in group_rows:
        samples, rate = clip(root, manifest[r["item_id"]])
        path = out / "listen" / f"group{r['group']:02d}" / f"{r['item_id'].replace('#', '_')}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, samples, rate)
        rows.append(
            {
                "group": r["group"],
                "item_id": r["item_id"],
                "clip": path.relative_to(out).as_posix(),
                "seconds": f"{len(samples) / rate:.1f}",
                "gender": r["gender"],
                "same_person": "",
            }
        )
    return rows


def listen_page(rows: list[dict]) -> str:
    """The listening page with the rows inlined; "</" escaped so no row can end the script."""
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    return PAGE.read_text().replace("/*ROWS*/[]", data)


def summary(pairs: list[dict], group_rows: list[dict], threshold: float) -> list[str]:
    lines = []
    for same, name in ((True, "same recording (different people)"), (False, "across recordings")):
        bands = Counter(p["band"] for p in pairs if p["same_recording"] == same)
        lines.append(
            f"{name}: {sum(bands.values())} pairs, "
            + ", ".join(f"{b} {bands[b]}" for b in ("STRONG", "PROBABLE", "WEAK", "NO_MATCH"))
        )
    sizes = Counter()
    mixed = set()
    for r in group_rows:
        sizes[r["group"]] = r["size"]
        if len(r["genders"].split()) > 1:
            mixed.add(r["group"])
    lines.append(f"link threshold {threshold:.2f} (highest pair inside one recording)")
    lines.append(
        f"{len(sizes)} groups, {sum(sizes.values())} occurrences, sizes "
        f"{sorted(sizes.values(), reverse=True)}; {len(mixed)} with mixed gender labels"
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Same-person candidates across recordings")
    parser.add_argument("root", type=Path, help="the folder of recordings")
    parser.add_argument("--run", type=Path, default=OUT / "reference", help="a slice-run OUT")
    parser.add_argument("--export", type=Path, help="the labelling tool's JSON export")
    parser.add_argument(
        "--manifest", type=Path, default=OUT / "manifest-reference.tsv", help="the run's items"
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    results = [r for r in read_tsv(args.run / "results.tsv") if r["usable"] == "True"]
    npz = np.load(args.run / "embeddings.npz")
    embeddings = dict(zip(npz["item_ids"], npz["embeddings"], strict=True))
    table = read_table(OPTABLE)
    cohorts = {
        c: np.load(COHORT_DIR / f"{cohort_of(c)}.npz")["embeddings"] for c in {c for c, _ in table}
    }
    pairs = score_pairs(results, embeddings, table, cohorts)
    genders = {}
    if args.export is not None:
        files = read_tsv(args.out / "files.tsv")
        genders = item_genders(read_export(args.export, files, args.root))
    threshold = link_threshold(pairs)
    group_rows = groups([r["item_id"] for r in results], pairs, genders, threshold)
    write_tsv(args.out / "pairs.tsv", PAIR_FIELDS, pairs)
    write_tsv(args.out / "groups.tsv", GROUP_FIELDS, group_rows)
    manifest = {r["item_id"]: r for r in read_tsv(args.manifest)}
    listen_rows = listen(args.root, args.out, manifest, group_rows)
    write_tsv(args.out / "listen.tsv", LISTEN_FIELDS, listen_rows)
    (args.out / "listen.html").write_text(listen_page(listen_rows))
    print(f"{len(results)} usable occurrences, pool {len(results) - 1}")
    print("\n".join(summary(pairs, group_rows, threshold)))
    print(f"-> {args.out}/pairs.tsv, groups.tsv, listen.tsv, listen.html, listen/")


if __name__ == "__main__":
    main()
