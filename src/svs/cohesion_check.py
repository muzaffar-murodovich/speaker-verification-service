"""The weekly cohesion check (handbook D18, SVS guide §8.5 and §11.2).

Cohesion is the mean cosine of a profile's occurrences to its centroid (svs.profile). A
profile that absorbed another person's occurrence has a lower cohesion, so the alert is
a growing left tail: more profiles below CFG_SV_COHESION_ALERT than the week before.

Each run reads every profile's stored cohesion, appends one line to history.tsv and
writes the profiles below the alert to a review list. Profiles with one occurrence are
left out: one occurrence is its own centroid, and its cohesion is always 1.0.
"""

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

from svs.profile import COHESION_ALERT
from svs.store import ProfileStore

OUT = Path("artifacts/cohesion")
HISTORY_FIELDS = ["date", "profiles", "checked", "p05", "median", "below_alert", "share_below"]


def read_cohesion(store: ProfileStore, page: int = 1024) -> tuple[int, dict[str, float]]:
    """All profiles in the store, and the cohesion of each with two or more occurrences."""
    total, cohesion, offset = 0, {}, None
    while True:
        points, offset = store.client.scroll(
            store.collection,
            limit=page,
            offset=offset,
            with_payload=["profile_id", "cohesion", "occurrences"],
            with_vectors=False,
        )
        for p in points:
            total += 1
            if len(p.payload["occurrences"]) >= 2:
                cohesion[p.payload["profile_id"]] = p.payload["cohesion"]
        if offset is None:
            return total, cohesion


def summarise(date: str, total: int, cohesion: dict[str, float]) -> dict:
    values = np.array(list(cohesion.values()))
    below = int((values < COHESION_ALERT).sum())
    return {
        "date": date,
        "profiles": total,
        "checked": len(values),
        "p05": f"{np.quantile(values, 0.05):.4f}" if len(values) else "",
        "median": f"{np.median(values):.4f}" if len(values) else "",
        "below_alert": below,
        "share_below": f"{below / len(values):.4f}" if len(values) else "",
    }


def growing(previous: dict | None, current: dict) -> bool:
    """The §11.1 alert: a larger share below the alert value than at the last check."""
    if previous is None or not previous["share_below"] or not current["share_below"]:
        return False
    return float(current["share_below"]) > float(previous["share_below"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly cohesion check of the profile store")
    parser.add_argument("--url", default="http://localhost:6333")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    total, cohesion = read_cohesion(ProfileStore(QdrantClient(url=args.url)))
    date = datetime.now(UTC).date().isoformat()
    current = summarise(date, total, cohesion)

    args.out.mkdir(parents=True, exist_ok=True)
    history = args.out / "history.tsv"
    rows = list(csv.DictReader(history.open(), delimiter="\t")) if history.exists() else []
    previous = rows[-1] if rows else None
    with history.open("a", newline="") as f:
        writer = csv.DictWriter(f, HISTORY_FIELDS, delimiter="\t")
        if not rows:
            writer.writeheader()
        writer.writerow(current)
    review = sorted((c, pid) for pid, c in cohesion.items() if c < COHESION_ALERT)
    (args.out / f"review-{date}.tsv").write_text(
        "profile_id\tcohesion\n" + "".join(f"{pid}\t{c:.4f}\n" for c, pid in review)
    )

    print("\t".join(HISTORY_FIELDS))
    for row in (previous, current):
        if row is not None:
            print("\t".join(str(row[k]) for k in HISTORY_FIELDS))
    print(f"{len(review)} profiles below {COHESION_ALERT} -> {args.out}/review-{date}.tsv")
    if growing(previous, current):
        print("ALERT: the left tail grew since the last check (runbook: cohesion tail growing)")


if __name__ == "__main__":
    main()
