"""The band of a normalised score: STRONG, PROBABLE, WEAK or NO_MATCH (SVS guide §4.5).

The thresholds depend on the condition and the duration bucket (§4.4), so one score can
be STRONG for long audio and WEAK for short. Every band comes with the row of the
operating-point table that decided it (handbook M2: bands traceable to the table).
"""

import csv
from pathlib import Path

from svs.optable import BAND_ORDER
from svs.trials import duration_bucket

OPTABLE = Path("docs/operating-points.tsv")
NO_MATCH = "NO_MATCH"


def read_table(path: str | Path = OPTABLE) -> dict[tuple[str, str], list[dict]]:
    """(condition, bucket) -> its rows, highest threshold first.

    Raises if a band is missing or the thresholds do not fall from STRONG to WEAK: the
    bands would then overlap.
    """
    table: dict[tuple[str, str], dict[str, dict]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            table.setdefault((row["condition"], row["bucket"]), {})[row["band"]] = row
    out = {}
    for key, by_band in table.items():
        if set(by_band) != set(BAND_ORDER):
            raise ValueError(f"{key}: bands {sorted(by_band)}, expected {list(BAND_ORDER)}")
        rows = [by_band[band] for band in BAND_ORDER]
        thresholds = [float(r["threshold"]) for r in rows]
        if thresholds != sorted(thresholds, reverse=True) or len(set(thresholds)) < 3:
            raise ValueError(f"{key}: thresholds {thresholds} do not fall from STRONG to WEAK")
        out[key] = rows
    return out


def band(
    score: float, speech_seconds: float, condition: str, table: dict[tuple[str, str], list[dict]]
) -> tuple[str, dict]:
    """The band of a score, and the table row that decided it.

    A score at or above a threshold is in that band (as in svs.metrics.error_rates).
    NO_MATCH returns the WEAK row: the floor the score fell below. Under 3.0 s of
    speech duration_bucket raises: the service refuses to score.
    """
    bucket = duration_bucket(speech_seconds)
    if (condition, bucket) not in table:
        raise ValueError(f"No operating points for condition {condition!r}")
    rows = table[(condition, bucket)]
    for row in rows:
        if score >= float(row["threshold"]):
            return row["band"], row
    return NO_MATCH, rows[-1]
