"""The operating-point table: EER joined to the thresholds of each band."""

import csv

import pytest

from svs.optable import band_precision, table_rows
from svs.trials import BUCKETS

FIELDS = ("condition", "bucket", "band", "far", "threshold", "false_accepts", "speaker_pairs")
POINT_FIELDS = (*FIELDS, "n_nontarget", "frr_%", "n_target", "recall", "precision_pool50000")


def test_band_precision_counts_only_the_scores_inside_the_band():
    strong = {"recall": "0.6", "false_accepts": "10", "precision": "0.857"}
    probable = {"recall": "0.8", "false_accepts": "30", "n_nontarget": "1000000"}
    assert band_precision(strong, None) == 0.857
    # In the band: recall 0.2, and 20 false accepts in 1,000,000 = 1.0 per 50,000-pool query.
    assert band_precision(probable, strong) == pytest.approx(0.2 / 1.2)


def write(path, fields, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(fields)
        writer.writerows(rows)


def test_table_rows_join_eer_and_bands_per_bucket(tmp_path):
    scores, points = tmp_path / "scores.tsv", tmp_path / "points.tsv"
    # Per bucket one missed target (0.1) and one accepted non-target (0.8): EER 25 %.
    write(
        scores,
        ("bucket", "gender", "target", "score"),
        [
            (b, "f", t, s)
            for b in BUCKETS
            for t, s in [(1, 0.1), (1, 0.6), (1, 0.7), (1, 0.9)]
            + [(0, 0.2), (0, 0.3), (0, 0.4), (0, 0.8)]
        ],
    )
    write(
        points,
        POINT_FIELDS,
        [
            ("g729", b, band, "x", thr, fa, 1, 1000, "x", 4, recall, "0.5")
            for b in BUCKETS
            for band, thr, fa, recall in [("WEAK", 5, 3, 0.9), ("STRONG", 7, 1, 0.5)]
            + [("PROBABLE", 6, 2, 0.7)]
        ],
    )
    rows = table_rows("g729", str(scores), str(points))
    assert [(r["bucket"], r["band"]) for r in rows[:3]] == [
        ("3-10s", "STRONG"),
        ("3-10s", "PROBABLE"),
        ("3-10s", "WEAK"),
    ]
    assert len(rows) == 12
    assert {r["eer_%"] for r in rows} == {"25.000"}
    assert {r["cohort"] for r in rows} == {"voxceleb2-dev-v1-g729"}
    with pytest.raises(ValueError, match="not the table of pcm8k"):
        table_rows("pcm8k", str(scores), str(points))
