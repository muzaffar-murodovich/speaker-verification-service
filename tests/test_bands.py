"""Bands from the operating-point table."""

import csv

import pytest

from svs.bands import NO_MATCH, OPTABLE, band, read_table
from svs.optable import BAND_ORDER
from svs.trials import BUCKETS

# Each bucket has its own thresholds (SVS guide §4.4).
THRESHOLDS = {"3-10s": (7.0, 6.0, 5.0), "30-120s": (6.0, 5.0, 4.0)}


def write_table(path, thresholds):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(("condition", "bucket", "band", "threshold"))
        for bucket, values in thresholds.items():
            for name, value in zip(BAND_ORDER, values, strict=True):
                writer.writerow(("g711a", bucket, name, value))


@pytest.fixture
def table(tmp_path):
    write_table(tmp_path / "t.tsv", THRESHOLDS)
    return read_table(tmp_path / "t.tsv")


def test_same_score_gets_the_band_of_its_bucket(table):
    assert band(6.5, 5.0, "g711a", table)[0] == "PROBABLE"
    assert band(6.5, 60.0, "g711a", table)[0] == "STRONG"


def test_a_score_on_a_threshold_is_in_that_band(table):
    assert [band(s, 5.0, "g711a", table)[0] for s in (7.0, 6.0, 5.0)] == list(BAND_ORDER)


def test_each_band_names_its_table_row(table):
    name, row = band(5.5, 5.0, "g711a", table)
    assert (name, row["bucket"], row["band"], row["threshold"]) == ("WEAK", "3-10s", "WEAK", "5.0")
    # Below the floor: NO_MATCH, traced to the WEAK row it fell below.
    name, row = band(4.99, 5.0, "g711a", table)
    assert (name, row["band"]) == (NO_MATCH, "WEAK")


def test_refuses_under_three_seconds_and_unknown_conditions(table):
    with pytest.raises(ValueError, match="must be finite and >= 3.0"):
        band(9.0, 2.99, "g711a", table)
    with pytest.raises(ValueError, match="No operating points for condition 'pcm8k'"):
        band(9.0, 5.0, "pcm8k", table)


def test_read_table_refuses_overlapping_bands(tmp_path):
    write_table(tmp_path / "t.tsv", {"3-10s": (6.0, 6.5, 5.0)})
    with pytest.raises(ValueError, match="do not fall from STRONG to WEAK"):
        read_table(tmp_path / "t.tsv")


def test_published_table_covers_every_condition_and_bucket():
    table = read_table(OPTABLE)
    conditions = {"pcm16k", "pcm8k", "g711a", "g711u", "g729", "opus", "noise15", "noise5"}
    assert set(table) == {(c, b) for c in conditions for b in BUCKETS}
