"""Bands from the operating-point table."""

import csv

import pytest

from svs.bands import NO_MATCH, OPTABLE, band, read_table
from svs.calibrate import POOL_TIERS
from svs.optable import BAND_ORDER
from svs.trials import BUCKETS

# Each bucket has its own thresholds (SVS guide §4.4); a larger pool has higher ones (§4.6).
THRESHOLDS = {
    ("3-10s", 1_000): (7.0, 6.0, 5.0),
    ("3-10s", 50_000): (8.0, 7.0, 6.0),
    ("30-120s", 1_000): (6.0, 5.0, 4.0),
}


def write_table(path, thresholds):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(("condition", "bucket", "pool", "band", "threshold"))
        for (bucket, pool), values in thresholds.items():
            for name, value in zip(BAND_ORDER, values, strict=True):
                writer.writerow(("g711a", bucket, pool, name, value))


@pytest.fixture
def table(tmp_path):
    write_table(tmp_path / "t.tsv", THRESHOLDS)
    return read_table(tmp_path / "t.tsv")


def test_same_score_gets_the_band_of_its_bucket(table):
    assert band(6.5, 5.0, "g711a", table, 1_000)[0] == "PROBABLE"
    assert band(6.5, 60.0, "g711a", table, 1_000)[0] == "STRONG"


def test_a_score_on_a_threshold_is_in_that_band(table):
    bands = [band(s, 5.0, "g711a", table, 1_000)[0] for s in (7.0, 6.0, 5.0)]
    assert bands == list(BAND_ORDER)


def test_each_band_names_its_table_row(table):
    name, row = band(5.5, 5.0, "g711a", table, 1_000)
    assert (name, row["bucket"], row["band"], row["threshold"]) == ("WEAK", "3-10s", "WEAK", "5.0")
    # Below the floor: NO_MATCH, traced to the WEAK row it fell below.
    name, row = band(4.99, 5.0, "g711a", table, 1_000)
    assert (name, row["band"]) == (NO_MATCH, "WEAK")


def test_a_larger_pool_needs_a_higher_score(table):
    # SVS guide §4.6: 7.5 is STRONG against 1,000 profiles, only PROBABLE against 50,000.
    assert band(7.5, 5.0, "g711a", table, 1_000)[0] == "STRONG"
    name, row = band(7.5, 5.0, "g711a", table, 50_000)
    assert (name, row["pool"]) == ("PROBABLE", "50000")


def test_the_pool_takes_the_smallest_tier_that_holds_it(table):
    # 1,001 profiles are more than the 1,000 tier allows: the 50,000 tier is used.
    assert [band(7.5, 5.0, "g711a", table, n)[1]["pool"] for n in (0, 1_000, 1_001)] == [
        "1000",
        "1000",
        "50000",
    ]
    # Above the largest tier, the largest (the caller decides what such a pool may do).
    assert band(7.5, 5.0, "g711a", table, 200_000)[1]["pool"] == "50000"
    # A bucket with one tier uses it for every pool.
    assert band(5.5, 60.0, "g711a", table, 50_000)[1]["pool"] == "1000"


def test_refuses_under_three_seconds_and_unknown_conditions(table):
    with pytest.raises(ValueError, match="must be finite and >= 3.0"):
        band(9.0, 2.99, "g711a", table, 1_000)
    with pytest.raises(ValueError, match="No operating points for condition 'pcm8k'"):
        band(9.0, 5.0, "pcm8k", table, 1_000)


def test_read_table_refuses_overlapping_bands(tmp_path):
    write_table(tmp_path / "t.tsv", {("3-10s", 1_000): (6.0, 6.5, 5.0)})
    with pytest.raises(ValueError, match="do not fall from STRONG to WEAK"):
        read_table(tmp_path / "t.tsv")


def test_published_table_covers_every_condition_bucket_and_pool_tier():
    table = read_table(OPTABLE)
    conditions = {"pcm16k", "pcm8k", "g711a", "g711u", "g729", "opus", "noise15", "noise5"}
    assert set(table) == {(c, b) for c in conditions for b in BUCKETS}
    assert {tuple(by_tier) for by_tier in table.values()} == {POOL_TIERS}
