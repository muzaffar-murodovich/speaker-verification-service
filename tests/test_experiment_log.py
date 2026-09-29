"""Guards for experiments.csv: every logged number must carry its context."""

import csv
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "experiments.csv"

COLUMNS = [
    "run_id",
    "date",
    "code_ref",
    "dataset",
    "condition",
    "config",
    "metric",
    "n_trials",
    "conclusion",
]


def read_rows():
    with LOG.open(newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def test_header_matches_required_columns():
    header, _ = read_rows()
    assert header == COLUMNS


def test_every_row_is_complete():
    _, rows = read_rows()
    for row in rows:
        empty = [col for col in COLUMNS if not (row.get(col) or "").strip()]
        assert not empty, f"run {row.get('run_id')!r} has empty columns: {empty}"


def test_run_ids_are_unique():
    _, rows = read_rows()
    ids = [row["run_id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_n_trials_is_a_positive_integer():
    _, rows = read_rows()
    for row in rows:
        assert row["n_trials"].isdigit() and int(row["n_trials"]) > 0, row["run_id"]
