"""The weekly cohesion check: one-occurrence profiles left out, the left tail compared."""

import numpy as np
import pytest
from qdrant_client import QdrantClient

from svs import cohesion_check
from svs.cohesion_check import growing, read_cohesion, summarise
from svs.embed import EMBEDDING_DIM, MODEL_VERSION
from svs.store import ProfileStore

# profile_id -> (occurrences, stored cohesion)
PROFILES = {
    "spk_a": (1, 1.0),
    "spk_b": (2, 0.91),
    "spk_c": (3, 0.52),
    "spk_d": (2, 0.55),
    "spk_e": (4, 0.30),
}


def store_with(profiles: dict) -> ProfileStore:
    store = ProfileStore(QdrantClient(":memory:"))
    store.create()
    vector = np.eye(EMBEDDING_DIM, dtype=np.float32)[0]
    for pid, (n, cohesion) in profiles.items():
        payload = {
            "status": "PROVISIONAL",
            "source": ["cc_north"],
            "last_seen": "2026-11-02T00:00:00Z",
            "evidence_seconds": 60.0 * n,
            "occurrences": [{}] * n,
            "cohesion": cohesion,
        }
        store.upsert(MODEL_VERSION, pid, vector, payload)
    return store


def test_one_occurrence_profiles_are_counted_but_not_checked():
    total, cohesion = read_cohesion(store_with(PROFILES), page=2)  # three pages
    assert total == 5
    assert cohesion == {"spk_b": 0.91, "spk_c": 0.52, "spk_d": 0.55, "spk_e": 0.30}


def test_below_alert_is_strictly_below_055():
    row = summarise("2026-11-02", 5, {"spk_b": 0.91, "spk_c": 0.52, "spk_d": 0.55, "spk_e": 0.3})
    assert row["below_alert"] == 2  # 0.52 and 0.30; 0.55 itself is not below
    assert row["share_below"] == "0.5000"
    assert row["median"] == "0.5350"


def test_an_empty_store_gives_a_row_without_numbers():
    row = summarise("2026-11-02", 0, {})
    assert (row["checked"], row["below_alert"], row["share_below"]) == (0, 0, "")


@pytest.mark.parametrize(
    ("before", "now", "alert"),
    [(None, "0.10", False), ("0.10", "0.10", False), ("0.10", "0.12", True), ("", "0.5", False)],
)
def test_the_alert_is_a_larger_share_than_last_time(before, now, alert):
    previous = None if before is None else {"share_below": before}
    assert growing(previous, {"share_below": now}) is alert


def test_two_runs_append_history_and_the_second_alerts(tmp_path, monkeypatch, capsys):
    stores = iter([store_with(PROFILES), store_with({**PROFILES, "spk_b": (2, 0.40)})])
    monkeypatch.setattr(cohesion_check, "ProfileStore", lambda client: next(stores))
    monkeypatch.setattr("sys.argv", ["cohesion_check", "--out", str(tmp_path)])
    cohesion_check.main()
    assert "ALERT" not in capsys.readouterr().out
    cohesion_check.main()
    assert "ALERT" in capsys.readouterr().out
    lines = (tmp_path / "history.tsv").read_text().splitlines()
    assert len(lines) == 3  # header and two runs
    (review,) = tmp_path.glob("review-*.tsv")
    assert review.read_text().splitlines()[1:] == [
        "spk_e\t0.3000",
        "spk_b\t0.4000",
        "spk_c\t0.5200",
    ]
