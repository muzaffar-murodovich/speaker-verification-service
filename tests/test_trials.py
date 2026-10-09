"""Trial guards: a same-session target trial must never be built (SVS guide §8.1)."""

import pytest

from svs.trials import (
    Trial,
    Utterance,
    duration_bucket,
    nontarget_trials,
    target_trials,
    write_trials,
)

# Speaker A has two utterances in video v1 and one in v2; speaker B has one session only.
# A and B are male, C and D are female. Durations are net speech seconds.
A1 = Utterance("a1", speaker="A", session="v1", gender="m", duration=8.0)
A2 = Utterance("a2", speaker="A", session="v1", gender="m", duration=12.0)
A3 = Utterance("a3", speaker="A", session="v2", gender="m", duration=45.0)
B1 = Utterance("b1", speaker="B", session="v3", gender="m", duration=150.0)
B2 = Utterance("b2", speaker="B", session="v3", gender="m", duration=20.0)
C1 = Utterance("c1", speaker="C", session="v4", gender="f", duration=5.0)
D1 = Utterance("d1", speaker="D", session="v5", gender="f", duration=35.0)
ALL = [A1, A2, A3, B1, B2, C1, D1]
# Same gender, different speaker: 3 x 2 male pairs and 1 female pair.
NONTARGET_PAIRS = {(a, b) for a in ("a1", "a2", "a3") for b in ("b1", "b2")} | {("c1", "d1")}


def test_trial_rejects_same_session_target():
    with pytest.raises(ValueError, match="Same-session"):
        Trial(A1, A2)


def test_trial_accepts_cross_session_target_and_nontarget():
    assert Trial(A1, A3).is_target
    assert not Trial(A1, B1).is_target


def test_target_trials_are_exactly_the_cross_session_pairs():
    pairs = [(t.enrol.utt_id, t.test.utt_id) for t in target_trials([B2, A3, A2, B1, A1])]
    assert pairs == [("a1", "a3"), ("a2", "a3")]


@pytest.mark.parametrize(
    ("seconds", "bucket"),
    [
        (3.0, "3-10s"),
        (9.99, "3-10s"),
        (10.0, "10-30s"),
        (29.99, "10-30s"),
        (30.0, "30-120s"),
        (119.99, "30-120s"),
        (120.0, ">120s"),
        (3600.0, ">120s"),
    ],
)
def test_duration_bucket_edges(seconds, bucket):
    assert duration_bucket(seconds) == bucket


@pytest.mark.parametrize("seconds", [2.99, 0.0, -1.0, float("nan"), float("inf")])
def test_duration_bucket_refuses_short_or_invalid(seconds):
    with pytest.raises(ValueError):
        duration_bucket(seconds)


def test_nontarget_trials_are_exactly_the_same_gender_pairs():
    trials = nontarget_trials(ALL, n=len(NONTARGET_PAIRS))
    assert {(t.enrol.utt_id, t.test.utt_id) for t in trials} == NONTARGET_PAIRS
    assert len(trials) == len(NONTARGET_PAIRS)


def test_nontarget_trials_are_reproducible_with_a_seed():
    assert nontarget_trials(ALL, n=4, seed=7) == nontarget_trials(ALL, n=4, seed=7)


def test_nontarget_trials_refuse_more_pairs_than_exist():
    with pytest.raises(ValueError, match="only"):
        nontarget_trials(ALL, n=len(NONTARGET_PAIRS) + 1)


def test_trial_bucket_is_decided_by_the_shorter_clip():
    assert Trial(A1, A3).bucket == "3-10s"  # 8 s and 45 s
    assert Trial(A3, B1).bucket == "30-120s"  # 45 s and 150 s


def test_trial_rejects_a_clip_below_the_minimum():
    short = Utterance("e1", speaker="E", session="v6", gender="m", duration=2.5)
    with pytest.raises(ValueError):
        Trial(A1, short)


def test_write_trials_writes_a_header_and_one_row_per_trial(tmp_path):
    path = tmp_path / "trials.tsv"
    assert write_trials(path, [Trial(A1, A3), Trial(A3, B1)]) == 2
    assert path.read_text().splitlines() == [
        "enrol\ttest\ttarget\tbucket",
        "a1\ta3\t1\t3-10s",
        "a3\tb1\t0\t30-120s",
    ]
