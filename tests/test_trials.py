"""Trial guards: a same-session target trial must never be built (SVS guide §8.1)."""

import pytest

from svs.trials import Trial, Utterance, duration_bucket, target_trials

# Speaker A has two utterances in video v1 and one in v2; speaker B has one session only.
A1 = Utterance("a1", speaker="A", session="v1")
A2 = Utterance("a2", speaker="A", session="v1")
A3 = Utterance("a3", speaker="A", session="v2")
B1 = Utterance("b1", speaker="B", session="v3")
B2 = Utterance("b2", speaker="B", session="v3")


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
