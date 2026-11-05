"""D17 trial kinds on the slice, the breakdown axes, and the slice cohort's exclusions."""

import numpy as np

from svs.slice_breakdown import facets, kind_of, people, slice_cohort_stats
from svs.slice_service import queries


def test_only_yes_answers_make_a_person():
    answers = [
        {"group": "0", "item_id": "a.mp3#ref0", "same_person": "yes"},
        {"group": "0", "item_id": "b.mp3#ref1", "same_person": "no"},
        {"group": "0", "item_id": "c.mp3#ref0", "same_person": "unsure"},
        {"group": "1", "item_id": "d.mp3#ref0", "same_person": "yes"},
    ]
    assert people(answers) == {"a.mp3#ref0": 0, "d.mp3#ref0": 1}


def test_kinds_come_from_labels_not_scores():
    person = {"a.mp3#ref0": 0, "b.mp3#ref0": 0, "c.mp3#ref0": 1}
    gender = {"a.mp3#ref0": "F", "b.mp3#ref0": "F", "c.mp3#ref0": "F", "x.mp3#ref1": "M"}
    assert kind_of("a.mp3#ref0", "b.mp3#ref0", person, gender) == "target"
    assert kind_of("a.mp3#ref0", "a.mp3#ref1", person, gender) == "same_recording"
    assert kind_of("a.mp3#ref0", "x.mp3#ref1", person, gender) == "cross_gender"
    # Two confirmed people of one gender may still be one person: left out.
    assert kind_of("a.mp3#ref0", "c.mp3#ref0", person, gender) is None
    assert kind_of("y.mp3#ref0", "z.mp3#ref0", person, {}) is None


def test_a_pair_is_as_short_as_its_shorter_side():
    a = {"seconds": 45.0, "format": "mp3", "rate": 8000, "layout": "stereo", "source": "S"}
    b = {"seconds": 12.0, "format": "wav", "rate": 8000, "layout": "mono", "source": "T"}
    assert facets(a, b) == {
        "all": "all",
        "bucket": "10-30s",
        "format": "mixed",
        "rate": "8k",
        "layout": "mixed",
        "source": "different",
    }


def test_the_slice_cohort_leaves_out_both_recordings_and_both_people():
    recordings = ["r1", "r2", "r3", "r4", "r5"]
    owner = [0, 1, 0, None, None]
    raw = np.array(
        [
            [1.0, 0.5, 0.9, 0.1, 0.3],
            [0.5, 1.0, 0.2, 0.4, 0.6],
            [0.9, 0.2, 1.0, 0.0, 0.0],
            [0.1, 0.4, 0.0, 1.0, 0.0],
            [0.3, 0.6, 0.0, 0.0, 1.0],
        ]
    )
    mu, sd = slice_cohort_stats(raw, recordings, owner, 0, 1, n=2)
    # r1, r2 are the pair; r3 is person 0 again: only r4 and r5 remain.
    assert mu.tolist() == [0.2, 0.5]
    assert np.allclose(sd, [0.1, 0.1])


def test_service_queries_enrol_once_per_person_and_skip_their_recordings():
    answers = [
        {"group": "0", "item_id": "a.mp3#ref0", "same_person": "yes"},
        {"group": "0", "item_id": "b.mp3#ref1", "same_person": "yes"},
        {"group": "1", "item_id": "c.mp3#ref0", "same_person": "yes"},
    ]
    usable = ["a.mp3#ref0", "a.mp3#ref1", "b.mp3#ref1", "c.mp3#ref0", "d.mp3#ref0"]
    enrol, known, unknown = queries(answers, usable)
    assert enrol == {0: "a.mp3#ref0", 1: "c.mp3#ref0"}
    assert known == [("b.mp3#ref1", 0)]
    assert unknown == ["d.mp3#ref0"]
