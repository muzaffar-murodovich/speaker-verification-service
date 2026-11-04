"""Same-person candidates: the link threshold, the groups, and the clips to listen to."""

import numpy as np
import soundfile as sf

from svs.slice_link import clip, groups, link_threshold, listen_page, recording_of


def pair(left, right, score):
    return {
        "left": left,
        "right": right,
        "same_recording": recording_of(left) == recording_of(right),
        "score": str(score),
    }


PAIRS = [
    pair("a.mp3#ref0", "a.mp3#ref1", 9.0),  # two people in one call: the threshold
    pair("a.mp3#ref0", "b.mp3#ref0", 12.0),
    pair("b.mp3#ref0", "c.wav#ch1", 10.0),
    pair("a.mp3#ref0", "c.wav#ch1", 4.0),
    pair("a.mp3#ref1", "b.mp3#ref0", 2.0),
    pair("a.mp3#ref1", "c.wav#ch1", 9.0),  # equal to the threshold: not a link
]
IDS = ["a.mp3#ref0", "a.mp3#ref1", "b.mp3#ref0", "c.wav#ch1"]


def test_the_threshold_is_the_highest_pair_inside_one_recording():
    assert link_threshold(PAIRS) == 9.0


def test_links_above_the_threshold_chain_into_one_group():
    rows = groups(IDS, PAIRS, {i: "F" for i in IDS}, threshold=9.0)
    # a-b and b-c link; a-c (4.0) does not, but the chain joins all three.
    assert [r["item_id"] for r in rows] == ["a.mp3#ref0", "b.mp3#ref0", "c.wav#ch1"]
    assert {r["min_score"] for r in rows} == {"4.0000"}
    assert {r["genders"] for r in rows} == {"F3"}


def test_mixed_gender_labels_show_in_the_group():
    genders = {"a.mp3#ref0": "F", "b.mp3#ref0": "M", "c.wav#ch1": "F"}
    assert {r["genders"] for r in groups(IDS, PAIRS, genders, threshold=9.0)} == {"F2 M1"}


def test_a_pair_inside_one_recording_never_links():
    rows = groups(IDS, PAIRS, {}, threshold=0.0)
    assert "a.mp3#ref1" in {r["item_id"] for r in rows}  # through b and c, not through a#ref0
    assert groups(["a.mp3#ref0", "a.mp3#ref1"], PAIRS[:1], {}, threshold=0.0) == []


def test_a_clip_takes_the_longest_segments_up_to_its_length_in_time_order(tmp_path):
    rate = 8000
    audio = np.zeros((10 * rate, 2), np.float32)
    audio[:, 1] = np.arange(10 * rate) / (10 * rate)  # the value tells the time
    sf.write(tmp_path / "c.wav", audio, rate, subtype="FLOAT")
    row = {"audio": "c.wav", "channel": "1", "segments": "0-1000,2000-5000,6000-8000"}
    samples, got_rate = clip(tmp_path, row, seconds=4.0)
    assert got_rate == rate and len(samples) == 4 * rate
    # 3 s from 2-5 s, then 1 s of the 2 s at 6-8 s; the 1 s at 0-1 s is left out.
    assert samples[0] == np.float32(0.2) and samples[-1] < np.float32(0.7)


def test_the_page_carries_every_row_and_no_row_can_close_the_script():
    rows = [
        {
            "group": 0,
            "item_id": "a.mp3#ref0",
            "clip": "listen/x.wav",
            "seconds": "10.0",
            "gender": "F",
            "same_person": "",
        },
        {
            "group": 0,
            "item_id": "</script><b>",
            "clip": "y",
            "seconds": "1",
            "gender": "",
            "same_person": "",
        },
    ]
    page = listen_page(rows)
    assert "/*ROWS*/" not in page and '"a.mp3#ref0"' in page
    assert page.count("</script>") == 1
