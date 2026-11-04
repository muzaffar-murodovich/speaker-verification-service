"""The minimal diarizer: regions, windows, clusters, turns, and DER against a reference."""

import json

import numpy as np
import pytest
import soundfile as sf

from svs import diarize
from svs.diarize import cluster, der, speech_of, speech_regions, turns, windows
from svs.extract import FRAME_SAMPLES, SAMPLE_RATE
from svs.slice_labels import merged, read_export, reference_manifest
from svs.slice_run import parse_segments, segments_text


def flags_of(text: str) -> np.ndarray:
    """ "11..1" -> frame flags; 1 is speech."""
    return np.array([c == "1" for c in text])


def test_a_short_silence_stays_inside_a_region_and_a_long_one_splits_it():
    flags = flags_of("0111" + "00" + "11" + "00000" + "1")
    assert speech_regions(flags, gap=2) == [(1, 8), (13, 14)]
    assert speech_regions(flags, gap=1) == [(1, 4), (6, 8), (13, 14)]


def test_a_region_is_cut_into_the_fewest_equal_windows():
    assert windows([(0, 150)], size=75) == [(0, 75), (75, 150)]
    # 160 frames need three windows of at most 75: 53, 54, 53.
    assert windows([(0, 160)], size=75) == [(0, 53), (53, 107), (107, 160)]
    assert windows([(10, 20)], size=75) == [(10, 20)]


def test_clusters_are_numbered_by_first_appearance():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert cluster(np.stack([b, b, a, b, a])).tolist() == [0, 0, 1, 0, 1]
    assert cluster(np.stack([a])).tolist() == [0]


def test_a_turn_spans_the_gaps_and_the_speech_does_not():
    labelled = [(0, 1000, 0), (1200, 2000, 0), (2000, 3000, 0), (3000, 4000, 1), (5000, 6000, 0)]
    assert turns(labelled) == {0: [(0, 3000), (5000, 6000)], 1: [(3000, 4000)]}
    assert speech_of(labelled) == {
        0: [(0, 1000), (1200, 3000), (5000, 6000)],
        1: [(3000, 4000)],
    }


class ToneEmbedder:
    """Two 'voices': the sign of the first sample tells them apart."""

    def embed(self, samples):
        return np.array([1.0, 0.0]) if samples[0] > 0 else np.array([0.0, 1.0])


def test_diarize_finds_two_alternating_speakers(monkeypatch):
    # 2 s of speaker A, 0.6 s silence, 2 s of B, 0.6 s silence, 2 s of A, at 16 kHz.
    second = SAMPLE_RATE
    a, b, quiet = np.full(2 * second, 0.5), np.full(2 * second, -0.5), np.zeros(6 * second // 10)
    audio = np.concatenate([a, quiet, b, quiet, a]).astype(np.float32)
    monkeypatch.setattr(diarize, "speech_frames", lambda x: np.abs(x[::FRAME_SAMPLES]) > 0)
    found = diarize.diarize(ToneEmbedder(), audio, SAMPLE_RATE)
    assert found == [
        (0, 1000, 0),
        (1000, 2000, 0),
        (2600, 3600, 1),
        (3600, 4600, 1),
        (5200, 6200, 0),
        (6200, 7200, 0),
    ]


def test_no_window_long_enough_means_no_speakers(monkeypatch):
    audio = np.full(SAMPLE_RATE // 4, 0.5, np.float32)  # 0.25 s, under MIN_WINDOW_FRAMES
    monkeypatch.setattr(diarize, "speech_frames", lambda x: np.abs(x[::FRAME_SAMPLES]) > 0)
    assert diarize.diarize(ToneEmbedder(), audio, SAMPLE_RATE) == []


def test_der_counts_missed_false_alarm_and_confusion():
    reference = {"x": [(0, 4000)], "y": [(4000, 8000)]}
    # 1 s of y missed, 1 s false alarm after the end, and 1 s of x given to y's speaker.
    hypothesis = {0: [(0, 3000)], 1: [(3000, 7000), (8000, 9000)]}
    r = der(reference, hypothesis)
    assert r["ref_seconds"] == pytest.approx(8.0)
    assert (r["missed"], r["false_alarm"], r["confusion"]) == pytest.approx((1.0, 1.0, 1.0))
    assert r["der"] == pytest.approx(3 / 8)


def test_der_pairs_speakers_whatever_their_names():
    reference = {"x": [(0, 1000)], "y": [(1000, 2000)]}
    assert der(reference, {7: [(1000, 2000)], 3: [(0, 1000)]})["der"] == 0
    assert der(reference, {})["missed"] == pytest.approx(2.0)


def test_segments_round_inwards_and_read_back():
    text = segments_text([(1200.4, 4400.6), (5000.2, 5000.9)])
    assert text == "1201-4400"
    assert parse_segments(text) == [(1201.0, 4400.0)]


def test_reference_segments_are_sorted_joined_and_clipped():
    assert merged([(500, 900), (0, 600), (1000, 1200), (1500, 2500)], 2000) == [
        (0, 900),
        (1000, 1200),
        (1500, 2000),
    ]


def test_the_export_gives_mono_speakers_and_stereo_channels(tmp_path):
    def seg(start, end, speaker_id):
        return {"start": start, "end": end, "speaker_id": speaker_id}

    export = {
        "data": [
            {"id": "m1", "segments": [seg(0, 1, "spk-a"), seg(1, 2, "spk-b"), seg(2, 3, "spk-a")]},
            {
                "id": "s1",
                "segments": [seg(0, 2, "speaker-right-channel"), seg(1, 3, "speaker-left-channel")],
            },
            {"id": "gone", "segments": [seg(0, 1, "spk-a")]},
        ]
    }
    (tmp_path / "export.json").write_text(json.dumps(export))
    # 2.9996 s: files.tsv would say 3.000, and a segment clipped there would end outside.
    sf.write(tmp_path / "m1.flac", np.zeros(47994, np.float32), 16000)
    sf.write(tmp_path / "s1.wav", np.zeros((24000, 2), np.float32), 8000)
    files = [
        {"audio": "m1.flac", "channels": "1", "seconds": "3.000", "error": ""},
        {"audio": "s1.wav", "channels": "2", "seconds": "3.000", "error": ""},
    ]
    reference = read_export(tmp_path / "export.json", files, tmp_path)
    assert reference["m1.flac"]["speakers"] == {
        "spk-a": [(0, 1000), (2000, 2999.625)],
        "spk-b": [(1000, 2000)],
    }
    rows = reference_manifest(reference)
    assert [(r["item_id"], r["channel"], r["segments"]) for r in rows] == [
        ("m1.flac#ref0", "", "0-1000,2000-2999"),
        ("m1.flac#ref1", "", "1000-2000"),
        ("s1.wav#ref0", 1, "0-2000"),
        ("s1.wav#ref1", 0, "1000-3000"),
    ]
