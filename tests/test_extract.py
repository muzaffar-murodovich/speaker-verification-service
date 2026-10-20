"""Segment guards: invalid ranges must never reach speech pooling."""

import numpy as np
import pytest
import soundfile as sf

from svs.extract import (
    SAMPLE_RATE,
    extract_speech,
    load_audio,
    speech_from_samples,
    validate_segments,
)


def test_accepts_adjacent_segments_gaps_and_audio_boundaries():
    validate_segments([(0, 100), (100, 200), (300, 1000)], duration_ms=1000)


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [(0,)],
        [(0, 100, 200)],
        [(-1, 100)],
        [(100, 100)],
        [(200, 100)],
        [(0, 1001)],
        [(0, float("inf"))],
        [(float("nan"), 100)],
        [(0, 100), (99, 200)],
        [(300, 400), (0, 100)],
    ],
)
def test_rejects_invalid_segments(segments):
    with pytest.raises(ValueError):
        validate_segments(segments, duration_ms=1000)


@pytest.mark.parametrize("duration_ms", [0, -1, float("inf"), float("nan")])
def test_rejects_invalid_audio_duration(duration_ms):
    with pytest.raises(ValueError):
        validate_segments([(0, 100)], duration_ms=duration_ms)


def write_audio(tmp_path, audio, rate=16000, subtype="FLOAT"):
    path = tmp_path / "audio.wav"
    sf.write(path, audio, rate, subtype=subtype)
    return path


def test_channel_selection_never_mixes_speakers(tmp_path):
    audio = np.column_stack([np.full(160, 0.25), np.full(160, -0.5)])
    path = write_audio(tmp_path, audio)
    with pytest.raises(ValueError, match="explicitly"):
        load_audio(path)
    samples, rate = load_audio(path, channel=1)
    assert rate == 16000
    np.testing.assert_array_equal(samples, audio[:, 1])
    for invalid in [-1, 2, True, 0.5]:
        with pytest.raises(ValueError):
            load_audio(path, invalid)


@pytest.mark.parametrize("audio", [np.empty(0), np.array([np.nan]), np.array([np.inf])])
def test_rejects_empty_or_nonfinite_audio(tmp_path, audio):
    with pytest.raises(ValueError):
        load_audio(write_audio(tmp_path, audio))


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000])
def test_real_vad_removes_silence_at_different_rates(tmp_path, rate):
    result = extract_speech(write_audio(tmp_path, np.zeros(rate), rate), [(0, 1000)])
    assert result.samples.dtype == np.float32
    assert result.net_speech_seconds == 0
    assert result.selected_seconds == 1
    assert result.speech_ratio == 0


def test_segment_pooling_excludes_gaps_and_padding(tmp_path, monkeypatch):
    calls = []

    class Detector:
        def __init__(self, mode):
            self.calls = 0

        def is_speech(self, pcm, rate):
            assert len(pcm) == 640
            assert rate == SAMPLE_RATE
            calls.append(np.frombuffer(pcm, dtype="<i2"))
            self.calls += 1
            return self.calls == 1  # Keep first frame of each independent segment.

    monkeypatch.setattr("svs.extract.webrtcvad.Vad", Detector)
    audio = np.arange(1600, dtype=np.float32) / 2000
    result = extract_speech(write_audio(tmp_path, audio), [(0, 40), (90, 100)])
    np.testing.assert_array_equal(result.samples, np.concatenate([audio[:320], audio[1440:]]))
    assert result.selected_seconds == 0.05
    assert result.net_speech_seconds == 0.03
    assert result.speech_ratio == 0.6
    assert np.all(calls[-1][160:] == 0)


def test_resampling_preserves_duration_and_filters_aliases(tmp_path, monkeypatch):
    class KeepAll:
        def __init__(self, mode):
            pass

        def is_speech(self, pcm, rate):
            return True

    monkeypatch.setattr("svs.extract.webrtcvad.Vad", KeepAll)
    time = np.arange(48000) / 48000
    low = extract_speech(
        write_audio(tmp_path, 0.5 * np.sin(2 * np.pi * 1000 * time), 48000), [(0, 1000)]
    )
    high = extract_speech(
        write_audio(tmp_path, 0.5 * np.sin(2 * np.pi * 12000 * time), 48000), [(0, 1000)]
    )
    assert len(low.samples) == 16000
    assert low.net_speech_seconds == 1
    assert np.std(high.samples[320:-320]) < np.std(low.samples[320:-320]) / 100


@pytest.mark.parametrize("mode", [-1, 4, True, 1.5, "2"])
def test_invalid_vad_mode(tmp_path, mode):
    with pytest.raises(ValueError):
        extract_speech(write_audio(tmp_path, np.zeros(1600)), [(0, 100)], vad_mode=mode)


def test_pipeline_validates_segments_before_vad(tmp_path):
    path = write_audio(tmp_path, np.zeros(16000))
    with pytest.raises(ValueError, match="overlap"):
        extract_speech(path, [(0, 600), (500, 1000)])
    with pytest.raises(ValueError, match="no samples"):
        extract_speech(path, [(0.001, 0.002)])


def test_samples_in_memory_give_the_same_speech_as_the_file(tmp_path):
    # Degraded audio (D6) enters through speech_from_samples; clean audio must not change.
    rng = np.random.default_rng(0)
    audio = (0.3 * rng.standard_normal(8000) * np.sin(np.linspace(0, 20, 8000))).astype(np.float32)
    path = write_audio(tmp_path, audio, 8000)
    from_file = extract_speech(path, [(0, 1000)])
    from_memory = speech_from_samples(load_audio(path)[0], 8000, [(0, 1000)])
    assert len(from_file.samples) > 0
    np.testing.assert_array_equal(from_memory.samples, from_file.samples)
    assert from_memory.selected_seconds == from_file.selected_seconds
