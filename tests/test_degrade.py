"""Degradation guards: each condition must change the audio the way its channel does."""

import numpy as np
import pytest

from svs import degrade as degrade_module
from svs.degrade import CONDITIONS, TELEPHONY_RATE, _opus_encode, degrade
from svs.extract import SAMPLE_RATE


def tone(hz: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def energy(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2))


def snr_db(reference: np.ndarray, decoded: np.ndarray) -> float:
    return 10 * np.log10(energy(reference) / energy(decoded - reference))


def test_clean_condition_returns_input_unchanged():
    x = tone(440)
    y, rate = degrade(x, "pcm16k")
    assert rate == SAMPLE_RATE
    np.testing.assert_array_equal(y, x)


@pytest.mark.parametrize("condition", [c for c in CONDITIONS if c not in ("pcm16k", "opus")])
def test_telephony_conditions_run_at_8khz_with_half_the_samples(condition):
    y, rate = degrade(tone(440), condition)
    assert rate == TELEPHONY_RATE
    assert len(y) == SAMPLE_RATE // 2


def test_8khz_keeps_speech_band():
    y, _ = degrade(tone(1000), "pcm8k")
    # Skip the filter's start-up samples at both ends.
    assert energy(y[200:-200]) == pytest.approx(energy(tone(1000)), rel=0.01)


def test_8khz_removes_content_above_4khz_instead_of_folding_it_down():
    # Dropping every second sample without a filter would fold 6 kHz to 2 kHz at full energy.
    y, _ = degrade(tone(6000), "pcm8k")
    assert energy(y[200:-200]) < 1e-4 * energy(tone(6000))


@pytest.mark.parametrize("condition", ["g711a", "g711u"])
def test_g711_adds_quantisation_error_of_the_expected_size(condition):
    narrow, _ = degrade(tone(440), "pcm8k")
    coded, _ = degrade(tone(440), condition)
    # 8-bit companded codecs keep roughly 35-40 dB SNR; 0 error means the codec was skipped.
    assert 30 < snr_db(narrow, coded) < 45


def test_a_law_and_mu_law_are_different_codecs():
    a, _ = degrade(tone(440), "g711a")
    u, _ = degrade(tone(440), "g711u")
    assert not np.array_equal(a, u)


def test_g729_is_a_low_bitrate_codec_not_g711():
    narrow, _ = degrade(tone(440), "pcm8k")
    coded, _ = degrade(tone(440), "g729")
    # G.729 keeps how speech sounds, not the waveform: far below G.711's ~36 dB.
    assert snr_db(narrow, coded) < 20


def test_g729_asks_for_the_build_step_when_the_tool_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(degrade_module, "G729_BINARY", tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="make g729"):
        degrade(tone(440), "g729")


def test_opus_stays_at_16khz_with_every_sample_kept():
    y, rate = degrade(tone(440), "opus")
    assert rate == SAMPLE_RATE
    assert len(y) == SAMPLE_RATE


def test_opus_changes_the_audio():
    x = tone(440)
    y, _ = degrade(x, "opus")
    assert np.max(np.abs(y - x)) > 1e-3


def test_opus_encodes_at_the_chosen_bitrate():
    # Noise keeps the variable bitrate near its target; ffmpeg's default would be ~65 kbit/s.
    noise = (0.1 * np.random.default_rng(0).standard_normal(2 * SAMPLE_RATE)).astype(np.float32)
    kbit_per_second = len(_opus_encode(noise)) * 8 / 2 / 1000
    assert 12 < kbit_per_second < 20


@pytest.mark.parametrize(
    "samples",
    [np.zeros(0, np.float32), np.zeros((10, 2), np.float32), np.array([0.0, np.nan], np.float32)],
)
def test_rejects_empty_multichannel_or_nonfinite_audio(samples):
    with pytest.raises(ValueError):
        degrade(samples, "pcm8k")


def test_rejects_unknown_condition():
    with pytest.raises(ValueError):
        degrade(tone(440), "g722")
