"""Quality score inputs, the Appendix C composite, and the usability gate."""

import math

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from svs.extract import SAMPLE_RATE
from svs.quality import (
    bandwidth_hz,
    clipping_ratio,
    quality,
    snr_db,
    turn_consistency,
    unusable_reason,
)
from svs.quality_cache import speaker_rows

FRAME = 320  # 20 ms at 16 kHz


def test_composite_uses_the_appendix_c_weights():
    assert quality(25, 0.7, 0.0, 3500, 0.8) == pytest.approx(1.0)
    assert quality(5, 0.2, 0.02, 2500, 0.4) == 0.0
    # Only SNR at its best: its weight alone.
    assert quality(25, 0.2, 0.02, 2500, 0.4) == pytest.approx(0.30)
    # Only turn consistency at its best.
    assert quality(5, 0.2, 0.02, 2500, 0.8) == pytest.approx(0.25)


def test_snr_takes_the_noise_out_of_the_speech_frames():
    # 10 quiet frames at power 1e-4 (the noise floor) and 90 frames at 1.01e-2, which is
    # speech 1e-2 plus that noise. Speech over noise is 100, so 20 dB exactly.
    quiet = np.full(10 * FRAME, 0.01)
    loud = np.full(90 * FRAME, math.sqrt(1.01e-2))
    assert snr_db(np.concatenate([quiet, loud]), SAMPLE_RATE, loud) == pytest.approx(20.0)


def test_snr_edge_cases():
    loud = np.full(9 * FRAME, 0.1)
    with_silence = np.concatenate([np.zeros(FRAME), loud])
    assert snr_db(with_silence, SAMPLE_RATE, loud) == math.inf
    # Speech no louder than the floor: no speech power left after the noise is taken off.
    assert snr_db(loud, SAMPLE_RATE, loud) == -math.inf
    assert snr_db(loud, SAMPLE_RATE, np.empty(0)) == -math.inf


def test_clipping_is_measured_against_the_peak():
    # A full-scale sine clipped at 0.5: |sin| >= 0.5 on 2/3 of a period, all at the peak
    # (0.6625 here: 106 of the 160 samples of a 100 Hz period).
    t = np.arange(16000) / 16000
    clipped = np.clip(np.sin(2 * np.pi * 100 * t), -0.5, 0.5)
    assert clipping_ratio(clipped) == pytest.approx(2 / 3, abs=0.01)
    # The same clipping after a decoder that tops out below full scale (A-law: 0.984).
    assert clipping_ratio(0.984 * clipped) == clipping_ratio(clipped)
    assert clipping_ratio(np.zeros(100)) == 0.0


def test_bandwidth_of_wideband_and_once_narrowband_audio():
    noise = np.random.default_rng(0).normal(0, 0.1, 4 * SAMPLE_RATE)
    assert bandwidth_hz(noise) == 8000.0
    # Down to 8 kHz and back (svs.degrade, then svs.extract): above 4 kHz only the
    # filters' transition band is left, which flat noise shows up to about 4.6 kHz.
    narrow = resample_poly(resample_poly(noise, 1, 2), 2, 1)
    assert 3500 <= bandwidth_hz(narrow) < 5000
    assert bandwidth_hz(noise[:100]) == 0.0


def test_turn_consistency_is_the_mean_pairwise_cosine():
    e1, e2 = np.eye(2)
    assert turn_consistency([e1]) == 1.0
    assert turn_consistency([e1, e2]) == 0.0
    # Pairs (e1, e1), (e1, e2), (e1, e2): cosines 1, 0, 0.
    assert turn_consistency([e1, e1, e2]) == pytest.approx(1 / 3)


def test_gate_refuses_in_the_guide_order():
    # The guide's example: low quality, but refused for too little speech.
    assert unusable_reason(2.9, 0.41, 1.0) == "E_SV_INSUFFICIENT_SPEECH"
    assert unusable_reason(10.0, 0.30, 0.50) == "E_SV_TURN_INCONSISTENT"
    assert unusable_reason(10.0, 0.44, 1.0) == "E_SV_LOW_QUALITY"


def test_gate_thresholds_are_inclusive_floors():
    assert unusable_reason(3.0, 0.45, 0.55) is None


def test_a_video_pools_its_clips_and_its_turns_are_the_clips(tmp_path):
    rng = np.random.default_rng(0)
    ids = ["id1/vidA/00001", "id1/vidA/00002", "id1/vidA/00003", "id1/vidB/00001"]
    for i, utt_id in enumerate(ids):
        path = tmp_path / f"{utt_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, rng.normal(0, 0.1, (4 + i) * SAMPLE_RATE), SAMPLE_RATE, subtype="PCM_16")
    e1, e2 = np.eye(192, dtype=np.float32)[:2]
    nan = np.full(192, np.nan, np.float32)  # a clip under 3.0 s: not a turn
    embeddings = dict(zip(ids, [e1, e2, nan, e1], strict=True))

    rows = speaker_rows(tmp_path, ids, "pcm16k", embeddings)

    assert [(r["source"], r["unit"]) for r in rows] == [
        *(("clip", u) for u in ids),
        ("video", "id1/vidA"),
        ("video", "id1/vidB"),
    ]
    by_unit = {r["unit"]: r for r in rows}
    assert by_unit["id1/vidA"]["turn_consistency"] == "0"
    assert by_unit["id1/vidB"]["turn_consistency"] == "1"
    clip_seconds = sum(float(by_unit[u]["net_seconds"]) for u in ids[:3])
    assert float(by_unit["id1/vidA"]["net_seconds"]) == pytest.approx(clip_seconds, abs=0.02)
