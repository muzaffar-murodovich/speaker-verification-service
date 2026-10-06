"""Speech extraction helpers. Segment boundaries use milliseconds (SVS guide §3.2)."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad
from numpy.typing import NDArray
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320  # WebRTC accepts 20 ms frames at 16 kHz.


@dataclass
class ExtractedSpeech:
    """Mono float32 speech at 16 kHz; durations refer to selected segments only."""

    samples: NDArray[np.float32]
    selected_seconds: float

    @property
    def net_speech_seconds(self) -> float:
        return len(self.samples) / SAMPLE_RATE

    @property
    def speech_ratio(self) -> float:
        return self.net_speech_seconds / self.selected_seconds


def load_audio(path: str | Path, channel: int | None = None) -> tuple[NDArray, int]:
    """Decode a local file without mixing speakers on different channels.

    channel is zero-based. None is accepted only for mono recordings.
    Supported codecs depend on the installed libsndfile; decode other codecs
    to PCM WAV with ffmpeg before calling this function.
    """
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty and contain only finite samples")
    if channel is None:
        if audio.shape[1] != 1:
            raise ValueError("Select a channel explicitly for multichannel audio")
        channel = 0
    if isinstance(channel, bool) or not isinstance(channel, int):
        raise ValueError("Channel must be a zero-based integer")
    if not 0 <= channel < audio.shape[1]:
        raise ValueError("Channel is outside the audio channel range")
    return audio[:, channel], sample_rate


def extract_speech(
    path: str | Path,
    segments: Sequence[Sequence[float]],
    channel: int | None = None,
    vad_mode: int = 2,
) -> ExtractedSpeech:
    """Select segments, resample with anti-alias filtering, and pool VAD speech.

    VAD runs separately on each segment so gaps never become artificial speech.
    A final partial frame is zero-padded for detection only; padding is never
    returned or counted. Empty speech is a valid result, not usable speech.
    Mode 2 is a baseline choice, not a measured optimum.
    """
    if isinstance(vad_mode, bool) or vad_mode not in (0, 1, 2, 3):
        raise ValueError("VAD mode must be an integer from 0 to 3")
    audio, source_rate = load_audio(path, channel)
    validate_segments(segments, len(audio) * 1000 / source_rate)
    divisor = math.gcd(source_rate, SAMPLE_RATE)
    if source_rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE // divisor, source_rate // divisor)

    speech = []
    selected_samples = 0
    for start, end in segments:
        # ceil selects samples whose timestamps belong to [start, end).
        chunk = audio[math.ceil(start * SAMPLE_RATE / 1000) : math.ceil(end * SAMPLE_RATE / 1000)]
        if not len(chunk):
            raise ValueError("Segment contains no samples at the target sample rate")
        selected_samples += len(chunk)
        detector = webrtcvad.Vad(vad_mode)
        for offset in range(0, len(chunk), FRAME_SAMPLES):
            frame = chunk[offset : offset + FRAME_SAMPLES]
            padded = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
            pcm = (np.clip(padded, -1, 1) * 32767).astype("<i2").tobytes()
            if detector.is_speech(pcm, SAMPLE_RATE):
                speech.append(frame)

    pooled = np.concatenate(speech).astype(np.float32) if speech else np.empty(0, np.float32)
    return ExtractedSpeech(pooled, selected_samples / SAMPLE_RATE)


def validate_segments(segments: Sequence[Sequence[float]], duration_ms: float) -> None:
    """Require nonempty, ordered, non-overlapping segments inside the audio.

    Intervals are half-open: [start, end), so adjacent segments are allowed.
    Reject overlaps instead of counting the same speech twice.
    """
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("Audio duration must be finite and positive")
    if not segments:
        raise ValueError("At least one segment is required")

    previous_end = 0.0
    for segment in segments:
        if len(segment) != 2:
            raise ValueError("Each segment must contain start and end in milliseconds")
        start, end = segment
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Segment boundaries must be finite")
        if not 0 <= start < end <= duration_ms:
            raise ValueError("Segments must have positive length and stay inside the audio")
        if start < previous_end:
            raise ValueError("Segments must be ordered and must not overlap")
        previous_end = end
