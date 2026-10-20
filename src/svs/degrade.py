"""Degradation harness: turn clean 16 kHz audio into telephony conditions (handbook D6).

Each condition returns the audio as the service would receive it: the codec is
encoded and decoded again, and the sample rate stays at the channel rate. Scoring
code resamples to 16 kHz itself, the same way it handles real 8 kHz input.
"""

import io
import subprocess

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

from svs.extract import SAMPLE_RATE

TELEPHONY_RATE = 8000
# Condition names follow the codec labels of the SVS guide (§3.3, §8.3).
CONDITIONS = ("pcm16k", "pcm8k", "g711a", "g711u", "opus")
_G711_SUBTYPES = {"g711a": "ALAW", "g711u": "ULAW"}
# The handbook and the SVS guide name Opus but no bitrate. 16 kbit/s at 16 kHz is the
# low end of the wideband-speech range in RFC 7587 §3.1.1.
OPUS_BITRATE = "16k"


def degrade(samples: NDArray[np.float32], condition: str) -> tuple[NDArray[np.float32], int]:
    """Return (samples, sample_rate) of 16 kHz mono audio after the given condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Samples must be nonempty, mono and finite")
    if condition == "pcm16k":
        return samples.astype(np.float32), SAMPLE_RATE
    if condition == "opus":
        return _opus_decode(_opus_encode(samples)), SAMPLE_RATE

    # resample_poly filters out everything above 4 kHz before dropping samples.
    narrow = resample_poly(samples, 1, SAMPLE_RATE // TELEPHONY_RATE).astype(np.float32)
    if condition == "pcm8k":
        return narrow, TELEPHONY_RATE
    return _codec_round_trip(narrow, _G711_SUBTYPES[condition]), TELEPHONY_RATE


def _codec_round_trip(samples: NDArray[np.float32], subtype: str) -> NDArray[np.float32]:
    """Encode to an in-memory WAV with the given codec and decode it again."""
    buffer = io.BytesIO()
    # The codec takes values in [-1, 1]; clip here so the overflow is visible in code.
    sf.write(buffer, np.clip(samples, -1, 1), TELEPHONY_RATE, format="WAV", subtype=subtype)
    buffer.seek(0)
    decoded, _ = sf.read(buffer, dtype="float32")
    return decoded


def _ffmpeg(args: list[str], data: bytes) -> bytes:
    command = ["ffmpeg", "-v", "error", *args]
    return subprocess.run(command, input=data, capture_output=True, check=True).stdout


def _opus_encode(samples: NDArray[np.float32]) -> bytes:
    """16 kHz mono float32 to an Ogg Opus stream in VoIP mode at OPUS_BITRATE."""
    raw = ["-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0"]
    codec = ["-c:a", "libopus", "-b:a", OPUS_BITRATE, "-application", "voip", "-f", "ogg"]
    return _ffmpeg([*raw, *codec, "pipe:1"], samples.astype("<f4").tobytes())


def _opus_decode(stream: bytes) -> NDArray[np.float32]:
    out = ["-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"]
    return np.frombuffer(_ffmpeg(["-i", "pipe:0", *out], stream), dtype="<f4").copy()
