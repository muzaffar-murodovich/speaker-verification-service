"""Degradation harness: turn clean 16 kHz audio into telephony conditions (handbook D6).

Each condition returns the audio as the service would receive it: the codec is
encoded and decoded again, and the sample rate stays at the channel rate. Scoring
code resamples to 16 kHz itself, the same way it handles real 8 kHz input.
"""

import io
import subprocess
import zlib
from functools import cache
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

from svs.extract import SAMPLE_RATE, load_audio
from svs.trials import MIN_SPEECH_SECONDS

TELEPHONY_RATE = 8000
# Condition names follow the codec labels of the SVS guide (§3.3, §8.3).
CONDITIONS = ("pcm16k", "pcm8k", "g711a", "g711u", "g729", "opus", "noise15", "noise5")
_G711_SUBTYPES = {"g711a": "ALAW", "g711u": "ULAW"}
# The handbook and the SVS guide name Opus but no bitrate. 16 kbit/s at 16 kHz is the
# low end of the wideband-speech range in RFC 7587 §3.1.1.
OPUS_BITRATE = "16k"
# ffmpeg cannot encode G.729; tools/g729.c wraps bcg729 and `make g729` builds it here.
G729_BINARY = Path("artifacts/bin/g729")
# MUSAN noise (make musan) mixed into clean 16 kHz audio without a codec, so noise and
# codec effects stay apart. The guides give no SNR levels; both lie inside the 5-25 dB
# range of the SVS guide's quality score (Appendix C), 5 dB at its worst end.
NOISE_DIR = Path("data/musan/noise")
NOISE_SNR_DB = {"noise15": 15.0, "noise5": 5.0}


def degrade(
    samples: NDArray[np.float32], condition: str, noise_key: str | None = None
) -> tuple[NDArray[np.float32], int]:
    """Return (samples, sample_rate) of 16 kHz mono audio after the given condition.

    Noise conditions need noise_key, usually the utt_id: it fixes which noise is used.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {CONDITIONS}")
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Samples must be nonempty, mono and finite")
    if condition == "pcm16k":
        return samples.astype(np.float32), SAMPLE_RATE
    if condition == "opus":
        return _opus_decode(_opus_encode(samples)), SAMPLE_RATE
    if condition in NOISE_SNR_DB:
        if noise_key is None:
            raise ValueError(f"{condition} needs a noise_key")
        noise, offset = pick_noise(noise_key)
        return add_noise(samples, noise, NOISE_SNR_DB[condition], offset), SAMPLE_RATE

    # resample_poly filters out everything above 4 kHz before dropping samples.
    narrow = resample_poly(samples, 1, SAMPLE_RATE // TELEPHONY_RATE).astype(np.float32)
    if condition == "pcm8k":
        return narrow, TELEPHONY_RATE
    if condition == "g729":
        return _g729_round_trip(narrow), TELEPHONY_RATE
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


def _g729_round_trip(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    """Encode and decode 8 kHz audio with G.729 through the bcg729 wrapper.

    The decoded audio lags by the codec's 5 ms look-ahead; the lag is kept, and the
    zero-padded tail of the last frame is trimmed so the length matches the input.
    """
    if not G729_BINARY.exists():
        raise FileNotFoundError(f"{G729_BINARY} is missing; run `make g729`")
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    out = subprocess.run([str(G729_BINARY)], input=pcm, capture_output=True, check=True).stdout
    return (np.frombuffer(out, dtype="<i2")[: len(samples)] / 32767).astype(np.float32)


def add_noise(
    samples: NDArray[np.float32], noise: NDArray[np.float32], snr_db: float, offset: int = 0
) -> NDArray[np.float32]:
    """Mix noise into samples so that their power ratio is exactly snr_db.

    Power is the mean square over the whole clip, silence included. Noise shorter
    than the clip is repeated; offset picks where in the noise the clip starts.
    """
    if not np.isfinite(snr_db):
        raise ValueError("SNR must be finite")
    signal_power = np.mean(samples.astype(np.float64) ** 2)
    if noise.ndim != 1 or not len(noise) or signal_power == 0:
        raise ValueError("Noise must be nonempty and mono, and samples must not be silent")
    piece = np.resize(np.roll(noise, -offset), len(samples)).astype(np.float64)
    noise_power = np.mean(piece**2)
    if noise_power == 0:
        raise ValueError("The noise piece is silent; its SNR is undefined")
    scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return (samples + scale * piece).astype(np.float32)


@cache
def noise_files(noise_dir: Path = NOISE_DIR) -> tuple[Path, ...]:
    """MUSAN noise files of at least MIN_SPEECH_SECONDS, in a fixed order.

    A shorter file cannot cover even the shortest scored clip without repeating, and
    a repeated short sound becomes a periodic click that real calls do not have.
    """
    files = tuple(
        path
        for path in sorted(noise_dir.rglob("*.wav"))
        if sf.info(str(path)).duration >= MIN_SPEECH_SECONDS
    )
    if not files:
        raise FileNotFoundError(f"No noise files in {noise_dir}; run `make musan`")
    return files


def pick_noise(key: str) -> tuple[NDArray[np.float32], int]:
    """The same key always gets the same noise file and start offset, on any machine.

    CRC32 is used instead of hash(), which Python randomises per process.
    """
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    files = noise_files(NOISE_DIR)
    noise, rate = load_audio(files[rng.integers(len(files))])
    if rate != SAMPLE_RATE:
        raise ValueError(f"Noise must be {SAMPLE_RATE} Hz, got {rate}")
    return noise, int(rng.integers(len(noise)))
