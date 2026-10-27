"""Quality score of extracted speech and the usability gate (SVS guide §3.2, Appendix C).

The composite and its weights are the guide's Appendix C starting point. How each input
is measured is not in the guide; the choices below are plain ones, logged in LOG.md.
A change to a weight or a measurement changes every quality, so it is a new version.
"""

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch

from svs.extract import SAMPLE_RATE, ExtractedSpeech
from svs.trials import MIN_SPEECH_SECONDS

QUALITY_VERSION = "appendix-c-v1"
MIN_QUALITY = 0.45  # CFG_SV_MIN_QUALITY
MIN_TURN_CONSISTENCY = 0.55  # CFG_SV_MIN_TURN_CONSISTENCY

FRAME_SECONDS = 0.02  # the VAD frame length
NOISE_QUANTILE = 0.1  # the quietest 10 % of frames stand for the noise floor
CLIP_LEVEL = 0.99  # a sample this close to the clip's peak counts as clipped
BANDWIDTH_DROP_DB = 40.0  # bandwidth ends where the spectrum falls this far below its peak


def clamp(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def quality(
    snr_db: float,
    speech_ratio: float,
    clipping_ratio: float,
    bandwidth_hz: float,
    turn_consistency: float,
) -> float:
    """Composite quality in 0-1, with the weights of SVS guide Appendix C."""
    q_snr = clamp((snr_db - 5.0) / 20.0)  # 5 dB -> 0, 25 dB -> 1
    q_spr = clamp((speech_ratio - 0.2) / 0.5)
    q_clip = clamp(1.0 - clipping_ratio / 0.02)
    q_bw = clamp((bandwidth_hz - 2500) / 1000.0)
    q_cons = clamp((turn_consistency - 0.4) / 0.4)
    return 0.30 * q_snr + 0.20 * q_spr + 0.15 * q_clip + 0.10 * q_bw + 0.25 * q_cons


def frame_powers(audio: NDArray, rate: int) -> NDArray[np.float64]:
    """Mean power of each whole 20 ms frame; a final partial frame is dropped."""
    size = round(FRAME_SECONDS * rate)
    n = len(audio) // size
    frames = np.asarray(audio[: n * size], np.float64).reshape(n, size)
    return np.mean(frames**2, axis=1)


def snr_db(audio: NDArray, rate: int, speech: NDArray) -> float:
    """Speech-to-noise power ratio in dB.

    Noise is the mean power of the quietest 10 % of frames of the selected audio, which
    exist even when VAD finds no pause. The speech frames hold speech plus noise, so the
    noise power is taken off before the ratio. Digital silence as the floor gives +inf.
    """
    powers = frame_powers(audio, rate)
    if not len(powers) or not len(speech):
        return -math.inf
    quiet = np.sort(powers)[: max(1, int(NOISE_QUANTILE * len(powers)))]
    noise = float(np.mean(quiet))
    signal = float(np.mean(np.asarray(speech, np.float64) ** 2)) - noise
    if signal <= 0:
        return -math.inf
    return math.inf if noise == 0 else 10 * math.log10(signal / noise)


def clipping_ratio(audio: NDArray) -> float:
    """Share of samples within 1 % of the clip's peak.

    Relative to the peak, not to full scale: a G.711 decoder never returns full scale
    (A-law tops out at 0.984), so a full-scale test would miss clipping that went
    through a codec. A clean clip has a handful of samples near its peak.
    """
    magnitude = np.abs(audio)
    peak = float(magnitude.max()) if len(magnitude) else 0.0
    if peak == 0:
        return 0.0
    return float(np.mean(magnitude >= CLIP_LEVEL * peak))


def bandwidth_hz(speech: NDArray) -> float:
    """Highest frequency of the 16 kHz speech whose power is within 40 dB of the peak.

    The average spectrum comes from Welch's method with 32 ms windows. Audio that was
    once 8 kHz has nothing above 4 kHz after resampling, whatever it is resampled to.
    """
    if len(speech) < 512:
        return 0.0
    freqs, power = welch(speech, SAMPLE_RATE, nperseg=512)
    above = np.nonzero(power >= power.max() * 10 ** (-BANDWIDTH_DROP_DB / 10))[0]
    return float(freqs[above[-1]])


def turn_consistency(turns: Sequence[NDArray]) -> float:
    """Mean pairwise cosine of the turn embeddings pooled into one embedding (§3.2).

    One turn has nothing to disagree with, so it scores 1.0. Turn embeddings are
    unit length (the embedder normalises), so a dot product is the cosine.
    """
    if len(turns) < 2:
        return 1.0
    stacked = np.stack(turns).astype(np.float64)
    gram = stacked @ stacked.T
    n = len(turns)
    return float((gram.sum() - np.trace(gram)) / (n * (n - 1)))


def quality_detail(
    audio: NDArray, rate: int, speech: ExtractedSpeech, consistency: float = 1.0
) -> dict[str, float]:
    """The five inputs of the composite, as in the /v1/embed quality_detail (§3.2).

    audio is the selected segments at their own rate; speech is what VAD kept of them.
    """
    return {
        "snr_db": snr_db(audio, rate, speech.samples),
        "speech_ratio": speech.speech_ratio,
        "clipping_ratio": clipping_ratio(audio),
        "bandwidth_hz": bandwidth_hz(speech.samples),
        "turn_consistency": consistency,
    }


def unusable_reason(net_speech_seconds: float, quality: float, consistency: float) -> str | None:
    """The usability gate: the error code that refuses this speech, or None if usable.

    Checked in the order of the guide's §3.2 example, where 6.2 s at quality 0.41 is
    refused for speech, not quality: too little speech first, then a probable second
    speaker (a diarization error), then low quality.
    """
    if net_speech_seconds < MIN_SPEECH_SECONDS:
        return "E_SV_INSUFFICIENT_SPEECH"
    if consistency < MIN_TURN_CONSISTENCY:
        return "E_SV_TURN_INCONSISTENT"
    if quality < MIN_QUALITY:
        return "E_SV_LOW_QUALITY"
    return None
