# Speaker Verification Service

A speaker verification (1:1) and identification (1:N) service built during the
MohirDev internship: speech extraction, ECAPA-TDNN embeddings, calibrated scoring,
speaker profiles and vector search.

**Status:** D2 extraction implemented and tested; D3 embeddings are next.
Progress and decisions are in [LOG.md](LOG.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and `make`.

```bash
make setup   # create .venv and install dependencies
make test    # run the tests
make eval    # reproduce the reported numbers (available from D5)
```

Run `make` with no target to list all targets.

## Layout

```
src/svs/          the package
tests/            tests, including guards that must fail if a safety check is removed
experiments.csv   one row per run that produces a number
LOG.md            decisions, dead ends and reasons
```

`data/` and `artifacts/` are local only and never committed.

## Reporting numbers

Every number in this repository states its dataset, condition, trial count and the
commit that produced it. A number without those is not reported.

## D2: extract speech

```python
from svs.extract import extract_speech

speech = extract_speech(
    "data/call.wav",
    segments=[(1200, 4400), (9100, 16300)],  # milliseconds
    channel=0,  # zero-based; omit only for mono audio
)
# speech.samples: mono float32 at 16 kHz, ready for D3
print(speech.net_speech_seconds, speech.speech_ratio)
```

Local audio is decoded with SoundFile. For codecs unsupported by the installed
libsndfile, first decode to PCM WAV using ffmpeg. No remote URI fetching is included.
Stereo requires explicit channel selection. Segments are validated, audio is
resampled with anti-alias filtering, and WebRTC VAD retains speech in 20 ms frames.
An empty result means no speech was detected; it must not be sent to the model.
VAD mode 2 is a baseline choice whose accuracy has not yet been measured.

See [audio fundamentals](docs/audio-basics.md) for sample rates, codecs, clipping,
channels and pooling. D3 adds embeddings; D2 does not compute quality or scores.
