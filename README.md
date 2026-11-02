# Speaker Verification Service

A speaker verification (1:1) and identification (1:N) service built during the
MohirDev internship: speech extraction, ECAPA-TDNN embeddings, calibrated scoring,
speaker profiles and vector search.

**Status:** VoxCeleb1 evaluation, calibration per condition, profiles, 1:N search and
the HTTP service are done (D1-D15), and so are the operations pieces of D18. Next is the
internal call-centre slice (D16-D17). Progress, decisions and open questions are in
[LOG.md](LOG.md); every reported number is a row of [experiments.csv](experiments.csv).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and `make`.

```bash
make setup   # create .venv and install dependencies
make test    # run the tests
make eval    # reproduce the reported EER per bucket and the DET plot
```

## The service

Requires Docker with `docker compose`.

```bash
make model     # download the embedding model (checksum-verified)
make service   # build and start Qdrant and the API on :8000
make smoke     # call every endpoint once (reads three VoxCeleb1 clips under data/)
```

| Endpoint | What it does |
|---|---|
| `POST /v1/embed` | Speech segments of a file under `data/` -> embedding, net speech, quality, usable or the reason it is not |
| `POST /v1/verify` | 1:1: raw cosine, AS-Norm score, band, and the operating-point row behind the band |
| `POST /v1/identify` | 1:N in a filtered pool: candidates with bands and `margin_to_next`, a decision, stage latencies |
| `POST /v1/profiles` | Enrol a PROVISIONAL profile (`Idempotency-Key` supported) |
| `PATCH /v1/profiles/{id}` | Add a STRONG-matched occurrence; promote when the evidence rule holds |
| `GET /v1/healthz` | Liveness and the model version |
| `GET /metrics` | Prometheus text: embed time and refusals, identify time by stage, decisions by band, profiles by status, cohesion |

Every JSON response carries `request_id` and `model_version`; errors are `E_SV_*` codes.
The thresholds come from [docs/operating-points.tsv](docs/operating-points.tsv); the AS-Norm
cohorts are committed in `cohorts/`. Operations: [docs/runbook.md](docs/runbook.md), and
`make cohesion-check` once a week. `docker compose down` stops both containers; profiles
stay in the `qdrant-storage` volume.

## Evaluation

`make eval` builds whatever is missing, in order: the model (24 MB), VoxCeleb1
(31 GB of archives), clip and video embeddings, the shared-audio video pairs, then
the trial list, scores, EER and DET. On a clean clone the embeddings take most of
the time: 12,487 s and 11,550 s on our home PC, plus 1,791 s for the video pairs on
an M3 Mac. Each long step writes a `.done` file, so a later `make eval` skips it and
takes about 20 s. At the end the trial list and scores are checked against
[eval.sha256](eval.sha256), the files of the run in LOG.md; another machine may
differ in the last float digits, so a mismatch is reported, not treated as a failure.

VoxCeleb comes from a Hugging Face mirror that needs an account since October 2026:
accept the dataset's terms on its page, then run `hf auth login` (or set `HF_TOKEN`).
Without it the download fails with HTTP 401.

Run `make` with no target to list all targets.

## Layout

```
src/svs/          the package
tests/            tests, including guards that must fail if a safety check is removed
docs/             operating-point table, figures, runbook, audio basics
cohorts/          the AS-Norm cohort embeddings per condition, with checksums
experiments.csv   one row per run that produces a number
LOG.md            decisions, dead ends and reasons
```

`data/` and `artifacts/` are local only and never committed.

## Reporting numbers

Every number in this repository states its dataset, condition, trial count and the
commit that produced it. A number without those is not reported.

## Extract speech in Python

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
channels and pooling.
