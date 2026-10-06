# Log

Decisions, dead ends and the reasons behind parameters. Newest entry at the bottom.
Numbers live in `experiments.csv`; this file explains them.

## 2026-09-29 — D1: repository reset

- Started this repository from an empty tree. An earlier AI-generated draft was archived
  in a separate private repository and is not used as a code source: it had a placeholder
  embedding backend, no VAD, no pretrained model and no evaluation, so none of it could
  produce an honest number.
- Tooling: Python 3.12 managed by `uv`, `pytest` for tests, `ruff` for lint.
  `make setup && make eval` is the reproduction path; `make eval` fails on purpose
  until the first EER exists (D5), rather than pretending to succeed.
- Runtime dependencies are added on the day they are first needed, so every entry in
  `pyproject.toml` has a reason in this log.
- `experiments.csv` has its header now and a test (`tests/test_experiment_log.py`) that
  rejects rows with a missing column, a duplicate `run_id` or no trial count.
- Code must be device-agnostic (cuda → mps → cpu): development is on a laptop,
  embedding extraction on the GPU node.

## 2026-09-30 — D2: segment validation

- Started the extraction pipeline with segment validation (SVS guide §3.2).
  Boundaries are milliseconds, and intervals include the start but exclude the end.
- Reject empty, non-finite, reversed, out-of-bounds, unordered or overlapping
  segments. Adjacent segments are valid. Rejecting overlaps prevents counting the
  same audio twice when pooling speech; guard tests cover these cases.
- This is only the first step: audio decoding, sample-rate handling, VAD and speech
  pooling are not implemented yet. No runtime dependency or measured result added.

## 2026-10-02 — D2: audio extraction

- Added local audio decoding with SoundFile, explicit zero-based channel selection,
  anti-aliased resampling with SciPy, and WebRTC VAD speech pooling. NumPy provides
  sample arrays; webrtcvad-wheels provides the native VAD without a local C build.
- Output is mono float32 at 16 kHz. Segment validation precedes slicing. Half-open
  boundaries use ceil to include samples whose timestamps fall inside the interval.
- Implementation decisions: multichannel audio requires a channel (no implicit
  mix); VAD mode 2, 20 ms frames; detector state resets across segments. The last
  partial frame is padded only for classification, never for returned duration.
  These choices are baseline policies, not additional SVS guide requirements.
- Return pooled samples, selected seconds, net speech seconds and speech ratio.
  Empty speech is explicit. Scoring's 3-second guard belongs downstream; quality,
  diarization, URI fetching and embeddings are outside this D2 implementation.
- Guard tests cover invalid audio/channels/modes, silence, boundaries, gaps,
  padding and anti-alias filtering. A manual smoke check with the public upstream
  py-webrtcvad test-audio.raw (8 kHz PCM) retained speech and removed non-speech.
  This is a functional check, not an accuracy benchmark or experiment result.
- Added docs/audio-basics.md and a runnable API example. Codec encoders and
  degradation analysis remain D6 work. No measured verification metrics reported.
