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

## 2026-10-07 — D3: pretrained ECAPA-TDNN

- Checkpoint: WeSpeaker `ECAPA_TDNN_GLOB_c512-ASTP-emb192`, large-margin fine-tuned,
  ONNX export, Hugging Face revision pinned in the Makefile with a SHA-256 check.
  Licence CC BY 4.0.
- Why not SpeechBrain `spkrec-ecapa-voxceleb`: its model card says it was trained on
  VoxCeleb1 + VoxCeleb2. Then 1,211 of the 1,251 VoxCeleb1 speakers were seen in
  training, and an EER on them would be flattering. The WeSpeaker model was trained on
  VoxCeleb2 dev only (its `run.sh` and `config.yaml`), so all VoxCeleb1 speakers are
  unseen and M1's ≥ 1,000 speakers can come from VoxCeleb1.
- Published reference for D5 (model card, LM, no AS-Norm): EER 0.878 % on
  vox1-O-clean. This is their number, not ours.
- Runtime: onnxruntime + kaldi-native-fbank instead of torch. Features follow
  WeSpeaker's ONNX example: 80-bin fbank, 25/10 ms, hamming, no dither, int16 scale,
  mean normalisation. One-off check on a 9 s synthetic clip: max fbank difference
  to torchaudio's Kaldi fbank 5.4e-4, embedding cosine 1.0 (rounded). A functional
  check, not an experiment.
- Embeddings are L2-normalised where they are produced (SVS guide §4.1).
  CUDA is used when onnxruntime-gpu provides it; otherwise CPU.

## 2026-10-08 — D4: same-session guard

- VoxCeleb1 audio is not available yet, so the trial builder is written and tested on
  small fake metadata first.
- A session is the recording source; for VoxCeleb it is the video id. `Trial` refuses
  to exist when both sides are the same speaker from the same session (SVS guide §8.1,
  pitfall #1). `target_trials` also skips those pairs, so the generator never trips the guard.
- Each pair is emitted once (enrol < test by utt_id): cosine is symmetric, and both orders
  would double the trial count without adding information.
- Both guards were mutation-checked: removing either one makes a test fail.

## 2026-10-08 — D4: duration buckets

- `duration_bucket()` maps net speech seconds to 3-10, 10-30, 30-120 and >120 s
  (SVS guide §4.4, §8.3). Intervals are half-open, like segments in D2: 10.0 s is 10-30.
- Below 3.0 s (`CFG_SV_MIN_SPEECH_SECONDS`) it raises instead of returning a bucket: the
  service refuses to score such clips, so they must not enter the evaluation either.
  NaN and infinity are rejected explicitly, because `NaN < 3.0` is false.

## 2026-10-08 — D4: same-gender non-target trials

- `nontarget_trials()` samples n distinct pairs of different speakers of the same gender
  (SVS guide §8.1): cross-gender impostors are easy to reject and flatter the EER.
  Enumerating all pairs is not possible at VoxCeleb scale, so pairs are drawn at random
  with a fixed seed and rejected if same-speaker or already drawn. After 100·n attempts
  it raises instead of looping forever or returning fewer trials than asked.
- The enrol side is drawn uniformly over utterances, so the gender mix follows the data.
- Deviation: the handbook also asks for same-language non-targets. VoxCeleb1 metadata
  has nationality, not language, so language is not matched yet. To raise with the mentor.
