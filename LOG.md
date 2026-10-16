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

## 2026-10-09 — D4: trial buckets and the trial-list file

- A trial's duration bucket is the bucket of its shorter clip. The shorter side limits
  how much the comparison can tell, and in the service the weak side is always the new
  occurrence while the enrolled side is a long profile. Both orders of a pair give the
  same bucket. To confirm with the mentor.
- `Trial` refuses a clip below 3.0 s, so no trial list can contain one.
- `write_trials()` writes a tab-separated list: enrol, test, target (1/0), bucket.
- Open problem: VoxCeleb1 clips are short (about 8 s on average per the dataset paper;
  to be measured on our copy). With one clip per side, the 30-120 s and >120 s buckets
  will be nearly empty. A likely fix is to join clips from the same video into longer
  clips; they stay one session, so the same-session guard still applies. Not built yet:
  it depends on the data layout and needs the mentor's agreement.
- D4 status: the builder is complete and tested on fake metadata. No trial list over
  real data exists yet, so M1's 100 000-trial count is not met.

## 2026-10-09 — Data: VoxCeleb1

- The official VoxCeleb server now requires credentials (HTTP 401). The mentor allowed
  downloading public data from the internet. VoxCeleb1 is kept because the published
  reference EER for our checkpoint is on VoxCeleb1, and none of its speakers were in
  the checkpoint's training data.
- Source: Hugging Face mirror `ProgramComputer/voxceleb`, revision pinned in the
  Makefile. The SHA-256 of every audio archive equals the official checksum that
  torchaudio's VoxCeleb1 loader uses, so the files are the original Oxford archives.
  `vox1_meta.csv` has no official checksum; its hash was recorded at download time.
- `make data` downloads with resume, checks `voxceleb1.sha256`, and unpacks to
  `data/voxceleb1/wav`. Metadata: 1,251 speakers (690 m, 561 f).
- Unpacked and counted on 2026-10-09: 1,251 speaker folders (identical to the ids in
  `vox1_meta.csv`), 22,496 videos, 153,516 wav files, 38 GB. Layout is
  `wav/<speaker>/<video>/<clip>.wav`; a sampled file is mono, 16 kHz, 16-bit PCM.

## 2026-10-12 — VoxCeleb1 catalogue

- `list_clips()` turns `wav/<speaker>/<video>/<clip>.wav` plus `vox1_meta.csv` into
  clips with speaker, session (video id) and gender. It reads paths only, not audio.
- utt_id keeps the video: clip names restart at 00001 in every video, so without it
  clips of one speaker from different videos would share an id.
- A speaker without gender in the metadata is an error, not a silent skip: it would
  otherwise drop out of same-gender non-target sampling unnoticed.
- Duration is not in the catalogue. Net speech seconds come from VAD when embeddings are
  computed, and only then does a clip become an `Utterance` with a duration bucket.
- Run on our copy: 153,516 clips with unique ids, 1,251 speakers, 22,496 videos;
  90,450 clips from male and 63,066 from female speakers.

## 2026-10-12 — Embedding cache

- `make embeddings` runs VAD and the model on every VoxCeleb1 clip once. It writes
  `artifacts/embeddings/voxceleb1/<speaker>.npz` with utt_ids, net speech seconds and
  192-dim embeddings. Trials and scores later read this cache instead of the audio.
- Net speech seconds are kept for every clip, so dropped clips can be counted. Clips
  under 3.0 s of speech get a NaN embedding: the service would refuse them, and NaN
  makes any accidental use visible as a NaN score.
- One file per speaker, written only after all its clips are done. An interrupted run
  leaves no partial file, and a rerun continues with the missing speakers.
- Speed probe before the full run (throwaway script, not committed): 100 random clips
  (seed 0) on the Mac (Apple M3), one process, 38 ms per clip; 83 % of the time is the
  model, not VAD. onnxruntime spreads the model over several CPU cores by default.
  Projected full run about 1.6 h. This is an estimate; the real duration is logged
  after the run.
- The Mac overheated during the full run, so the run moved to a home Linux PC
  (Intel i5-1235U, no GPU). The same 100 clips: 46.9 ms per clip there, 35.2 ms on a
  second Mac run. Both machines gave the same audio and net speech totals, so they
  read the same files and VAD agreed.

## 2026-10-13 — Embedding cache: full VoxCeleb1 run

- `make embeddings` ran on the home Linux PC (commit 4df7274): 1,251 speaker files,
  0 errors in the log, 12,487 s wall time (about 3.5 h). The 100-clip probe projected
  2.0 h for this machine, so the probe underestimated by about 1.7×. The cause was not
  measured.
- Cross-machine check: the 145 speakers the Mac finished before it was stopped were
  compared with the home files. Same utt_ids, same net seconds (max difference 0.0),
  same NaN rows, and every cosine between the two embeddings of a clip rounds to
  1.000000 (17,348 embeddings).
- Cache: 153,516 clips, all ids unique. 1,342 clips have under 3.0 s of speech and get
  NaN embeddings (NaN rows = 1,342, the same set). 152,174 clips remain, from all 1,251
  speakers and 22,398 of 22,496 videos. Per speaker: 43 to 995 clips, median 101.
  89,629 clips from male and 62,545 from female speakers.
- Net speech over all clips: median 5.86 s, 90th percentile 13.16 s, max 134.70 s.
- Kept clips per bucket: 3-10s 123,282; 10-30s 27,815; 30-120s 1,076; >120s 1.
  With single clips, the two long buckets are close to empty (open item 3).
- If the kept clips of each video were joined: 3-10s 5,497; 10-30s 6,501;
  30-120s 8,204; >120s 2,196 videos. Joining is not built; it is the candidate fix for
  open item 3.

## 2026-10-14 — Embedding cache: one embedding per video

- Long buckets were close to empty with single clips (see 2026-10-13), so each video's
  clips are joined: VAD runs per clip, the speech is concatenated and embedded once.
  A video is one session, so joining never mixes sessions. Averaging clip embeddings
  was rejected: the model would never hear the long audio the bucket names.
- Speech is capped at 300 s per video. Measured on the Mac with one model call: peak
  memory 0.85 GB at 120 s, 1.67 GB at 300 s, 8.13 GB at 1,718 s (the longest video).
  The home PC has 7 GB. 300 s is a plain choice, not tuned; the cached seconds are what
  the model saw.
- Missed step: raw file sizes (1.8 s to read) already showed one clip over 120 s and a
  1,940 s longest video. Checking them first would have settled joining and the cap
  before the first run.
- `make embeddings-videos` ran on the home PC (commit 6102634): 1,251 speaker files,
  0 errors, 11,550 s wall time.
- Checks: 22,496 videos, the same set as in the clip cache. Each video's seconds equal
  the sum of its clips' seconds, capped at 300 s (max difference 1.1e-13). 356 videos
  hit the cap. 92 videos have under 3.0 s of speech and get NaN rows.
- Kept videos per bucket: 3-10s 5,464; 10-30s 6,509; 30-120s 8,230; >120s 2,201.
  These differ slightly from the 2026-10-13 estimate, which summed only clips of 3 s
  or more and had no cap. All 1,251 speakers keep 7 to 36 videos (median 17).

## 2026-10-15 — The VoxCeleb1 trial list

- Short buckets (3-10s, 10-30s) come from single clips, long buckets (30-120s, >120s)
  from whole videos. A trial never mixes the two: a clip lies inside its video, so the
  two sides would share audio. Each row records its source (SVS guide §8.1).
- Per bucket: up to 50,000 target trials and 100,000 non-target trials, so every
  bucket has at least 100,000 trials. Plain round numbers, not tuned.
- Targets are drawn by reservoir sampling, so every cross-session pair of the bucket
  has the same chance; taking the first 50,000 would have used only the first few
  speakers in sorted order. Non-targets are same-gender, sampled per bucket.
- Available target pairs: 3-10s 10,871,284 (clips); 10-30s 496,483 (clips);
  30-120s 47,165 (videos); >120s 3,186 (videos, 535 speakers). The >120s bucket uses
  all of them; its EER will rest on few target trials and is less stable.
- `make trials` (seed 0) wrote 550,351 trials in 13 s. SHA-256 of
  artifacts/trials/voxceleb1.tsv: fd6cd63e8ebc4131499c8cd156354275fb70cb1fda0df70411da883200ab28cb.
  A rebuild gives the identical file.
- Independent check of the file: 0 duplicate pairs, 0 same-session targets, 0 wrong
  labels, 0 cross-gender non-targets, 0 trials mixing clip and video.
- Not met: same-language non-targets. VoxCeleb1 has nationality, not language
  (open item 2). Not built yet: the cohort-disjointness test; the cohort comes with
  AS-Norm.

## 2026-10-16 — Cosine scores for the VoxCeleb1 trial list

- `make scores` writes artifacts/scores/voxceleb1-cosine.tsv: the trial list with a
  score column. The score is the cosine of the enrol and test embeddings. The source
  column picks the cache (clip or video) for both sides. Scores keep every digit, so
  rounding cannot create ties.
- Run on the Mac: 550,351 scores in 3.7 s, peak memory 4.1 GB. SHA-256 of the file:
  f7dc03a3de47d49c2f79887400f7d5a3d0223d80f0df7084cfe3a7e76c44ef70.
- Independent check (plain dot products from the .npz files, not svs.score): rows in
  the same order as the trial list; max difference 1.2e-7 on 2,000 random rows; 0 NaN.
- Mean score per bucket, target / non-target: 3-10s 0.646 / 0.062; 10-30s 0.755 / 0.079;
  30-120s 0.811 / 0.088; >120s 0.849 / 0.090.
- 22 target trials score above 0.999 (10-30s 7, 30-120s 6, >120s 9), each from two
  different video ids, e.g. id10683/Fy5YN2c7608/00008 vs id10683/OMyNDgBxHSI/00008
  scores 1.0. These look like the same audio uploaded twice under two video ids, which
  the same-session guard cannot see. Not fixed yet: the first EER uses the list as it
  is, then the effect of removing such pairs is measured.

## 2026-10-16 — First EER per bucket (before duplicate-audio cleanup)

- `make eer` (commit b7b879e) reads the score file and prints the EER per bucket.
  A trial is accepted when score >= threshold. The EER is the mean of the miss and
  false-alarm rates at the threshold where they are closest.
- Results, raw cosine, duplicate audio NOT removed (experiments.csv d5-eer-*):

  | bucket  | EER %  | threshold | n_target | n_nontarget |
  |---------|--------|-----------|----------|-------------|
  | 3-10s   | 1.582  | 0.3476    | 50,000   | 100,000     |
  | 10-30s  | 0.592  | 0.4265    | 50,000   | 100,000     |
  | 30-120s | 0.638  | 0.4524    | 47,165   | 100,000     |
  | >120s   | 0.345  | 0.4873    | 3,186    | 100,000     |

- Gap between miss and false-alarm rate at the chosen threshold: at most 0.001 pp.
- Independent check: scikit-learn roc_curve (not a project dependency) gives the same
  numbers with drop_intermediate=False. With its default it skips thresholds and gives
  0.339 % for >120s; our code keeps every threshold.
- In >120s one target error moves the miss rate by 0.031 pp; 0.345 % rests on about
  11 missed targets. This number is less stable than the others.
- 10-30s (clips) is lower than 30-120s (videos). Not explained yet; the two buckets
  come from different sources.
- Not comparable to the published 0.878 % (vox1-O-clean, a different trial list).
- These numbers are a baseline. The duplicate-audio pairs make target trials easier,
  so the EER can only go up when they are removed. That cleanup is the next step.
