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

## 2026-10-19 — Videos that hold the same audio under two video ids

- Some interviews were uploaded twice. The same-session guard compares video ids, so a
  target trial across two such videos passes it although both sides are one session.
- Embedding cosine alone cannot find them. Cross-video clip pairs of one speaker
  fall off smoothly with cosine, with no gap; same-video clip pairs also reach
  cosine >= 0.9999 (38 pairs). Cosine measures how alike two voices are, not whether
  the audio is the same.
- Decision rule: normalised cross-correlation of the two waveforms at the best lag
  (`shared_audio`). 500 random cross-video pairs of one speaker reached at most 0.131
  (median 0.04). Among the 3,933 pairs at cosine >= 0.95, 3,506 were above 0.2 and most
  above 0.5. MIN_SHARED = 0.2.
- Filter: only clip pairs at cosine >= MIN_COSINE are correlated. Samples found shared
  audio in 41 of 1,000 pairs at 0.90-0.95 (those 41 gave 10 video pairs not found at
  >= 0.95) and in 0 of 500 per band at 0.85-0.90, 0.80-0.85, 0.7-0.8, 0.6-0.7,
  0.5-0.6, < 0.5. MIN_COSINE = 0.85.
- Two videos are one session if one clip pair shares audio, so the other clip pairs of
  a confirmed video pair are not read.
- `make duplicates` (Mac): 235,820 candidate clip pairs, 1,055 video pairs of 551
  speakers, 1,791 s. File artifacts/trials/voxceleb1-shared-videos.tsv, SHA-256
  e0d8cec59cd681d7642928b638c41563d0386d4cef0ba2e629de919e7bd32a67.
- Checks: 0 pairs across speakers, 0 pairs of a video with itself, 0 repeated pairs.
- Video pairs by the cosine of their confirming clip pair: >= 0.95 858; 0.90-0.95 139;
  0.85-0.90 58. The sample saw 0 of 500 at 0.85-0.90, yet the full band held 58 video
  pairs: a sample of 500 cannot see a rate near 0.03 %.
- Known limit: pairs below 0.85 are likely missed; how many is not measured. Kept
  at 0.85 because the 58 pairs of the lowest band touch at most 36 targets per bucket
  (0.07 %); 0.80 would cost about 1.5 h more (estimate from the 1,791 s run).
- Target trials of the current list across a found video pair (direct pairs only, no
  chains yet): 3-10s 651 of 50,000 (1.30 %); 10-30s 807 of 50,000 (1.61 %);
  30-120s 358 of 47,165 (0.76 %); >120s 81 of 3,186 (2.54 %).

## 2026-10-19 — EER after merging videos that share audio

- `merge_sessions` (commit 2763d60) joins video pairs from `make duplicates` into
  groups, chains included (A-B and B-C make A, B, C one session). The session is the
  group's first video id in sorted order, so the result does not depend on pair order.
  1,055 video pairs give 761 groups over 1,712 videos; sizes 2: 630, 3: 101, 4: 15,
  5: 11, 6: 2, 7: 1, 15: 1. The 15-video group (id11157) links 14 videos to one video,
  3IdsvAQv_3c, mostly at shared >= 0.95; it looks like a compilation. A chain through
  a compilation also merges videos that are different interviews, which removes some
  valid targets but never keeps a same-session one.
- `make trials` (commit 4497fc0) now takes the video-pair file. New list: 549,856 trials;
  targets 3-10s 50,000; 10-30s 50,000; 30-120s 46,754; >120s 3,102. SHA-256
  5828838be4bf26b2b1a1b250b278176765a405d81c70968b7ba8e1b110b3450d, identical on
  5 rebuilds. Scores SHA-256 6c5213efb74726507f4a16c36b699d8b7ee4266c8cba18dc78188f0d80a2058a.
- One of 6 runs of `make trials` crashed at exit (libc++abi recursive_mutex lock
  failed, exit 134) after the file was written; the file was identical to the other
  runs. Cause not found.
- EER %, raw cosine (experiments.csv d5-eer-clean-*):

  | bucket  | old list | old list minus duplicates | new list (headline) |
  |---------|----------|---------------------------|---------------------|
  | 3-10s   | 1.582    | 1.590 (686 removed)       | 1.560               |
  | 10-30s  | 0.592    | 0.600 (862 removed)       | 0.602               |
  | 30-120s | 0.638    | 0.642 (411 removed)       | 0.642               |
  | >120s   | 0.345    | 0.355 (84 removed)        | 0.355               |

- Pure effect of the duplicates (same list, duplicates removed): EER goes up by
  0.004 to 0.010 pp in every bucket.
- In 3-10s and 10-30s the new list is mostly a new target sample: 237 of 50,000 and
  4,904 of 50,000 old targets remain. Removing pairs shifts every later reservoir
  decision. Two samples of 3-10s give 1.590 and 1.560, so the sampling spread is
  larger than the duplicate effect there.
- Open item for the mentor: the third decimal of the short-bucket EERs depends on
  the sample. An interval (e.g. bootstrap) would show how far it can move; not built.

## 2026-10-20 — DET plot, make eval, D5 summary

- `make det` (commit af1b12d) draws one DET line per bucket from the score file, on
  normal-deviate axes from 0.001 % to 40 %; the legend gives EER and trial counts.
  Points where a rate is 0 or 1 lie at infinity on these axes and are left out, so the
  >120s line ends near 2.5 % false alarms, where its miss rate reaches 0. Each step of
  that line is one of its 3,102 target trials. New dependency: matplotlib, for this plot.
- `make eval` (commit d19bfe0) builds what is missing, in order: model, VoxCeleb1,
  clip and video embeddings, shared-audio video pairs. Each long step writes a file
  when it finishes (.done for the caches), so make skips it next time and an
  interrupted run resumes. Then it always reruns trials, scores, EER and DET, and
  checks the files against eval.sha256 (the run of 2026-10-19). A mismatch prints a
  warning and does not fail: another machine may round embeddings differently.
- Checked on the Mac: first `make eval` 30.6 s (embedding steps found every speaker
  file and wrote none), second 18.9 s; same EER as d5-eer-clean-*, all three files OK.
  A dry run (`make -n eval`) in an empty directory lists every step in order.
  Not checked: a full run from a clean clone (31 GB download, ~7 h of embedding).
- M1 status: >= 1,000 speakers and >= 10,000 utterances embedded (1,251 / 153,516);
  549,856 trials over four buckets; same-session guard test fails when the guard is
  removed; EER with DET and trial counts; every bucket below 2 x 0.878 % (published
  figure for this checkpoint, on a different trial list). Missing: the cohort
  disjointness test (comes with AS-Norm, D8); a full clean-clone run of make eval.

### Open items for the mentor

1. Bucket by the shorter side of a trial (accepted earlier).
2. No same-language non-targets: VoxCeleb1 has nationality, not language.
3. Long buckets use whole videos, capped at 300 s; >120s has only 3,102 targets, so its
   EER is the least stable.
4. Videos that hold the same audio: 1,055 pairs found with cosine >= 0.85 and
   waveform correlation >= 0.2; pairs below 0.85 are likely missed (count unknown).
5. The third decimal of the short-bucket EERs depends on the target sample (two
   samples of 3-10s: 1.590 and 1.560). No interval yet; bootstrap is one option.
6. Faster reproduction: may the embedding caches (154 MB) go to a private Hugging Face
   dataset, so that a `make fetch-cache` step replaces the ~7 h of embedding?
7. One of 6 runs of `make trials` crashed at exit (libc++abi mutex) after writing a
   correct file; cause not found.

## 2026-10-20 — EER per gender, DET figure in the repo, exit crash

- A full re-read of the handbook and SVS guide after D5 found that SVS guide §8.3
  requires a gender breakdown in every evaluation (for fairness, not for tuning).
- The trial list now has a gender column (commit 77e8db0). Every trial is same-gender
  (targets are one speaker, non-targets are drawn same-gender), and `write_trials`
  refuses a cross-gender trial. The trials themselves did not change: the first five
  columns are identical to the 2026-10-19 file. New SHA-256 in eval.sha256.
- EER %, raw cosine (experiments.csv d5-eer-*-f / -m), female / male:
  3-10s 1.881 / 1.368; 10-30s 0.714 / 0.539; 30-120s 0.595 / 0.692; >120s 0.555 / 0.198.
  Targets female / male: 17,692 / 32,308; 19,038 / 30,962; 19,304 / 27,450;
  1,080 / 2,022. In >120s one female target error is 0.093 pp, so 0.555 % rests on
  about 6 missed targets. Female EER is higher in three buckets of four; the cause is
  not analysed. Thresholds stay shared across genders.
- DET figure of the reported run: docs/figures/voxceleb1-det-d5.png (scores SHA-256
  4bc1910b…, same bucket EERs as 2026-10-19). `make eval` still writes to artifacts/;
  the docs copy is the figure of this report and is replaced only by a new report.
- Exit crash (open item 7): `make trials` and `make scores` loaded onnxruntime
  through embed_cache without using it. Before: 3 crashes in 19 `make trials` runs,
  one of them stopping `make eval`. Commit 0ce4aa9 imports onnxruntime only when a model
  is built; after: 0 crashes in 30 runs. The crash is gone in practice; its exact
  cause (a libc++ mutex at interpreter exit) is not known.

## 2026-10-21 — D6 degradation harness and per-condition scoring

Handbook D6 asks for 8 kHz, G.711 A-law/µ-law, G.729, Opus and noise at a controlled
SNR, and for the eval set to be rebuilt and tagged by condition. The code is done
(`src/svs/degrade.py`, `make degraded-embeddings`, `make degraded-scores`); the
embedding run is not, it waits for the GPU decision. No EER under any condition yet.

**Conditions** (names follow the codec labels of SVS guide §8.3):

| condition | what is done to clean 16 kHz audio | rate given to VAD |
| --- | --- | --- |
| pcm16k | nothing (the D5 baseline) | 16 kHz |
| pcm8k | resample to 8 kHz with an anti-alias filter | 8 kHz |
| g711a / g711u | pcm8k, then G.711 A-law / µ-law encode and decode (libsndfile) | 8 kHz |
| g729 | pcm8k, then G.729 Annex A encode and decode (bcg729) | 8 kHz |
| opus | Opus 16 kbit/s, VoIP mode, encode and decode (ffmpeg libopus) | 16 kHz |
| noise15 / noise5 | MUSAN noise added at 15 / 5 dB SNR, no codec | 16 kHz |

**Choices where the handbook and the SVS guide are silent** (the mentor may disagree):

1. Opus bitrate 16 kbit/s: the low end of RFC 7587 §3.1.1's wideband-speech range.
2. G.729: ffmpeg 9.0.2 decodes but cannot encode it. `tools/g729.c` wraps bcg729
   (Homebrew/Ubuntu `libbcg729`, GPL-3); `make g729` builds it into artifacts/, so no
   GPL binary is committed. Its VAD/DTX is off; its 5 ms look-ahead lag is kept.
3. SNR levels 15 and 5 dB: inside the 5-25 dB range of the guide's quality score
   (Appendix C), 5 dB at its worst end. Power is the mean square over the whole
   clip, silence included, so the SNR within speech is somewhat better.
4. Noise files shorter than 3.0 s are not used (155 of 930, 3.9 of 374 minutes,
   including the one near-silent file): repeated, they become a periodic click.
5. The utt_id picks the noise file and offset through CRC32 (Python's hash() changes
   per process), so every machine, and both SNR levels, use the same noise piece.
6. Trials keep the pairs and duration buckets of the clean list; only the audio
   changes. A trial whose degraded audio has under 3.0 s of speech is refused and
   counted, as the service would refuse it (SVS guide §4.4). Each condition can
   therefore score slightly fewer trials; every EER will carry its own trial count.

**MUSAN** (`make musan`, CC BY 4.0, OpenSLR MD5 checked): noise/ has 930 files
(free-sound 843, sound-bible 87), all 16 kHz mono, 6.23 h; 775 are used.

**Checks.**
- The clean score file now has a `condition` column (pcm16k). Without it the file is
  byte-identical to the D5 run (SHA-256 4bc1910b…), and `make eer` prints the same
  EERs; eval.sha256 holds the new SHA-256 (c6e2455e…). 0 clean trials refused.
- Smoke run, one speaker (id10899, 168 clips, Mac): clips under 3.0 s of speech —
  clean 1, g729 0, noise5 9. Under noise5 the VAD found more speech in 71 clips (up to
  +1.26 s) and less in 87 (down to −6.82 s): WebRTC VAD both takes noise for speech and
  loses speech in noise. Under g729 the change was −0.46 to +1.36 s. One speaker only;
  the refusal counts of the full run will be reported per condition in D7.

**Cost.** The trial list uses 145,337 audio items (134,906 clips, 10,431 videos):
91.2 % / 85.7 % of the clip / video speech seconds. Scaled from the measured D3/D4 runs
(12,487 s / 11,550 s on the home PC), one condition is about 5.9 h, seven about 41 h,
on CPU. Opus costs ~127 ms per clip on the Mac against ~27 ms for the others (two
ffmpeg processes per clip, one-clip timing); worth fixing only if CPU-bound.

**Open items for the mentor**

8. GPU (L40S) access for the ~41 h of degraded embeddings (decision expected 2026-10-21).
9. The choices 1-6 above.

**Next.** D7 (EER per condition) needs the degraded embeddings, so D8 (AS-Norm) comes
first, on clean audio. All 1,251 VoxCeleb1 speakers are in the trial list, so a
disjoint impostor cohort cannot come from VoxCeleb1; the plan is VoxCeleb2, with
disjointness checked before use.

## 2026-10-21 — D8 AS-Norm with a fixed, versioned, disjoint cohort

Handbook D8: a fixed, versioned impostor cohort (2,000-5,000), top-n cohort scores,
symmetric z-score, profile-side statistics cached. SVS guide §4.3 gives the formula,
n = 300 (CFG_SV_COHORT_TOP_N) and the 1e-6 on the spread; both are used as given.

**Cohort `voxceleb2-dev-v1`** (`make cohort`, list in `cohorts/voxceleb2-dev-v1.tsv`):
5,000 VoxCeleb2 dev speakers, one clip each. Net speech per clip: 3.00-60.14 s,
median 5.34 s (4,343 in 3-10 s, 635 in 10-30 s, 22 over 30 s); 3,052 male, 1,948
female. 52 clips had under 3.0 s of speech and gave their place to the next speaker.
Build: 19 min on the Mac (download 1.1 GB of wav after decoding); a second build
from the cached wavs gave a byte-identical list and identical embeddings.

**Disjointness.** VoxCeleb1 and VoxCeleb2 number speakers separately, so ids prove
nothing, and vox2_meta.csv has no names. Names come from VGGFace2's identity_meta.csv
(pinned revision, SHA-256 checked) and are compared in one spelling (case, `_`, `-`,
`.` ignored). VoxCeleb2 *test* holds two VoxCeleb1 trial speakers spelled differently:
Candace_Cameron-Bure = id10123, Maria_Canals_Barrera = id10736 (1,473 trial rows). An
exact-string check would have missed both. VoxCeleb2 dev (5,994 speakers) shares no
name with the 1,251 VoxCeleb1 speakers; `svs.cohort` asserts this on every build
(SVS guide pitfall 15), and tests/test_cohort.py uses the real Candace pair.

**Choices where the handbook and the SVS guide are silent** (the mentor may disagree):

10. Size 5,000, the handbook's upper end; the guide's 20,000 default is for production.
11. One clip per speaker, so the top 300 impostors of an audio come from many voices.
    Speakers and clips are picked in CRC32 order (same on every machine).
12. VoxCeleb2 dev, although the model was trained on it: it is the only large source
    with names to prove disjointness, and VoxCeleb2 test has 120 speakers, two of them
    in our trials. WeSpeaker's recipe does the same (examples/voxceleb/v2/run.sh:
    `--cohort_set vox2_dev`, `top_n=300`).
13. The archive is 77.5 GB in eight parts; only its directory (145.7 MB) and the 5,100
    chosen members are read, by HTTP range requests, each checked against its CRC32.
    AAC is decoded to 16 kHz wav by ffmpeg 9.0.2; another ffmpeg version may decode a
    clip slightly differently, and `make cohort` fails if the list changes.
14. Cohort audio is clean 16 kHz. The guide wants the cohort to match the production
    condition mix; for degraded conditions it must be degraded the same way (M2, later).

**Result, clean 16 kHz** (VoxCeleb1 trials sha256 7eb0eefa, 549,856 trials, 0 refused;
AS-Norm scores sha256 367d84aa):

| bucket | EER cosine | EER AS-Norm | n_target / n_nontarget |
| --- | --- | --- | --- |
| 3-10 s | 1.560 % | 1.448 % | 50,000 / 100,000 |
| 10-30 s | 0.602 % | 0.582 % | 50,000 / 100,000 |
| 30-120 s | 0.642 % | 0.559 % | 46,754 / 100,000 |
| > 120 s | 0.355 % | 0.322 % | 3,102 / 100,000 |

Lower in every bucket; per gender lower in 7 of 8 cells, equal in > 120 s female
(0.555 %, only 1,080 targets). The gain on clean wideband audio is small, as the guide
expects (it says AS-Norm is worth several points on telephony). The EER threshold now
grows from 1.99 (3-10 s) to 3.97 (> 120 s): AS-Norm does not remove the need for
duration-dependent thresholds (D9).

**Open items for the mentor**

10. The choices 10-14 above.

**Next.** M2 needs EER with and without AS-Norm per condition, with the gain on degraded
8 kHz audio: the degraded embeddings (night runs on the home PC since 2026-10-21) and a
cohort degraded per condition.

## 2026-10-22 — D9 (in progress): thresholds from FRR at a fixed FAR, clean audio

Handbook D9: calibrate per condition, choose thresholds from FRR at a fixed FAR (not
from EER), four bands. SVS guide §8.4 step 4: STRONG at the FAR where
`N_expected × FAR ≤ 0.1`, PROBABLE at about 10 × that FAR. Only clean 16 kHz (pcm16k)
so far: the degraded embeddings are still running on the home PC.

**The FAR chosen, and why.** The guide ties STRONG to the target pool size (§4.6) but
names no size for an evaluation. N = 50,000 = `CFG_SV_MAX_AUTO_POOL`: above it the
service never assigns automatically, so STRONG must hold for every pool up to it.
STRONG FAR = 0.1 / 50,000 = 2e-6; PROBABLE FAR = 2e-5. Constants in `svs.calibrate`.

**Why the trial list's non-targets are not enough.** With 100,000 non-targets per
bucket, a FAR of 1e-5 allows one false accept, so its threshold is set by a single
trial. Probe on 3-10s (AS-Norm): the thresholds for FAR 1e-3 / 1e-4 / 1e-5 from the
100,000 gave 1.18e-3 / 0.96e-4 / 1.9e-5 on 1,000,000 fresh non-targets. The first two
held (they rest on 99 and 10 false accepts); the last was off by 2×.

**Non-targets for calibration** (commit 3dcc0b4): `nontarget_pairs` draws distinct
same-gender, different-speaker pairs of one bucket in numpy, with the rules of the
trial list, seed 0. They are not written to a file; the seed and the code give the same
pairs. Available pairs: 3-10s 5,747,898,297; 10-30s 211,472,369; 30-120s 26,429,032;
>120s 1,244,644. 5,000,000 per bucket gives about 10 false accepts at 2e-6; >120s uses
all of its 1,244,644.

**Operating points** (commit 06dba57, `make operating-points`, 28 s, 3.6 GB on the Mac;
artifacts/calibration/voxceleb1-pcm16k.tsv, SHA-256 8906f8b4…, identical on 3 runs).
AS-Norm, cohort voxceleb2-dev-v1; FRR over the trial list's targets
(experiments.csv d9-op-*):

| bucket | PROBABLE threshold | FRR | STRONG threshold | FRR | false accepts at STRONG (speaker pairs) | non-targets / targets |
| --- | --- | --- | --- | --- | --- | --- |
| 3-10 s | 6.0756 | 23.90 % | 6.6948 | 33.30 % | 10 (9) | 5,000,000 / 50,000 |
| 10-30 s | 6.4298 | 9.30 % | 7.2218 | 17.38 % | 10 (10) | 5,000,000 / 50,000 |
| 30-120 s | 6.7363 | 6.49 % | 7.6520 | 13.36 % | 10 (8) | 5,000,000 / 46,754 |
| > 120 s | 6.7582 | 4.35 % | 7.4228 | 8.09 % | 2 (2) | 1,244,644 / 3,102 |

- At the same FAR, FRR falls from 33.30 % (3-10 s) to 8.09 % (> 120 s): one threshold
  for all durations would be wrong for at least one of them (SVS guide §4.4).
- The false accepts come from many speaker pairs (8-10 of 10), not one look-alike pair.
- > 120 s STRONG rests on 2 false accepts and is not reliable; that is why its threshold
  is below the 30-120 s one. VoxCeleb1 has no more long videos.
- The thresholds are far above the guide's reference STRONG values (0.86-1.34). Not
  explained yet; the reference comes from different data and the guide says ours will
  differ.

**Choices where the handbook and the SVS guide are silent** (the mentor may disagree):

15. N = 50,000 (`CFG_SV_MAX_AUTO_POOL`) as the pool size of the STRONG rule.
16. Calibration non-targets are fresh pairs (5,000,000 per bucket), not the trial
    list's 100,000, so the STRONG threshold rests on about 10 false accepts.

**Not done yet:** precision and recall per threshold, the WEAK floor, the band function,
per-condition tables (needs the degraded embeddings), thresholds for smaller pools.

**Open items for the mentor**

11. The choices 15-16 above.

## 2026-10-26 — D6 done: degraded embeddings and per-condition scores

The seven degraded conditions are embedded and scored. No GPU came (open item 8), so
the run used the home PC's CPU on four nights, 2026-10-21 to 2026-10-25, through a
local cron job that stopped at 08:55 and resumed the next evening. 17,514 speaker
files (7 conditions × clips and videos × 1,251 speakers), 1.0 GB, copied to the Mac.

**Two power cuts left five empty files.** The home PC lost power on 2026-10-22 (07:16)
and 2026-10-23 (01:37). Files are written as `.npz.part` and renamed, but without an
fsync the rename reached the disk and the data did not: g729 id10747-id10749 and opus
id10841-id10842 were 0-byte `id*.npz` files, which the next night skipped as done.
`make degraded-scores` found them (EOFError). Commit 6e01e43 adds the fsync before the
rename, with a test that fails without it. The five speakers were recomputed on the
home PC (32 s + 46 s), so every condition comes from one machine.

**Degraded audio depends on library versions.** Before the recompute, the same speakers
were made on the Mac for comparison:

- bcg729 1.1.2 (Mac, Homebrew) vs 1.1.1 (home, Ubuntu 1.1.1-2build1, no code patches):
  on id10746 (60 clips) the G.729 output differed in 32 clips, the VAD speech length in
  8, lowest embedding cosine 0.932. With 1.1.1 built from source on the Mac: 2 clips,
  1 clip, 0.984. The rest comes from `resample_poly` float rounding on ARM vs x86 (same
  numpy 2.5.3 and scipy 1.18.1 on both).
- ffmpeg 9.0.2 (Mac) vs 6.1.1 (home) for Opus: VAD speech length differed in 79 of 96
  and 48 of 68 clips of the two speakers, lowest cosine 0.91.

The published degraded results are therefore tied to the home PC: bcg729 1.1.1, ffmpeg
6.1.1 with libopus 1.4 (Ubuntu 24.04; the Mac has libopus 1.6.1), x86. A clean clone on
another machine will give slightly different scores for g729 and opus (open item 12).
pcm8k, g711a/u and noise use only numpy/scipy/libsndfile and differ only by float
rounding.

**Trials refused** (degraded audio with under 3.0 s of speech, of 549,856; clean 0):

| condition | refused | scored |
| --- | --- | --- |
| pcm8k | 207 | 549,649 |
| g711u | 455 | 549,401 |
| g711a | 751 | 549,105 |
| g729 | 300 | 549,556 |
| opus | 292 | 549,564 |
| noise15 | 2,049 | 547,807 |
| noise5 | 6,234 | 543,622 |

Noise makes the VAD lose speech, as the one-speaker smoke run of 2026-10-21 showed.

**Open items for the mentor**

8. Closed: no GPU was needed; four nights on the home PC's CPU.
12. Degraded results depend on bcg729 and ffmpeg versions (above). Record the versions
    with the results, or pin them (e.g. build bcg729 1.1.1 from source in `make g729`)?

**Next.** D7: EER per condition and bucket, with and without AS-Norm.

## 2026-10-26 — D7: EER per condition, and the clean threshold on degraded audio

Handbook D7: EER per codec, per sample rate, per duration bucket. `make eer-conditions`
(`svs.damage`, commit a6f0962, 19 s) reads the clean and the seven degraded score files
of the same trial list (sha256 7eb0eefa) and prints, per condition and bucket (and per
bucket × gender): EER, its threshold, trial counts, trials refused, and the FRR and FAR
that the **clean** EER threshold of that bucket gives under the condition. Raw cosine
only; AS-Norm per condition needs a degraded cohort (next).

**EER %, raw cosine** (each condition on the trials it scored; counts and refusals in
artifacts/damage/voxceleb1-cosine.tsv, sha256 2b893027):

| condition | 3-10 s | 10-30 s | 30-120 s | > 120 s |
| --- | --- | --- | --- | --- |
| pcm16k (clean) | 1.560 | 0.602 | 0.642 | 0.355 |
| pcm8k | 3.345 | 1.418 | 1.418 | 0.709 |
| g711a | 3.467 | 1.472 | 1.450 | 0.735 |
| g711u | 3.444 | 1.466 | 1.457 | 0.710 |
| g729 | 6.393 | 2.962 | 2.693 | 1.289 |
| opus (16 kHz) | 2.058 | 0.774 | 0.818 | 0.427 |
| noise15 | 2.007 | 0.770 | 0.727 | 0.419 |
| noise5 | 3.296 | 1.307 | 0.873 | 0.516 |

- Sample rate: 8 kHz alone doubles the EER in every bucket (1.560 → 3.345 in 3-10 s).
- G.711 adds little on top of 8 kHz (≤ 0.13 points). G.729 doubles it again: 3.6-4.9×
  clean, depending on the bucket.
- Opus at 16 kbit/s keeps 16 kHz and costs 0.07-0.50 points.
- noise5 hurts short audio most (3-10 s 2.1×, > 120 s 1.5×), and the VAD loses speech:
  6,234 trials refused, 6,174 of them in 3-10 s.

**The clean threshold on degraded audio** (FAR / FRR at the clean EER threshold of the
bucket; on clean both are the EER):

| condition | 3-10 s FAR | FRR | 10-30 s FAR | FRR | > 120 s FAR | FRR |
| --- | --- | --- | --- | --- | --- | --- |
| pcm16k | 1.560 | 1.560 | 0.603 | 0.602 | 0.355 | 0.355 |
| pcm8k | 6.130 | 1.903 | 3.711 | 0.706 | 2.735 | 0.290 |
| g711a | 6.338 | 1.917 | 3.850 | 0.712 | 2.857 | 0.290 |
| g729 | 19.942 | 1.805 | 15.785 | 0.512 | 14.105 | 0.032 |
| noise5 | 1.315 | 6.068 | 0.592 | 2.109 | 0.511 | 0.516 |

Narrowband audio moves *all* cosines up, impostors included: the EER threshold grows
from 0.348 (clean) to 0.391 (pcm8k) and 0.446 (G.729) in 3-10 s. A threshold fitted on
clean 16 kHz therefore accepts 4-40× more impostors on 8 kHz telephony (G.729 > 120 s:
0.355 % → 14.1 %), while FRR barely moves (and falls under G.729), so nothing in the
missed targets warns of it. Noise moves the cosines down and turns the same threshold
into missed targets. This is the table for the handbook's "why a clean threshold is
worthless on 8 kHz": thresholds must be per condition (D9).

**Gender.** Female EER is higher than male in all four buckets under pcm8k and G.729
(e.g. G.729 > 120 s: 2.315 % vs 0.796 %); on clean it was higher in 3 of 4.

**Choice where the handbook and the SVS guide are silent** (the mentor may disagree):

17. Each condition's EER uses the trials it scored, as the service would. On the
    542,410 trials that no condition refused (`make eer-conditions COMMON=1`, sha256
    c79fb2a2) every EER moves by at most 0.05 points (largest: pcm8k 3-10 s 3.345 →
    3.296), so the refusals do not change the comparison.

**Next.** M2: AS-Norm per condition (cohort degraded the same way, on the home PC for
the same library versions), its gain on 8 kHz, then the D9 per-condition operating-point
table.

## 2026-10-26 — AS-Norm per condition, with the cohort degraded the same way

M2 asks for EER with and without AS-Norm per condition, with the gain on degraded
8 kHz audio. SVS guide §4.3: the cohort should match the audio it normalises, so each
condition gets its own cohort: the 5,000 committed members of `voxceleb2-dev-v1`,
degraded like the trials (`svs.cohort_condition`, commit 8746349, `make
degraded-cohorts`). It ran on the home PC (same bcg729 1.1.1 and ffmpeg 6.1.1 as the
trial embeddings): 55 min, opus 1,296 s, the others 300-451 s each.

**Members left out** (under 3.0 s of speech after degradation; not replaced, so the
cohort stays the committed list): pcm8k 3, g711a 8, g711u 3, g729 2, opus 4, noise15
28, noise5 67 of 5,000.

**EER %, raw cosine → AS-Norm** (`make eer-conditions-asnorm`, 19 s; table
artifacts/damage/voxceleb1-asnorm.tsv, sha256 012ec36c; relative change in brackets):

| condition | 3-10 s | 10-30 s | 30-120 s | > 120 s |
| --- | --- | --- | --- | --- |
| pcm16k | 1.560 → 1.448 (−7 %) | 0.602 → 0.582 (−3 %) | 0.642 → 0.559 (−13 %) | 0.355 → 0.322 (−9 %) |
| pcm8k | 3.345 → 2.941 (−12 %) | 1.418 → 1.350 (−5 %) | 1.418 → 1.249 (−12 %) | 0.709 → 0.806 (+14 %) |
| g711a | 3.467 → 3.039 (−12 %) | 1.472 → 1.408 (−4 %) | 1.450 → 1.285 (−11 %) | 0.735 → 0.838 (+14 %) |
| g711u | 3.444 → 3.037 (−12 %) | 1.466 → 1.392 (−5 %) | 1.457 → 1.269 (−13 %) | 0.710 → 0.838 (+18 %) |
| g729 | 6.393 → 5.299 (−17 %) | 2.962 → 2.488 (−16 %) | 2.693 → 2.248 (−17 %) | 1.289 → 1.325 (+3 %) |
| opus | 2.058 → 1.777 (−14 %) | 0.774 → 0.702 (−9 %) | 0.818 → 0.687 (−16 %) | 0.427 → 0.355 (−17 %) |
| noise15 | 2.007 → 1.847 (−8 %) | 0.770 → 0.732 (−5 %) | 0.727 → 0.648 (−11 %) | 0.419 → 0.387 (−8 %) |
| noise5 | 3.296 → 3.025 (−8 %) | 1.307 → 1.191 (−9 %) | 0.873 → 0.776 (−11 %) | 0.516 → 0.481 (−7 %) |

- On 8 kHz telephony the gain in the three buckets up to 120 s is 4-17 %, larger than on
  clean audio (3-13 %), and largest under G.729 (16-17 % in every one of them).
- \> 120 s on 8 kHz gets worse, but by 3-4 missed targets of 3,102 (pcm8k 22 → 25,
  g711u 22 → 26, g729 40 → 41 at the EER point): inside the noise of this bucket (open
  item 3). The four narrowband rows share the same trials, so they are not four
  independent signs. A possible cause, not tested: the cohort is short clips (median
  5.3 s), the > 120 s audio is not.

**The clean threshold under AS-Norm** (FAR / FRR at the clean AS-Norm EER threshold):

| condition | 3-10 s FAR | FRR | 10-30 s FAR | FRR | > 120 s FAR | FRR |
| --- | --- | --- | --- | --- | --- | --- |
| pcm16k | 1.448 | 1.448 | 0.582 | 0.582 | 0.322 | 0.322 |
| pcm8k | 1.359 | 5.184 | 0.502 | 2.452 | 0.238 | 1.580 |
| g729 | 1.299 | 13.590 | 0.499 | 6.864 | 0.266 | 3.804 |
| noise5 | 1.328 | 5.144 | 0.592 | 1.849 | 0.411 | 0.613 |

With the matched cohort, the impostor scores of 8 kHz audio land where clean ones do:
under G.729 the FAR of the clean threshold falls from 19.94 % (raw cosine) to 1.30 % in
3-10 s. The targets do not follow; their FRR at that threshold rises to 13.59 %. So
AS-Norm fixes the false accepts that made the raw clean threshold dangerous, but the
clean threshold is still wrong on 8 kHz, now in the other direction: thresholds stay per
condition (D9).

**Choice where the handbook and the SVS guide are silent** (the mentor may disagree):

18. A cohort member with under 3.0 s of speech after degradation is left out of that
    condition's cohort, not replaced by a spare: the service never scores such audio,
    and the member list stays the committed version.

Score files (sha256): pcm8k e5745ba1, g711a e5a77c6c, g711u 52a9263c, g729 ff4d83ce,
opus 733b2158, noise15 d9820a45, noise5 2bb6a4d7. Cohorts: pcm8k 689a64eb, g711a
47454ddf, g711u d26e9ecb, g729 98e394a1, opus e19f1c60, noise15 8f0ac71f, noise5
133f81ba.

**Open items for the mentor**

13. Choice 18 above; and whether > 120 s needs a cohort of long audio.

**Next.** D9 per condition: operating points (STRONG/PROBABLE from FRR at fixed FAR) for
every condition on AS-Norm scores, precision and recall, WEAK floor, band function.

## 2026-10-26 — D9: operating points per condition

Handbook D9: calibrate per condition. The clean operating points (2026-10-22 entry)
now exist for the seven degraded conditions too, with the same FARs (STRONG 2e-6,
PROBABLE 2e-5, choice 15) and the same fresh-non-target method (choice 16).

**How** (commit 0e035aa, `make operating-points-conditions`, 203 s, 3.7 GB on the Mac):
per condition, 5,000,000 fresh non-target pairs per bucket (seed 0) from that
condition's embeddings, AS-Norm against that condition's cohort; FRR over the targets
of that condition's AS-Norm score file. Only stored embeddings are read, no audio is
degraded, so the Mac's codec versions play no part. pcm8k rerun: identical output.
Tables artifacts/calibration/voxceleb1-<condition>.tsv (sha256 pcm8k 3299b628, g711a
fbe3881d, g711u 696ef01f, g729 aa95f428, opus 954db6cb, noise15 cd1d0837, noise5
7a6c2162; pcm16k unchanged 8906f8b4). experiments.csv d9-op-<condition>.

**FRR % at STRONG** (FAR 2e-6):

| condition | 3-10 s | 10-30 s | 30-120 s | > 120 s |
| --- | --- | --- | --- | --- |
| pcm16k | 33.30 | 17.38 | 13.36 | 8.09 |
| pcm8k | 66.80 | 47.88 | 37.82 | 35.01 |
| g711a | 69.29 | 47.91 | 42.23 | 35.94 |
| g711u | 69.18 | 44.08 | 42.40 | 35.65 |
| g729 | 84.77 | 62.67 | 54.58 | 42.75 |
| opus | 48.77 | 24.84 | 14.45 | 10.96 |
| noise15 | 48.93 | 24.15 | 15.41 | 6.48 |
| noise5 | 63.54 | 35.98 | 17.69 | 9.45 |

**FRR % at PROBABLE** (FAR 2e-5):

| condition | 3-10 s | 10-30 s | 30-120 s | > 120 s |
| --- | --- | --- | --- | --- |
| pcm16k | 23.90 | 9.30 | 6.49 | 4.35 |
| pcm8k | 48.52 | 26.94 | 22.23 | 16.38 |
| g711a | 52.87 | 28.84 | 22.40 | 16.12 |
| g711u | 50.07 | 29.12 | 21.55 | 16.22 |
| g729 | 70.58 | 45.00 | 38.06 | 28.69 |
| opus | 31.33 | 12.83 | 7.57 | 5.64 |
| noise15 | 32.37 | 13.03 | 7.43 | 4.80 |
| noise5 | 47.39 | 22.30 | 9.42 | 6.06 |

**STRONG thresholds** (AS-Norm score):

| condition | 3-10 s | 10-30 s | 30-120 s | > 120 s |
| --- | --- | --- | --- | --- |
| pcm16k | 6.6948 | 7.2218 | 7.6520 | 7.4228 |
| pcm8k | 6.5339 | 6.9375 | 7.0810 | 7.4862 |
| g711a | 6.5875 | 6.8809 | 7.2354 | 7.4999 |
| g711u | 6.5892 | 6.7149 | 7.2539 | 7.4988 |
| g729 | 6.1673 | 6.3029 | 6.6637 | 6.7689 |
| opus | 6.9346 | 7.2511 | 7.2848 | 7.3242 |
| noise15 | 7.0030 | 7.2565 | 7.6601 | 7.0959 |
| noise5 | 7.0250 | 7.2148 | 7.5788 | 7.3628 |

- On 8 kHz telephony STRONG misses two thirds or more of the true 3-10 s pairs (pcm8k
  66.80 %, G.729 84.77 %), 2.0-2.5 times the clean rate. The 8 kHz thresholds are lower than the
  clean ones, yet the FRR is higher: the targets fall further than the impostors.
- Opus and noise at 15 dB stay close to clean from 30 s up; the damage is concentrated
  in 8 kHz codecs and short audio.
- Each condition has its own threshold per bucket (3-10 s: 6.17 under G.729 to 7.03
  under noise 5 dB), so the service looks the threshold up by condition and bucket.
- At STRONG 3-10 s to 30-120 s, the 10 false accepts come from 5-10 speaker pairs.
  \> 120 s rests on 2 false accepts, and on 8 kHz both come from one speaker pair: these
  four thresholds are not reliable (as on clean audio; VoxCeleb1 has no more long videos).

**Durations move under degradation.** The non-target pairs are put into buckets by
the speech seconds measured on the degraded audio; the targets keep the clean trial
list's buckets (D6 choice 6). Clips and videos whose bucket differs from the clean one
(including those that fall below 3.0 s): pcm8k 0.19 % of clips / 0.18 % of videos,
g711a 0.73 / 0.62, g711u 0.44 / 0.34, g729 0.76 / 1.45, opus 0.42 / 0.50, noise15
2.14 / 1.84, noise5 4.48 / 3.86 (median change −0.12 to +0.22 s).

**Choice where the handbook and the SVS guide are silent** (the mentor may disagree):

19. Calibration non-targets are bucketed by the speech duration of the degraded audio,
    as the service will bucket a query it measures; the targets keep the clean
    buckets. At most 4.5 % of clips (noise 5 dB) are in a different bucket.

**Not done yet:** precision and recall per threshold, the WEAK floor, the band
function, thresholds for smaller pools.

**Open items for the mentor**

14. Choice 19 above; and the > 120 s STRONG thresholds, which rest on 2 false accepts.

## 2026-10-26 — D9: precision and recall at each threshold

SVS guide §8.4 step 5: record precision and recall at each threshold, with the trial
count. Commit 525610e adds two columns to every operating-point table: `recall`
(1 − FRR over the targets) and `precision_pool50000`. The earlier ten columns are
unchanged (same SHA-256 when cut to them). New tables (sha256): pcm16k bfbb2060, pcm8k
d1289308, g711a 311b5c44, g711u 21fcb711, g729 ffa1b0bf, opus 3de29622, noise15
8bbcbf2b, noise5 b88673a5.

**What the precision means.** On the trial list (two non-targets per target) precision
is close to 1 and says nothing about the service. Here it is the share of right answers
for one identify query against a pool of 50,000 profiles that holds the true speaker:
the true profile passes with probability recall, each other profile with the measured
FAR (false accepts / non-targets), so precision = recall / (recall + 50,000 × FAR).
Example, clean 3-10 s STRONG: 0.667 / (0.667 + 50,000 × 10 / 5,000,000) = 0.870.

**Precision / recall at STRONG** (FAR 2e-6):

| condition | 3-10 s | 10-30 s | 30-120 s | > 120 s |
| --- | --- | --- | --- | --- |
| pcm16k | 0.870 / 0.6670 | 0.892 / 0.8262 | 0.897 / 0.8664 | 0.920 / 0.9191 |
| pcm8k | 0.769 / 0.3320 | 0.839 / 0.5212 | 0.861 / 0.6218 | 0.889 / 0.6499 |
| g711a | 0.754 / 0.3071 | 0.839 / 0.5209 | 0.852 / 0.5777 | 0.885 / 0.6406 |
| g711u | 0.755 / 0.3082 | 0.848 / 0.5592 | 0.852 / 0.5760 | 0.887 / 0.6435 |
| g729 | 0.604 / 0.1523 | 0.789 / 0.3733 | 0.820 / 0.4542 | 0.881 / 0.5725 |
| opus | 0.837 / 0.5123 | 0.883 / 0.7516 | 0.895 / 0.8555 | 0.917 / 0.8904 |
| noise15 | 0.836 / 0.5107 | 0.884 / 0.7585 | 0.894 / 0.8459 | 0.918 / 0.9352 |
| noise5 | 0.785 / 0.3646 | 0.865 / 0.6402 | 0.892 / 0.8231 | 0.916 / 0.9055 |

**Precision / recall at PROBABLE** (FAR 2e-5):

| condition | 3-10 s | 10-30 s | 30-120 s | > 120 s |
| --- | --- | --- | --- | --- |
| pcm16k | 0.435 / 0.7610 | 0.478 / 0.9070 | 0.486 / 0.9351 | 0.498 / 0.9565 |
| pcm8k | 0.342 / 0.5148 | 0.425 / 0.7306 | 0.440 / 0.7777 | 0.463 / 0.8362 |
| g711a | 0.323 / 0.4713 | 0.418 / 0.7116 | 0.439 / 0.7760 | 0.456 / 0.8388 |
| g711u | 0.335 / 0.4993 | 0.417 / 0.7088 | 0.442 / 0.7845 | 0.460 / 0.8378 |
| g729 | 0.229 / 0.2942 | 0.357 / 0.5500 | 0.385 / 0.6194 | 0.425 / 0.7131 |
| opus | 0.410 / 0.6867 | 0.468 / 0.8717 | 0.483 / 0.9243 | 0.493 / 0.9436 |
| noise15 | 0.406 / 0.6763 | 0.468 / 0.8697 | 0.483 / 0.9257 | 0.498 / 0.9520 |
| noise5 | 0.347 / 0.5261 | 0.440 / 0.7770 | 0.478 / 0.9058 | 0.486 / 0.9394 |

Trial counts per row: the n_target and n_nontarget columns of the tables (above, and
2026-10-22 entry).

- The STRONG rule (N × FAR ≤ 0.1) caps precision at recall / (recall + 0.1), so below
  1 / 1.1 = 0.909 even at recall 1 (> 120 s goes slightly above it only because its
  measured FAR, 1.5-1.7e-6, is under 2e-6). No row reaches the guide's typical STRONG
  precision of ≥ 0.97 (§4.5); that needs about 0.03 false accepts per query.
- At PROBABLE about one impostor per query passes (50,000 × 2e-5), so precision is
  0.23-0.50, far below the guide's 0.85-0.95.
- 8 kHz lowers precision because it lowers recall: the false accepts per query are fixed
  by the FAR, the true accepts fall (G.729 3-10 s: recall 0.152, precision 0.604).
- A smaller pool raises precision directly (pool 500 at PROBABLE: 0.01 false accepts per
  query). That is the guide's "filter first" (§4.6); pool size is D14 work.

**Choice where the handbook and the SVS guide are silent** (the mentor may disagree):

20. Precision is for one identify query against a pool of 50,000 (choice 15) that holds
    the true speaker, from the measured FAR. If the speaker may be missing from the pool,
    precision is lower, so these values are upper bounds for that pool.

**Open items for the mentor**

15. Choice 20 above. And: with STRONG from N × FAR ≤ 0.1 at N = 50,000, precision
    cannot reach the guide's ≥ 0.97 for STRONG or 0.85-0.95 for PROBABLE. Are the
    guide's typical precisions meant for filtered pools?

**Next:** WEAK floor, band function (score + condition → band) with a test.

## 2026-10-26 — D9: the WEAK floor

SVS guide §8.4: `WEAK` → the floor, chosen for a review workload people will actually
do. A WEAK candidate gets no automatic action; it is shown to a person (§4.5). The
guide gives no number for the workload.

**The ladder** (clean 3-10 s, the same 5,000,000 fresh non-targets; impostor
candidates per query = 50,000 × measured FAR):

| FAR | threshold | recall | impostor candidates per query |
| --- | --- | --- | --- |
| 2e-6 (STRONG) | 6.6948 | 0.6670 | 0.10 |
| 2e-5 (PROBABLE) | 6.0756 | 0.7610 | 0.99 |
| 4e-5 | 5.7978 | 0.7956 | 1.99 |
| 1e-4 | 5.4092 | 0.8384 | 5.00 |
| 2e-4 | 5.0321 | 0.8727 | 10.00 |
| 1e-3 | 4.0753 | 0.9344 | 49.99 |

A floor at FAR 1e-3 would show the true speaker 93 % of the time, next to about 50
impostors per query; at about 20 minutes per false match (§4.4) nobody reviews that.

**The floor** (commit 0fc5168): `WEAK_FAR = PROBABLE_FAR + 1 / POOL_SIZE` = 4e-5, so
the WEAK band adds one impostor candidate per query. Tables recomputed (`make
operating-points operating-points-conditions`, 3 min 55 s); without the WEAK rows each
file has the same SHA-256 as before (STRONG and PROBABLE unchanged). New tables
(sha256): pcm16k d80ad401, pcm8k 90e4aa8e, g711a 97232196, g711u 85d3d716, g729
de07ffc5, opus 8bbd71da, noise15 8cbe2e69, noise5 be1facd1.

**Recall / precision at the WEAK floor** (precision for a pool of 50,000, choice 20;
all accepts above the floor):

| condition | 3-10 s | 10-30 s | 30-120 s | > 120 s |
| --- | --- | --- | --- | --- |
| pcm16k | 0.7956 / 0.286 | 0.9276 / 0.318 | 0.9488 / 0.323 | 0.9681 / 0.330 |
| pcm8k | 0.5670 / 0.222 | 0.7793 / 0.281 | 0.8251 / 0.293 | 0.8672 / 0.304 |
| g711a | 0.5384 / 0.213 | 0.7657 / 0.278 | 0.8236 / 0.293 | 0.8617 / 0.301 |
| g711u | 0.5570 / 0.219 | 0.7673 / 0.278 | 0.8246 / 0.293 | 0.8620 / 0.305 |
| g729 | 0.3519 / 0.150 | 0.6105 / 0.235 | 0.6746 / 0.253 | 0.7721 / 0.282 |
| opus | 0.7328 / 0.269 | 0.9062 / 0.313 | 0.9385 / 0.320 | 0.9545 / 0.325 |
| noise15 | 0.7165 / 0.265 | 0.8948 / 0.310 | 0.9430 / 0.322 | 0.9607 / 0.329 |
| noise5 | 0.5824 / 0.226 | 0.8104 / 0.289 | 0.9253 / 0.317 | 0.9536 / 0.324 |

- Each floor rests on 47-51 false accepts in > 120 s and 199 in the other buckets, so
  it is far better measured than STRONG.
- About two impostors and at most one true speaker pass the floor per query, so the
  precision above it stays under 1/3; the guide's 0.4-0.7 for WEAK needs a smaller pool
  (open item 15).

**Choice where the handbook and the SVS guide are silent** (the mentor may disagree):

21. WEAK floor at FAR = PROBABLE FAR + 1 / 50,000 = 4e-5: one impostor candidate per
    query on top of PROBABLE's, for a pool of 50,000. All three bands are then set in
    one unit, impostors per query: STRONG 0.1, PROBABLE 1, WEAK 2.

**Open items for the mentor**

16. Choice 21 above: is one extra impostor per query the review workload the analysts
    will accept?

**Next:** band function (score + condition + bucket → band) with a test; the
per-condition table with EER joined to the operating points.

## 2026-10-26 — D9 done: operating-point table, band function, DET per condition

**The operating-point table** (handbook M2; commit 994e5ee, `make operating-points-table`,
8 s): docs/operating-points.tsv (sha256 0b077bd0), committed. 96 rows = 8 conditions × 4
buckets × 3 bands. Each row: model (`voxceleb_ECAPA512_LM@a2f3dcb1`), cohort
(`voxceleb2-dev-v1`, or `-<condition>`), threshold, FAR, false accepts and their speaker
pairs, non-target and target counts, recall, precision (pool 50,000, choice 20), band
precision, and the condition's AS-Norm EER with its trial counts. The EERs equal the
2026-10-26 AS-Norm table. A new model or cohort means a new table (SVS guide §8.4 step 6).

**Band precision** is the precision of the scores inside one band only (above its
threshold, below the next). At pool 50,000 it is 0.019-0.190 in PROBABLE and
0.009-0.063 in WEAK: almost every score inside these two bands is an impostor. Clean
3-10 s: STRONG 0.870, PROBABLE 0.096, WEAK 0.033. This is open item 15 again, now per
band: the guide's 0.85-0.95 (PROBABLE) and 0.4-0.7 (WEAK) need far smaller pools.

**The band function** (commit fff1dae, `svs.bands`): score, speech seconds and condition
→ band and the table row that decided it (M2: every band traceable to a row). A score
on a threshold is in that band; NO_MATCH returns the WEAK row it fell below. Under 3.0 s
of speech it raises (the service refuses to score); an unknown condition raises;
reading a table whose thresholds do not fall from STRONG to WEAK raises. A test checks
that the committed table covers all 8 × 4 condition-buckets. Examples on the real table,
score 6.5: clean 60 s → WEAK, clean 5 s → PROBABLE, G.729 5 s → STRONG.

**Our thresholds rise with duration; the guide's fall.** Clean STRONG: 3-10 s 6.69,
30-120 s 7.65; guide reference 1.34 and 0.94 (§4.4). So here the same score is a higher
band on short audio, the reverse of the guide's example. Not explained. One candidate,
not tested: AS-Norm divides by the cohort spread, and long audio against a cohort of
short clips may give larger z-scores for impostors too (see open item 13).

**DET per condition** (commit 839883f, `make det-conditions`, 9 s): one plot per bucket,
one line per condition, AS-Norm scores; docs/figures/voxceleb1-det-conditions-d9.png
(sha256 8c81cc7e). The four 8 kHz lines (pcm8k, G.711 A/µ, and G.729 above them) lie
above the wideband ones at nearly every FAR, not only at the EER (in 10-30 s, noise
5 dB reaches them below FAR 0.005 %); opus and noise 15 dB lie close
to clean. The curves use the trial list's 100,000 non-targets, so they stop at FAR 1e-5;
the STRONG point (2e-6) is only in the table.

**Handbook D9 status:** calibrate per condition, DET per condition, thresholds from FRR
at a fixed FAR (choice 15), four bands — done. M2 left: minDCF with prior and costs
(D10); the 3.0 s refusal as a service test comes with the API.

**Open items for the mentor**

17. Thresholds rise with duration here and fall in the guide's reference (above).

## 2026-10-27 — D10: minDCF, the quality score and the usability gate

**minDCF** (commit 577e2bd; `make eer-conditions`, `make eer-conditions-asnorm`, and a
new column in docs/operating-points.tsv). Prior and costs: p_target 0.01, c_miss 1,
c_fa 1 (choice 22). Normalised by the better of accept-all and reject-all, so 1.0 means
the scores are worth nothing. AS-Norm, per bucket (3-10 / 10-30 / 30-120 / >120 s):

| condition | minDCF                          |
| --------- | ------------------------------- |
| pcm16k    | 0.1227 / 0.0475 / 0.0412 / 0.0285 |
| pcm8k     | 0.2695 / 0.1330 / 0.1109 / 0.0784 |
| g711a     | 0.2828 / 0.1402 / 0.1078 / 0.0862 |
| g711u     | 0.2781 / 0.1409 / 0.1086 / 0.0839 |
| g729      | 0.4424 / 0.2333 / 0.1906 / 0.1374 |
| opus      | 0.1711 / 0.0652 / 0.0558 / 0.0333 |
| noise15   | 0.1643 / 0.0674 / 0.0486 / 0.0301 |
| noise5    | 0.2507 / 0.1132 / 0.0603 / 0.0369 |

Trial counts as in the D7 table (up to 50,000 target / 100,000 non-target per bucket;
>120 s has 3,102 targets). AS-Norm lowers minDCF in 3-10 s under every condition
(clean 0.1349 → 0.1227, pcm8k 0.3068 → 0.2695, G.729 0.5444 → 0.4424); in >120 s it
does not always (clean 0.0265 → 0.0285), where the targets are few. minDCF ranks the
conditions as EER does. At p_target 0.01 one false accept in 100,000 non-targets adds
about 0.001, so the >120 s values rest on a handful of trials.

p_target 0.01 does not describe our service: one query against a pool of 50,000 has a
prior near 1/50,000. minDCF here is for comparing models and conditions; the service's
operating points stay the D9 thresholds from FRR at a fixed FAR (choice 15).

**Quality score** (commit 5488795, `svs.quality`, `make quality`; ~15 min per
condition on the Mac, 6 workers). The Appendix C composite with its weights, version
`appendix-c-v1`. How each input is measured is choice 23. Every VoxCeleb1 clip
(153,516) and video (22,496) under pcm16k, pcm8k, noise15 and noise5, in
artifacts/quality/voxceleb1-<condition>.tsv (sha256 89dc8f5b, 88ca9339, f620f954,
ffd51fef). g711a/u, G.729 and Opus depend on codec library versions, so their quality
runs on the home PC with their embeddings (not done yet).

Clips of at least 3.0 s, median quality: pcm16k 0.967, pcm8k 0.973, noise15 0.889,
noise5 0.829; lowest 0.572-0.606. Median SNR estimate: 22.9 / 23.4 / 17.6 / 13.7 dB.
Bandwidth: median 7,938 Hz wideband, 4,125 Hz on pcm8k; Appendix C gives full marks
from 3,500 Hz, so this input does not separate 8 kHz from 16 kHz. Median turn
consistency of videos with two or more clips: 0.783 clean, 0.691 noise5.

**Does quality predict errors?** (commit c80511b, `make quality-eval`, 12 s). Trial
quality = the worse side; deciles within each bucket (choice 25). EER of the worst
tenth against the best tenth:

| condition | 3-10 s        | 10-30 s       | 30-120 s      | >120 s        |
| --------- | ------------- | ------------- | ------------- | ------------- |
| pcm16k    | 2.093 / 1.113 | 0.879 / 0.354 | 1.062 / 0.538 | 0.089 / 0.211 |
| pcm8k     | 4.259 / 2.343 | 1.934 / 1.133 | 2.111 / 0.926 | 0.796 / 0.572 |
| noise15   | 2.228 / 1.210 | 1.013 / 0.600 | 1.200 / 0.539 | 0.405 / 0.378 |
| noise5    | 4.333 / 1.843 | 1.834 / 0.787 | 1.307 / 0.586 | 0.780 / 0.198 |

Each tenth holds about 4,100-5,900 targets and 8,900-10,900 non-targets, except
>120 s: about 230-520 targets. In the three shorter buckets the worst tenth has 1.7 to
2.5 times the EER of the best, under every condition: the composite does rank trials
by how likely they are to fail. In >120 s the result is mixed (clean is reversed) on
too few targets to tell. Many trials share quality 1.0 (in clean 10-30 s more than a
tenth), so the best tenth is a seeded random sample of them; the trial list is grouped
by type, and an earlier version that kept file order put 739 targets and 14,261
non-targets into one best tenth.

**The usability gate** (`svs.quality.unusable_reason`): refuse under 3.0 s of speech
(`E_SV_INSUFFICIENT_SPEECH`), then turn consistency under 0.55
(`E_SV_TURN_INCONSISTENT`), then quality under 0.45 (`E_SV_LOW_QUALITY`); defaults
from SVS guide Appendix A, order from its §3.2 example (choice 24).

- The quality floor refuses **no** clip or video under these four conditions: no unit
  of at least 3.0 s with consistent turns falls below 0.45.
- Turn consistency refuses multi-clip videos, so only the 30-120 s and >120 s buckets
  (the video buckets): 30-120 s refuses 417 trials clean (102 targets), 597 pcm8k,
  1,031 noise15, 8,703 noise5 (2,471 targets). EER on what passes: clean
  0.559 → 0.536 %, noise5 0.776 → 0.727 %; in >120 s it changes by 0.02 at most.
- VoxCeleb1 videos are meant to hold one speaker, so most of these refusals are
  probably one speaker whose clips differ (noise lowers the clip-to-clip cosine), not
  two people. The gate then refuses true speakers; on this data the 0.55 default is a
  noise detector as much as a diarization check (open item 19).

**Choices where the handbook and the SVS guide are silent** (the mentor may disagree):

22. minDCF at p_target 0.01, c_miss = c_fa = 1: the defaults of WeSpeaker's
    wespeaker/bin/compute_metrics.py (checked in its source, 2026-10-27), with the same
    normalisation, so our numbers are computed the way the model's own toolkit does.
23. Quality inputs: SNR = speech power (VAD frames, noise taken off) over the mean
    power of the quietest 10 % of 20 ms frames; clipping = share of samples within 1 %
    of the clip's peak (relative to the peak, because G.711 never decodes to full
    scale); bandwidth = highest frequency within 40 dB of the spectral peak (Welch,
    32 ms windows); a clip is one turn (consistency 1.0), a video's turns are its clips
    (cached clip embeddings; clips under 3.0 s are not turns).
24. Gate order: speech, then turn consistency, then quality.
25. Trial quality = the lower of its two sides; deciles within each duration bucket,
    so duration does not pose as quality.

**Open items for the mentor**

18. The Appendix C weights are the guide's starting point, not fitted. The guide says
    to fit them against our own accuracy and freeze them; the decile table shows the
    composite already ranks errors. Fit them now on VoxCeleb1, or wait for internal
    data (D16-D17)?
19. On VoxCeleb1 the quality floor refuses nothing, and turn consistency refuses mostly
    one-speaker videos under noise (above). Keep the guide defaults 0.45 / 0.55?
20. The SNR estimate reads high under added noise: noise5 is mixed at 5 dB, its median
    estimate is 13.7 dB. MUSAN noise varies in level, so the quietest frames
    under-state it. An estimate, not a measurement; it still ranks noise5 below
    noise15 below clean.
21. Quality for g711a/u, G.729 and Opus still to run on the home PC.

**Handbook D10 status:** minDCF with prior and costs — done (M2 item closed). Quality
score and usability gate — done, measured on four conditions. M2 left: the 3.0 s
refusal as a service test, with the API (week 3). Mid-program checkpoint: the mentor
meeting itself.
