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
