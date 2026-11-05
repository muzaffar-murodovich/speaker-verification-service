# Speaker Verification Service: technical report

MohirDev internship, speaker verification track, 2026-09-29 to 2026-11-05.
Every number in this report has a row in `experiments.csv` and an entry in `LOG.md`,
with its dataset, condition and trial count. Section numbers like "§4.3" refer to the
SVS engineering guide; "choice N" and "item N" refer to the numbered choices and open
items in `LOG.md`.

## 0. Summary

I built a speaker verification (1:1) and identification (1:N) service: VAD and speech
pooling, a pretrained ECAPA-TDNN embedding, AS-Norm with a fixed and disjoint cohort,
operating points per condition, duration bucket and pool size, four bands, a profile
store in Qdrant with an evidence rule, and an HTTP service with five endpoints, metrics
and a runbook. On VoxCeleb1 it was measured on 549,856 trials under eight conditions.
Calibration per condition is the reason the bands mean something on 8 kHz audio: a
threshold fitted on clean audio accepts 4 to 40 times more impostors on telephony. On the
internal call-centre slice the VoxCeleb operating points did not hold: different people
reached STRONG far more often than the table says, and I could show why only in part.

| Headline number | Value | Data, condition, n |
|---|---|---|
| EER, AS-Norm, 3-10 s | 1.448 % clean; 2.941 % 8 kHz; 5.299 % G.729 | VoxCeleb1, 50,000 target / 100,000 non-target trials each |
| FRR at STRONG (FAR 2e-6, pool 50,000), 3-10 s | 33.3 % clean; 66.8 % 8 kHz; 84.8 % G.729 | VoxCeleb1, 5,000,000 non-target pairs, 50,000 targets |
| Different people at STRONG on the internal slice | 35.8 % in one call; 1.6 % cross-gender in two calls (table: 0.01 %) | 109 and 11,597 pairs, 217 occurrences |

## 1. What the system is, and what it is not

| In scope | Out of scope |
|---|---|
| Embedding of given speech segments (`/v1/embed`) with a usability gate | Diarization (a minimal one exists for the slice only, outside the service) |
| 1:1 verify with a calibrated score and a band | Authentication or anti-spoofing: not built to resist a deliberate impostor |
| 1:N identify in a filtered pool, with a decision | Names, phone numbers, case data: profiles are opaque ids |
| Profile enrolment, the evidence rule, STRONG-only updates | Merge and split of profiles (§5.6-5.8) |
| Operating points per condition × duration × pool size | Fine-tuning or training a model |
| `/metrics`, a weekly cohesion check, a runbook page | Occurrence collections (§7.2 hot/cold), the §3.7 endpoints |

## 2. Pipeline

```
audio file + segments (ms)            POST /v1/embed
   │  load, pick channel, resample to 16 kHz (anti-aliased)
   ▼
WebRTC VAD (mode 2, 20 ms frames) ── speech pooled per occurrence
   ▼
ECAPA-TDNN (WeSpeaker, ONNX) ── one 192-dim unit vector per occurrence; one per turn
   ▼                                for turn consistency (transient, never stored)
usability gate: < 3.0 s → INSUFFICIENT_SPEECH; turns < 0.55 → TURN_INCONSISTENT;
                quality < 0.45 → LOW_QUALITY
   ▼                                                POST /v1/identify
mandatory filter (source, last_seen_after) → exact pool count → pool tier
   ▼
recall: Qdrant HNSW (m 32, ef 256), 600 candidates, float32, no Qdrant scores used
   ▼
exact rescore: float64 cosine of the 600, keep 200
   ▼
AS-Norm: query vs the condition's cohort, profile stats from its payload (top 300)
   ▼
condition (from {codec, sample_rate}; none → most conservative) × duration bucket × pool tier
   ▼
band (STRONG / PROBABLE / WEAK / NO_MATCH) + the table row behind it + margin_to_next
   ▼
decision: pool > 50,000 → REVIEW_REQUIRED; top < PROBABLE → NO_MATCH;
          two at PROBABLE+ → REVIEW_REQUIRED; else MATCH
   ▼                                       POST /v1/profiles, PATCH /v1/profiles/{id}
enrol PROVISIONAL → update only at STRONG → CONFIRMED when the evidence rule holds
```

| Component | Version |
|---|---|
| Embedding model | WeSpeaker `ECAPA_TDNN_GLOB_c512-ASTP-emb192`, large-margin, ONNX; `voxceleb_ECAPA512_LM@a2f3dcb1` (SHA-256 checked) |
| Features | kaldi-native-fbank, 80 bins, 25/10 ms |
| Runtime | Python 3.12, onnxruntime ≥ 1.30, numpy, scipy, webrtcvad-wheels |
| Vector store | Qdrant v1.19.2 (Docker), qdrant-client 1.19.1 |
| Service | FastAPI + uvicorn, one worker; prometheus-client |
| Cohort | `voxceleb2-dev-v1`, 5,000 speakers, plus one degraded copy per condition |
| Operating points | `docs/operating-points.tsv`, 384 rows |

## 3. Data

| Dataset | Use | Size |
|---|---|---|
| VoxCeleb1 (all) | evaluation, calibration, retrieval bench | 1,251 speakers (690 m, 561 f), 22,496 videos, 153,516 clips |
| VoxCeleb2 dev | AS-Norm cohort | 5,000 speakers, one clip each |
| MUSAN noise | noise conditions | 930 files, 6.23 h; the 775 of ≥ 3 s used |
| Internal call-centre slice (T2) | D16-D17 | 117 recordings, 6.42 h; 37 stereo, 80 mono; 8-48 kHz mp3/wav; labelled turns |

The model was trained on VoxCeleb2 dev only, so every VoxCeleb1 speaker is unseen. A
model trained on VoxCeleb1 + 2 was rejected for that reason (LOG D3).

**Trial list.** 549,856 trials in four duration buckets (3-10, 10-30, 30-120, > 120 s
of net speech; a trial is in the bucket of its shorter side). Short buckets use single
clips, long buckets whole videos (clips joined, capped at 300 s); a trial never mixes the
two. Up to 50,000 targets (reservoir-sampled from all cross-session pairs) and 100,000
same-gender non-targets per bucket; > 120 s has only 3,102 targets.

**Same-session guard.** A `Trial` cannot be built from two clips of one speaker and one
session; the generator never tries, and a test fails if the guard is removed. Video ids
are not enough: 1,055 video pairs hold the same audio under two ids (found by waveform
cross-correlation ≥ 0.2 among clip pairs at cosine ≥ 0.85). They are merged into one
session before trials are drawn. Pairs below cosine 0.85 are likely missed (item 4).

**Cohort disjointness.** VoxCeleb1 and VoxCeleb2 number speakers separately, so names
are compared (from VGGFace2's identity list, one spelling: case, `_`, `-`, `.` ignored).
VoxCeleb2 *test* holds two VoxCeleb1 trial speakers under other spellings; VoxCeleb2
*dev* shares no name with VoxCeleb1. The build asserts it; `tests/test_cohort.py` uses a
real cross-spelled pair.

**Degradation.** Seven conditions, applied to clips, videos and the cohort alike:

| Condition | What is done |
|---|---|
| pcm8k | resample to 8 kHz |
| g711a / g711u | 8 kHz, then G.711 A-law / µ-law |
| g729 | 8 kHz, then G.729 Annex A (bcg729 1.1.1) |
| opus | Opus 16 kbit/s VoIP (ffmpeg 6.1.1, libopus 1.4) |
| noise15 / noise5 | MUSAN noise at 15 / 5 dB SNR |

**The internal slice.** Labels say who spoke when inside a recording; speaker ids are
local to one recording. Same people across recordings were found by scoring every pair
and listening to the candidates (12 groups); the listener confirmed 9 people in 26
occurrences, 27 target pairs (D16). These targets were selected by score, so they are
used only for counts, never for an error rate.

## 4. Scoring and calibration

**Raw cosine → AS-Norm.** Each side's top-300 cohort scores give a mean and spread; the
score is the mean of the two z-scores (§4.3). Cohort size 5,000 (the handbook's upper
end; the guide's 20,000 is for production); n = 300 from the guide. Top-n, not all
5,000: the impostors that cause false accepts are the ones close to the voice; far
voices pull the mean down and make an ordinary score look strong. Each condition is
normalised with a cohort degraded the same way. Profile-side statistics are cached in
the profile payload.

**Thresholds** come from FRR at a fixed FAR, not from the EER. The FAR follows the pool
size N: STRONG at 0.1 / N (0.1 expected impostors per query, §8.4), PROBABLE at 1 / N,
the WEAK floor at 2 / N (one more impostor candidate per query to review; choice 21).
Four pool tiers: 1,000, 10,000, 50,000, 100,000; a pool between tiers takes the stricter
one. Each threshold is measured on 5,000,000 fresh same-gender non-target pairs per
bucket, so STRONG at pool 50,000 (2e-6) rests on about 10 false accepts.

STRONG thresholds (AS-Norm, pool 50,000) and FRR at them:

| Condition | 3-10 s | 10-30 s | 30-120 s | > 120 s |
|---|---|---|---|---|
| pcm16k | 6.69 (33.3 %) | 7.22 (17.4 %) | 7.65 (13.4 %) | 7.42 (8.1 %) |
| pcm8k | 6.53 (66.8 %) | 6.94 (47.9 %) | 7.08 (37.8 %) | 7.49 (35.0 %) |
| g729 | 6.17 (84.8 %) | 6.30 (62.7 %) | 6.66 (54.6 %) | 6.77 (42.8 %) |
| opus | 6.93 (48.8 %) | 7.25 (24.8 %) | 7.28 (14.5 %) | 7.32 (11.0 %) |
| noise5 | 7.03 (63.5 %) | 7.21 (36.0 %) | 7.58 (17.7 %) | 7.36 (9.5 %) |

All 8 conditions × 4 buckets × 4 tiers × 3 bands, with recall, precision, band precision,
EER and minDCF, are in `docs/operating-points.tsv`. The > 120 s STRONG rows rest on 2
false accepts and are not reliable.

**Bands.**

| Band | Meaning | Action |
|---|---|---|
| STRONG | ≥ the 0.1 / N threshold | may be assigned automatically; the only band that may update a centroid |
| PROBABLE | ≥ the 1 / N threshold | assigned but flagged; never updates a centroid |
| WEAK | ≥ the 2 / N floor | listed for a person to review; no automatic action |
| NO_MATCH | below the floor | new profile |

Every band returned by the service carries the table row that decided it.

## 5. Results

**EER %, raw cosine → AS-Norm** (VoxCeleb1; 50,000 / 100,000 trials per bucket, > 120 s
3,102 targets; each condition scores the trials its audio allows, at most 6,234 of
549,856 refused under noise5):

| Condition | 3-10 s | 10-30 s | 30-120 s | > 120 s |
|---|---|---|---|---|
| pcm16k | 1.560 → 1.448 | 0.602 → 0.582 | 0.642 → 0.559 | 0.355 → 0.322 |
| pcm8k | 3.345 → 2.941 | 1.418 → 1.350 | 1.418 → 1.249 | 0.709 → 0.806 |
| g711a | 3.467 → 3.039 | 1.472 → 1.408 | 1.450 → 1.285 | 0.735 → 0.838 |
| g711u | 3.444 → 3.037 | 1.466 → 1.392 | 1.457 → 1.269 | 0.710 → 0.838 |
| g729 | 6.393 → 5.299 | 2.962 → 2.488 | 2.693 → 2.248 | 1.289 → 1.325 |
| opus | 2.058 → 1.777 | 0.774 → 0.702 | 0.818 → 0.687 | 0.427 → 0.355 |
| noise15 | 2.007 → 1.847 | 0.770 → 0.732 | 0.727 → 0.648 | 0.419 → 0.387 |
| noise5 | 3.296 → 3.025 | 1.307 → 1.191 | 0.873 → 0.776 | 0.516 → 0.481 |

minDCF (p_target 0.01, c_miss = c_fa = 1, AS-Norm), 3-10 s: clean 0.123, pcm8k 0.270,
G.729 0.442. DET curves: `docs/figures/voxceleb1-det-d5.png` (clean, per bucket) and
`docs/figures/voxceleb1-det-conditions-d9.png` (per condition).

- **Sample rate** doubles the EER on its own; G.711 adds almost nothing; G.729 doubles it
  again. Opus at 16 kHz and noise at 15 dB stay near clean.
- **The clean threshold on 8 kHz** (raw cosine, 3-10 s): FAR 1.56 % → 6.13 % (pcm8k),
  19.94 % (G.729), while FRR barely moves, so nothing in missed targets warns of it.
  With a matched cohort AS-Norm brings the G.729 FAR back to 1.30 %, but FRR rises to
  13.6 %: the clean threshold is wrong in the other direction. Hence thresholds per
  condition.
- **AS-Norm** gains 4-17 % on 8 kHz up to 120 s, more than on clean (3-13 %). In
  > 120 s on 8 kHz it loses by 3-4 missed targets of 3,102, inside that bucket's noise.
- **Gender**: female EER is higher in 3 of 4 clean buckets and in all four under 8 kHz
  and G.729 (G.729 > 120 s: 2.315 % vs 0.796 %). Not analysed further.
- **Language** and **session gap**: VoxCeleb1 has neither (nationality only; video ids
  carry no dates).

**The internal slice (D16-D17).** The slice measures false accepts only (no unbiased
targets). Share at STRONG, pool tier 1,000 (table FAR 1e-4, same-gender):

| Non-target pairs | n | verify (no condition) | pcm8k |
|---|---|---|---|
| two speakers of one call | 109 | 35.8 % | 22.0 % |
| two calls, different annotator gender | 11,597 | 1.6 % | 0.8 % |

| Breakdown (cross-gender pairs, verify) | STRONG share |
|---|---|
| same source folder / different folders | 5.4 % / 1.1 % (1,494 / 10,103 pairs) |
| both 8 kHz / both wider (re-encoded) | 0.3 % / 6.2 % |
| both stereo channels / both mono | 5.7 % / 1.7 % |
| 3-10 s / 30-120 s (shorter side) | 0.2 % / 1.9 % |

- **Cross-source** is the one breakdown the handbook says predicts production best. Two
  different people from one source folder pass STRONG 5 times as often as from two:
  the embeddings carry the recording chain as well as the voice. Of the 27 confirmed
  target pairs, 4 cross sources; all four are STRONG, but they were found by score.
- **Session gap**: not measurable (8 of 117 files carry a date, no confirmed person among
  them). **Language**: not measurable (segment labels 6,151 Uzbek, 15 Russian, 5 English).
- **Through the service** (9 profiles, identify pool 9): known speakers 17 of 17 top-1
  right, verify STRONG 17 of 17, but MATCH only 2 (15 REVIEW_REQUIRED: a second profile
  reaches PROBABLE). Recordings with no confirmed person: 41 of 172 assigned to one of
  the 9, 24 at STRONG; one certainly wrong (other gender).
- **Diarization** (mono files, 80): DER 34.1 % with Ward clustering; the usability gate
  refuses 54 of 160 diarized mono items, so it catches many diarization errors.

**Every gap between VoxCeleb and the slice, with its hypothesis:**

| Gap | Hypothesis | Status |
|---|---|---|
| STRONG false accepts 1e-4 → 1.6-36 % | The cohort is not like this audio: a slice voice's top-50 mean against the other slice voices is 0.494 (sd 0.092), against the VoxCeleb2 8 kHz cohort 0.278 (sd 0.051), so AS-Norm inflates every slice score | tested, supported. A slice cohort moves 0 of 11,706 non-targets to STRONG, but also 26 of 27 targets below it: cohort and thresholds must both come from in-domain audio |
| the same | The service picked the most conservative condition, not 8 kHz | tested: pcm8k halves the false accepts; the gap stays |
| the same, worst for same-call and same-source pairs | A shared line, crosstalk in stereo, a shared recording chain | supported by the breakdowns, not isolated |
| EER and FRR on real audio | unknown | not measurable: no unbiased target trials |
| 2 of 17 known speakers MATCH vs 59 % on VoxCeleb (pool 1,000) | Impostor profiles reach PROBABLE (the first hypothesis), so the two-assignable rule fires | follows from the first row |

## 6. The store and the service

**Collection design.** One Qdrant collection per model version:
`spk_profile_<version, lowercased>_<first 8 hex of SHA-256 of the exact version>`.
Cosine, 192 dims, HNSW m 32, ef_construct 256, `full_scan_threshold` 20,000 (KB), keyword
indexes on `status` and `source`, datetime on `last_seen`, float on `evidence_seconds`.
A store opened for one version refuses any other with `E_SV_MODEL_VERSION_MISMATCH`, so
vectors of two models can never be compared. Profiles are kept in full precision; int8
quantization was measured on the benchmark collection only.

**Recall@200 vs brute force** (151,174 VoxCeleb1 clip vectors, 1,000 held-out queries):

| Vectors | hnsw_ef | recall@200 mean / min | same decision | recall stage p50 / p95 | exact stage p50 / p95 |
|---|---|---|---|---|---|
| float32 | 256 | 0.9996 / 0.980 | 1000 / 1000 | 8.0 / 8.5 ms | 30.5 / 31.8 ms |
| int8 | 256 | 0.9996 / 0.975 | 1000 / 1000 | 6.9 / 7.2 ms | 30.6 / 31.8 ms |

Filters of 10 %, 1 % and 0.1 % give recall 1.0: below `full_scan_threshold` Qdrant scans
the filtered points and the graph is not used, so filtered graph search is not measured
(item 25). The exact stage is almost all the fetch of 600 vectors. The exact pool count
adds 24 ms p95 on the whole pool. Mac M3 with Docker Desktop.

**Base rate** (clean 3-10 s, 2,000 queries, pools of nested VoxCeleb1 clips):

| Pool | Fixed pool-1,000 threshold: impostors at STRONG per query | Pool's own threshold: impostors per query | FRR at STRONG |
|---|---|---|---|
| 1,000 | 0.046 | 0.046 | 19.65 % |
| 10,000 | 0.495 | 0.061 | 33.50 % |
| 50,000 | 2.631 | 0.155 | 39.30 % |
| 100,000 | 5.151 | 0.075 | 51.15 % |

False accepts grow as N × FAR with a fixed threshold; the pool rule holds them near 0.1
and pays in FRR, which is what filtering first buys back. Identify refuses a call with no
filter (`E_SV_FILTER_REQUIRED`) and returns REVIEW_REQUIRED above 50,000.

**Service.** `/v1/embed`, `/v1/verify`, `/v1/identify`, `/v1/profiles`,
`PATCH /v1/profiles/{id}`, `/v1/healthz`, `/metrics`; `request_id` and `model_version`
in every response, `E_SV_*` errors, idempotency keys on both profile writes. `make
service` + `make smoke` on the Mac. `/metrics` exposes embed duration, rejection reasons,
identify duration by stage, decisions by band, profiles by status, cohesion, pool size.
`docs/runbook.md` has seven symptoms in the symptom → first checks → likely cause form.

## 7. Profiles

**Centroid.** `sum_vec` (float64) and `sum_w` are stored; the centroid is
`sum_vec / |sum_vec|`. An occurrence weighs `min(quality × log1p(seconds), 3.0)`: log
because the tenth minute of one call adds less new evidence than the first, capped so
one long call cannot outweigh several sessions (0.9 quality at 600 s gives 5.76, capped
to 3.0; a linear weight would give 540).

**Evidence rule** (to CONFIRMED): ≥ 120 s of speech, ≥ 2 distinct recordings, mean
quality ≥ 0.6, cohesion ≥ 0.65. Two recordings, because one recording's channel and
session are baked into every one of its embeddings; a second session shows the profile is
the person, not the line.

**What is refused and why.**

| Refusal | Code | Why |
|---|---|---|
| promotion before the rule holds | `E_SV_ENROL_RULE_UNMET` (+ what is missing) | one recording is not evidence of a person |
| centroid update below STRONG | `E_SV_UPDATE_BAND_TOO_LOW` | a PROBABLE match is wrong often enough (band precision 0.10 at pool 50,000) that updating on it pulls the centroid towards impostors, and every later match then inherits the error |
| an occurrence under 3.0 s or quality < 0.45 | `E_SV_INSUFFICIENT_SPEECH`, `E_SV_LOW_QUALITY` | too little or too poor speech to be evidence |
| the same occurrence twice | `E_SV_DUPLICATE_OCCURRENCE` | its seconds would count twice towards 120 s |

Measured on VoxCeleb1 (all 1,251 speakers): the rule promotes every speaker in six
conditions, one refused each under noise5 and G.729. Cohesion catches one wrong video in
three for only 0.6-9.8 % of speakers (item 23): the real protection is the STRONG-only
update, not cohesion.

## 8. What I got wrong

1. **I treated the VoxCeleb operating points as the service's operating points.** From
   D9 (2026-10-26) to D16 I described STRONG as "FAR 2e-6 at pool 50,000, 1e-4 at pool
   1,000", and the README and the service return those numbers with every band. On the
   internal slice (2026-11-04) two different people in one call reached STRONG in 39 of
   109 pairs (35.8 %), and cross-gender pairs from two calls in 1.6 %. The table was
   right for VoxCeleb and silent about every other audio, and I did not say so where it
   is used. The service now has the evidence; the fix (in-domain cohort and thresholds)
   is not built.
2. **I chose the diarizer's clustering on the same labels I then scored it with**
   (2026-11-04). Average linkage gave DER 55.9 %, Ward 34.1 %; both measured on the
   slice's only labels, so 34.1 % is optimistic. There was no held-out part to choose on.
3. **I trusted video ids as sessions** (2026-10-15). 1,055 video pairs hold the same
   audio under two ids; 0.8-2.5 % of target trials per bucket crossed one. The first EERs
   (2026-10-16) included them. Removing them changed the EER by only 0.004-0.010 points,
   but I found them by noticing 22 targets above cosine 0.999, not by checking.
4. **I wrote files without fsync** (D6). Two power cuts on the home PC (2026-10-22,
   2026-10-23) left five 0-byte embedding files that the next night skipped as done.
   `make degraded-scores` found them only because loading failed. Fixed in 6e01e43 with a
   test; five speakers recomputed.
5. **I assumed degraded audio is the same on every machine.** bcg729 1.1.2 (Mac) vs
   1.1.1 (home PC) changed G.729 output in 32 of 60 clips (lowest embedding cosine
   0.932); ffmpeg 9 vs 6 changed Opus VAD lengths in most clips. Found 2026-10-26 by
   comparing machines. The degraded results are tied to the home PC's library versions,
   and a clean clone elsewhere will differ slightly (item 12).
6. **I did not look at file sizes before the first embedding run** (2026-10-13). The two
   long buckets came out nearly empty (1,076 and 1 clips); the sizes, readable in 2 s,
   already showed it. It cost a second full run (11,550 s) for video embeddings.
7. **I expected the turn-consistency gate to detect a second speaker.** On VoxCeleb1 it
   refuses mostly one-speaker videos under noise (8,703 trials in noise5 30-120 s); on the
   slice it refuses 12 % of real single speakers (22 of 181 mono). At 0.55 it is as much
   a noise detector as a diarization check (items 19, 40).

## 9. What the production SVS does that mine does not

| Item | Why it exists |
|---|---|
| Occurrence collections (hot / cold) next to profiles (§7.2) | Identify can then match a voice heard once and never enrolled, and a profile can be rebuilt, merged or split from its occurrences. Mine keeps occurrences inside the profile payload only. |
| Duplicate detection, merge and split with a review queue (§5.6-5.8) | A person heard first on a bad line and later on a good one gets two weak profiles; without merge the store fills with fragments and identify gets worse. Merge is reviewed because one wrong merge joins two people for good. Mine never merges. |
| A cohort of about 20,000 matched to the production condition mix (§4.3), and thresholds measured on its own audio (§0.5) | Section 5 shows why: with a VoxCeleb cohort and VoxCeleb thresholds, call-centre voices look like each other and STRONG stops meaning 1e-4. |

## 10. What I would build next

1. **An in-domain cohort and in-domain operating points.** Section 5's first gap is the
   largest error in the system. It needs unlabelled call-centre audio with source labels
   for the cohort and a labelled set with speaker ids across calls for thresholds
   (items 44, 45). Until then the service should say that its bands are VoxCeleb bands.
2. **A source and session label on every occurrence, and same-source impostors in
   calibration.** Same-source pairs pass 5× as often; calibrating on them makes STRONG
   hold where it is hardest.
3. **A real diarizer upstream** (§1.1), not mine: 34 % DER feeds wrong speakers into
   profiles, and the gate is the only thing that catches it.
4. **An interval on every EER and FAR** (bootstrap over speakers, item 5): the > 120 s
   and STRONG numbers rest on a handful of trials.
5. **Delete**: the noise conditions from the service's condition list, since no request
   can name them (choice 50), and the in-process idempotency store, which a second worker
   or a restart breaks (item 34).

## Appendix: configuration reference

| Parameter | Value | Source |
|---|---|---|
| `CFG_SV_MIN_SPEECH_SECONDS` | 3.0 s | guide Appendix A |
| `CFG_SV_MIN_TURN_CONSISTENCY` | 0.55 | guide Appendix A |
| `CFG_SV_MIN_QUALITY` | 0.45 | guide Appendix A |
| quality formula | Appendix C weights, version `appendix-c-v1` | guide Appendix C; inputs: choice 23 |
| VAD | WebRTC mode 2, 20 ms frames, 16 kHz | choice in LOG D2 |
| `CFG_SV_COHORT_TOP_N` | 300 | guide §4.3 |
| cohort | `voxceleb2-dev-v1`, 5,000 speakers, one clip each | handbook D8 range; choices 10-14 |
| pool tiers | 1,000 / 10,000 / 50,000 / 100,000 | choice 41 |
| band FARs | STRONG 0.1 / N, PROBABLE 1 / N, WEAK 2 / N | guide §8.4; choices 15, 21, 42 |
| calibration non-targets | 5,000,000 same-gender pairs per bucket, seed 0 | choice 16 |
| `CFG_SV_MAX_AUTO_POOL` | 50,000 | guide Appendix A |
| recall_stage_k / `CFG_SV_OVERSAMPLING` | 200 / 3.0 (600 candidates) | handbook D13; guide Appendix A |
| `CFG_SV_HNSW_EF_SEARCH_UI` | 256 | choice 37 |
| HNSW m / ef_construct | 32 / 256 | handbook D11 |
| `CFG_SV_MAX_OCC_WEIGHT` | 3.0 | guide §5.4 |
| promotion: seconds / sessions / quality / cohesion | 120 / 2 / 0.6 / 0.65 | guide §5.2 |
| `CFG_SV_MIN_UPDATE_BAND` | STRONG | guide §5.3 |
| `CFG_SV_COHESION_ALERT` | 0.55 | guide Appendix A |
| verify pool | 1 (tier 1,000) | choice 52 |
| no condition block | most conservative condition of the bucket | choice 51 |
| minDCF | p_target 0.01, c_miss = c_fa = 1 | choice 22 |

## Appendix: how to reproduce

```bash
git clone <repo> && cd speaker-verification-service
git checkout v1.0
make setup        # .venv with uv
make test         # 358 tests (the Qdrant ones skip without Docker)
hf auth login     # VoxCeleb's Hugging Face mirror needs an account (since 2026-10)
make eval         # model, VoxCeleb1 (31 GB), embeddings (~7 h CPU), trials, scores, EER, DET
make service && make smoke   # the five endpoints on VoxCeleb1 audio
```

`make eval` checks the trial list and scores against `eval.sha256`. The degraded
conditions (`make degraded-embeddings`, ~41 h CPU) depend on bcg729 1.1.1 and ffmpeg
6.1.1 for G.729 and Opus. The internal slice is not in the repository; its targets are in
the README.
