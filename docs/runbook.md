# Runbook

One row per symptom, in the format of SVS guide §11.2: what you see, what to check
first, and what it usually turns out to be. The metric names are those of `GET /metrics`
([src/svs/monitoring.py](../src/svs/monitoring.py)). A `rate()` or a p95 is read in
Prometheus over a window; the service itself only counts since it started.

| Symptom | First checks | Likely cause |
|---|---|---|
| Identify p95 above 120 ms | `svs_identify_duration_seconds` by `stage`: which stage grew. `count` → `svs_identify_pool_size`. `recall` → Qdrant segment count, `ef_search`, whether the pool is under `full_scan_threshold` (20,000). `exact` → the number rescored (`searched.rescored`, at most 200). | Wider filters (larger pools), or a collection that needs optimising. On D13 the exact stage was mostly fetching the 200 full-precision vectors. |
| Embed refusals rising | `svs_embed_rejected_total` by `reason`, against the weeks before. | `E_SV_INSUFFICIENT_SPEECH`: shorter turns from diarization, or wrong segment times. `E_SV_TURN_INCONSISTENT`: two voices in one speaker's segments, e.g. a mono line carrying agent and caller. `E_SV_LOW_QUALITY`: a new codec, line or noise; `quality_detail` of refused items names the component. |
| Band mix shifts towards `NO_MATCH` (more new profiles) | `svs_decisions_total` by `band`; the `warnings` of identify responses (no `condition` sent means the strictest thresholds); `svs_embed_rejected_total` for the same period. | Audio from a source with another codec or sample rate. The thresholds are per condition (docs/operating-points.tsv); a condition the table does not have is refused, not guessed. |
| `REVIEW_REQUIRED` rising | `svs_identify_pool_size` above 50,000; candidate lists with two candidates at PROBABLE or above. | Filters too wide: above CFG_SV_MAX_AUTO_POOL nothing is assigned automatically. Or one speaker in two profiles (a duplicate), which shows as two assigning candidates. |
| Cohesion left tail growing | `make cohesion-check` (history.tsv, review list); recent profile updates of the profiles on the list; `svs_profile_cohesion`. | An occurrence of another person in a profile. Only STRONG moves a centroid (E_SV_UPDATE_BAND_TOO_LOW below it), so check the diarization of the added occurrences first. |
| `svs_profiles` missing from `/metrics`, calls fail with `E_SV_STORE_UNAVAILABLE` | `docker compose ps`; `curl localhost:6333/`; the service log. | Qdrant is down or unreachable. The gauge is left out on purpose rather than showing the last count. |
| Calls fail with `E_SV_MODEL_VERSION_MISMATCH` | `model_version` in the failing requests against `GET /v1/healthz`. | A caller holds embeddings of another model version. They cannot be compared; they are re-embedded with the current model (SVS guide §9). |

## Weekly

Run `make cohesion-check` once a week. It appends one line to
`artifacts/cohesion/history.tsv` and prints `ALERT` when the share of profiles below
0.55 (CFG_SV_COHESION_ALERT) is larger than at the last check. The profiles below 0.55
are in `artifacts/cohesion/review-<date>.tsv`, for a person to look at.
