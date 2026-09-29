# Speaker Verification Service

A speaker verification (1:1) and identification (1:N) service built during the
MohirDev internship: speech extraction, ECAPA-TDNN embeddings, calibrated scoring,
speaker profiles and vector search.

**Status:** D1, skeleton only. Progress and decisions are in [LOG.md](LOG.md).

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
