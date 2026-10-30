"""The HTTP service (handbook D15): the five endpoints of SVS guide §3.2-3.6.

Every response, errors included, carries request_id and model_version (§3.1). An error is
one of the E_SV_* codes of §3.8: the code is what a caller acts on, the message is for a
person. The scoring logic lives in the other svs modules; this file only turns JSON into
their arguments, and their results and exceptions back into JSON.
"""

import base64
import binascii
import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from svs.asnorm import cohort_stats
from svs.bands import NO_MATCH, OPTABLE, band, read_table
from svs.embed import EMBEDDING_DIM, MODEL_VERSION, Embedder
from svs.extract import load_audio, speech_from_samples
from svs.optable import BAND_ORDER, CLEAN, cohort_of
from svs.profile import (
    CONFIRMED,
    MIN_UPDATE_BAND,
    PROMOTE_MIN_COHESION,
    PROMOTE_MIN_QUALITY,
    PROMOTE_MIN_SECONDS,
    PROMOTE_MIN_SESSIONS,
    PROVISIONAL,
    EnrolRuleUnmet,
    Occurrence,
    Profile,
    UnusableOccurrence,
    UpdateBandTooLow,
)
from svs.quality import quality, quality_detail, turn_consistency, unusable_reason
from svs.retrieve import RECALL_K, TOP_K, FilterRequired, IdentifyFilter, identify
from svs.store import ModelVersionMismatch, ProfileStore
from svs.trials import MIN_SPEECH_SECONDS, duration_bucket

# §3.8: each code's HTTP status and whether the same call can succeed later.
ERRORS = {
    "E_SV_INSUFFICIENT_SPEECH": (422, "after change"),
    "E_SV_LOW_QUALITY": (422, "after change"),
    "E_SV_TURN_INCONSISTENT": (422, "after change"),
    "E_SV_MODEL_VERSION_MISMATCH": (409, "no"),
    "E_SV_UNKNOWN_CONDITION": (422, "after change"),
    "E_SV_PROFILE_NOT_FOUND": (404, "no"),
    "E_SV_ENROL_RULE_UNMET": (422, "after change"),
    "E_SV_UPDATE_BAND_TOO_LOW": (422, "no"),
    "E_SV_STORE_UNAVAILABLE": (503, "yes"),
    "E_SV_INDEX_BUILDING": (503, "yes"),
    # Not in §3.8 (LOG, D15).
    "E_SV_FILTER_REQUIRED": (422, "after change"),
    "E_SV_INVALID_REQUEST": (422, "after change"),
    "E_SV_IDEMPOTENCY_KEY_REUSED": (409, "no"),
    "E_SV_DUPLICATE_OCCURRENCE": (409, "no"),
}


class ServiceError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# Exceptions that already carry an E_SV_* code where the rule they guard is written.
CODED = (
    ServiceError,
    ModelVersionMismatch,
    FilterRequired,
    EnrolRuleUnmet,
    UpdateBandTooLow,
    UnusableOccurrence,
)


# The (codec, sample rate) of a request's condition block (§3.3) -> the condition name of
# the operating-point table (svs.degrade). The noise conditions have no codec to name them.
CONDITIONS = {
    ("pcm", 16000): "pcm16k",
    ("pcm", 8000): "pcm8k",
    ("g711a", 8000): "g711a",
    ("g711u", 8000): "g711u",
    ("g729", 8000): "g729",
    ("opus", 16000): "opus",
}
# A 1:1 claim compares the audio with one profile: the smallest pool tier applies (LOG, D15).
VERIFY_POOL = 1
# A unit vector's norm, up to float32 rounding after base64 and back.
NORM_TOLERANCE = 1e-3
# Items per /v1/embed call: the batch size of SVS guide §3.1.
MAX_EMBED_ITEMS = 256
# Idempotency keys remembered; the oldest is forgotten first (LOG, D15).
IDEMPOTENCY_KEYS = 10_000


class Idempotency:
    """Responses of mutating calls by Idempotency-Key, in this process's memory.

    The lock is held for the whole call, so a retry that arrives while the first try is
    still running waits for it and gets its response. It also serialises every write, so
    two updates of one profile cannot read the same sum_vec and lose one of the two.
    """

    def __init__(self, size: int = IDEMPOTENCY_KEYS) -> None:
        self.lock = threading.Lock()
        self.size = size
        self.done: OrderedDict[str, tuple[str, int, dict]] = OrderedDict()


@dataclass
class Services:
    """What the endpoints work with; tests build it with fakes."""

    store: ProfileStore | None = None
    table: dict = field(default_factory=dict)  # svs.bands.read_table
    cohorts: dict[str, NDArray] = field(default_factory=dict)  # condition -> cohort embeddings
    idempotency: Idempotency = field(default_factory=Idempotency)
    embedder: Embedder | None = None
    audio_root: Path | None = None  # CFG_SV_AUDIO_ROOT: audio_uri paths are inside it


class Condition(BaseModel):
    codec: str
    sample_rate: int


class Side(BaseModel):
    """One side of a verify call: an embedding with its net speech, or a stored profile."""

    embedding: str | None = None
    net_speech_seconds: float | None = None
    profile_id: str | None = None


class VerifyRequest(BaseModel):
    model_version: str
    left: Side
    right: Side
    condition: Condition | None = None


class Filter(BaseModel):
    """The §3.4 filter. last_seen_after needs a time zone, e.g. 2026-05-01T00:00:00Z."""

    source: list[str]
    last_seen_after: datetime
    status: list[str] = ["CONFIRMED", "PROVISIONAL"]
    min_evidence_seconds: float = 0.0


class IdentifyRequest(BaseModel):
    model_version: str
    embedding: str
    net_speech_seconds: float
    # Optional here so that a missing filter gets its own code, not E_SV_INVALID_REQUEST.
    filter: Filter | None = None
    top_k: int = Field(TOP_K, ge=1, le=RECALL_K)
    min_band: str = "WEAK"  # the lowest band listed among the candidates
    condition: Condition | None = None


class EmbedItem(BaseModel):
    id: str
    audio_uri: str  # a path relative to CFG_SV_AUDIO_ROOT
    segments: list[tuple[float, float]]  # milliseconds, [start, end)
    channel: int | None = None  # zero-based; None only for mono audio
    purpose: Literal["TURN", "OCCURRENCE"]


class EmbedRequest(BaseModel):
    model_version: str
    items: list[EmbedItem] = Field(min_length=1, max_length=MAX_EMBED_ITEMS)


class OccurrenceIn(BaseModel):
    """One speaker in one recording, as /v1/embed returned it (§3.5)."""

    audio_id: str
    speaker_ref: str
    embedding: str
    net_speech_seconds: float
    quality: float = Field(ge=0.0, le=1.0)


class EnrolRequest(BaseModel):
    model_version: str
    occurrence: OccurrenceIn
    # Where the audio came from: the identify filter needs it (handbook D11 payload).
    source: str
    status: str = PROVISIONAL
    attributes: dict = {}


class UpdateRequest(BaseModel):
    model_version: str
    occurrence: OccurrenceIn
    matched_band: str
    source: str | None = None


def decode_embedding(text: str) -> NDArray[np.float32]:
    """Base64 little-endian float32[192] (§3.1), checked to be finite and unit length.

    Embeddings are normalised when produced and never at query time (§4.1): a vector that
    is not unit length was made by something else, and is refused rather than repaired.
    """
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ServiceError("E_SV_INVALID_REQUEST", f"embedding is not base64: {exc}") from exc
    if len(raw) != 4 * EMBEDDING_DIM:
        raise ServiceError(
            "E_SV_INVALID_REQUEST", f"embedding has {len(raw)} bytes, not {4 * EMBEDDING_DIM}"
        )
    vector = np.frombuffer(raw, "<f4").astype(np.float32)
    if not np.isfinite(vector).all() or abs(np.linalg.norm(vector) - 1) > NORM_TOLERANCE:
        raise ServiceError("E_SV_INVALID_REQUEST", "embedding is not a finite unit vector")
    return vector


def encode_embedding(vector: NDArray) -> str:
    return base64.b64encode(np.asarray(vector, "<f4").tobytes()).decode()


def check_version(model_version: str) -> None:
    if model_version != MODEL_VERSION:
        raise ModelVersionMismatch(f"{model_version!r} request to the {MODEL_VERSION!r} service")


def check_speech(seconds: float) -> None:
    """Below 3.0 s of net speech the service refuses to score (handbook M2)."""
    if not seconds >= MIN_SPEECH_SECONDS:
        raise ServiceError(
            "E_SV_INSUFFICIENT_SPEECH", f"{seconds} s of speech, {MIN_SPEECH_SECONDS} s needed"
        )


def resolve_condition(
    condition: Condition | None, seconds: float, table: dict
) -> tuple[str, list[str]]:
    """The table's condition name for the request, and any warning to return with it.

    Without a condition block the call gets the condition whose STRONG threshold is the
    highest in this bucket, over all pool tiers: the most conservative one (§3.3).
    """
    if condition is not None:
        name = CONDITIONS.get((condition.codec.lower(), condition.sample_rate))
        if name is None or (name, duration_bucket(seconds)) not in table:
            raise ServiceError(
                "E_SV_UNKNOWN_CONDITION",
                f"codec {condition.codec!r} at {condition.sample_rate} Hz is not calibrated",
            )
        return name, []
    bucket = duration_bucket(seconds)
    name = max(
        (c for c, b in table if b == bucket),
        key=lambda c: max(float(rows[0]["threshold"]) for rows in table[(c, bucket)].values()),
    )
    return name, [f"no condition given: thresholds of {name}, the most conservative"]


def operating_point(row: dict) -> dict:
    """The operating-point table row behind a band (handbook M2: bands traceable to rows)."""
    return {
        "name": f"{row['condition']}_{row['bucket']}_pool{row['pool']}_{row['band']}",
        "condition": row["condition"],
        "bucket": row["bucket"],
        "pool": int(row["pool"]),
        "far": float(row["far"]),
        "recall": float(row["recall"]),
        "precision": float(row["precision"]),
        "eer": float(row["eer_%"]) / 100,
        "trials": int(row["n_target"]) + int(row["n_nontarget"]),
        "cohort": row["cohort"],
    }


def store_of(services: Services) -> ProfileStore:
    if services.store is None:
        raise ServiceError("E_SV_STORE_UNAVAILABLE", "no vector store configured")
    return services.store


def profile_side(services: Services, profile_id: str) -> tuple[NDArray, float, float, float]:
    """A stored profile's centroid, evidence seconds and cached cohort mean and spread."""
    found = store_of(services).get(MODEL_VERSION, profile_id)
    if found is None:
        raise ServiceError("E_SV_PROFILE_NOT_FOUND", f"no profile {profile_id!r}")
    vector, payload = found
    return vector, payload["evidence_seconds"], payload["cohort_mu"], payload["cohort_sd"]


def verify(services: Services, req: VerifyRequest) -> dict:
    """1:1 (§3.3): the raw cosine, its AS-Norm score, the band and the row behind it.

    The bucket is that of the shorter side, as in the trial list (svs.trials). Each
    embedding side is normalised with the condition's cohort; a profile side uses the
    statistics cached in its payload when it was written (§4.3).
    """
    check_version(req.model_version)
    if req.left.embedding is None or req.left.net_speech_seconds is None:
        raise ServiceError("E_SV_INVALID_REQUEST", "left needs embedding and net_speech_seconds")
    left = decode_embedding(req.left.embedding)
    if req.right.profile_id is not None:
        right, right_seconds, mu_r, sd_r = profile_side(services, req.right.profile_id)
    elif req.right.embedding is not None and req.right.net_speech_seconds is not None:
        right, right_seconds = decode_embedding(req.right.embedding), req.right.net_speech_seconds
        mu_r = sd_r = None
    else:
        raise ServiceError(
            "E_SV_INVALID_REQUEST", "right needs a profile_id, or embedding and net_speech_seconds"
        )
    seconds = min(req.left.net_speech_seconds, right_seconds)
    check_speech(seconds)
    name, warnings = resolve_condition(req.condition, seconds, services.table)
    cohort = services.cohorts[name]
    (mu_l,), (sd_l,) = cohort_stats(left[None, :], cohort)
    if mu_r is None:
        (mu_r,), (sd_r,) = cohort_stats(right[None, :], cohort)
    raw = float(left.astype(np.float64) @ right.astype(np.float64))
    # svs.asnorm.as_norm, with the two z-scores kept for the explanation.
    z_left = (raw - mu_l) / (sd_l + 1e-6)
    z_right = (raw - mu_r) / (sd_r + 1e-6)
    score = 0.5 * (z_left + z_right)
    name_of_band, row = band(score, seconds, name, services.table, VERIFY_POOL)
    return {
        "raw_score": raw,
        "normalised_score": score,
        "band": name_of_band,
        "threshold_used": float(row["threshold"]),
        "operating_point": operating_point(row),
        "explain": {"cohort_size": len(cohort), "z_left": z_left, "z_right": z_right},
        "warnings": warnings,
    }


def reply(request: Request, body: dict, status: int = 200) -> JSONResponse:
    """A JSON response with the request_id and model_version every response carries."""
    body = {**body, "model_version": MODEL_VERSION, "request_id": request.state.request_id}
    return JSONResponse(body, status_code=status)


def error(request: Request, code: str, message: str) -> JSONResponse:
    status, retryable = ERRORS[code]
    return reply(request, {"error": code, "message": message, "retryable": retryable}, status)


def identify_one(services: Services, req: IdentifyRequest) -> dict:
    """1:N (§3.4) through svs.retrieve.identify: filter, pool count, two stages, decision.

    The decision is made on every ranked candidate; min_band only shortens the list the
    caller sees. searched tells "nobody matched" apart from "almost nothing was searched".
    """
    check_version(req.model_version)
    if req.filter is None:
        raise FilterRequired("identify needs a filter; an unfiltered search is refused")
    if req.min_band not in (*BAND_ORDER, NO_MATCH):
        raise ServiceError("E_SV_INVALID_REQUEST", f"min_band {req.min_band!r} is not a band")
    query = decode_embedding(req.embedding)
    check_speech(req.net_speech_seconds)
    name, warnings = resolve_condition(req.condition, req.net_speech_seconds, services.table)
    query_filter = IdentifyFilter(
        source=tuple(req.filter.source),
        last_seen_after=req.filter.last_seen_after,
        status=tuple(req.filter.status),
        min_evidence_seconds=req.filter.min_evidence_seconds,
    )
    found = identify(
        store_of(services),
        req.model_version,
        query,
        req.net_speech_seconds,
        name,
        services.cohorts[name],
        services.table,
        query_filter,
        req.top_k,
    )
    shown = (*BAND_ORDER, NO_MATCH)[: (*BAND_ORDER, NO_MATCH).index(req.min_band) + 1]
    return {
        "candidates": [asdict(c) for c in found.candidates if c.band in shown],
        "decision": found.decision,
        "decision_profile_id": found.decision_profile_id,
        "searched": {
            "candidate_pool": found.candidate_pool,
            "pool_tier": found.pool_tier,
            "recall_stage_k": RECALL_K,
            "rescored": found.rescored,
        },
        "condition": name,
        "latency_ms": {
            "count": found.count_ms,
            "recall": found.recall_ms,
            "exact": found.exact_ms,
        },
        "warnings": warnings,
    }


def audio_path(root: Path | None, uri: str) -> Path:
    """The file an audio_uri names. Paths that leave the audio root are refused."""
    if root is None:
        raise ServiceError("E_SV_INVALID_REQUEST", "this service has no audio root")
    path = (root / uri).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ServiceError("E_SV_INVALID_REQUEST", f"no audio file {uri!r} under the audio root")
    return path


def embed_item(embedder: Embedder, root: Path | None, item: EmbedItem) -> dict:
    """One speaker's segments of one recording -> embedding, net speech and quality (§3.2).

    Each segment is a turn. Turns with 3.0 s of speech or more get their own embedding,
    and their mean pairwise cosine is the turn consistency, as in svs.quality_cache.
    Under 3.0 s of pooled speech there is no embedding: the result says why.
    """
    try:
        audio, rate = load_audio(audio_path(root, item.audio_uri), item.channel)
        pooled = speech_from_samples(audio, rate, item.segments)
        turns = [speech_from_samples(audio, rate, [segment]) for segment in item.segments]
    except ValueError as exc:  # bad segments, channel or audio, as svs.extract words it
        raise ServiceError("E_SV_INVALID_REQUEST", f"item {item.id}: {exc}") from exc
    turn_embeddings = [
        embedder.embed(t.samples) for t in turns if t.net_speech_seconds >= MIN_SPEECH_SECONDS
    ]
    consistency = turn_consistency(turn_embeddings)
    # The selected segments at their own rate, cut as svs.extract cuts them: quality looks
    # at the audio as it arrived.
    selected = np.concatenate(
        [audio[math.ceil(a * rate / 1000) : math.ceil(b * rate / 1000)] for a, b in item.segments]
    )
    detail = quality_detail(selected, rate, pooled, consistency)
    score = quality(**detail)
    seconds = pooled.net_speech_seconds
    reason = unusable_reason(seconds, score, consistency)
    result = {
        "id": item.id,
        "purpose": item.purpose,
        "embedding": (
            encode_embedding(embedder.embed(pooled.samples))
            if seconds >= MIN_SPEECH_SECONDS
            else None
        ),
        "net_speech_seconds": seconds,
        "quality": score,
        "quality_detail": detail,
        "usable": reason is None,
    }
    if reason is not None:
        result["unusable_reason"] = reason
    return result


def embed(services: Services, req: EmbedRequest) -> dict:
    check_version(req.model_version)
    if services.embedder is None:
        raise RuntimeError("the service was started without the embedding model")
    return {"results": [embed_item(services.embedder, services.audio_root, i) for i in req.items]}


def occurrence_of(occ: OccurrenceIn) -> Occurrence:
    """A session is a recording: audio_id is the Occurrence's recording_id (§5.2)."""
    return Occurrence(
        occ.audio_id, decode_embedding(occ.embedding), occ.net_speech_seconds, occ.quality
    )


def promotion_rule(profile: Profile) -> dict:
    missing = profile.promotion_missing()
    return {
        "min_evidence_seconds": PROMOTE_MIN_SECONDS,
        "min_sessions": PROMOTE_MIN_SESSIONS,
        "min_mean_quality": PROMOTE_MIN_QUALITY,
        "min_cohesion": PROMOTE_MIN_COHESION,
        "met": not missing,
        "missing": missing,
    }


def summary(profile: Profile) -> dict:
    return {
        "profile_id": profile.profile_id,
        "status": profile.status,
        "evidence_seconds": profile.evidence_seconds,
        "sessions": profile.sessions,
        "promotion_rule": promotion_rule(profile),
    }


def save_profile(services: Services, profile: Profile, payload: dict) -> None:
    """Write the centroid and everything needed to update it exactly later (Appendix B).

    sum_vec and sum_w are stored, not only the centroid, so an update adds to the same
    sums. The occurrences are kept for sessions and cohesion. The cohort statistics are
    those of the clean cohort, cached here for every later verify and identify (§4.3).
    """
    (mu,), (sd,) = cohort_stats(profile.centroid[None, :], services.cohorts[CLEAN])
    payload = {
        **payload,
        "status": profile.status,
        "last_seen": datetime.now(UTC).isoformat(),
        "evidence_seconds": profile.evidence_seconds,
        "sessions": profile.sessions,
        "cohesion": profile.cohesion,
        "cohort_mu": float(mu),
        "cohort_sd": float(sd),
        "sum_vec": base64.b64encode(profile.sum_vec.astype("<f8").tobytes()).decode(),
        "sum_w": profile.sum_w,
    }
    store_of(services).upsert(MODEL_VERSION, profile.profile_id, profile.centroid, payload)


def load_profile(services: Services, profile_id: str) -> tuple[Profile, dict]:
    found = store_of(services).get(MODEL_VERSION, profile_id)
    if found is None:
        raise ServiceError("E_SV_PROFILE_NOT_FOUND", f"no profile {profile_id!r}")
    _, payload = found
    occurrences = [
        Occurrence(
            o["audio_id"], decode_embedding(o["embedding"]), o["net_speech_seconds"], o["quality"]
        )
        for o in payload["occurrences"]
    ]
    profile = Profile(
        profile_id,
        np.frombuffer(base64.b64decode(payload["sum_vec"]), "<f8").copy(),
        payload["sum_w"],
        payload["status"],
        occurrences,
    )
    return profile, payload


def enrol(services: Services, req: EnrolRequest) -> dict:
    """A new profile from one occurrence (§3.5); PROVISIONAL unless the rule already holds.

    Asking for CONFIRMED runs the evidence rule: one occurrence is one session, so the
    request is refused with E_SV_ENROL_RULE_UNMET and nothing is written.
    """
    check_version(req.model_version)
    if req.status not in (PROVISIONAL, CONFIRMED):
        raise ServiceError("E_SV_INVALID_REQUEST", f"status {req.status!r}")
    profile = Profile(f"spk_{uuid.uuid4().hex}")
    profile.accept(occurrence_of(req.occurrence))
    if req.status == CONFIRMED:
        profile.promote()
    now = datetime.now(UTC).isoformat()
    payload = {
        "source": [req.source],
        "first_seen": now,
        "attributes": req.attributes,
        "occurrences": [req.occurrence.model_dump()],
    }
    save_profile(services, profile, payload)
    return summary(profile)


def update(services: Services, profile_id: str, req: UpdateRequest) -> dict:
    """Add a matched occurrence to a profile (§3.6), only at STRONG, then try promotion.

    The same speaker of the same recording is refused a second time: its seconds would
    count twice towards the 120 s of the evidence rule.
    """
    check_version(req.model_version)
    profile, payload = load_profile(services, profile_id)
    key = (req.occurrence.audio_id, req.occurrence.speaker_ref)
    if key in {(o["audio_id"], o["speaker_ref"]) for o in payload["occurrences"]}:
        raise ServiceError(
            "E_SV_DUPLICATE_OCCURRENCE", f"{key[0]}:{key[1]} is already in {profile_id}"
        )
    profile.update(occurrence_of(req.occurrence), req.matched_band)
    promoted = profile.status == PROVISIONAL and not profile.promotion_missing()
    if promoted:
        profile.promote()
    sources = payload["source"]
    if req.source is not None and req.source not in sources:
        sources = [*sources, req.source]
    payload = {
        **payload,
        "source": sources,
        "occurrences": [*payload["occurrences"], req.occurrence.model_dump()],
    }
    save_profile(services, profile, payload)
    return {**summary(profile), "promoted": promoted, "min_update_band": MIN_UPDATE_BAND}


def idempotent(
    services: Services, request: Request, key: str | None, status: int, run
) -> JSONResponse:
    """Run a mutating call once per Idempotency-Key (§3.1).

    A retry with the same key and body gets the first response again, request_id
    included, and nothing is written twice. The same key with another body is refused.
    An error is not remembered: the caller can fix the request and retry with the key.
    """
    cache = services.idempotency
    with cache.lock:
        if key is None:
            return reply(request, run(), status)
        fingerprint = f"{request.method} {request.url.path} {request.state.body_hash}"
        if key in cache.done:
            seen, seen_status, seen_body = cache.done[key]
            if seen != fingerprint:
                raise ServiceError(
                    "E_SV_IDEMPOTENCY_KEY_REUSED", f"key {key!r} was used for another request"
                )
            return JSONResponse(seen_body, seen_status, headers={"Idempotent-Replayed": "true"})
        response = reply(request, run(), status)
        cache.done[key] = (fingerprint, status, json.loads(response.body))
        if len(cache.done) > cache.size:
            cache.done.popitem(last=False)
        return response


# The committed cohort embeddings (cohorts/cohorts.sha256), one file per condition.
COHORT_DIR = Path("cohorts")
# Seconds the service waits for Qdrant at start, e.g. while docker compose starts both.
STORE_WAIT = 60


def load_services() -> Services:
    """The model, the operating-point table, each condition's cohort and the profile store.

    CFG_SV_QDRANT_URL and CFG_SV_AUDIO_ROOT come from the environment. The profile
    collection of this model version is created if it does not exist (svs.store main).
    """
    table = read_table(OPTABLE)
    conditions = {condition for condition, _ in table}
    cohorts = {
        c: np.load(COHORT_DIR / f"{cohort_of(c)}.npz")["embeddings"] for c in sorted(conditions)
    }
    client = QdrantClient(url=os.environ.get("CFG_SV_QDRANT_URL", "http://localhost:6333"))
    store = ProfileStore(client)
    for attempt in range(STORE_WAIT):
        try:
            exists = client.collection_exists(store.collection)
            break
        except ResponseHandlingException:
            if attempt == STORE_WAIT - 1:
                raise
            time.sleep(1)
    if not exists:
        store.create()
    root = Path(os.environ.get("CFG_SV_AUDIO_ROOT", "data"))
    return Services(store, table, cohorts, embedder=Embedder(), audio_root=root)


def service() -> FastAPI:
    """The app uvicorn runs: uvicorn svs.api:service --factory."""
    return create_app(load_services())


def create_app(services: Services) -> FastAPI:
    app = FastAPI(title="Speaker verification service", version=MODEL_VERSION)
    app.state.svs = services

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = f"req_{uuid.uuid4().hex}"
        # The idempotency fingerprint: the raw body, before any parsing (Starlette caches it).
        request.state.body_hash = hashlib.sha256(await request.body()).hexdigest()
        response = await call_next(request)
        # A header too, so a response this file did not build (an unknown path) has one.
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    async def coded(request: Request, exc: Exception) -> JSONResponse:
        return error(request, exc.code, str(exc))

    for coded_exception in CODED:
        app.add_exception_handler(coded_exception, coded)

    @app.exception_handler(ResponseHandlingException)
    async def store_down(request: Request, exc: ResponseHandlingException) -> JSONResponse:
        return error(request, "E_SV_STORE_UNAVAILABLE", f"vector store unreachable: {exc}")

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return error(request, "E_SV_INVALID_REQUEST", problems)

    @app.get("/v1/healthz")
    def healthz(request: Request) -> JSONResponse:
        return reply(request, {"status": "ok"})

    @app.post("/v1/embed")
    def embed_endpoint(request: Request, body: EmbedRequest) -> JSONResponse:
        return reply(request, embed(services, body))

    @app.post("/v1/identify")
    def identify_endpoint(request: Request, body: IdentifyRequest) -> JSONResponse:
        return reply(request, identify_one(services, body))

    @app.post("/v1/profiles")
    def enrol_endpoint(
        request: Request,
        body: EnrolRequest,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        return idempotent(services, request, idempotency_key, 201, lambda: enrol(services, body))

    @app.patch("/v1/profiles/{profile_id}")
    def update_endpoint(
        request: Request,
        profile_id: str,
        body: UpdateRequest,
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        return idempotent(
            services, request, idempotency_key, 200, lambda: update(services, profile_id, body)
        )

    @app.post("/v1/verify")
    def verify_endpoint(request: Request, body: VerifyRequest) -> JSONResponse:
        return reply(request, verify(services, body))

    return app
