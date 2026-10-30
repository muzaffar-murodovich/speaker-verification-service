"""The HTTP service: request_id and model_version on every response, §3.8 error codes."""

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from svs.api import (
    CODED,
    CONDITIONS,
    ERRORS,
    Services,
    create_app,
    decode_embedding,
    encode_embedding,
)
from svs.asnorm import as_norm, cohort_stats
from svs.bands import read_table
from svs.embed import EMBEDDING_DIM, MODEL_PATH, MODEL_VERSION, Embedder
from svs.profile import Occurrence, Profile
from svs.quality import quality, unusable_reason
from svs.store import ProfileStore

RNG = np.random.default_rng(0)
needs_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `make model` first")


def unit(n: int) -> np.ndarray:
    x = RNG.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


COHORT = unit(400)
# Thresholds on the AS-Norm scale. pcm8k is the stricter condition.
THRESHOLDS = {"pcm16k": (5.0, 3.0, 1.0), "pcm8k": (8.0, 6.0, 4.0)}


def table() -> dict:
    out = {}
    for condition, thresholds in THRESHOLDS.items():
        for bucket in ("3-10s", "10-30s", "30-120s", ">120s"):
            rows = [
                {
                    "condition": condition,
                    "bucket": bucket,
                    "pool": "1000",
                    "band": name,
                    "threshold": str(t),
                    "far": "1e-4",
                    "recall": "0.9",
                    "precision": "0.99",
                    "eer_%": "2.5",
                    "n_target": "100",
                    "n_nontarget": "1000",
                    "cohort": f"cohort-{condition}",
                }
                for name, t in zip(("STRONG", "PROBABLE", "WEAK"), thresholds, strict=True)
            ]
            out[(condition, bucket)] = {1000: rows}
    return out


@pytest.fixture
def services():
    store = ProfileStore(QdrantClient(":memory:"))
    store.create()
    return Services(store, table(), {"pcm16k": COHORT, "pcm8k": COHORT[::-1].copy()})


@pytest.fixture
def api(services):
    return TestClient(create_app(services))


def side(vector: np.ndarray, seconds: float) -> dict:
    return {"embedding": encode_embedding(vector), "net_speech_seconds": seconds}


def add_profile(services: Services, profile_id: str, vector: np.ndarray, seconds: float):
    mu, sd = cohort_stats(vector[None, :], COHORT)
    payload = {
        "status": "PROVISIONAL",
        "source": "cc_north",
        "last_seen": "2026-10-30T00:00:00Z",
        "evidence_seconds": seconds,
        "cohort_mu": float(mu[0]),
        "cohort_sd": float(sd[0]),
    }
    services.store.upsert(MODEL_VERSION, profile_id, vector, payload)


PCM16K = {"codec": "pcm", "sample_rate": 16000}


def test_every_response_carries_request_id_and_model_version(api):
    first = api.get("/v1/healthz").json()
    second = api.get("/v1/healthz").json()
    assert first["model_version"] == MODEL_VERSION
    assert first["request_id"].startswith("req_")
    assert first["request_id"] != second["request_id"]


def test_a_response_the_service_did_not_build_still_has_a_request_id(api):
    response = api.get("/no/such/path")
    assert response.status_code == 404
    assert response.headers["X-Request-ID"].startswith("req_")


def test_every_code_the_modules_raise_has_a_status():
    codes = {cls.code for cls in CODED if hasattr(cls, "code")}
    codes |= {unusable_reason(1.0, 1.0, 1.0), unusable_reason(9.0, 0.0, 1.0)}
    codes |= {unusable_reason(9.0, 1.0, 0.0)}
    assert codes <= ERRORS.keys()


def verify(api, left, right, condition=PCM16K, version=MODEL_VERSION):
    body = {"model_version": version, "left": left, "right": right}
    if condition is not None:
        body["condition"] = condition
    return api.post("/v1/verify", json=body)


def test_verify_refuses_under_3_seconds_of_speech(api):
    """The M2 refusal, as a service test: 2.9 s on one side is refused, 3.0 s is scored."""
    a, b = unit(2)
    refused = verify(api, side(a, 2.9), side(b, 60.0))
    assert refused.status_code == 422
    assert refused.json()["error"] == "E_SV_INSUFFICIENT_SPEECH"
    assert refused.json()["request_id"].startswith("req_")
    assert verify(api, side(a, 3.0), side(b, 60.0)).status_code == 200


def test_verify_refuses_short_audio_against_a_long_profile(api, services):
    a, b = unit(2)
    add_profile(services, "spk_long", b, 1800.0)
    response = verify(api, side(a, 2.0), {"profile_id": "spk_long"})
    assert response.json()["error"] == "E_SV_INSUFFICIENT_SPEECH"


def test_verify_score_is_as_norm_of_the_cosine(api):
    a, b = unit(2)
    out = verify(api, side(a, 12.0), side(b, 40.0)).json()
    mu, sd = cohort_stats(np.stack([a, b]), COHORT)
    raw = float(a.astype(np.float64) @ b)
    assert out["raw_score"] == pytest.approx(raw)
    assert out["normalised_score"] == pytest.approx(as_norm(raw, mu[0], sd[0], mu[1], sd[1]))
    assert out["model_version"] == MODEL_VERSION


def test_verify_band_comes_from_the_shorter_sides_bucket_row(api):
    a = unit(1)[0]
    out = verify(api, side(a, 12.0), side(a, 40.0)).json()
    assert out["band"] == "STRONG"
    assert out["threshold_used"] == 5.0
    assert out["operating_point"]["name"] == "pcm16k_10-30s_pool1000_STRONG"


def test_verify_against_a_profile_uses_its_cached_statistics(api, services):
    a, b = unit(2)
    add_profile(services, "spk_a", b, 200.0)
    by_profile = verify(api, side(a, 12.0), {"profile_id": "spk_a"}).json()
    by_vector = verify(api, side(a, 12.0), side(b, 200.0)).json()
    assert by_profile["normalised_score"] == pytest.approx(by_vector["normalised_score"])


def test_verify_unknown_profile_is_404(api):
    response = verify(api, side(unit(1)[0], 12.0), {"profile_id": "spk_none"})
    assert response.status_code == 404
    assert response.json()["error"] == "E_SV_PROFILE_NOT_FOUND"


def test_verify_refuses_another_model_version(api):
    a, b = unit(2)
    response = verify(api, side(a, 12.0), side(b, 12.0), version="ecapa-v3")
    assert response.status_code == 409
    assert response.json()["error"] == "E_SV_MODEL_VERSION_MISMATCH"


def test_verify_refuses_an_uncalibrated_condition(api):
    a, b = unit(2)
    for condition in (
        {"codec": "amr", "sample_rate": 8000},
        {"codec": "pcm", "sample_rate": 44100},
    ):
        response = verify(api, side(a, 12.0), side(b, 12.0), condition)
        assert response.json()["error"] == "E_SV_UNKNOWN_CONDITION"


def test_verify_without_condition_gets_the_strictest_thresholds_and_a_warning(api):
    a = unit(1)[0]
    out = verify(api, side(a, 12.0), side(a, 12.0), condition=None).json()
    assert out["operating_point"]["condition"] == "pcm8k"
    assert out["threshold_used"] in THRESHOLDS["pcm8k"]
    assert out["warnings"]


@pytest.mark.parametrize(
    "embedding",
    [
        "not base64!",
        encode_embedding(np.ones(10, np.float32)),
        encode_embedding(2 * unit(1)[0]),
        encode_embedding(np.full(EMBEDDING_DIM, np.nan, np.float32)),
    ],
)
def test_verify_refuses_a_malformed_embedding(api, embedding):
    response = verify(
        api, {"embedding": embedding, "net_speech_seconds": 12.0}, side(unit(1)[0], 12.0)
    )
    assert response.status_code == 422
    assert response.json()["error"] == "E_SV_INVALID_REQUEST"


def test_a_body_with_a_missing_field_is_an_invalid_request(api):
    response = api.post("/v1/verify", json={"model_version": MODEL_VERSION})
    assert response.status_code == 422
    assert response.json()["error"] == "E_SV_INVALID_REQUEST"
    assert "left" in response.json()["message"]


def test_every_request_condition_is_in_the_operating_point_table():
    calibrated = {condition for condition, _ in read_table()}
    assert set(CONDITIONS.values()) <= calibrated


NORTH = {"source": ["cc_north"], "last_seen_after": "2026-01-01T00:00:00Z"}


def identify(api, vector, seconds=12.0, filter=NORTH, **extra):
    body = {
        "model_version": MODEL_VERSION,
        "embedding": encode_embedding(vector),
        "net_speech_seconds": seconds,
        "condition": PCM16K,
        **extra,
    }
    if filter is not None:
        body["filter"] = filter
    return api.post("/v1/identify", json=body)


@pytest.fixture
def profiles(services):
    vectors = unit(5)
    for i, vector in enumerate(vectors):
        add_profile(services, f"spk_{i}", vector, 200.0)
    return vectors


def test_identify_finds_the_profile_and_gives_a_margin(api, profiles):
    out = identify(api, profiles[2], min_band="NO_MATCH").json()
    assert out["decision"] == "MATCH"
    assert out["decision_profile_id"] == "spk_2"
    first, second = out["candidates"][:2]
    assert first["profile_id"] == "spk_2"
    assert first["band"] == "STRONG"
    assert first["margin_to_next"] == pytest.approx(
        first["normalised_score"] - second["normalised_score"]
    )
    assert out["searched"]["candidate_pool"] == 5
    assert out["request_id"].startswith("req_")


def test_identify_without_a_filter_is_refused(api, profiles):
    for missing in (None, {**NORTH, "source": []}):
        response = identify(api, profiles[0], filter=missing)
        assert response.status_code == 422
        assert response.json()["error"] == "E_SV_FILTER_REQUIRED"


def test_identify_refuses_a_date_without_a_time_zone(api, profiles):
    response = identify(api, profiles[0], filter={**NORTH, "last_seen_after": "2026-01-01"})
    assert response.json()["error"] == "E_SV_FILTER_REQUIRED"


def test_identify_on_an_empty_pool_says_nothing_was_searched(api, profiles):
    out = identify(api, profiles[0], filter={**NORTH, "source": ["cc_south"]}).json()
    assert out["candidates"] == []
    assert out["decision"] == "NO_MATCH"
    assert out["searched"]["candidate_pool"] == 0


def test_identify_refuses_under_3_seconds_of_speech(api, profiles):
    response = identify(api, profiles[0], seconds=2.5)
    assert response.json()["error"] == "E_SV_INSUFFICIENT_SPEECH"


def test_min_band_shortens_the_list_but_not_the_decision(api, profiles):
    everything = identify(api, profiles[1], min_band="NO_MATCH").json()
    strong_only = identify(api, profiles[1], min_band="STRONG").json()
    assert len(everything["candidates"]) == 5
    assert [c["band"] for c in strong_only["candidates"]] == ["STRONG"]
    assert strong_only["decision"] == everything["decision"] == "MATCH"


def occurrence(vector, audio_id, seconds=61.9, quality=0.78, speaker_ref="S1"):
    return {
        "audio_id": audio_id,
        "speaker_ref": speaker_ref,
        "embedding": encode_embedding(vector),
        "net_speech_seconds": seconds,
        "quality": quality,
    }


def enrol(api, occ, key=None, **extra):
    body = {"model_version": MODEL_VERSION, "occurrence": occ, "source": "cc_north", **extra}
    headers = {"Idempotency-Key": key} if key else {}
    return api.post("/v1/profiles", json=body, headers=headers)


def patch(api, profile_id, occ, band="STRONG", key=None):
    body = {"model_version": MODEL_VERSION, "occurrence": occ, "matched_band": band}
    headers = {"Idempotency-Key": key} if key else {}
    return api.patch(f"/v1/profiles/{profile_id}", json=body, headers=headers)


def stored(services) -> int:
    return services.store.client.count(services.store.collection).count


def near(vector, scale=0.02):
    """Another recording of the same voice: the vector plus a little noise."""
    x = vector + scale * RNG.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return x / np.linalg.norm(x)


def test_enrol_creates_a_provisional_profile_and_names_what_is_missing(api, services):
    response = enrol(api, occurrence(unit(1)[0], "aud_1"))
    assert response.status_code == 201
    out = response.json()
    assert out["status"] == "PROVISIONAL"
    assert out["sessions"] == 1
    assert out["promotion_rule"]["met"] is False
    assert set(out["promotion_rule"]["missing"]) == {"evidence_seconds", "sessions"}
    assert out["request_id"].startswith("req_")
    assert stored(services) == 1


def test_enrol_as_confirmed_is_refused_and_writes_nothing(api, services):
    response = enrol(api, occurrence(unit(1)[0], "aud_1", seconds=300.0), status="CONFIRMED")
    assert response.status_code == 422
    assert response.json()["error"] == "E_SV_ENROL_RULE_UNMET"
    assert stored(services) == 0


@pytest.mark.parametrize(
    ("seconds", "quality", "code"),
    [(2.0, 0.9, "E_SV_INSUFFICIENT_SPEECH"), (60.0, 0.3, "E_SV_LOW_QUALITY")],
)
def test_enrol_refuses_unusable_speech(api, services, seconds, quality, code):
    response = enrol(api, occurrence(unit(1)[0], "aud_1", seconds, quality))
    assert response.json()["error"] == code
    assert stored(services) == 0


def test_update_below_strong_is_refused_and_changes_nothing(api, services):
    voice = unit(1)[0]
    profile_id = enrol(api, occurrence(voice, "aud_1")).json()["profile_id"]
    before = services.store.get(MODEL_VERSION, profile_id)
    response = patch(api, profile_id, occurrence(near(voice), "aud_2"), band="PROBABLE")
    assert response.status_code == 422
    assert response.json()["error"] == "E_SV_UPDATE_BAND_TOO_LOW"
    after = services.store.get(MODEL_VERSION, profile_id)
    assert np.array_equal(before[0], after[0])
    assert before[1]["sum_w"] == after[1]["sum_w"]


def test_a_strong_update_from_a_second_recording_promotes(api, services):
    voice = unit(1)[0]
    first, second = near(voice), near(voice)
    profile_id = enrol(api, occurrence(first, "aud_1")).json()["profile_id"]
    out = patch(api, profile_id, occurrence(second, "aud_2")).json()
    assert out["promoted"] is True
    assert out["status"] == "CONFIRMED"
    assert out["sessions"] == 2
    # The stored centroid is the one svs.profile computes from the same two occurrences.
    local = Profile(profile_id)
    local.accept(Occurrence("aud_1", first, 61.9, 0.78))
    local.accept(Occurrence("aud_2", second, 61.9, 0.78))
    vector, payload = services.store.get(MODEL_VERSION, profile_id)
    assert np.allclose(vector, local.centroid, atol=1e-6)
    assert payload["sum_w"] == pytest.approx(local.sum_w)


def test_the_same_occurrence_twice_is_refused(api):
    voice = unit(1)[0]
    profile_id = enrol(api, occurrence(voice, "aud_1")).json()["profile_id"]
    response = patch(api, profile_id, occurrence(near(voice), "aud_1"))
    assert response.status_code == 409
    assert response.json()["error"] == "E_SV_DUPLICATE_OCCURRENCE"


def test_update_of_an_unknown_profile_is_404(api):
    response = patch(api, "spk_none", occurrence(unit(1)[0], "aud_1"))
    assert response.json()["error"] == "E_SV_PROFILE_NOT_FOUND"


def test_an_enrol_retry_with_the_same_key_creates_one_profile(api, services):
    occ = occurrence(unit(1)[0], "aud_1")
    first = enrol(api, occ, key="idem_1")
    retry = enrol(api, occ, key="idem_1")
    assert retry.status_code == 201
    assert retry.json() == first.json()
    assert retry.headers["Idempotent-Replayed"] == "true"
    assert stored(services) == 1
    # Without a key, the same body is a second enrolment.
    enrol(api, occ)
    assert stored(services) == 2


def test_an_update_retry_with_the_same_key_counts_the_speech_once(api, services):
    voice = unit(1)[0]
    profile_id = enrol(api, occurrence(voice, "aud_1")).json()["profile_id"]
    occ = occurrence(near(voice), "aud_2")
    first = patch(api, profile_id, occ, key="idem_2")
    retry = patch(api, profile_id, occ, key="idem_2")
    assert retry.status_code == 200
    assert retry.json() == first.json()
    assert services.store.get(MODEL_VERSION, profile_id)[1]["evidence_seconds"] == 2 * 61.9


def test_a_key_reused_for_another_body_is_refused(api, services):
    enrol(api, occurrence(unit(1)[0], "aud_1"), key="idem_3")
    response = enrol(api, occurrence(unit(1)[0], "aud_9"), key="idem_3")
    assert response.status_code == 409
    assert response.json()["error"] == "E_SV_IDEMPOTENCY_KEY_REUSED"
    assert stored(services) == 1


def test_a_failed_call_does_not_use_up_its_key(api, services):
    occ = occurrence(unit(1)[0], "aud_1")
    assert enrol(api, occ, key="idem_4", status="CONFIRMED").status_code == 422
    assert enrol(api, occ, key="idem_4", status="CONFIRMED").status_code == 422
    assert stored(services) == 0


def test_an_enrolled_profile_is_found_by_identify_and_verify(api):
    voice = unit(1)[0]
    profile_id = enrol(api, occurrence(voice, "aud_1")).json()["profile_id"]
    found = identify(api, near(voice)).json()
    assert found["candidates"][0]["profile_id"] == profile_id
    checked = verify(api, side(near(voice), 12.0), {"profile_id": profile_id}).json()
    assert checked["band"] == "STRONG"


class SignEmbedder:
    """A stand-in model: one voice for audio with a positive mean, another for negative."""

    def embed(self, samples):
        return np.eye(EMBEDDING_DIM, dtype=np.float32)[0 if samples.mean() > 0 else 1]


@pytest.fixture
def audio_api(services, tmp_path, monkeypatch):
    class KeepAll:  # every frame is speech, as in tests/test_extract.py
        def __init__(self, mode):
            pass

        def is_speech(self, pcm, rate):
            return True

    monkeypatch.setattr("svs.extract.webrtcvad.Vad", KeepAll)
    rng = np.random.default_rng(1)
    noise = 0.05 * rng.standard_normal(16000 * 20)
    # 0-10 s one voice (mean +0.2), 10-20 s another (mean -0.2).
    audio = np.concatenate([noise[:160000] + 0.2, noise[160000:] - 0.2]).astype(np.float32)
    sf.write(tmp_path / "call.wav", audio, 16000, subtype="FLOAT")
    services.embedder = SignEmbedder()
    services.audio_root = tmp_path
    return TestClient(create_app(services))


def embed(api, segments, uri="call.wav"):
    item = {"id": "aud_1:S1", "audio_uri": uri, "segments": segments, "purpose": "OCCURRENCE"}
    return api.post("/v1/embed", json={"model_version": MODEL_VERSION, "items": [item]})


def test_embed_returns_a_unit_embedding_with_speech_and_quality(audio_api):
    out = embed(audio_api, [[0, 4000], [5000, 9000]]).json()
    (result,) = out["results"]
    assert result["usable"] is True
    assert result["net_speech_seconds"] == pytest.approx(8.0)
    assert set(result["quality_detail"]) == {
        "snr_db",
        "speech_ratio",
        "clipping_ratio",
        "bandwidth_hz",
        "turn_consistency",
    }
    assert result["quality"] == pytest.approx(quality(**result["quality_detail"]))
    assert np.linalg.norm(decode_embedding(result["embedding"])) == pytest.approx(1.0)
    assert out["request_id"].startswith("req_")


def test_embed_marks_two_voices_in_one_speaker_unusable(audio_api):
    (result,) = embed(audio_api, [[0, 5000], [12000, 17000]]).json()["results"]
    assert result["quality_detail"]["turn_consistency"] == pytest.approx(0.0)
    assert result["usable"] is False
    assert result["unusable_reason"] == "E_SV_TURN_INCONSISTENT"


def test_embed_under_3_seconds_has_no_embedding(audio_api):
    (result,) = embed(audio_api, [[0, 2500]]).json()["results"]
    assert result["embedding"] is None
    assert result["unusable_reason"] == "E_SV_INSUFFICIENT_SPEECH"


@pytest.mark.parametrize("uri", ["missing.wav", "../call.wav", "/etc/passwd"])
def test_embed_reads_only_files_under_the_audio_root(audio_api, uri):
    response = embed(audio_api, [[0, 4000]], uri=uri)
    assert response.status_code == 422
    assert response.json()["error"] == "E_SV_INVALID_REQUEST"


def test_embed_refuses_segments_outside_the_audio(audio_api):
    response = embed(audio_api, [[0, 25000]])
    assert response.json()["error"] == "E_SV_INVALID_REQUEST"
    assert "aud_1:S1" in response.json()["message"]


@needs_model
def test_embed_with_the_real_model_matches_svs_embed(audio_api, services):
    services.embedder = Embedder()
    (result,) = embed(audio_api, [[0, 4000]]).json()["results"]
    audio, rate = sf.read(services.audio_root / "call.wav", dtype="float32")
    expected = services.embedder.embed(audio[:64000])
    assert np.allclose(decode_embedding(result["embedding"]), expected, atol=1e-5)
