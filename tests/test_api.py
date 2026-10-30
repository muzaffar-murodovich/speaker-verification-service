"""The HTTP service: request_id and model_version on every response, §3.8 error codes."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from svs.api import CODED, CONDITIONS, ERRORS, Services, create_app, encode_embedding
from svs.asnorm import as_norm, cohort_stats
from svs.bands import read_table
from svs.embed import EMBEDDING_DIM, MODEL_VERSION
from svs.quality import unusable_reason
from svs.store import ProfileStore

RNG = np.random.default_rng(0)


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
