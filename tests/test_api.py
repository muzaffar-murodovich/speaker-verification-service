"""The HTTP service: request_id and model_version on every response, §3.8 error codes."""

import pytest
from fastapi.testclient import TestClient

from svs.api import CODED, ERRORS, Services, create_app
from svs.embed import MODEL_VERSION
from svs.quality import unusable_reason


@pytest.fixture
def api():
    return TestClient(create_app(Services()))


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
