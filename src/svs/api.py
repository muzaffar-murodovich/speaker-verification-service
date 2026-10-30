"""The HTTP service (handbook D15): the five endpoints of SVS guide §3.2-3.6.

Every response, errors included, carries request_id and model_version (§3.1). An error is
one of the E_SV_* codes of §3.8: the code is what a caller acts on, the message is for a
person. The scoring logic lives in the other svs modules; this file only turns JSON into
their arguments, and their results and exceptions back into JSON.
"""

import uuid
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from svs.embed import MODEL_VERSION
from svs.profile import EnrolRuleUnmet, UnusableOccurrence, UpdateBandTooLow
from svs.retrieve import FilterRequired
from svs.store import ModelVersionMismatch

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


@dataclass
class Services:
    """What the endpoints work with; tests build it with fakes."""


def reply(request: Request, body: dict, status: int = 200) -> JSONResponse:
    """A JSON response with the request_id and model_version every response carries."""
    body = {**body, "model_version": MODEL_VERSION, "request_id": request.state.request_id}
    return JSONResponse(body, status_code=status)


def error(request: Request, code: str, message: str) -> JSONResponse:
    status, retryable = ERRORS[code]
    return reply(request, {"error": code, "message": message, "retryable": retryable}, status)


def create_app(services: Services) -> FastAPI:
    app = FastAPI(title="Speaker verification service", version=MODEL_VERSION)
    app.state.svs = services

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = f"req_{uuid.uuid4().hex}"
        response = await call_next(request)
        # A header too, so a response this file did not build (an unknown path) has one.
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    async def coded(request: Request, exc: Exception) -> JSONResponse:
        return error(request, exc.code, str(exc))

    for coded_exception in CODED:
        app.add_exception_handler(coded_exception, coded)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return error(request, "E_SV_INVALID_REQUEST", problems)

    @app.get("/v1/healthz")
    def healthz(request: Request) -> JSONResponse:
        return reply(request, {"status": "ok"})

    return app
