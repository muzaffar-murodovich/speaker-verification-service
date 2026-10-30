"""Call all five endpoints of a running service once, on VoxCeleb1 audio (handbook M3).

Speaker id10001 is enrolled from one video and tested with another; id10002 is the
impostor. Each run uses its own source name, so earlier runs' profiles stay out of
the identify pool. Usage: make smoke (after docker compose up).
"""

import argparse
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import soundfile as sf

ENROL = "voxceleb1/wav/id10001/7w0IBEWc9Qw/00001.wav"
SAME = "voxceleb1/wav/id10001/1zcIwhmdeo4/00002.wav"
OTHER = "voxceleb1/wav/id10002/0_laIeN-Q44/00001.wav"
PCM16K = {"codec": "pcm", "sample_rate": 16000}


def show(name: str, response: httpx.Response, *fields: str) -> dict:
    body = response.json()
    picked = {f: body.get(f) for f in fields}
    print(f"{name}: HTTP {response.status_code} {picked} request_id={body['request_id']}")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--data", default="data", help="the host folder mounted as audio root")
    args = parser.parse_args()
    api = httpx.Client(base_url=args.url, timeout=60)
    version = show("healthz", api.get("/v1/healthz"), "status")["model_version"]
    source = f"smoke_{uuid.uuid4().hex[:8]}"

    items = []
    for uri, segment_ms in ((ENROL, None), (SAME, None), (OTHER, None), (SAME, 2000)):
        ms = segment_ms or int(sf.info(f"{args.data}/{uri}").duration * 1000)
        items.append(
            {"id": f"{uri}:{ms}", "audio_uri": uri, "segments": [[0, ms]], "purpose": "OCCURRENCE"}
        )
    embedded = api.post("/v1/embed", json={"model_version": version, "items": items}).json()
    enrol, same, other, short = embedded["results"]
    for r in embedded["results"]:
        print(
            f"embed {r['id']}: {r['net_speech_seconds']:.2f} s, quality {r['quality']:.3f}, "
            f"usable {r['usable']} {r.get('unusable_reason', '')}"
        )

    def occurrence(result: dict, audio_id: str) -> dict:
        return {
            "audio_id": audio_id,
            "speaker_ref": "S1",
            "embedding": result["embedding"],
            "net_speech_seconds": result["net_speech_seconds"],
            "quality": result["quality"],
        }

    body = {
        "model_version": version,
        "occurrence": occurrence(enrol, "7w0IBEWc9Qw"),
        "source": source,
    }
    key = {"Idempotency-Key": f"idem_{source}"}
    created = show(
        "enrol", api.post("/v1/profiles", json=body, headers=key), "profile_id", "status"
    )
    retry = api.post("/v1/profiles", json=body, headers=key)
    show("enrol retry, same key", retry, "profile_id")
    print(f"  replayed: {retry.headers.get('Idempotent-Replayed')}")
    profile = {"profile_id": created["profile_id"]}

    def side(result: dict) -> dict:
        return {
            "embedding": result["embedding"],
            "net_speech_seconds": result["net_speech_seconds"],
        }

    for name, result in (("same speaker", same), ("other speaker", other)):
        body = {
            "model_version": version,
            "left": side(result),
            "right": profile,
            "condition": PCM16K,
        }
        show(f"verify {name}", api.post("/v1/verify", json=body), "normalised_score", "band")
    body = {
        "model_version": version,
        "left": {"embedding": same["embedding"], "net_speech_seconds": 2.0},
        "right": profile,
        "condition": PCM16K,
    }
    show("verify 2.0 s", api.post("/v1/verify", json=body), "error")

    since = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    body = {
        "model_version": version,
        **side(same),
        "filter": {"source": [source], "last_seen_after": since},
        "condition": PCM16K,
        "min_band": "NO_MATCH",
    }
    found = show("identify", api.post("/v1/identify", json=body), "decision", "searched")
    for c in found["candidates"]:
        print(
            f"  {c['profile_id']} {c['band']} {c['normalised_score']:.2f}",
            f"margin_to_next {c['margin_to_next']}",
        )

    for band in ("PROBABLE", "STRONG"):
        body = {
            "model_version": version,
            "occurrence": occurrence(same, "1zcIwhmdeo4"),
            "matched_band": band,
        }
        show(
            f"update at {band}",
            api.patch(f"/v1/profiles/{profile['profile_id']}", json=body),
            "error",
            "sessions",
            "promotion_rule",
        )


if __name__ == "__main__":
    main()
