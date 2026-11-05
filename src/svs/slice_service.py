"""The internal slice through the running service, end to end (D16, handbook M4).

For each person the listener confirmed (listen-answers.tsv), the first occurrence is
embedded with POST /v1/embed and enrolled with POST /v1/profiles. Every other confirmed
occurrence is a known query, and every usable occurrence of a recording with no confirmed
person is an unknown query (nobody enrolled is in that recording). Each query is embedded,
then searched with POST /v1/identify in this run's source only, and the known ones are
also checked with POST /v1/verify against their own profile.

A known query is right when identify returns MATCH on its own profile; an unknown query
should end in NO_MATCH. No condition block is sent: the slice has none, so the service
uses the most conservative condition (D15). The pool is the 9 enrolled profiles, so the
pool tier is 1,000. Usage: make service, then make slice-service.
"""

import argparse
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from svs.slice_link import recording_of
from svs.slice_run import OUT, parse_segments, read_tsv, write_tsv

FIELDS = [
    "item_id",
    "kind",
    "person",
    "decision",
    "top_person",
    "top_band",
    "top_score",
    "verify_band",
    "verify_score",
]


def queries(answers: list[dict], usable: list[str]) -> tuple[dict, list, list]:
    """Enrolment item per person, known queries (item, person), unknown query items."""
    person = {r["item_id"]: int(r["group"]) for r in answers if r["same_person"] == "yes"}
    enrol: dict[int, str] = {}
    known = []
    for item, p in person.items():
        if p not in enrol:
            enrol[p] = item
        else:
            known.append((item, p))
    taken = {recording_of(i) for i in person}
    unknown = [i for i in usable if recording_of(i) not in taken]
    return enrol, known, unknown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--uri-prefix",
        default="audio/Validation_Dataset_audio",
        help="the slice folder relative to the service's audio root (data/)",
    )
    parser.add_argument("--run", type=Path, default=OUT / "reference")
    parser.add_argument("--manifest", type=Path, default=OUT / "manifest-reference.tsv")
    parser.add_argument("--answers", type=Path, default=OUT / "listen-answers.tsv")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    api = httpx.Client(base_url=args.url, timeout=120)
    version = api.get("/v1/healthz").json()["model_version"]
    source = f"slice_{uuid.uuid4().hex[:8]}"
    manifest = {r["item_id"]: r for r in read_tsv(args.manifest)}
    usable = [r["item_id"] for r in read_tsv(args.run / "results.tsv") if r["usable"] == "True"]
    enrol, known, unknown = queries(read_tsv(args.answers), usable)

    def embed(item_id: str) -> dict:
        row = manifest[item_id]
        body = {
            "model_version": version,
            "items": [
                {
                    "id": item_id,
                    "audio_uri": f"{args.uri_prefix}/{row['audio']}",
                    "segments": parse_segments(row["segments"]),
                    "channel": int(row["channel"]) if str(row["channel"]) else None,
                    "purpose": "OCCURRENCE",
                }
            ],
        }
        response = api.post("/v1/embed", json=body)
        response.raise_for_status()
        return response.json()["results"][0]

    profile_of, person_of = {}, {}
    for p, item in enrol.items():
        e = embed(item)
        body = {
            "model_version": version,
            "source": source,
            "occurrence": {
                "audio_id": recording_of(item),
                "speaker_ref": item.split("#")[1],
                "embedding": e["embedding"],
                "net_speech_seconds": e["net_speech_seconds"],
                "quality": e["quality"],
            },
        }
        created = api.post("/v1/profiles", json=body).json()
        profile_of[p] = created["profile_id"]
        person_of[created["profile_id"]] = p
    print(f"{len(profile_of)} profiles enrolled in source {source}")

    since = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    rows = []
    for item, p, kind in [(i, p, "known") for i, p in known] + [
        (i, None, "unknown") for i in unknown
    ]:
        e = embed(item)
        body = {
            "model_version": version,
            "embedding": e["embedding"],
            "net_speech_seconds": e["net_speech_seconds"],
            "filter": {"source": [source], "last_seen_after": since},
            "min_band": "NO_MATCH",
        }
        found = api.post("/v1/identify", json=body).json()
        top = found["candidates"][0] if found["candidates"] else {}
        row = {
            "item_id": item,
            "kind": kind,
            "person": "" if p is None else p,
            "decision": found["decision"],
            "top_person": person_of.get(top.get("profile_id"), ""),
            "top_band": top.get("band", ""),
            "top_score": f"{top['normalised_score']:.3f}" if top else "",
            "verify_band": "",
            "verify_score": "",
        }
        if p is not None:
            body = {
                "model_version": version,
                "left": {
                    "embedding": e["embedding"],
                    "net_speech_seconds": e["net_speech_seconds"],
                },
                "right": {"profile_id": profile_of[p]},
            }
            v = api.post("/v1/verify", json=body).json()
            row["verify_band"], row["verify_score"] = v["band"], f"{v['normalised_score']:.3f}"
        rows.append(row)
    write_tsv(args.out / "service.tsv", FIELDS, rows)
    for kind in ("known", "unknown"):
        part = [r for r in rows if r["kind"] == kind]
        decisions = Counter(r["decision"] for r in part)
        line = f"{kind}: {len(part)} queries, decisions {dict(decisions)}"
        if kind == "known":
            right = sum(r["decision"] == "MATCH" and r["top_person"] == r["person"] for r in part)
            top1 = sum(r["top_person"] == r["person"] for r in part)
            verify = Counter(r["verify_band"] for r in part)
            line += f"; right MATCH {right}, right top-1 {top1}; verify bands {dict(verify)}"
        else:
            line += f"; top band {dict(Counter(r['top_band'] for r in part))}"
        print(line)
    print(f"-> {args.out}/service.tsv")


if __name__ == "__main__":
    main()
