"""Run the pipeline over a folder of call recordings (handbook D16).

inspect reads only the file headers: format, sample rate, channels, duration. It costs
seconds, and it says what the audio is before anything is embedded. A file libsndfile
cannot open is listed with its error, not skipped: D16 is about writing down
everything that breaks.

run embeds each item of a manifest as /v1/embed does (svs.api.embed_item) and records
what came out or why it failed. An item is one speaker's segments of one recording.
Without a manifest, each channel of each readable file is one item over its whole
length: right for stereo calls with agent and caller on separate channels. A whole
channel is one turn, so turn consistency cannot see a second voice in it, and hold
music counts as speech wherever VAD keeps it.

Only failures that the input causes are recorded and passed over. Anything else is a
bug in this code, and stops the run.
"""

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from svs.api import EmbedItem, ServiceError, decode_embedding, embed_item
from svs.embed import EMBEDDING_DIM, Embedder

OUT = Path("artifacts/slice")
FILE_FIELDS = ["audio", "format", "subtype", "sample_rate", "channels", "seconds", "error"]
# segments: "start-end" in milliseconds, comma-separated; empty means the whole channel.
# speaker: a label, when the slice has one (D17 trials need it); empty otherwise.
MANIFEST_FIELDS = ["item_id", "audio", "channel", "segments", "speaker"]
DETAIL_FIELDS = ["snr_db", "speech_ratio", "clipping_ratio", "bandwidth_hz", "turn_consistency"]
RESULT_FIELDS = [
    *("item_id", "audio", "channel", "speaker", "net_speech_seconds", "quality"),
    *DETAIL_FIELDS,
    *("usable", "reason", "error"),
]


def one_line(exc: Exception) -> str:
    """An error message that fits in one TSV cell."""
    return " ".join(str(exc).split())


def inspect_file(root: Path, path: Path) -> dict:
    row = dict.fromkeys(FILE_FIELDS, "")
    row["audio"] = path.relative_to(root).as_posix()
    try:
        info = sf.info(path)
    except RuntimeError as exc:  # sf.LibsndfileError: an unknown format or codec
        row["error"] = one_line(exc)
        return row
    row.update(
        format=info.format,
        subtype=info.subtype,
        sample_rate=info.samplerate,
        channels=info.channels,
        seconds=f"{info.frames / info.samplerate:.3f}",
    )
    return row


def inspect_folder(root: Path) -> list[dict]:
    """Every file under root except hidden ones (.DS_Store), in path order."""
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    return [inspect_file(root, p) for p in paths]


def write_tsv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def inspect_summary(rows: list[dict]) -> list[str]:
    readable = [r for r in rows if not r["error"]]
    hours = sum(float(r["seconds"]) for r in readable) / 3600
    kinds = Counter((r["sample_rate"], r["channels"], r["subtype"]) for r in readable)
    lines = [f"{len(rows)} files, {len(rows) - len(readable)} unreadable, {hours:.2f} h readable"]
    for (rate, channels, subtype), n in kinds.most_common():
        lines.append(f"{n}\t{rate} Hz\t{channels} ch\t{subtype}")
    return lines


def default_manifest(files: list[dict]) -> list[dict]:
    """One item per channel of each readable file; a mono file has no channel to name."""
    rows = []
    for f in files:
        if f["error"]:
            continue
        channels = int(f["channels"])
        for ch in range(channels):
            rows.append(
                {
                    "item_id": f["audio"] if channels == 1 else f"{f['audio']}#ch{ch}",
                    "audio": f["audio"],
                    "channel": "" if channels == 1 else ch,
                    "segments": "",
                    "speaker": "",
                }
            )
    return rows


def parse_segments(text: str) -> list[tuple[float, float]]:
    """ "1200-4400,9100-16300" -> [(1200.0, 4400.0), (9100.0, 16300.0)]; "" -> []."""
    segments = []
    for part in filter(None, text.split(",")):
        start, end = part.split("-")
        segments.append((float(start), float(end)))
    return segments


def segments_text(segments: list[tuple[float, float]]) -> str:
    """parse_segments reversed, in whole ms rounded inwards, so no end passes the audio.

    A segment shorter than 1 ms can round to nothing; it is left out.
    """
    whole = [(math.ceil(s), math.floor(e)) for s, e in segments]
    return ",".join(f"{s}-{e}" for s, e in whole if s < e)


def whole_length(path: Path) -> list[tuple[float, float]]:
    info = sf.info(path)
    return [(0.0, info.frames * 1000 / info.samplerate)]


def run_item(embedder, root: Path, row: dict) -> tuple[dict, np.ndarray | None]:
    """One manifest row -> a results row, and the embedding when there is one."""
    out = dict.fromkeys(RESULT_FIELDS, "")
    out.update({k: row[k] for k in ("item_id", "audio", "channel", "speaker")})
    try:
        segments = parse_segments(row["segments"]) or whole_length(root / row["audio"])
        item = EmbedItem(
            id=row["item_id"],
            audio_uri=row["audio"],
            segments=segments,
            channel=int(row["channel"]) if str(row["channel"]) else None,
            purpose="OCCURRENCE",
        )
        result = embed_item(embedder, root, item)
    except ServiceError as exc:  # bad segments or channel, a file it cannot decode
        out["error"] = f"{exc.code}: {one_line(exc)}"
        return out, None
    except sf.LibsndfileError as exc:  # whole_length of a file it cannot open
        out["error"] = f"E_SV_INVALID_REQUEST: {one_line(exc)}"
        return out, None
    except ValueError as exc:  # a manifest cell it cannot parse
        out["error"] = f"manifest: {one_line(exc)}"
        return out, None
    out.update(
        net_speech_seconds=f"{result['net_speech_seconds']:.3f}",
        quality=f"{result['quality']:.4f}",
        usable=result["usable"],
        reason=result.get("unusable_reason", ""),
    )
    out.update({k: f"{result['quality_detail'][k]:.4f}" for k in DETAIL_FIELDS})
    embedding = result["embedding"]
    return out, None if embedding is None else decode_embedding(embedding)


def run_manifest(embedder, root: Path, manifest: list[dict]) -> tuple[list[dict], dict]:
    """Every row, in order; embeddings keyed by item_id."""
    rows, embeddings = [], {}
    for i, row in enumerate(manifest, 1):
        out, embedding = run_item(embedder, root, row)
        rows.append(out)
        if embedding is not None:
            embeddings[row["item_id"]] = embedding
        if i % 100 == 0:
            print(f"{i}/{len(manifest)}", flush=True)
    return rows, embeddings


def run_summary(rows: list[dict]) -> list[str]:
    errors = [r for r in rows if r["error"]]
    usable = sum(r["usable"] is True for r in rows)
    lines = [f"{len(rows)} items, {usable} usable, {len(errors)} failed"]
    reasons = Counter(r["reason"] for r in rows if r["reason"])
    codes = Counter(r["error"].split(":")[0] for r in errors)
    lines += [f"{n}\t{reason}" for reason, n in (reasons + codes).most_common()]
    return lines


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description="The pipeline over a folder of recordings")
    parser.add_argument("step", choices=["inspect", "run"])
    parser.add_argument("root", type=Path, help="the folder of recordings")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--manifest", type=Path, help="default: one item per channel")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step == "inspect":
        rows = inspect_folder(args.root)
        write_tsv(args.out / "files.tsv", FILE_FIELDS, rows)
        print("\n".join(inspect_summary(rows)))
        print(f"-> {args.out}/files.tsv")
        return

    if args.manifest is None:
        manifest = default_manifest(read_tsv(args.out / "files.tsv"))
        write_tsv(args.out / "manifest.tsv", MANIFEST_FIELDS, manifest)
    else:
        manifest = read_tsv(args.manifest)
    rows, embeddings = run_manifest(Embedder(), args.root, manifest)
    write_tsv(args.out / "results.tsv", RESULT_FIELDS, rows)
    np.savez(
        args.out / "embeddings.npz",
        item_ids=np.array(list(embeddings)),
        embeddings=np.array(list(embeddings.values()), np.float32).reshape(-1, EMBEDDING_DIM),
    )
    print("\n".join(run_summary(rows)))
    print(f"-> {args.out}/results.tsv, {args.out}/embeddings.npz")


if __name__ == "__main__":
    main()
