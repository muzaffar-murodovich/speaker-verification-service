"""Run the pipeline over a folder of call recordings (handbook D16).

inspect reads only the file headers: format, sample rate, channels, duration. It costs
seconds, and it says what the audio is before anything is embedded. A file libsndfile
cannot open is listed with its error, not skipped: D16 is about writing down
everything that breaks.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import soundfile as sf

OUT = Path("artifacts/slice")
FILE_FIELDS = ["audio", "format", "subtype", "sample_rate", "channels", "seconds", "error"]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="The pipeline over a folder of recordings")
    parser.add_argument("step", choices=["inspect"])
    parser.add_argument("root", type=Path, help="the folder of recordings")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = inspect_folder(args.root)
    write_tsv(args.out / "files.tsv", FILE_FIELDS, rows)
    print("\n".join(inspect_summary(rows)))
    print(f"-> {args.out}/files.tsv")


if __name__ == "__main__":
    main()
