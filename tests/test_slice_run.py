"""The pipeline over a folder of recordings: headers first, nothing stops at a broken file."""

import numpy as np
import soundfile as sf

from svs import slice_run
from svs.slice_run import inspect_folder, inspect_summary


def call_folder(root):
    """An 8 kHz stereo call, a 16 kHz mono FLAC, a file that is not audio, a hidden file."""
    (root / "day1").mkdir()
    sf.write(root / "day1" / "call.wav", np.zeros((16000, 2), np.float32), 8000, subtype="PCM_16")
    sf.write(root / "voice.flac", np.zeros(48000, np.float32), 16000)
    (root / "labels.txt").write_text("not audio\n")
    (root / ".DS_Store").write_bytes(b"\0")
    return root


def test_headers_of_every_file_and_the_error_of_the_unreadable_one(tmp_path):
    rows = inspect_folder(call_folder(tmp_path))
    assert [r["audio"] for r in rows] == ["day1/call.wav", "labels.txt", "voice.flac"]
    call, text, voice = rows
    assert (call["sample_rate"], call["channels"], call["subtype"], call["seconds"]) == (
        8000,
        2,
        "PCM_16",
        "2.000",
    )
    assert call["error"] == ""
    assert text["error"] and text["sample_rate"] == ""
    assert (voice["format"], voice["sample_rate"], voice["channels"]) == ("FLAC", 16000, 1)


def test_the_summary_counts_each_kind_of_audio(tmp_path):
    lines = inspect_summary(inspect_folder(call_folder(tmp_path)))
    assert lines[0] == "3 files, 1 unreadable, 0.00 h readable"
    assert sorted(lines[1:]) == ["1\t16000 Hz\t1 ch\tPCM_16", "1\t8000 Hz\t2 ch\tPCM_16"]


def test_inspect_writes_files_tsv(tmp_path, monkeypatch, capsys):
    (tmp_path / "in").mkdir()
    root, out = call_folder(tmp_path / "in"), tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["slice_run", "inspect", str(root), "--out", str(out)])
    slice_run.main()
    lines = (out / "files.tsv").read_text().splitlines()
    assert lines[0].split("\t") == slice_run.FILE_FIELDS
    assert len(lines) == 4
    assert "1 unreadable" in capsys.readouterr().out
