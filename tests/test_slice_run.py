"""The pipeline over a folder of recordings: headers first, nothing stops at a broken file."""

import numpy as np
import pytest
import soundfile as sf

from svs import slice_run
from svs.embed import EMBEDDING_DIM
from svs.slice_run import (
    default_manifest,
    inspect_folder,
    inspect_summary,
    parse_segments,
    read_tsv,
    run_manifest,
    run_summary,
)


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


class KeepAll:  # every frame is speech, as in tests/test_api.py
    def __init__(self, mode):
        pass

    def is_speech(self, pcm, rate):
        return True


class SignEmbedder:
    """A stand-in model: one voice for audio with a positive mean, another for negative."""

    def embed(self, samples):
        return np.eye(EMBEDDING_DIM, dtype=np.float32)[0 if samples.mean() > 0 else 1]


@pytest.fixture
def calls(tmp_path, monkeypatch):
    """A stereo call: agent (mean +0.2) on channel 0, caller (mean -0.2) on channel 1,
    10 s each; and a 2 s mono file."""
    monkeypatch.setattr("svs.extract.webrtcvad.Vad", KeepAll)
    noise = 0.05 * np.random.default_rng(1).standard_normal((160000, 2))
    stereo = (noise + [0.2, -0.2]).astype(np.float32)
    sf.write(tmp_path / "call.wav", stereo, 16000, subtype="FLOAT")
    sf.write(tmp_path / "short.wav", stereo[:32000, 0], 16000, subtype="FLOAT")
    (tmp_path / "labels.txt").write_text("not audio\n")
    return tmp_path


def test_the_default_manifest_is_one_item_per_channel_of_each_readable_file(calls):
    manifest = default_manifest(inspect_folder(calls))
    assert [(m["item_id"], m["channel"]) for m in manifest] == [
        ("call.wav#ch0", 0),
        ("call.wav#ch1", 1),
        ("short.wav", ""),
    ]
    assert {m["segments"] for m in manifest} == {""}


def test_segments_are_start_end_in_ms_and_empty_means_none():
    assert parse_segments("1200-4400,9100-16300") == [(1200.0, 4400.0), (9100.0, 16300.0)]
    assert parse_segments("") == []


def test_each_channel_is_its_own_speaker_and_short_speech_is_refused(calls):
    rows, embeddings = run_manifest(SignEmbedder(), calls, default_manifest(inspect_folder(calls)))
    agent, caller, short = rows
    assert agent["net_speech_seconds"] == caller["net_speech_seconds"] == "10.000"
    assert embeddings["call.wav#ch0"] @ embeddings["call.wav#ch1"] == 0  # two voices
    assert short["reason"] == "E_SV_INSUFFICIENT_SPEECH" and short["error"] == ""
    assert "short.wav" not in embeddings


def test_a_broken_row_is_recorded_and_the_run_goes_on(calls):
    def row(item_id, audio, channel="", segments=""):
        return {
            "item_id": item_id,
            "audio": audio,
            "channel": channel,
            "segments": segments,
            "speaker": "",
        }

    manifest = [
        row("past_the_end", "call.wav", 0, "0-25000"),
        row("no_channel", "call.wav"),
        row("not_audio", "labels.txt"),
        row("bad_cell", "call.wav", 0, "0to4000"),
        row("fine", "call.wav", 1, "0-4000"),
    ]
    rows, embeddings = run_manifest(SignEmbedder(), calls, manifest)
    errors = [r["error"].split(":")[0] for r in rows]
    assert errors == ["E_SV_INVALID_REQUEST"] * 3 + ["manifest", ""]
    assert list(embeddings) == ["fine"]
    assert run_summary(rows)[0] == "5 items, 1 usable, 4 failed"


def test_run_writes_the_manifest_results_and_embeddings(calls, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    monkeypatch.setattr(slice_run, "Embedder", SignEmbedder)
    for step in ("inspect", "run"):
        monkeypatch.setattr("sys.argv", ["slice_run", step, str(calls), "--out", str(out)])
        slice_run.main()
    assert len(read_tsv(out / "manifest.tsv")) == 3
    assert [r["usable"] for r in read_tsv(out / "results.tsv")] == ["True", "True", "False"]
    saved = np.load(out / "embeddings.npz")
    assert list(saved["item_ids"]) == ["call.wav#ch0", "call.wav#ch1"]
    assert saved["embeddings"].shape == (2, EMBEDDING_DIM)
    assert "3 items, 2 usable, 0 failed" in capsys.readouterr().out
