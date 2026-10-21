"""Split zip: members are read across part boundaries and checked against their CRC32."""

import io
import zipfile

import pytest

from svs.split_zip import SplitZip

CONTENT = {
    "dev/aac/id1/v1/00001.m4a": bytes(range(256)) * 3,  # deflated
    "dev/aac/id1/v1/00002.m4a": b"stored as is" * 20,
    "dev/aac/id2/v9/00001.m4a": b"x" * 500,
}


def split_archive(part_size: int) -> tuple[list[bytes], SplitZip, list[int]]:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        for name, data in CONTENT.items():
            kind = zipfile.ZIP_STORED if "00002" in name else zipfile.ZIP_DEFLATED
            z.writestr(name, data, compress_type=kind)
    whole = buffer.getvalue()
    parts = [whole[i : i + part_size] for i in range(0, len(whole), part_size)]
    calls = []

    def fetch(part, first, last):
        calls.append(part)
        return parts[part][first : last + 1]

    return parts, SplitZip([len(p) for p in parts], fetch), calls


def test_every_member_comes_back_whole_across_small_parts():
    _, archive, calls = split_archive(part_size=97)
    infos = archive.members()
    assert [i.filename for i in infos] == list(CONTENT)
    for info in infos:
        calls.clear()
        assert archive.read_member(info) == CONTENT[info.filename]
        assert len(set(calls)) > 1  # this member really came from several parts


def test_corrupt_member_is_refused():
    parts, archive, _ = split_archive(part_size=10_000)
    info = archive.members()[1]  # stored, so a flipped byte reaches the CRC check
    data_at = info.header_offset + 30 + len(info.filename) + 5
    whole = bytearray(parts[0])
    whole[data_at] ^= 0xFF
    broken = SplitZip([len(whole)], lambda part, first, last: bytes(whole[first : last + 1]))
    with pytest.raises(ValueError, match="CRC32 mismatch"):
        broken.read_member(info)


def test_short_reply_from_the_server_is_an_error():
    parts, _, _ = split_archive(part_size=10_000)
    short = SplitZip([len(parts[0])], lambda part, first, last: parts[0][first:last])
    with pytest.raises(OSError, match="got"):
        short.read_at(0, 100)
