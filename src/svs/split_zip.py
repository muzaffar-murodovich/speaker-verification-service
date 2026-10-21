"""Read single members of a zip archive that is split into byte parts, without the whole.

VoxCeleb2 dev is one 77.5 GB zip cut into eight files (cat them to get the zip). The
cohort needs about 5,000 of its 1,092,009 members, so the zip directory and those
members are read with HTTP range requests instead of downloading every part.
"""

import io
import os
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path

# fetch(part index, first byte, last byte inclusive) -> those bytes of that part.
Fetch = Callable[[int, int, int], bytes]
# A local file header is 30 bytes plus name and extra field; the extra field is not in
# the directory entry, so this much is read past the name and trimmed after.
EXTRA_SLACK = 1024


class SplitZip:
    def __init__(self, sizes: list[int], fetch: Fetch) -> None:
        self.sizes = sizes
        self.starts = [sum(sizes[:i]) for i in range(len(sizes))]
        self.total = sum(sizes)
        self.fetch = fetch

    def read_at(self, offset: int, n: int) -> bytes:
        """n bytes from offset in the joined archive, across part boundaries."""
        out = []
        end = min(offset + n, self.total)
        while offset < end:
            part = max(i for i, start in enumerate(self.starts) if start <= offset)
            first = offset - self.starts[part]
            last = min(self.sizes[part], end - self.starts[part]) - 1
            chunk = self.fetch(part, first, last)
            if len(chunk) != last - first + 1:
                raise OSError(f"part {part}: asked {last - first + 1} bytes, got {len(chunk)}")
            out.append(chunk)
            offset += len(chunk)
        return b"".join(out)

    def members(self) -> list[zipfile.ZipInfo]:
        """The zip directory, read by zipfile itself (zip64 included)."""
        return zipfile.ZipFile(_View(self)).infolist()

    def read_member(self, info: zipfile.ZipInfo) -> bytes:
        """One member's content, inflated and checked against the CRC32 in the directory."""
        size = 30 + len(info.orig_filename.encode()) + EXTRA_SLACK + info.compress_size
        raw = self.read_at(info.header_offset, size)
        if raw[:4] != b"PK\x03\x04":
            raise ValueError(f"{info.filename}: no local file header at {info.header_offset}")
        name_len, extra_len = struct.unpack("<HH", raw[26:30])
        start = 30 + name_len + extra_len
        if start + info.compress_size > len(raw):
            raw += self.read_at(
                info.header_offset + len(raw), start + info.compress_size - len(raw)
            )
        data = raw[start : start + info.compress_size]
        if info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(data, -15)
        elif info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{info.filename}: compression type {info.compress_type}")
        if zlib.crc32(data) != info.CRC:
            raise ValueError(f"{info.filename}: CRC32 mismatch")
        return data


class _View(io.RawIOBase):
    """The joined archive as a seekable file, so zipfile can read the directory."""

    def __init__(self, archive: SplitZip) -> None:
        self.archive = archive
        self.pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.archive.total + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        n = self.archive.total - self.pos if n < 0 else n
        data = self.archive.read_at(self.pos, n)
        self.pos += len(data)
        return data


def hf_token() -> str:
    """HF_TOKEN, else the token `hf auth login` saved. The mirror is gated (auto-approve)."""
    if token := os.environ.get("HF_TOKEN"):
        return token
    path = Path.home() / ".cache" / "huggingface" / "token"
    if not path.exists():
        raise RuntimeError("No Hugging Face token: accept the dataset terms, then `hf auth login`")
    return path.read_text().strip()


def http_fetch(urls: list[str], token: str, tries: int = 5) -> Fetch:
    """Range requests against one URL per part, retried with a growing pause."""

    def fetch(part: int, first: int, last: int) -> bytes:
        headers = {"Authorization": f"Bearer {token}", "Range": f"bytes={first}-{last}"}
        for attempt in range(tries):
            try:
                request = urllib.request.Request(urls[part], headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise OSError(f"{urls[part]}: HTTP {response.status}, not a range reply")
                    return response.read()
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == tries - 1:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    return fetch
