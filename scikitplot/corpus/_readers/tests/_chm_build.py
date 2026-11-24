# scikitplot/corpus/_readers/tests/_chm_build.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Build small CHM files in tests, with a minimal LZX encoder.

The encoder writes every block type the decoder reads (verbatim, aligned
offset, uncompressed), new and repeated match offsets, blocks that span
frames, and the ``E8`` header, so the decoder is covered without binary
fixtures. Its Huffman trees are complete fixed-shape codes (any symbol
can be written); compression ratio is not the point.

**Developer note:** a decoder and an encoder written from one reading of
the format can share a misreading. Real help files compiled by the
Microsoft compiler are the independent check (see the CHM reader's pull
request); this module guards against regressions.
"""

from __future__ import annotations

import struct

from .._chm_format import (
    _ALIGNED_SIZE,
    _EXTRA_BITS,
    _LENGTH_SIZE,
    _NUM_CHARS,
    _POSITION_BASE,
    _PRETREE_SIZE,
    _SLOTS,
    FRAME,
)

VERBATIM, ALIGNED, UNCOMPRESSED = 1, 2, 3
_CONTROL = "::DataSpace/Storage/MSCompressed/ControlData"
_CONTENT = "::DataSpace/Storage/MSCompressed/Content"
_RESET = (
    "::DataSpace/Storage/MSCompressed/Transform/"
    "{7FC28940-9D31-11D0-9B27-00A0C91E9C7C}/InstanceData/ResetTable"
)


def complete_lengths(n: int) -> list[int]:
    """Return code lengths of a complete prefix code for *n* symbols."""
    bits = max(1, (n - 1).bit_length())
    short = (1 << bits) - n
    return [bits - 1] * short + [bits] * (n - short)


def _codes(lengths: list[int]) -> dict[int, tuple[int, int]]:
    """Canonical codes: symbol -> (code, length)."""
    out: dict[int, tuple[int, int]] = {}
    code = 0
    for length in range(1, max(lengths) + 1):
        for symbol, size in enumerate(lengths):
            if size == length:
                out[symbol] = (code, length)
                code += 1
        code <<= 1
    return out


class _Bits:
    """MSB-first bit writer over little-endian 16-bit words."""

    def __init__(self) -> None:
        self.out = bytearray()
        self.acc = 0
        self.n = 0

    def write(self, value: int, bits: int) -> None:
        for i in range(bits - 1, -1, -1):
            self.acc = (self.acc << 1) | ((value >> i) & 1)
            self.n += 1
            if self.n == 16:
                self.out += struct.pack("<H", self.acc)
                self.acc = self.n = 0

    def align(self, force: bool = False) -> None:
        """Pad to 16 bits; with *force*, add 16 bits when already aligned."""
        if self.n:
            self.write(0, 16 - self.n)
        elif force:
            self.write(0, 16)

    @property
    def total(self) -> int:
        return len(self.out) * 8 + self.n


def _slot(formatted: int) -> int:
    slot = 0
    while slot + 1 < len(_POSITION_BASE) and _POSITION_BASE[slot + 1] <= formatted:
        slot += 1
    return slot


class Encoder:
    """
    LZX encoder over a token list, one reset interval at a time.

    Tokens are ``int`` (a literal byte) or ``(length, offset)`` matches.
    """

    def __init__(self, window_bits: int = 16) -> None:
        self.window_bits = window_bits
        self.main_size = _NUM_CHARS + _SLOTS[window_bits] * 8
        self.main_lengths = complete_lengths(self.main_size)
        self.length_lengths = complete_lengths(_LENGTH_SIZE)
        self.aligned_lengths = [3] * _ALIGNED_SIZE
        self.main = _codes(self.main_lengths)
        self.length = _codes(self.length_lengths)
        self.aligned = _codes(self.aligned_lengths)
        self.reset()

    def reset(self) -> None:
        self.r = [1, 1, 1]
        self.prev_main = [0] * self.main_size
        self.prev_length = [0] * _LENGTH_SIZE
        self.header = False

    def _tree(self, bits: _Bits, new: list[int], prev: list[int]) -> None:
        pre = complete_lengths(_PRETREE_SIZE)
        codes = _codes(pre)
        for length in pre:
            bits.write(length, 4)
        for i, length in enumerate(new):
            z = (prev[i] - length) % 17
            bits.write(*codes[z])
        prev[:] = new

    def _match(self, bits: _Bits, kind: int, length: int, offset: int) -> None:
        r = self.r
        if offset == r[0]:
            slot = 0
        elif offset == r[1]:
            slot = 1
            r[0], r[1] = r[1], r[0]
        elif offset == r[2]:
            slot = 2
            r[0], r[2] = r[2], r[0]
        else:
            slot = max(3, _slot(offset + 2))
            r[2], r[1], r[0] = r[1], r[0], offset
        header = min(length - 2, 7)
        bits.write(*self.main[_NUM_CHARS + slot * 8 + header])
        if header == 7:
            bits.write(*self.length[length - 2 - 7])
        if slot < 3:
            return
        extra = _EXTRA_BITS[slot]
        value = offset + 2 - _POSITION_BASE[slot]
        if kind == ALIGNED and extra >= 3:
            if extra > 3:
                bits.write(value >> 3, extra - 3)
            bits.write(*self.aligned[value & 7])
        else:
            bits.write(value, extra)

    def frame(  # noqa: PLR0913
        self,
        tokens: list,
        kind: int,
        *,
        block_length: int | None,
        intel_size: int = 0,
        raw: bytes = b"",
    ) -> bytes:
        """
        Encode one 32 KiB frame; open a block of *block_length* when given.

        For ``UNCOMPRESSED``, *raw* is the frame's bytes.
        """
        bits = _Bits()
        if not self.header:
            if intel_size:
                bits.write(1, 1)
                bits.write(intel_size >> 16, 16)
                bits.write(intel_size & 0xFFFF, 16)
            else:
                bits.write(0, 1)
            self.header = True
        if block_length is not None:
            bits.write(kind, 3)
            bits.write(block_length >> 8, 16)
            bits.write(block_length & 0xFF, 8)
            if kind == ALIGNED:
                for length in self.aligned_lengths:
                    bits.write(length, 3)
            if kind in (VERBATIM, ALIGNED):
                self._tree(bits, self.main_lengths[:_NUM_CHARS], self.prev_main[:_NUM_CHARS])
                tail = self.prev_main[_NUM_CHARS:]
                self._tree(bits, self.main_lengths[_NUM_CHARS:], tail)
                self.prev_main[_NUM_CHARS:] = tail
                self.prev_main[:_NUM_CHARS] = self.main_lengths[:_NUM_CHARS]
                self._tree(bits, self.length_lengths, self.prev_length)
            else:
                bits.align(force=True)
                out = bytes(bits.out) + struct.pack("<3I", *self.r)
                out += raw
                if block_length & 1 and len(raw) == block_length:
                    out += b"\x00"
                return out
        if kind == UNCOMPRESSED:
            return bytes(raw)
        for token in tokens:
            if isinstance(token, int):
                bits.write(*self.main[token])
            else:
                self._match(bits, kind, *token)
        bits.align()
        return bytes(bits.out)


def tokenize(data: bytes, window: int = 1 << 16) -> list:
    """Greedy tokens with matches up to 257 bytes, never across a frame."""
    tokens: list = []
    last: dict[bytes, int] = {}
    i = 0
    while i < len(data):
        frame_end = (i // FRAME + 1) * FRAME
        key = data[i : i + 3]
        j = last.get(key)
        last[key] = i
        if j is not None and len(key) == 3 and i - j <= window - 3:
            n = 0
            limit = min(257, len(data) - i, frame_end - i)
            while n < limit and data[j + n] == data[i + n]:
                n += 1
            if n >= 3:
                tokens.append((n, i - j))
                i += n
                continue
        tokens.append(data[i])
        i += 1
    return tokens


def compress(
    stream: bytes,
    *,
    kind: int = VERBATIM,
    reset_frames: int = 2,
    frames_per_block: int = 1,
    intel_size: int = 0,
) -> tuple[bytes, list[int]]:
    """Return ``(content, frame offsets)`` for *stream*."""
    enc = Encoder()
    content = bytearray()
    offsets: list[int] = []
    frames = [stream[i : i + FRAME] for i in range(0, len(stream), FRAME)]
    interval: list[list] = []
    remaining = 0
    for index, frame in enumerate(frames):
        if index % reset_frames == 0:
            enc.reset()
            remaining = 0
            base = index * FRAME
            group = stream[base : base + reset_frames * FRAME]
            tokens = tokenize(group) if kind != UNCOMPRESSED else []
            interval = [[] for _ in range(reset_frames)]
            position = 0
            for token in tokens:
                interval[position // FRAME].append(token)
                position += 1 if isinstance(token, int) else token[0]
        block = None
        if remaining == 0:
            left = len(stream) - index * FRAME
            block = min(left, frames_per_block * FRAME, (reset_frames - index % reset_frames) * FRAME)
            remaining = block
        remaining -= len(frame)
        offsets.append(len(content))
        content += enc.frame(
            interval[index % reset_frames],
            kind,
            block_length=block,
            intel_size=intel_size,
            raw=frame,
        )
    return bytes(content), offsets


def _encint(value: int) -> bytes:
    groups = [value & 0x7F]
    value >>= 7
    while value:
        groups.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(groups))


def build_chm(  # noqa: PLR0913
    files: dict[str, bytes],
    *,
    stored: dict[str, bytes] | None = None,
    kind: int = VERBATIM,
    reset_frames: int = 2,
    frames_per_block: int = 1,
    intel_size: int = 0,
    lcid: int = 0x0409,
    chunk_size: int = 4096,
    next_chunk: int = -1,
) -> bytes:
    """
    Return the bytes of a CHM archive.

    *files* go to the compressed section (in order), *stored* to section 0.
    """
    stream = b"".join(files.values())
    content, offsets = compress(
        stream,
        kind=kind,
        reset_frames=reset_frames,
        frames_per_block=frames_per_block,
        intel_size=intel_size,
    )
    control = struct.pack("<I4s5I", 6, b"LZXC", 2, reset_frames, 2, 2, 0)
    table = struct.pack("<4I3Q", 2, len(offsets), 8, 0x28, len(stream), len(content), FRAME)
    table += struct.pack(f"<{len(offsets)}Q", *offsets)
    section0: dict[str, bytes] = {_CONTENT: content, _CONTROL: control, _RESET: table}
    section0.update(stored or {})
    entries = []
    at = 0
    for name, data in section0.items():
        entries.append((name, 0, at, len(data)))
        at += len(data)
    blob0 = b"".join(section0.values())
    at = 0
    for name, data in files.items():
        entries.append((name, 1, at, len(data)))
        at += len(data)
    body = b"".join(
        _encint(len(n.encode())) + n.encode() + _encint(s) + _encint(o) + _encint(n_)
        for n, s, o, n_ in entries
    )
    if 0x14 + len(body) > chunk_size:
        raise ValueError("test CHM: directory larger than one chunk")
    free = chunk_size - 0x14 - len(body)
    chunk = struct.pack("<4sIIii", b"PMGL", free, 0, -1, next_chunk) + body
    chunk += b"\x00" * (chunk_size - len(chunk))
    itsp = struct.pack(
        "<4sIIIIIIiiiiI", b"ITSP", 1, 0x54, 10, chunk_size, 2, 1, -1, 0, 0, -1, 1
    )
    itsp += b"\x00" * (0x54 - len(itsp))
    directory = itsp + chunk
    dir_offset = 0x60
    content_offset = dir_offset + len(directory)
    itsf = struct.pack("<4sIIIII", b"ITSF", 3, 0x60, 1, 0, lcid) + b"\x00" * 32
    itsf += struct.pack("<5Q", 0, 0, dir_offset, len(directory), content_offset)
    return itsf + directory + blob0
