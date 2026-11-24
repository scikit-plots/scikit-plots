# scikitplot/corpus/_readers/_chm_format.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# Binary format: header offsets, field sizes and code values are the
# format's own numbers, named in the comments next to them.
# ruff: noqa: PLR2004

"""
Microsoft Compiled HTML Help (``.chm``) container and LZX decoder.

Pure Python, standard library only, written from the published format
descriptions. It does what :class:`~scikitplot.corpus._readers.CHMReader`
needs and no more: list the files of an archive and return the bytes of
the ones asked for, within explicit limits.

**Container (ITSF).** A header gives the offsets of the directory and of
the content area. The directory is a chain of fixed-size ``PMGL`` chunks
listing ``(name, section, offset, length)``. Section 0 is stored as is;
section 1 (``MSCompressed``) is one LZX stream, cut into 32 KiB output
frames whose compressed offsets a reset table lists, with a decoder reset
every *n* frames. So a file is read by decoding from the reset point
before its first frame to its end, never the whole stream.

**LZX.** Bits are read most significant first from little-endian 16-bit
words. Blocks are *verbatim*, *aligned offset* or *uncompressed*; code
lengths are sent as deltas through a 20-symbol pre-tree and persist
between blocks until a reset; matches use the three most recent offsets.
Each frame starts on a 16-bit boundary at its reset-table offset. The
optional x86 ``E8`` call translation is undone per frame.

Every failure is a ``ValueError`` naming what is wrong. Nothing here
trusts a length, an offset or a code from the file: each is checked
against the bytes present and the caller's limits before it is used.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple  # noqa: F401

__all__: list[str] = []

# ---------------------------------------------------------------------------
# Container constants
# ---------------------------------------------------------------------------

_ITSF = b"ITSF"
_ITSP = b"ITSP"
_PMGL = b"PMGL"
_LZXC = b"LZXC"

_CONTENT = "::DataSpace/Storage/MSCompressed/Content"
_CONTROL = "::DataSpace/Storage/MSCompressed/ControlData"
_RESET_TABLE = (
    "::DataSpace/Storage/MSCompressed/Transform/"
    "{7FC28940-9D31-11D0-9B27-00A0C91E9C7C}/InstanceData/ResetTable"
)

#: Bound on directory chunks and entries, far above real help files.
_MAX_DIRECTORY_CHUNKS = 1 << 20
_MAX_NAME_BYTES = 4096

#: LZX output frame size; every CHM uses it.
FRAME = 0x8000

# ---------------------------------------------------------------------------
# LZX constants
# ---------------------------------------------------------------------------

_MIN_MATCH = 2
_NUM_CHARS = 256
_PRETREE_SIZE = 20
_LENGTH_SIZE = 249
_ALIGNED_SIZE = 8
_MAX_CODE_BITS = 16
_VERBATIM, _ALIGNED, _UNCOMPRESSED = 1, 2, 3

#: Position slots per window size (``window_bits`` 15..21).
_SLOTS = {15: 30, 16: 32, 17: 34, 18: 36, 19: 38, 20: 42, 21: 50}


def _slot_tables() -> tuple[list[int], list[int]]:
    """Return ``(extra_bits, position_base)`` for the 51 position slots."""
    extra: list[int] = []
    bits = 0
    for i in range(0, 52, 2):
        extra += [bits, bits]
        if i != 0 and bits < 17:
            bits += 1
    base: list[int] = []
    position = 0
    for i in range(51):
        base.append(position)
        position += 1 << extra[i]
    return extra[:51], base


_EXTRA_BITS, _POSITION_BASE = _slot_tables()


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChmEntry:
    """
    One directory entry of a CHM archive.

    Parameters
    ----------
    name : str
        Path inside the archive, for example ``"/index.html"``.
    section : int
        0 (stored) or 1 (LZX compressed).
    offset : int
        Offset in the section's uncompressed space.
    length : int
        Length in bytes.
    """

    name: str
    section: int
    offset: int
    length: int


def _u32(data: bytes, at: int) -> int:
    return struct.unpack_from("<I", data, at)[0]


def _i32(data: bytes, at: int) -> int:
    return struct.unpack_from("<i", data, at)[0]


def _u64(data: bytes, at: int) -> int:
    return struct.unpack_from("<Q", data, at)[0]


def _encint(data: bytes, at: int, end: int) -> tuple[int, int]:
    """
    Decode a CHM variable-length integer (7 bits per byte, high bit = more).

    Returns
    -------
    tuple of (int, int)
        The value and the offset after it.
    """
    value = 0
    for _ in range(10):  # 70 bits: more than any 64-bit field
        if at >= end:
            raise ValueError("CHM: a directory entry runs past its chunk")
        byte = data[at]
        at += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, at
    raise ValueError("CHM: a directory integer is longer than 64 bits")


class ChmArchive:
    """
    Read-only view of a CHM archive held in a bytes-like buffer.

    Parameters
    ----------
    data : bytes-like
        The whole file (``bytes`` or a read-only ``mmap``).
    name : str, optional
        Used in error messages.

    Raises
    ------
    ValueError
        If the buffer is not a well-formed CHM archive.

    Notes
    -----
    Construction parses the headers and the directory only; no file is
    decompressed until :meth:`read_many` asks for it. The compressed
    content is read through a view of *data*, not copied: use the archive
    as a context manager (or call :meth:`close`) before closing a memory
    map it reads.
    """

    def __init__(self, data: bytes, name: str = "<chm>") -> None:
        self._data = data
        self.name = name
        size = len(data)
        if size < 0x60 or bytes(data[:4]) != _ITSF:
            raise ValueError(f"CHM: {name} is not a CHM file (no ITSF header)")
        version = _u32(data, 4)
        if version not in (2, 3):
            raise ValueError(f"CHM: {name} has unsupported ITSF version {version}")
        self.lcid = _u32(data, 0x14)
        dir_offset = _u64(data, 0x48)
        dir_length = _u64(data, 0x50)
        content = _u64(data, 0x58) if version == 3 else dir_offset + dir_length
        if dir_offset + dir_length > size or content > size:
            raise ValueError(f"CHM: {name} is truncated (directory past the end)")
        self._content = content
        self.entries = self._read_directory(dir_offset, dir_length)
        self._by_name = {e.name.lower(): e for e in self.entries}
        self._lzx: _LzxSection | None = None
        self._views: list[memoryview] = []

    # -- directory ---------------------------------------------------------

    def _read_directory(self, offset: int, length: int) -> list[ChmEntry]:
        data = self._data
        if length < 0x54 or bytes(data[offset : offset + 4]) != _ITSP:
            raise ValueError(f"CHM: {self.name} has no ITSP directory header")
        header_len = _u32(data, offset + 8)
        chunk_size = _u32(data, offset + 0x10)
        first = _i32(data, offset + 0x20)
        count = _u32(data, offset + 0x2C)
        if not 0x200 <= chunk_size <= 0x10000 or chunk_size & (chunk_size - 1):
            raise ValueError(f"CHM: {self.name} has a bad chunk size {chunk_size}")
        start = offset + header_len
        if (
            count > _MAX_DIRECTORY_CHUNKS
            or start + count * chunk_size > offset + length
        ):
            raise ValueError(f"CHM: {self.name} has a directory larger than its area")
        entries: list[ChmEntry] = []
        seen: set[int] = set()
        index = first
        while 0 <= index < count:
            if index in seen:
                raise ValueError(f"CHM: {self.name} has a directory chunk loop")
            seen.add(index)
            base = start + index * chunk_size
            if bytes(data[base : base + 4]) != _PMGL:
                raise ValueError(f"CHM: {self.name} chunk {index} is not a PMGL chunk")
            free = _u32(data, base + 4)
            end = base + chunk_size - free
            if free > chunk_size - 0x14:
                raise ValueError(f"CHM: {self.name} chunk {index} has a bad free space")
            at = base + 0x14
            while at < end:
                name_len, at = _encint(data, at, end)
                if name_len > _MAX_NAME_BYTES or at + name_len > end:
                    raise ValueError(f"CHM: {self.name} has a bad entry name length")
                raw_name = bytes(data[at : at + name_len])
                at += name_len
                section, at = _encint(data, at, end)
                file_offset, at = _encint(data, at, end)
                file_length, at = _encint(data, at, end)
                entries.append(
                    ChmEntry(
                        raw_name.decode("utf-8", errors="replace"),
                        section,
                        file_offset,
                        file_length,
                    )
                )
            index = _i32(data, base + 0x10)
        return entries

    def entry(self, name: str) -> ChmEntry | None:
        """Return the entry for *name* (case-insensitive), or ``None``."""
        return self._by_name.get(name.lower())

    # -- reading -----------------------------------------------------------

    def _stored(self, entry: ChmEntry) -> bytes:
        start = self._content + entry.offset
        if entry.section != 0 or start + entry.length > len(self._data):
            raise ValueError(f"CHM: {self.name}: {entry.name} lies past the end")
        return bytes(self._data[start : start + entry.length])

    def _view(self, entry: ChmEntry) -> memoryview:
        start = self._content + entry.offset
        if entry.section != 0 or start + entry.length > len(self._data):
            raise ValueError(f"CHM: {self.name}: {entry.name} lies past the end")
        view = memoryview(self._data)[start : start + entry.length]
        self._views.append(view)
        return view

    def close(self) -> None:
        """Release the views into the buffer, so a memory map can close."""
        for view in self._views:
            view.release()
        self._views.clear()
        self._lzx = None

    def __enter__(self) -> ChmArchive:  # noqa: PYI034 (no typing.Self on 3.8)
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _section(self) -> _LzxSection:
        if self._lzx is None:
            self._lzx = _LzxSection(self)
        return self._lzx

    def read_many(
        self,
        entries: list[ChmEntry],
        *,
        max_decoded_bytes: int,
    ) -> Iterator[tuple[ChmEntry, bytes]]:
        """
        Yield ``(entry, bytes)`` for *entries*, in their order in the stream.

        Parameters
        ----------
        entries : list of ChmEntry
            Entries of this archive, section 0 or 1.
        max_decoded_bytes : int
            Most bytes the LZX decoder may produce in total (including
            bytes decoded only to reach a file). Reaching it raises.

        Yields
        ------
        tuple of (ChmEntry, bytes)
            Each entry with its content.

        Raises
        ------
        ValueError
            If the archive is corrupt or the decoding budget is exceeded.
        """
        for entry in entries:
            if entry.section == 0:
                yield entry, self._stored(entry)
        compressed = sorted(
            (e for e in entries if e.section == 1), key=lambda e: e.offset
        )
        if compressed:
            yield from self._section().read(compressed, max_decoded_bytes)


# ---------------------------------------------------------------------------
# Section 1: LZX stream with a reset table
# ---------------------------------------------------------------------------


class _LzxSection:
    """The compressed section of one archive: its stream, frames and resets."""

    def __init__(self, archive: ChmArchive) -> None:
        name = archive.name
        need = {n: archive.entry(n) for n in (_CONTENT, _CONTROL, _RESET_TABLE)}
        missing = [n for n, e in need.items() if e is None]
        if missing:
            raise ValueError(f"CHM: {name} has compressed files but no {missing[0]}")
        control = archive._stored(need[_CONTROL])
        table = archive._stored(need[_RESET_TABLE])
        if len(control) < 0x1C or control[4:8] != _LZXC:
            raise ValueError(f"CHM: {name} has no LZXC control data")
        version = _u32(control, 8)
        reset_interval = _u32(control, 12)
        window = _u32(control, 16)
        windows_per_reset = _u32(control, 20)
        if version == 2:
            reset_interval *= FRAME
            window *= FRAME
        window_bits = window.bit_length() - 1
        if window != 1 << window_bits or window_bits not in _SLOTS:
            raise ValueError(f"CHM: {name} has an unsupported LZX window {window}")
        if reset_interval <= 0 or reset_interval % FRAME:
            raise ValueError(f"CHM: {name} has a bad LZX reset interval")
        del windows_per_reset  # informational; the interval decides
        if len(table) < 0x28:
            raise ValueError(f"CHM: {name} has a truncated reset table")
        count = _u32(table, 4)
        entry_size = _u32(table, 8)
        header = _u32(table, 12)
        self.uncompressed = _u64(table, 16)
        compressed = _u64(table, 24)
        block = _u64(table, 32)
        if block != FRAME or entry_size != 8 or header + 8 * count > len(table):
            raise ValueError(f"CHM: {name} has an unsupported reset table")
        self.offsets = list(struct.unpack_from(f"<{count}Q", table, header))
        frames = -(-self.uncompressed // FRAME)
        if count < frames or any(b < a for a, b in zip(self.offsets, self.offsets[1:])):
            raise ValueError(f"CHM: {name} has an inconsistent reset table")
        # A view, not a copy: the content is most of the file.
        self.content = archive._view(need[_CONTENT])
        if compressed > len(self.content) or (count and self.offsets[-1] > compressed):
            raise ValueError(f"CHM: {name} has a reset table past its content")
        self.compressed = compressed
        self.window_bits = window_bits
        self.reset_frames = reset_interval // FRAME
        self.name = name

    def read(
        self, entries: list[ChmEntry], budget: int
    ) -> Iterator[tuple[ChmEntry, bytes]]:
        """
        Yield the content of *entries* (sorted by offset) within *budget*.

        Decoding continues from the last frame when the next file is near,
        and restarts at the reset point before it when that is later, so
        each frame is decoded at most once and gaps of whole reset
        intervals are skipped.
        """
        decoder = _LzxDecoder(self.window_bits, self.name)
        decoded = 0
        next_frame: int | None = None
        cache: dict[int, bytes] = {}
        for entry in entries:
            end = entry.offset + entry.length
            if end > self.uncompressed:
                raise ValueError(
                    f"CHM: {self.name}: {entry.name} lies past the compressed section"
                )
            if entry.length == 0:
                yield entry, b""
                continue
            first = entry.offset // FRAME
            last = (end - 1) // FRAME
            for stale in [f for f in cache if f < first]:
                del cache[stale]
            reset = first - first % self.reset_frames
            if next_frame is None or reset > next_frame:
                next_frame = reset
            while next_frame <= last:
                if next_frame % self.reset_frames == 0:
                    decoder.reset()
                out = decoder.frame(
                    self.content,
                    self.offsets[next_frame],
                    self._frame_end(next_frame),
                    min(FRAME, self.uncompressed - next_frame * FRAME),
                )
                decoded += len(out)
                if decoded > budget:
                    raise ValueError(
                        f"CHM: {self.name}: decoding needs more than"
                        f" {budget:,} bytes; pass a larger limit to read it"
                    )
                if next_frame >= first:
                    cache[next_frame] = out
                next_frame += 1
            joined = b"".join(cache[f] for f in range(first, last + 1))
            lo = entry.offset - first * FRAME
            yield entry, joined[lo : lo + entry.length]

    def _frame_end(self, frame: int) -> int:
        if frame + 1 < len(self.offsets):
            return self.offsets[frame + 1]
        return self.compressed


# ---------------------------------------------------------------------------
# LZX decoder
# ---------------------------------------------------------------------------


def _decode_table(lengths: list[int], what: str) -> tuple[list[int], int]:
    """
    Build a lookup table for canonical Huffman *lengths*.

    Returns
    -------
    tuple of (list of int, int)
        ``table[peek]`` is ``symbol << 5 | length`` (``-1`` for a code that
        is not assigned) for a peek of ``bits`` bits.

    Raises
    ------
    ValueError
        If the lengths over-subscribe the code space.
    """
    bits = max(lengths, default=0)
    if bits == 0:
        return [], 0
    if bits > _MAX_CODE_BITS:
        raise ValueError(f"LZX: {what} code longer than 16 bits")
    table = [-1] * (1 << bits)
    code = 0
    for length in range(1, bits + 1):
        for symbol, size in enumerate(lengths):
            if size != length:
                continue
            span = 1 << (bits - length)
            start = code << (bits - length)
            if start + span > len(table):
                raise ValueError(f"LZX: {what} code lengths over-subscribe the tree")
            table[start : start + span] = [symbol << 5 | length] * span
            code += 1
        code <<= 1
    return table, bits


class _LzxDecoder:
    """
    LZX decoder state that persists across the frames of one reset interval.

    Parameters
    ----------
    window_bits : int
        Base-2 logarithm of the window size (15..21).
    name : str
        Archive name for error messages.
    """

    def __init__(self, window_bits: int, name: str) -> None:
        self.window = 1 << window_bits
        self.main_size = _NUM_CHARS + _SLOTS[window_bits] * 8
        self.name = name
        self.reset()

    def reset(self) -> None:
        """Return to the state at the start of a reset interval."""
        self.r = [1, 1, 1]
        self.main_lengths = [0] * self.main_size
        self.length_lengths = [0] * _LENGTH_SIZE
        self.header_read = False
        self.intel_size = 0
        self.intel_started = False
        self.intel_position = 0
        self.block_type = 0
        self.block_remaining = 0
        self.block_length = 0
        self.history = bytearray()
        self.cursor = 0  # where the next frame's output starts in history
        self.produced = 0  # bytes produced since the reset
        empty: tuple[list[int], int] = ([], 0)
        self._main = self._length = self._aligned = empty

    # -- one frame ---------------------------------------------------------

    def frame(  # noqa: C901, PLR0912, PLR0915
        self,
        data: bytes,
        start: int,
        end: int,
        out_len: int,
    ) -> bytes:
        """
        Decode the frame whose compressed bytes are ``data[start:end]``.

        Returns
        -------
        bytes
            ``out_len`` bytes (32 KiB except for the last frame).
        """
        if not 0 <= start <= end <= len(data):
            raise ValueError(f"LZX: {self.name}: frame offsets outside the content")
        src = bytes(data[start:end]) + b"\x00" * 8
        limit = end - start + 4  # a little zero padding is read at the end
        pos = 0
        buf = 0
        left = 0
        history = self.history
        out_start = self.cursor
        out_end = out_start + out_len
        r0, r1, r2 = self.r

        def need(n: int) -> None:
            nonlocal pos, buf, left
            while left < n:
                if pos >= limit:
                    raise ValueError(f"LZX: {self.name}: compressed data ends early")
                buf = (buf << 16) | src[pos] | (src[pos + 1] << 8)
                pos += 2
                left += 16

        def read(n: int) -> int:
            nonlocal buf, left
            if n == 0:
                return 0
            need(n)
            left -= n
            value = buf >> left
            buf &= (1 << left) - 1
            return value

        def symbol(table: list[int], bits: int, what: str) -> int:
            nonlocal buf, left
            if bits == 0:
                raise ValueError(f"LZX: {self.name}: an empty {what} tree is used")
            if left < bits:
                need(bits)
            entry = table[buf >> (left - bits)]
            if entry < 0:
                raise ValueError(f"LZX: {self.name}: an invalid {what} code")
            left -= entry & 31
            buf &= (1 << left) - 1
            return entry >> 5

        def lengths(target: list[int], first: int, last: int) -> None:
            pre = [read(4) for _ in range(_PRETREE_SIZE)]
            table, bits = _decode_table(pre, "pre-tree")
            x = first
            while x < last:
                z = symbol(table, bits, "pre-tree")
                if z == 17:
                    run = read(4) + 4
                    value = None
                elif z == 18:
                    run = read(5) + 20
                    value = None
                elif z == 19:
                    run = read(1) + 4
                    z = symbol(table, bits, "pre-tree")
                    value = (target[x] - z) % 17
                else:
                    run = 1
                    value = (target[x] - z) % 17
                if x + run > last:
                    raise ValueError(f"LZX: {self.name}: code lengths overrun the tree")
                for i in range(x, x + run):
                    target[i] = 0 if value is None else value
                x += run

        if not self.header_read:
            if read(1):
                self.intel_size = (read(16) << 16) | read(16)
            self.header_read = True

        main_table: list[int] = []
        main_bits = 0
        length_table: list[int] = []
        length_bits = 0
        aligned_table: list[int] = []
        aligned_bits = 0
        if self.block_remaining and self.block_type != _UNCOMPRESSED:
            main_table, main_bits = self._main
            length_table, length_bits = self._length
            aligned_table, aligned_bits = self._aligned

        while len(history) < out_end:
            if self.block_remaining == 0:
                if self.block_type == _UNCOMPRESSED:
                    if self.block_length & 1:
                        pos += 1  # pad byte after an odd-length raw block
                    buf = left = 0
                self.block_type = read(3)
                self.block_length = self.block_remaining = (read(16) << 8) | read(8)
                kind = self.block_type
                if kind == _ALIGNED:
                    aligned = [read(3) for _ in range(_ALIGNED_SIZE)]
                    aligned_table, aligned_bits = _decode_table(aligned, "aligned")
                if kind in (_VERBATIM, _ALIGNED):
                    lengths(self.main_lengths, 0, _NUM_CHARS)
                    lengths(self.main_lengths, _NUM_CHARS, self.main_size)
                    main_table, main_bits = _decode_table(self.main_lengths, "main")
                    if self.main_lengths[0xE8]:
                        self.intel_started = True
                    lengths(self.length_lengths, 0, _LENGTH_SIZE)
                    length_table, length_bits = _decode_table(
                        self.length_lengths, "length"
                    )
                    self._main = (main_table, main_bits)
                    self._length = (length_table, length_bits)
                    self._aligned = (aligned_table, aligned_bits)
                elif kind == _UNCOMPRESSED:
                    self.intel_started = True
                    # Align to 16 bits: drop the partial word; an aligned
                    # stream carries one 16-bit word of padding instead.
                    pos -= 2 * (left // 16)
                    if left % 16 == 0:
                        pos += 2
                    buf = left = 0
                    if pos + 12 > limit:
                        raise ValueError(
                            f"LZX: {self.name}: compressed data ends early"
                        )
                    r0, r1, r2 = struct.unpack_from("<3I", src, pos)
                    pos += 12
                else:
                    raise ValueError(f"LZX: {self.name}: unknown block type {kind}")
                if self.block_remaining == 0:
                    raise ValueError(f"LZX: {self.name}: an empty block")

            run_start = len(history)
            run_end = run_start + min(self.block_remaining, out_end - run_start)
            kind = self.block_type
            if kind == _UNCOMPRESSED:
                take = run_end - run_start
                if pos + take > end - start:
                    raise ValueError(f"LZX: {self.name}: compressed data ends early")
                history += src[pos : pos + take]
                pos += take
            else:
                while len(history) < run_end:
                    main = symbol(main_table, main_bits, "main")
                    if main < _NUM_CHARS:
                        history.append(main)
                        continue
                    main -= _NUM_CHARS
                    match_length = main & 7
                    if match_length == 7:
                        match_length += symbol(length_table, length_bits, "length")
                    match_length += _MIN_MATCH
                    slot = main >> 3
                    if slot > 2:
                        extra = _EXTRA_BITS[slot]
                        offset = _POSITION_BASE[slot] - 2
                        if kind == _ALIGNED and extra >= 3:
                            if extra > 3:
                                offset += read(extra - 3) << 3
                            offset += symbol(aligned_table, aligned_bits, "aligned")
                        else:
                            offset += read(extra)
                        r2, r1, r0 = r1, r0, offset
                    elif slot == 0:
                        offset = r0
                    elif slot == 1:
                        offset = r1
                        r1, r0 = r0, r1
                    else:
                        offset = r2
                        r2, r0 = r0, r2
                    here = len(history)
                    source = here - offset
                    if offset <= 0 or source < 0 or offset > self.window:
                        raise ValueError(f"LZX: {self.name}: a match before the data")
                    if offset >= match_length:
                        history += history[source : source + match_length]
                    else:
                        pattern = bytes(history[source:here])
                        reps = -(-match_length // offset)
                        history += (pattern * reps)[:match_length]
            produced = len(history) - run_start
            self.block_remaining -= produced
            if self.block_remaining < 0:
                raise ValueError(f"LZX: {self.name}: a match runs past its block")

        self.r = [r0, r1, r2]
        out = bytes(history[out_start:out_end])
        # A match may run past the frame: those bytes open the next frame.
        self.cursor = out_end
        # Keep one window of history for back-references.
        if len(history) > 2 * self.window:
            drop = len(history) - self.window
            del history[:drop]
            self.cursor -= drop
        out = self._undo_e8(out)
        self.produced += out_len
        return out

    def _undo_e8(self, out: bytes) -> bytes:
        """Undo the x86 ``E8`` call translation of one frame, if it is on."""
        position = self.intel_position
        self.intel_position += len(out)
        if not (self.intel_started and self.intel_size) or len(out) <= 10:
            return out
        size = self.intel_size
        data = bytearray(out)
        i = 0
        stop = len(data) - 10
        while i < stop:
            i = data.find(0xE8, i, stop)
            if i < 0:
                break
            absolute = struct.unpack_from("<i", data, i + 1)[0]
            current = position + i
            if -current <= absolute < size:
                relative = absolute - current if absolute >= 0 else absolute + size
                struct.pack_into("<i", data, i + 1, relative)
            i += 5
        return bytes(data)
