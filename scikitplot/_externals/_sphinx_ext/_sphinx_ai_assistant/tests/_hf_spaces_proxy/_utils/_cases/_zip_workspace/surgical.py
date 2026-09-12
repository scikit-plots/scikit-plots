"""Run 138 — surgical ZIP local-record preservation and central rebuild."""
from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import io
import os
import pathlib
import stat
import struct
import sys
import zipfile

import pytest

_PROXY = RUNTIME_ROOT / "_hf_spaces_proxy"
if str(_PROXY) not in sys.path:
    sys.path.insert(0, str(_PROXY))

from _utils import _zip_workspace as zw  # noqa: E402


def _extra(field_id: int, payload: bytes) -> bytes:
    return struct.pack("<HH", field_id, len(payload)) + payload


def _info(name: str, *, compression: int = zipfile.ZIP_DEFLATED, extra: bytes = b"") -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 9, 4, 16, 20, 10))
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = ((stat.S_IFDIR | 0o755) if name.endswith("/") else (stat.S_IFREG | 0o640)) << 16
    info.internal_attr = 0 if name.endswith("/") else 1
    info.comment = b"entry-meta"
    info.extra = extra
    return info


def _zip(rows, *, comment: bytes = b"run138") -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", allowZip64=False) as zf:
        zf.comment = comment
        for first, payload in rows:
            info = first if isinstance(first, zipfile.ZipInfo) else _info(first)
            zf.writestr(info, payload)
    return out.getvalue()


def _records(raw: bytes) -> tuple[dict[str, bytes], dict[str, int], list[str]]:
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        infos = zf.infolist()
        start_dir = zf.start_dir
        central_order = [info.filename for info in infos]
        physical = sorted(infos, key=lambda info: info.header_offset)
        records: dict[str, bytes] = {}
        offsets: dict[str, int] = {}
        for index, info in enumerate(physical):
            boundary = physical[index + 1].header_offset if index + 1 < len(physical) else start_dir
            records[info.filename] = raw[info.header_offset:boundary]
            offsets[info.filename] = info.header_offset
        return records, offsets, central_order


def _artifact_bytes(source: bytes, replacements: dict[str, bytes]) -> bytes:
    with zw.rewrite_zip_workspace(io.BytesIO(source), replacements) as artifact:
        artifact.file.seek(0)
        return artifact.file.read()


def test_unchanged_local_records_are_byte_identical_when_middle_entry_changes() -> None:
    source = _zip([
        ("pkg/a.bin", os.urandom(4096)),
        ("pkg/edit.txt", b"old"),
        ("pkg/z.bin", os.urandom(4096)),
    ])
    before, before_offsets, _ = _records(source)
    output = _artifact_bytes(source, {"pkg/edit.txt": b"replacement-" * 200})
    after, after_offsets, _ = _records(output)

    assert after["pkg/a.bin"] == before["pkg/a.bin"]
    assert after["pkg/z.bin"] == before["pkg/z.bin"]
    assert after["pkg/edit.txt"] != before["pkg/edit.txt"]
    assert after_offsets["pkg/z.bin"] != before_offsets["pkg/z.bin"]


def test_205_file_stress_preserves_all_203_untouched_local_records() -> None:
    rows = [(f"root/d{i // 25:02d}/f{i:03d}.txt", (f"payload-{i}-" * 3).encode()) for i in range(205)]
    source = _zip(rows)
    before, _, _ = _records(source)
    first = rows[0][0]
    last = rows[-1][0]
    output = _artifact_bytes(source, {first: b"FIRST", last: b"LAST-LONGER" * 20})
    after, _, _ = _records(output)

    for name, _ in rows:
        if name not in {first, last}:
            assert after[name] == before[name]


def test_unchanged_data_descriptor_record_is_preserved_exactly() -> None:
    class Unseekable(io.BytesIO):
        def seekable(self) -> bool:
            return False

        def seek(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise io.UnsupportedOperation

    out = Unseekable()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=False) as zf:
        zf.writestr("keep.txt", b"keep-" * 400)
        zf.writestr("edit.txt", b"old")
    source = out.getvalue()
    with zipfile.ZipFile(io.BytesIO(source)) as zf:
        assert zf.getinfo("keep.txt").flag_bits & 0x08

    before, _, _ = _records(source)
    output = _artifact_bytes(source, {"edit.txt": b"changed" * 100})
    after, _, _ = _records(output)
    assert after["keep.txt"] == before["keep.txt"]
    with zipfile.ZipFile(io.BytesIO(output)) as zf:
        assert zf.read("keep.txt") == b"keep-" * 400
        assert zf.read("edit.txt") == b"changed" * 100
        assert not (zf.getinfo("edit.txt").flag_bits & 0x08)


def test_unchanged_deflated_entry_is_not_recompressed(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _zip([
        (_info("keep.bin", compression=zipfile.ZIP_DEFLATED), os.urandom(3000)),
        (_info("edit.bin", compression=zipfile.ZIP_STORED), b"old"),
    ])

    def forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("untouched DEFLATE records must not be recompressed")

    monkeypatch.setattr(zw.zlib, "compressobj", forbidden)
    output = _artifact_bytes(source, {"edit.bin": b"new"})
    with zipfile.ZipFile(io.BytesIO(output)) as zf:
        assert zf.read("keep.bin")
        assert zf.read("edit.bin") == b"new"


def test_stale_zip64_transport_extra_is_removed_from_central_but_raw_local_record_survives() -> None:
    zip64 = _extra(0x0001, struct.pack("<Q", 123))
    semantic = _extra(0x5455, b"\x01\x00\x00\x00\x00")
    source = _zip([
        (_info("keep.txt", extra=zip64 + semantic), b"keep"),
        (_info("edit.txt"), b"old"),
    ])
    before, _, _ = _records(source)
    output = _artifact_bytes(source, {"edit.txt": b"new"})
    after, _, _ = _records(output)

    assert after["keep.txt"] == before["keep.txt"]
    with zipfile.ZipFile(io.BytesIO(output)) as zf:
        assert zf.getinfo("keep.txt").extra == semantic
        assert zf.read("keep.txt") == b"keep"


def test_self_extracting_preamble_is_not_propagated_but_entry_records_are_preserved() -> None:
    base = _zip([("keep.txt", b"keep"), ("edit.txt", b"old")])
    preamble = b"MZ\x90\x00UNTRUSTED-STUB" * 4
    source = preamble + base
    before, _, _ = _records(source)
    output = _artifact_bytes(source, {"edit.txt": b"new-value"})
    after, _, _ = _records(output)

    assert not output.startswith(preamble)
    assert after["keep.txt"] == before["keep.txt"]
    with zipfile.ZipFile(io.BytesIO(output)) as zf:
        assert zf.read("edit.txt") == b"new-value"


def test_trailing_junk_after_eocd_is_rejected_instead_of_propagated() -> None:
    source = _zip([("a.txt", b"a")]) + b"HIDDEN-TRAILER"
    with pytest.raises(zw.ZipWorkspaceError, match="end-of-central-directory"):
        zw.rewrite_zip_workspace(io.BytesIO(source), {})


def test_unsupported_general_purpose_flags_fail_closed() -> None:
    raw = bytearray(_zip([("a.txt", b"payload")]))
    local = raw.find(b"PK\x03\x04")
    central = raw.find(b"PK\x01\x02")
    assert local >= 0 and central >= 0
    local_flags = struct.unpack_from("<H", raw, local + 6)[0]
    central_flags = struct.unpack_from("<H", raw, central + 8)[0]
    struct.pack_into("<H", raw, local + 6, local_flags | 0x0010)
    struct.pack_into("<H", raw, central + 8, central_flags | 0x0010)
    with pytest.raises(zw.ZipWorkspaceError, match="unsupported general-purpose flags"):
        zw.rewrite_zip_workspace(io.BytesIO(raw), {})


def test_central_directory_keeps_logical_entry_order_after_offsets_move() -> None:
    source = _zip([("a.txt", b"A"), ("b.txt", b"B"), ("c.txt", b"C")])
    _, _, before_order = _records(source)
    output = _artifact_bytes(source, {"a.txt": os.urandom(8000)})
    _, _, after_order = _records(output)
    assert before_order == ["a.txt", "b.txt", "c.txt"]
    assert after_order == before_order


def test_corrupt_untouched_payload_still_fails_despite_raw_copy() -> None:
    source = bytearray(_zip([("keep.bin", os.urandom(4096)), ("edit.txt", b"old")]))
    with zipfile.ZipFile(io.BytesIO(source)) as zf:
        info = zf.getinfo("keep.bin")
        start = info.header_offset
    name_len = struct.unpack_from("<H", source, start + 26)[0]
    extra_len = struct.unpack_from("<H", source, start + 28)[0]
    payload_start = start + 30 + name_len + extra_len
    source[payload_start + 1] ^= 0xFF

    with pytest.raises(zw.ZipWorkspaceError, match="payload is malformed or corrupt"):
        zw.rewrite_zip_workspace(io.BytesIO(source), {"edit.txt": b"new"})


def test_replaced_entry_cannot_bypass_malformed_local_extra_validation() -> None:
    semantic = _extra(0x5455, b"ABCD")
    raw = bytearray(_zip([(_info("edit.txt", extra=semantic), b"old")]))
    local = raw.find(b"PK\x03\x04")
    assert local >= 0
    name_len = struct.unpack_from("<H", raw, local + 26)[0]
    extra_len = struct.unpack_from("<H", raw, local + 28)[0]
    assert extra_len == len(semantic)
    extra_start = local + 30 + name_len
    # Keep the local extra byte count stable, but make its first TLV claim a
    # payload larger than the local extra region. Central metadata stays valid.
    struct.pack_into("<H", raw, extra_start + 2, extra_len + 10)
    with pytest.raises(zw.ZipWorkspaceError, match="malformed extra-field"):
        zw.rewrite_zip_workspace(io.BytesIO(raw), {"edit.txt": b"replacement"})


def test_central_directory_digital_signature_or_hidden_record_is_rejected() -> None:
    raw = bytearray(_zip([("a.txt", b"a")]))
    eocd = raw.rfind(b"PK\x05\x06")
    assert eocd >= 0
    cd_size = struct.unpack_from("<I", raw, eocd + 12)[0]
    digital_signature = b"PK\x05\x05" + struct.pack("<H", 4) + b"SIGN"
    raw[eocd:eocd] = digital_signature
    new_eocd = eocd + len(digital_signature)
    struct.pack_into("<I", raw, new_eocd + 12, cd_size + len(digital_signature))

    # The workspace must never silently discard/re-authorize central-directory
    # adjunct records, whether the stdlib rejects them first or our structural
    # validator does.
    with pytest.raises(zw.ZipWorkspaceError, match="malformed|corrupt|unsupported central-directory"):
        zw.rewrite_zip_workspace(io.BytesIO(raw), {})


def test_directory_entry_with_hidden_payload_is_rejected() -> None:
    source = _zip([(_info("pkg/"), b"hidden-directory-body")])
    with pytest.raises(zw.ZipWorkspaceError, match="directory entries must not carry"):
        zw.rewrite_zip_workspace(io.BytesIO(source), {})


def test_server_snapshot_hash_detects_mutation_during_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    source = io.BytesIO(_zip([("a.txt", b"a")]))
    original = zw._snapshot_source

    def mutating_snapshot(src, *, limits, should_cancel):
        raw = bytearray(source.getvalue())
        raw[-1] ^= 1
        source.seek(0)
        source.write(raw)
        source.seek(0)
        return original(src, limits=limits, should_cancel=should_cancel)

    monkeypatch.setattr(zw, "_snapshot_source", mutating_snapshot)
    with pytest.raises(zw.ZipWorkspaceError, match="server snapshot"):
        zw.rewrite_zip_workspace(source, {})
