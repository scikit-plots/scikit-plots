"""Run 137 — bounded, tree-preserving ZIP edit workspace."""
from __future__ import annotations

from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._paths import RUNTIME_ROOT

import ast
import hashlib
import io
import os
import pathlib
import stat
import struct
import sys
import zipfile
from dataclasses import replace

import pytest

_PROXY = RUNTIME_ROOT / "_hf_spaces_proxy"
if str(_PROXY) not in sys.path:
    sys.path.insert(0, str(_PROXY))

from _utils import _zip_workspace as zw  # noqa: E402


def _extra(field_id: int, payload: bytes) -> bytes:
    return struct.pack("<HH", field_id, len(payload)) + payload


def _info(
    name: str,
    *,
    compression: int = zipfile.ZIP_DEFLATED,
    directory: bool | None = None,
    mode: int | None = None,
    extra: bytes = b"",
    comment: bytes = b"entry-comment",
) -> zipfile.ZipInfo:
    is_dir = name.endswith("/") if directory is None else directory
    info = zipfile.ZipInfo(name, date_time=(2026, 9, 4, 12, 34, 56))
    info.compress_type = compression
    info.comment = comment
    info.extra = extra
    info.create_system = 3
    if mode is None:
        mode = (stat.S_IFDIR | 0o755) if is_dir else (stat.S_IFREG | 0o640)
    info.external_attr = mode << 16
    info.internal_attr = 1 if not is_dir else 0
    return info


def _zip(rows, *, archive_comment=b"archive-comment") -> io.BytesIO:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", allowZip64=False) as zf:
        zf.comment = archive_comment
        for row in rows:
            first, data = row
            if isinstance(first, zipfile.ZipInfo):
                info = first
            else:
                name = first
                info = _info(name, directory=name.endswith("/"))
            zf.writestr(info, data)
    out.seek(0)
    return out


def _read_all(fileobj) -> tuple[list[zipfile.ZipInfo], dict[str, bytes], bytes]:
    fileobj.seek(0)
    with zipfile.ZipFile(fileobj, "r") as zf:
        infos = zf.infolist()
        payloads = {info.filename: zf.read(info) for info in infos if not info.is_dir()}
        return infos, payloads, zf.comment


def _small_limits(**changes) -> zw.ZipWorkspaceLimits:
    base = zw.ZipWorkspaceLimits(
        max_entries=50,
        max_entry_uncompressed_bytes=1024,
        max_total_uncompressed_bytes=4096,
        max_replacement_total_bytes=2048,
        max_compression_ratio=500.0,
        max_source_bytes=1024 * 1024,
        chunk_bytes=64,
        output_spool_bytes=512,
        max_path_chars=256,
    )
    return replace(base, **changes)


def test_nested_tree_fidelity_metadata_and_bounded_receipt() -> None:
    semantic_extra = _extra(0x5455, b"\x01\x00\x00\x00\x00")
    src = _zip([
        (_info("pkg/", directory=True, extra=semantic_extra), b""),
        (_info("pkg/a.py", extra=semantic_extra), b"print('a')\n"),
        (_info("pkg/data.bin", compression=zipfile.ZIP_STORED, extra=semantic_extra), b"\x00\xff\x01"),
    ])
    source_hash = hashlib.sha256(src.getvalue()).hexdigest()
    with zw.rewrite_zip_workspace(src, {"pkg/a.py": b"print('changed')\n"}) as artifact:
        infos, payloads, comment = _read_all(artifact.file)
        assert [i.filename for i in infos] == ["pkg/", "pkg/a.py", "pkg/data.bin"]
        assert payloads["pkg/a.py"] == b"print('changed')\n"
        assert payloads["pkg/data.bin"] == b"\x00\xff\x01"
        assert comment == b"archive-comment"
        assert infos[1].date_time == (2026, 9, 4, 12, 34, 56)
        assert infos[1].comment == b"entry-comment"
        assert infos[1].extra == semantic_extra
        assert infos[1].external_attr == (_info("x").external_attr)
        receipt = artifact.receipt.as_dict()
        assert receipt == {
            "source_sha256": source_hash,
            "output_sha256": artifact.receipt.output_sha256,
            "entry_count": 3,
            "changed_paths": ["pkg/a.py"],
            "unchanged_count": 2,
            "tree_preserved": True,
            "unchanged_content_preserved": True,
            "metadata_preserved": True,
        }
        assert len(artifact.receipt.output_sha256) == 64


def test_205_file_stress_changes_only_first_and_last() -> None:
    rows = [(f"root/d{i // 25:02d}/f{i:03d}.txt", f"payload-{i}".encode()) for i in range(205)]
    src = _zip(rows)
    names = [name for name, _ in rows]
    with zw.rewrite_zip_workspace(src, {names[0]: b"FIRST", names[-1]: b"LAST"}) as artifact:
        infos, payloads, _ = _read_all(artifact.file)
        assert [i.filename for i in infos] == names
        assert payloads[names[0]] == b"FIRST"
        assert payloads[names[-1]] == b"LAST"
        assert payloads[names[102]] == b"payload-102"
        assert artifact.receipt.entry_count == 205
        assert artifact.receipt.changed_paths == (names[0], names[-1])
        assert artifact.receipt.unchanged_count == 203


def test_unknown_add_and_directory_replacement_are_rejected() -> None:
    src = _zip([("pkg/", b""), ("pkg/a.txt", b"a")])
    with pytest.raises(zw.ZipWorkspaceError, match="existing entry"):
        zw.rewrite_zip_workspace(src, {"pkg/new.txt": b"new"})
    with pytest.raises(zw.ZipWorkspaceError, match="directories"):
        zw.rewrite_zip_workspace(src, {"pkg/": b"no"})


def test_binary_non_text_replacement_is_exact() -> None:
    raw = b"\x89PNG\r\n\x1a\n\x00\xff\x10\x80"
    src = _zip([("assets/image.png", b"old")])
    with zw.rewrite_zip_workspace(src, {"assets/image.png": raw}) as artifact:
        _, payloads, _ = _read_all(artifact.file)
        assert payloads["assets/image.png"] == raw


def test_traversal_root_drive_backslash_and_ambiguous_segments_rejected() -> None:
    bad = ["../evil", "/root", "C:/evil", "a\\evil", "a/./evil", "a//evil"]
    for name in bad:
        src = _zip([(name, b"x")])
        with pytest.raises(zw.ZipWorkspaceError):
            zw.rewrite_zip_workspace(src, {})


def test_bidi_nul_and_control_paths_rejected() -> None:
    for name in ["safe\u202e.txt", "bad\x01.txt"]:
        src = _zip([(name, b"x")])
        with pytest.raises(zw.ZipWorkspaceError, match="control|bidi"):
            zw.rewrite_zip_workspace(src, {})
    # zipfile itself truncates embedded NUL names; authority must still never
    # accept such a caller replacement alias as a way to reach another path.
    src = _zip([("normal.txt", b"x")])
    with pytest.raises(zw.ZipWorkspaceError, match="existing entry"):
        zw.rewrite_zip_workspace(src, {"normal.txt\x00evil": b"y"})


def test_case_and_trailing_space_dot_aliases_rejected() -> None:
    src = _zip([("A.txt", b"a"), ("a.TXT", b"b")])
    with pytest.raises(zw.ZipWorkspaceError, match="aliased"):
        zw.rewrite_zip_workspace(src, {})
    for name in ["a.txt ", "a.txt."]:
        src = _zip([(name, b"x")])
        with pytest.raises(zw.ZipWorkspaceError, match="trailing-dot/space"):
            zw.rewrite_zip_workspace(src, {})


def test_unicode_nfc_nfd_aliases_rejected() -> None:
    src = _zip([("caf\u00e9.txt", b"a"), ("cafe\u0301.txt", b"b")])
    with pytest.raises(zw.ZipWorkspaceError, match="aliased"):
        zw.rewrite_zip_workspace(src, {})


def test_file_descendant_and_file_directory_collisions_rejected() -> None:
    src = _zip([("foo", b"file"), ("foo/bar.py", b"child")])
    with pytest.raises(zw.ZipWorkspaceError, match="file/descendant"):
        zw.rewrite_zip_workspace(src, {})
    src = _zip([("foo", b"file"), ("foo/", b"")])
    with pytest.raises(zw.ZipWorkspaceError, match="aliased"):
        zw.rewrite_zip_workspace(src, {})


def test_symlink_and_special_unix_files_rejected() -> None:
    symlink = _info("link", mode=stat.S_IFLNK | 0o777)
    src = _zip([(symlink, b"target")])
    with pytest.raises(zw.ZipWorkspaceError, match="symlinks"):
        zw.rewrite_zip_workspace(src, {})
    fifo = _info("pipe", mode=stat.S_IFIFO | 0o600)
    src = _zip([(fifo, b"")])
    with pytest.raises(zw.ZipWorkspaceError, match="special"):
        zw.rewrite_zip_workspace(src, {})


def test_unix_type_and_path_shape_inconsistency_rejected() -> None:
    fake_dir = _info("not-a-dir", directory=False, mode=stat.S_IFDIR | 0o755)
    src = _zip([(fake_dir, b"")])
    with pytest.raises(zw.ZipWorkspaceError, match="disagrees"):
        zw.rewrite_zip_workspace(src, {})
    fake_file = _info("looks-dir/", directory=True, mode=stat.S_IFREG | 0o644)
    src = _zip([(fake_file, b"")])
    with pytest.raises(zw.ZipWorkspaceError, match="disagrees"):
        zw.rewrite_zip_workspace(src, {})


def test_compression_bomb_ratio_rejected() -> None:
    src = _zip([("bomb.txt", b"0" * (1024 * 1024))])
    with pytest.raises(zw.ZipWorkspaceError, match="compression-ratio"):
        zw.rewrite_zip_workspace(src, {})


def test_entry_count_limit_rejected() -> None:
    src = _zip([("a", b"1"), ("b", b"2"), ("c", b"3")])
    with pytest.raises(zw.ZipWorkspaceError, match="entry-count"):
        zw.rewrite_zip_workspace(src, {}, limits=_small_limits(max_entries=2))


def test_per_entry_and_aggregate_uncompressed_limits_rejected() -> None:
    src = _zip([("a", b"12345")], archive_comment=b"")
    with pytest.raises(zw.ZipWorkspaceError, match="entry exceeds"):
        zw.rewrite_zip_workspace(src, {}, limits=_small_limits(max_entry_uncompressed_bytes=4))
    src = _zip([("a", b"1234"), ("b", b"5678")], archive_comment=b"")
    with pytest.raises(zw.ZipWorkspaceError, match="total uncompressed"):
        zw.rewrite_zip_workspace(src, {}, limits=_small_limits(max_total_uncompressed_bytes=7))


def test_replacement_per_entry_and_aggregate_limits_rejected() -> None:
    src = _zip([("a", b"1"), ("b", b"2")], archive_comment=b"")
    with pytest.raises(zw.ZipWorkspaceError, match="per-entry"):
        zw.rewrite_zip_workspace(src, {"a": b"12345"}, limits=_small_limits(max_entry_uncompressed_bytes=4))
    with pytest.raises(zw.ZipWorkspaceError, match="aggregate"):
        zw.rewrite_zip_workspace(
            src,
            {"a": b"123456", "b": b"abcdef"},
            limits=_small_limits(max_replacement_total_bytes=10),
        )


def test_resulting_output_growth_limit_is_checked_before_write() -> None:
    src = _zip([("a", b"1"), ("b", b"123456789")], archive_comment=b"")
    limits = _small_limits(max_total_uncompressed_bytes=20, max_entry_uncompressed_bytes=20)
    with pytest.raises(zw.ZipWorkspaceError, match="Rewritten ZIP"):
        zw.rewrite_zip_workspace(src, {"a": b"x" * 15}, limits=limits)


def test_zip64_extra_removed_but_semantic_extra_preserved() -> None:
    zip64 = _extra(0x0001, struct.pack("<Q", 123))
    semantic = _extra(0x5455, b"\x01\x00\x00\x00\x00")
    src = _zip([(_info("a.txt", extra=zip64 + semantic), b"hello")])
    with zw.rewrite_zip_workspace(src, {"a.txt": b"changed"}) as artifact:
        infos, _, _ = _read_all(artifact.file)
        assert infos[0].extra == semantic
        assert b"\x01\x00" not in infos[0].extra[:2]


def test_malformed_extra_tlv_is_rejected() -> None:
    malformed = struct.pack("<HH", 0x5455, 10) + b"tiny"
    src = _zip([(_info("a.txt", extra=malformed), b"hello")])
    with pytest.raises(zw.ZipWorkspaceError, match="malformed extra-field"):
        zw.rewrite_zip_workspace(src, {})


def test_malformed_zip_is_bounded_error() -> None:
    with pytest.raises(zw.ZipWorkspaceError, match="malformed|corrupt"):
        zw.rewrite_zip_workspace(io.BytesIO(b"not a zip"), {})


def _damage_first_payload(raw: bytes) -> bytes:
    data = bytearray(raw)
    off = data.find(b"PK\x03\x04")
    assert off >= 0
    name_len = struct.unpack_from("<H", data, off + 26)[0]
    extra_len = struct.unpack_from("<H", data, off + 28)[0]
    start = off + 30 + name_len + extra_len
    data[start + 1] ^= 0xFF
    return bytes(data)


def test_damaged_deflate_payload_maps_to_one_workspace_error() -> None:
    src = _zip([("a.txt", os.urandom(2048))], archive_comment=b"")
    damaged = io.BytesIO(_damage_first_payload(src.getvalue()))
    with pytest.raises(zw.ZipWorkspaceError, match="payload is malformed or corrupt"):
        zw.rewrite_zip_workspace(damaged, {})


def test_cancellation_closes_partial_output_and_surfaces_bounded_error() -> None:
    src = _zip([("a.bin", os.urandom(1024))], archive_comment=b"")
    calls = 0

    def cancel() -> bool:
        nonlocal calls
        calls += 1
        return calls > 3

    with pytest.raises(zw.ZipWorkspaceError, match="cancelled"):
        zw.rewrite_zip_workspace(src, {}, limits=_small_limits(max_entry_uncompressed_bytes=2048), should_cancel=cancel)


def test_concurrent_source_mutation_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    src = _zip([("a.txt", b"hello")], archive_comment=b"MUTABLE")
    original = zw._hash_source
    calls = 0

    def wrapped(source, *, limits, should_cancel):
        nonlocal calls
        calls += 1
        if calls == 2:
            raw = bytearray(src.getvalue())
            raw[-1] ^= 1  # mutate archive comment without changing structure/length
            src.seek(0)
            src.write(raw)
            src.seek(0)
        return original(source, limits=limits, should_cancel=should_cancel)

    monkeypatch.setattr(zw, "_hash_source", wrapped)
    with pytest.raises(zw.ZipWorkspaceError, match="changed after inspection"):
        zw.rewrite_zip_workspace(src, {})


def test_no_filesystem_extraction_api_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("filesystem extraction must never be used")

    monkeypatch.setattr(zipfile.ZipFile, "extract", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden)
    src = _zip([("pkg/a.txt", b"a")])
    with zw.rewrite_zip_workspace(src, {"pkg/a.txt": b"b"}) as artifact:
        _, payloads, _ = _read_all(artifact.file)
        assert payloads == {"pkg/a.txt": b"b"}

    tree = ast.parse(pathlib.Path(zw.__file__).read_text(encoding="utf-8"))
    forbidden_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"extract", "extractall"}
    ]
    assert forbidden_calls == []


def test_output_spools_to_disk_after_threshold() -> None:
    payload = os.urandom(4096)
    src = _zip([(_info("blob.bin", compression=zipfile.ZIP_STORED), payload)], archive_comment=b"")
    limits = _small_limits(
        max_entry_uncompressed_bytes=8192,
        max_total_uncompressed_bytes=8192,
        max_replacement_total_bytes=8192,
        output_spool_bytes=128,
    )
    with zw.rewrite_zip_workspace(src, {"blob.bin": payload[::-1]}, limits=limits) as artifact:
        assert getattr(artifact.file, "_rolled", False) is True


def test_source_size_and_unsupported_compression_are_fail_closed() -> None:
    src = _zip([("a.txt", b"hello")], archive_comment=b"")
    with pytest.raises(zw.ZipWorkspaceError, match="source byte limit"):
        zw.rewrite_zip_workspace(src, {}, limits=_small_limits(max_source_bytes=10))

    info = _info("b.txt", compression=zipfile.ZIP_BZIP2)
    src = _zip([(info, b"hello")], archive_comment=b"")
    with pytest.raises(zw.ZipWorkspaceError, match="unsupported compression"):
        zw.rewrite_zip_workspace(src, {})

# Large-contract case fragments are collected only through this canonical owner.
from scikitplot._externals._sphinx_ext._sphinx_ai_assistant.tests._case_loader import export_case_tests as _export_case_tests

_export_case_tests(globals(), package=__package__, case_package='_cases._zip_workspace', cases=('surgical',))
del _export_case_tests
