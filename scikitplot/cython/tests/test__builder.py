# scikitplot/cython/tests/test__builder.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :mod:`~scikitplot.cython._builder`.

Covers
------
- ``_to_path()``                    : str / bytes / Path / tilde / absolute
- ``_normalize_extra_sources()``    : None, empty, valid C/C++, bad suffix,
                                      missing file, duplicate, bytes path
- ``_support_files_digest()``       : str/bytes content, sorted, invalid name
- ``_support_paths_digest()``       : file hash, missing file, sorted, determinism
- ``_validate_support_filename()``  : valid / empty / slash / backslash / bad char
- ``_write_support_files()``        : str/bytes content, None no-op, collision
- ``_copy_support_paths()``         : copy, None no-op, missing, dup basename,
                                      reserved collision
- ``_copy_extra_sources()``         : C file copy, None → empty, invalid suffix
- ``_ensure_package()``             : simple / dotted / already-registered
- ``_utc_now_iso()``                : format, no microseconds
- ``_find_built_extension()``       : found / absent
- ``_find_annotation()``            : found / absent
- ``_clean_build_artifacts()``      : removes ext/C, keeps reserved, skips dirs
- ``DEFAULT_COMPILER_DIRECTIVES``   : language_level, embedsignature, is mapping
- ``compile_and_load()``            : smoke tests (requires Cython toolchain)
- Scenario 5 — C-API sources        : ``collect_c_api_sources``,
                                      ``collect_header_dirs``
"""
from __future__ import annotations

import os
import sys
import tempfile
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import pytest

from .._builder import (
    DEFAULT_COMPILER_DIRECTIVES,
    _ALLOWED_EXTRA_SOURCE_SUFFIXES,
    _ALLOWED_SUPPORT_NAME,
    _clean_build_artifacts,
    _copy_extra_sources,
    _copy_support_paths,
    _ensure_package,
    _find_annotation,
    _find_built_extension,
    _normalize_extra_sources,
    _support_files_digest,
    _support_paths_digest,
    _to_path,
    _utc_now_iso,
    _validate_support_filename,
    _write_support_files,
)
from .._cache import (
    is_valid_key,
)
from .._custom_compiler import (
    CApiCompiler,
    CustomCompilerProtocol,
    c_api_prereqs,
    collect_c_api_sources,
    collect_header_dirs,
)

requires_compiler = pytest.mark.skipif(
    not (
        __import__("importlib").util.find_spec("Cython") is not None
        and __import__("importlib").util.find_spec("setuptools") is not None
    ),
    reason="Cython and setuptools required for compiler tests",
)


class TestNormalizeExtraSources:
    """Tests for :func:`~scikitplot.cython._builder._normalize_extra_sources`."""

    def test_none_returns_empty(self) -> None:
        assert _normalize_extra_sources(None) == []

    def test_empty_list_returns_empty(self) -> None:
        assert _normalize_extra_sources([]) == []

    def test_valid_c_file(self, tmp_path: Path) -> None:
        src = tmp_path / "helper.c"
        src.write_text("int x = 0;", encoding="utf-8")
        result = _normalize_extra_sources([src])
        assert result[0] == src.resolve()

    @pytest.mark.parametrize("ext", [".c", ".cc", ".cpp", ".cxx"])
    def test_all_allowed_suffixes(self, tmp_path: Path, ext: str) -> None:
        src = tmp_path / f"helper{ext}"
        src.write_text("", encoding="utf-8")
        result = _normalize_extra_sources([src])
        assert len(result) == 1

    def test_invalid_suffix_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "helper.py"
        bad.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="extra source"):
            _normalize_extra_sources([bad])

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _normalize_extra_sources([tmp_path / "nonexistent.c"])

    def test_duplicate_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "helper.c"
        src.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate"):
            _normalize_extra_sources([src, src])


class TestNormalizeExtraSourcesBytesPath:
    """_normalize_extra_sources must handle bytes paths via os.fsdecode."""

    def test_bytes_path_accepted(self, tmp_path: Path) -> None:
        from .._builder import _normalize_extra_sources

        src = tmp_path / "extra.c"
        src.write_text("int x = 1;", encoding="utf-8")
        result = _normalize_extra_sources([os.fsencode(str(src))])
        assert len(result) == 1
        assert result[0] == src.resolve()


class TestSupportFilesDigest:
    """Tests for :func:`~scikitplot.cython._builder._support_files_digest`."""

    def test_none_returns_empty(self) -> None:
        assert _support_files_digest(None) == []

    def test_str_content(self) -> None:
        result = _support_files_digest({"header.pxd": "cdef int x"})
        assert len(result) == 1
        name, digest = result[0]
        assert name == "header.pxd"
        assert len(digest) == 64

    def test_bytes_content(self) -> None:
        result = _support_files_digest({"data.bin": b"\x00\x01\x02"})
        assert len(result) == 1

    def test_sorted_by_name(self) -> None:
        result = _support_files_digest({"z.pxd": "z", "a.pxd": "a"})
        assert result[0][0] == "a.pxd"
        assert result[1][0] == "z.pxd"

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError):
            _support_files_digest({"../escape.pxd": "bad"})


class TestSupportPathsDigest:
    """Tests for :func:`~scikitplot.cython._builder._support_paths_digest`."""

    def test_none_returns_empty(self) -> None:
        assert _support_paths_digest(None) == []

    def test_valid_file(self, tmp_path: Path) -> None:
        f = tmp_path / "helper.pxi"
        f.write_bytes(b"cdef int y")
        result = _support_paths_digest([f])
        assert len(result) == 1
        assert result[0][0] == "helper.pxi"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _support_paths_digest([tmp_path / "ghost.pxi"])

    def test_sorted_by_name(self, tmp_path: Path) -> None:
        fa = tmp_path / "a.pxi"
        fz = tmp_path / "z.pxi"
        fa.write_bytes(b"a")
        fz.write_bytes(b"z")
        result = _support_paths_digest([fz, fa])
        assert result[0][0] == "a.pxi"


class TestValidateSupportFilename:
    """Tests for :func:`~scikitplot.cython._builder._validate_support_filename`."""

    @pytest.mark.parametrize(
        "name",
        ["helper.pxd", "common.pxi", "header.h", "file_1.c", "a-b.c"],
    )
    def test_valid_names(self, name: str) -> None:
        _validate_support_filename(name)  # must not raise

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _validate_support_filename("")

    def test_slash_raises(self) -> None:
        with pytest.raises(ValueError, match="directories"):
            _validate_support_filename("sub/file.pxd")

    def test_backslash_raises(self) -> None:
        with pytest.raises(ValueError, match="directories"):
            _validate_support_filename("sub\\file.pxd")

    def test_unsupported_char_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported character"):
            _validate_support_filename("file with space.pxd")


class TestWriteSupportFiles:
    """Tests for :func:`~scikitplot.cython._builder._write_support_files`."""

    def test_writes_str_content(self, tmp_path: Path) -> None:
        _write_support_files(tmp_path, {"common.pxi": "cdef int y"}, reserved=set())
        assert (tmp_path / "common.pxi").read_text() == "cdef int y"

    def test_writes_bytes_content(self, tmp_path: Path) -> None:
        _write_support_files(tmp_path, {"data.bin": b"\x01\x02"}, reserved=set())
        assert (tmp_path / "data.bin").read_bytes() == b"\x01\x02"

    def test_none_is_no_op(self, tmp_path: Path) -> None:
        _write_support_files(tmp_path, None, reserved=set())

    def test_collision_with_reserved_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="reserved"):
            _write_support_files(
                tmp_path, {"main.pyx": "x"}, reserved={"main.pyx"}
            )


class TestCopySupportPaths:
    """Tests for :func:`~scikitplot.cython._builder._copy_support_paths`."""

    def test_copies_file(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        f = src_dir / "helper.pxi"
        f.write_bytes(b"content")
        dst_dir = tmp_path / "dst"
        dst_dir.mkdir()
        _copy_support_paths(dst_dir, [f], reserved=set())
        assert (dst_dir / "helper.pxi").read_bytes() == b"content"

    def test_none_is_no_op(self, tmp_path: Path) -> None:
        _copy_support_paths(tmp_path, None, reserved=set())

    def test_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _copy_support_paths(tmp_path, [tmp_path / "ghost.pxi"], reserved=set())

    def test_duplicate_basename_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        f1 = src / "dup.pxi"
        f2 = src / "dup.pxi"
        f1.write_bytes(b"a")
        dst = tmp_path / "dst"
        dst.mkdir()
        with pytest.raises(ValueError, match="duplicate"):
            _copy_support_paths(dst, [f1, f2], reserved=set())

    def test_reserved_collision_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        f = src / "reserved.pxi"
        f.write_bytes(b"a")
        with pytest.raises(ValueError, match="reserved"):
            _copy_support_paths(tmp_path, [f], reserved={"reserved.pxi"})


class TestCopyExtraSources:
    """Tests for :func:`~scikitplot.cython._builder._copy_extra_sources`."""

    def test_copies_c_file(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        c = src_dir / "helper.c"
        c.write_text("int y = 0;", encoding="utf-8")
        dst_dir = tmp_path / "dst"
        dst_dir.mkdir()
        out = _copy_extra_sources(dst_dir, [c], reserved=set())
        assert len(out) == 1
        assert (dst_dir / "helper.c").exists()

    def test_none_returns_empty(self, tmp_path: Path) -> None:
        assert _copy_extra_sources(tmp_path, None, reserved=set()) == []

    def test_invalid_suffix_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.py"
        bad.write_bytes(b"")
        with pytest.raises(ValueError, match="extra source"):
            _copy_extra_sources(tmp_path, [bad], reserved=set())


class TestEnsurePackage:
    """Tests for :func:`~scikitplot.cython._builder._ensure_package`."""

    def test_simple_package_registered(self, tmp_path: Path) -> None:
        pkg_dir = tmp_path / "mypkg"
        pkg_dir.mkdir()
        _ensure_package("mypkg", pkg_dir)
        assert "mypkg" in sys.modules

    def test_dotted_package_registered(self, tmp_path: Path) -> None:
        leaf = tmp_path / "a" / "b"
        leaf.mkdir(parents=True)
        _ensure_package("a.b", leaf)
        assert "a" in sys.modules
        assert "a.b" in sys.modules

    def test_already_registered_skipped(self, tmp_path: Path) -> None:
        pkg_dir = tmp_path / "existing_pkg"
        pkg_dir.mkdir()
        _ensure_package("existing_pkg", pkg_dir)
        mod = sys.modules["existing_pkg"]
        _ensure_package("existing_pkg", pkg_dir)  # second call must not replace
        assert sys.modules["existing_pkg"] is mod


class TestBuilderUtcNowIso:
    """Tests for :func:`~scikitplot.cython._builder._utc_now_iso`."""

    def test_format(self) -> None:
        ts = _utc_now_iso()
        assert ts.endswith("Z")
        assert "T" in ts

    def test_no_microseconds(self) -> None:
        ts = _utc_now_iso()
        # Should not contain fractional seconds
        assert "." not in ts


class TestFindBuiltExtension:
    """Tests for :func:`~scikitplot.cython._builder._find_built_extension`."""

    def test_finds_existing_extension(self, tmp_path: Path) -> None:
        so = tmp_path / f"mymod{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"ELF")
        result = _find_built_extension(tmp_path, "mymod")
        assert result == so

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        result = _find_built_extension(tmp_path, "ghost")
        assert result is None


class TestFindAnnotation:
    """Tests for :func:`~scikitplot.cython._builder._find_annotation`."""

    def test_finds_html_file(self, tmp_path: Path) -> None:
        html = tmp_path / "mymod.html"
        html.write_text("<html/>", encoding="utf-8")
        result = _find_annotation(tmp_path, "mymod")
        assert result == html

    def test_returns_none_when_absent(self, tmp_path: Path) -> None:
        result = _find_annotation(tmp_path, "ghost")
        assert result is None


class TestCleanBuildArtifacts:
    """Tests for :func:`~scikitplot.cython._builder._clean_build_artifacts`."""

    def test_removes_extension_artifact(self, tmp_path: Path) -> None:
        so = tmp_path / f"mymod{EXTENSION_SUFFIXES[0]}"
        so.write_bytes(b"ELF")
        _clean_build_artifacts(build_dir=tmp_path, name="mymod", keep=set())
        assert not so.exists()

    def test_keeps_reserved_files(self, tmp_path: Path) -> None:
        pyx = tmp_path / "mymod.pyx"
        pyx.write_text("def f(): pass", encoding="utf-8")
        _clean_build_artifacts(build_dir=tmp_path, name="mymod", keep={"mymod.pyx"})
        assert pyx.exists()

    def test_removes_c_file(self, tmp_path: Path) -> None:
        c_file = tmp_path / "mymod.c"
        c_file.write_text("/* generated */", encoding="utf-8")
        _clean_build_artifacts(build_dir=tmp_path, name="mymod", keep=set())
        assert not c_file.exists()

    def test_does_not_remove_unrelated_files(self, tmp_path: Path) -> None:
        unrelated = tmp_path / "other_module.c"
        unrelated.write_text("/* unrelated */", encoding="utf-8")
        _clean_build_artifacts(build_dir=tmp_path, name="mymod", keep=set())
        assert unrelated.exists()

    def test_skips_directories(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        _clean_build_artifacts(build_dir=tmp_path, name="mymod", keep=set())
        assert subdir.exists()


class TestDefaultCompilerDirectives:
    """Sanity-check canonical default directives."""

    def test_language_level_3(self) -> None:
        assert DEFAULT_COMPILER_DIRECTIVES["language_level"] == 3

    def test_embedsignature_true(self) -> None:
        assert DEFAULT_COMPILER_DIRECTIVES["embedsignature"] is True

    def test_is_mapping(self) -> None:
        assert hasattr(DEFAULT_COMPILER_DIRECTIVES, "__getitem__")


@requires_compiler
class TestCompileAndLoadSmoke:
    """
    Smoke tests that require a working C compiler + Cython.

    These are deliberately minimal — they verify the end-to-end pipeline works
    without testing every flag combination.
    """

    _SIMPLE_PYX = """\
# cython: language_level=3
def add(int a, int b):
    return a + b
"""

    def test_compile_and_load_returns_module(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load

        mod = compile_and_load(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        assert hasattr(mod, "add")
        assert mod.add(2, 3) == 5

    def test_compile_uses_cache_on_second_call(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load_result

        r1 = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        r2 = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        assert r1.key == r2.key
        assert r2.used_cache is True

    def test_force_rebuild_recompiles(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load_result

        r1 = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        r2 = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            force_rebuild=True,
            numpy_support=False,
            verbose=-1,
        )
        assert r2.used_cache is False

    def test_build_result_fields_populated(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load_result

        r = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        assert is_valid_key(r.key)
        assert r.artifact_path.exists()
        assert r.build_dir.exists()
        assert r.module_name != ""
        assert r.source_sha256 is not None
        assert r.fingerprint is not None

    def test_profile_release_works(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load

        mod = compile_and_load(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            profile="release",
            numpy_support=False,
            verbose=-1,
        )
        assert mod.add(10, 20) == 30

    def test_explicit_module_name(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load_result

        r = compile_and_load_result(
            self._SIMPLE_PYX,
            module_name="explicit_adder",
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        assert r.module_name == "explicit_adder"

    def test_import_cached_roundtrip(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load_result, import_cached

        r = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        mod2 = import_cached(r.key, cache_dir=tmp_cache)
        assert hasattr(mod2, "add")

    def test_pin_and_import_pinned(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load_result, import_pinned
        from .._pins import pin

        r = compile_and_load_result(
            self._SIMPLE_PYX,
            cache_dir=tmp_cache,
            numpy_support=False,
            verbose=-1,
        )
        pin(r.key, alias="my_adder", cache_dir=tmp_cache)
        mod = import_pinned("my_adder", cache_dir=tmp_cache)
        assert mod.add(1, 2) == 3

    def test_invalid_pyx_raises_runtime_error(self, tmp_cache: Path) -> None:
        from .._public import compile_and_load

        bad_source = "this is not valid cython @@@@"
        with pytest.raises(RuntimeError):
            compile_and_load(
                bad_source,
                cache_dir=tmp_cache,
                numpy_support=False,
                verbose=-1,
            )


class TestScenario5CApiSources:
    """
    Master scenario: C/C++ source and header collection.

    Notes
    -----
    **User note**: :func:`collect_c_api_sources` accepts files, directories,
    or glob patterns.  :func:`collect_header_dirs` gives you the
    ``include_dirs`` list from a source tree automatically.
    **Dev note**: all paths are returned as absolute, deduplicated, sorted.
    """

    # --- c_api_prereqs ---

    def test_c_api_prereqs_has_required_keys(self) -> None:
        result = c_api_prereqs()
        for key in ("cython", "numpy", "setuptools"):
            assert key in result

    def test_c_api_prereqs_all_ok_are_bool(self) -> None:
        result = c_api_prereqs()
        for key, val in result.items():
            assert isinstance(val["ok"], bool)

    # --- Scenario 5a: single file ---

    def test_single_c_file(self, tmp_path: Path) -> None:
        f = tmp_path / "foo.c"
        f.write_text("int foo() { return 1; }")
        result = collect_c_api_sources(str(f))
        assert len(result) == 1
        assert result[0] == f.resolve()

    def test_single_cpp_file(self, tmp_path: Path) -> None:
        f = tmp_path / "bar.cpp"
        f.write_text("int bar() { return 2; }")
        result = collect_c_api_sources(str(f))
        assert len(result) == 1
        assert result[0].suffix == ".cpp"

    def test_single_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            collect_c_api_sources(str(tmp_path / "nonexistent.c"))

    def test_single_file_bad_suffix_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.txt"
        f.write_text("not a C file")
        with pytest.raises(ValueError, match="unsupported source suffix"):
            collect_c_api_sources(str(f))

    def test_header_file_ignored_in_default_suffixes(self, tmp_path: Path) -> None:
        h = tmp_path / "foo.h"
        h.write_text("#pragma once")
        with pytest.raises(ValueError, match="unsupported source suffix"):
            collect_c_api_sources(str(h))

    # --- Scenario 5b: multiple files ---

    def test_multiple_explicit_files(self, tmp_path: Path) -> None:
        files = []
        for name in ("a.c", "b.cpp", "c.cxx"):
            f = tmp_path / name
            f.write_text("void stub() {}")
            files.append(str(f))
        result = collect_c_api_sources(*files)
        assert len(result) == 3

    def test_deduplication_of_same_file_twice(self, tmp_path: Path) -> None:
        f = tmp_path / "dup.c"
        f.write_text("int dup() { return 0; }")
        result = collect_c_api_sources(str(f), str(f))
        assert len(result) == 1

    # --- Scenario 5c: directory ---

    def test_directory_collects_all_c_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.c").write_text("int a() {}")
        (tmp_path / "b.cpp").write_text("int b() {}")
        (tmp_path / "README.md").write_text("# docs")
        result = collect_c_api_sources(str(tmp_path))
        names = {p.name for p in result}
        assert "a.c" in names
        assert "b.cpp" in names
        assert "README.md" not in names

    def test_directory_skips_headers_by_default(self, tmp_path: Path) -> None:
        (tmp_path / "a.c").write_text("int a() {}")
        (tmp_path / "mylib.h").write_text("#pragma once")
        result = collect_c_api_sources(str(tmp_path))
        names = {p.name for p in result}
        assert "mylib.h" not in names

    # --- Scenario 5d: nested folder tree ---

    def test_recursive_directory(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.c").write_text("int r() {}")
        (sub / "child.c").write_text("int c() {}")
        result = collect_c_api_sources(str(tmp_path), recursive=True)
        names = {p.name for p in result}
        assert "root.c" in names
        assert "child.c" in names

    def test_non_recursive_directory_excludes_subdirs(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.c").write_text("int r() {}")
        (sub / "child.c").write_text("int c() {}")
        result = collect_c_api_sources(str(tmp_path), recursive=False)
        names = {p.name for p in result}
        assert "root.c" in names
        assert "child.c" not in names

    # --- exclude_patterns ---

    def test_exclude_pattern_applied(self, tmp_path: Path) -> None:
        (tmp_path / "main.c").write_text("int main() {}")
        (tmp_path / "test_helper.c").write_text("int helper() {}")
        result = collect_c_api_sources(str(tmp_path), exclude_patterns=["test_*.c"])
        names = {p.name for p in result}
        assert "main.c" in names
        assert "test_helper.c" not in names

    def test_custom_suffixes(self, tmp_path: Path) -> None:
        (tmp_path / "a.c").write_text("int a() {}")
        (tmp_path / "b.f90").write_text("real b")
        result = collect_c_api_sources(
            str(tmp_path),
            suffixes=frozenset({".f90"}),
        )
        names = {p.name for p in result}
        assert "b.f90" in names
        assert "a.c" not in names

    def test_results_are_absolute_paths(self, tmp_path: Path) -> None:
        (tmp_path / "x.c").write_text("int x() {}")
        result = collect_c_api_sources(str(tmp_path))
        for p in result:
            assert p.is_absolute()

    # --- collect_header_dirs ---

    def test_collect_header_dirs_single_dir(self, tmp_path: Path) -> None:
        (tmp_path / "mylib.h").write_text("#pragma once")
        result = collect_header_dirs(str(tmp_path))
        assert len(result) == 1
        assert result[0] == tmp_path.resolve()

    def test_collect_header_dirs_nested(self, tmp_path: Path) -> None:
        sub = tmp_path / "include"
        sub.mkdir()
        (sub / "api.h").write_text("#pragma once")
        result = collect_header_dirs(str(tmp_path), recursive=True)
        assert sub.resolve() in result

    def test_collect_header_dirs_deduplicated(self, tmp_path: Path) -> None:
        (tmp_path / "a.h").write_text("#pragma once")
        (tmp_path / "b.h").write_text("#pragma once")
        result = collect_header_dirs(str(tmp_path))
        assert result.count(tmp_path.resolve()) == 1

    def test_collect_header_dirs_no_headers_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "main.c").write_text("int main() {}")
        result = collect_header_dirs(str(tmp_path))
        assert result == []

    def test_collect_header_dirs_results_sorted(self, tmp_path: Path) -> None:
        for sub in ("z_inc", "a_inc"):
            d = tmp_path / sub
            d.mkdir()
            (d / "h.h").write_text("#pragma once")
        result = collect_header_dirs(str(tmp_path), recursive=True)
        assert result == sorted(result)

    def test_collect_header_dirs_explicit_file(self, tmp_path: Path) -> None:
        h = tmp_path / "mylib.h"
        h.write_text("#pragma once")
        result = collect_header_dirs(str(h))
        assert tmp_path.resolve() in result

    def test_c_api_compiler_name(self) -> None:
        cc = CApiCompiler()
        assert cc.name == "custom_c_api"

    def test_c_api_compiler_satisfies_protocol(self) -> None:
        cc = CApiCompiler()
        assert isinstance(cc, CustomCompilerProtocol)
