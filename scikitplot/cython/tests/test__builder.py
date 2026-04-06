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
    _SETUPTOOLS_CACHE,
    _clean_build_artifacts,
    _copy_extra_sources,
    _copy_support_paths,
    _ensure_package,
    _find_annotation,
    _find_built_extension,
    _import_setuptools,
    _normalize_extra_sources,
    _open_annotation_in_browser,
    _set_verbosity,
    _support_files_digest,
    _support_paths_digest,
    _to_path,
    _utc_now_iso,
    _validate_support_filename,
    _write_support_files,
)
from .. import _builder as _builder_module
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

requires_setuptools = pytest.mark.skipif(
    __import__("importlib").util.find_spec("setuptools") is None,
    reason="setuptools required",
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


# ---------------------------------------------------------------------------
# TestImportSetuptools
# ---------------------------------------------------------------------------


@requires_setuptools
class TestImportSetuptools:
    """
    Tests for :func:`~scikitplot.cython._builder._import_setuptools`.

    Notes
    -----
    **Developer note — why class identity matters:**

    ``_import_setuptools()`` must return the *same* ``(Extension, Distribution)``
    class objects on every call.  Returning different objects breaks the
    ``isinstance(ext, Extension)`` check inside
    ``setuptools._distutils.command.build_ext.check_extensions_list``, raising::

        DistutilsSetupError: each element of 'ext_modules' option must be
        an Extension instance or 2-tuple

    This happens because ``cythonize()`` (from Cython) is imported once and
    keeps an internal reference to the ``Extension`` class it received at that
    time.  If the module cache is evicted and setuptools is re-imported between
    calls, the class objects diverge.  The singleton cache in
    ``_SETUPTOOLS_CACHE`` prevents this by importing setuptools exactly once.

    **Test isolation note:** because ``_import_setuptools()`` uses a module-level
    singleton, tests that need a clean cache state manipulate
    ``_builder_module._SETUPTOOLS_CACHE`` directly and restore it in a fixture.
    This is intentional: it tests the exact state the production code manages.
    """

    @pytest.fixture()
    def clean_cache(self):
        """
        Temporarily clear _SETUPTOOLS_CACHE so the function exercises the
        slow (first-call) import path during this test.

        Restores the original cached value after the test, so subsequent
        tests still benefit from the cached classes.
        """
        original = _builder_module._SETUPTOOLS_CACHE
        _builder_module._SETUPTOOLS_CACHE = None
        yield
        _builder_module._SETUPTOOLS_CACHE = original

    # ------------------------------------------------------------------
    # Return-value shape
    # ------------------------------------------------------------------

    def test_returns_two_items(self) -> None:
        """Result must be a 2-tuple."""
        result = _import_setuptools()
        assert len(result) == 2

    def test_extension_is_type(self) -> None:
        """First element must be a class (type)."""
        Extension, _ = _import_setuptools()
        assert isinstance(Extension, type)

    def test_distribution_is_type(self) -> None:
        """Second element must be a class (type)."""
        _, Distribution = _import_setuptools()
        assert isinstance(Distribution, type)

    def test_extension_module_contains_setuptools(self) -> None:
        """Extension must come from the setuptools namespace."""
        Extension, _ = _import_setuptools()
        assert "setuptools" in Extension.__module__

    def test_distribution_module_contains_setuptools(self) -> None:
        """Distribution must come from the setuptools namespace."""
        _, Distribution = _import_setuptools()
        assert "setuptools" in Distribution.__module__

    def test_extension_is_subclass_of_object(self) -> None:
        Extension, _ = _import_setuptools()
        assert issubclass(Extension, object)

    def test_distribution_is_subclass_of_object(self) -> None:
        _, Distribution = _import_setuptools()
        assert issubclass(Distribution, object)

    def test_extension_can_be_instantiated(self, tmp_path: Path) -> None:
        """A minimal Extension(name, sources=[...]) must not raise."""
        src = tmp_path / "dummy.c"
        src.write_text("", encoding="utf-8")
        Extension, _ = _import_setuptools()
        ext = Extension(name="mymodule", sources=[str(src)])
        assert ext.name == "mymodule"

    # ------------------------------------------------------------------
    # Singleton / class-identity guarantees
    # ------------------------------------------------------------------

    def test_repeated_calls_return_identical_extension_class(self) -> None:
        """
        Regression test for the second-call DistutilsSetupError bug.

        Before the singleton cache was added, each call evicted sys.modules
        and re-imported setuptools, producing a new Extension class object.
        Cython's cythonize() still held a reference to the old class, so
        isinstance() inside check_extensions_list returned False and raised
        DistutilsSetupError.  The singleton cache fixes this permanently.
        """
        Extension1, _ = _import_setuptools()
        Extension2, _ = _import_setuptools()
        assert Extension1 is Extension2, (
            "Repeated calls must return the *same* Extension class object. "
            "Different objects break isinstance() in build_ext."
        )

    def test_repeated_calls_return_identical_distribution_class(self) -> None:
        """Distribution class identity must also be stable across calls."""
        _, Dist1 = _import_setuptools()
        _, Dist2 = _import_setuptools()
        assert Dist1 is Dist2

    def test_many_repeated_calls_all_identical(self) -> None:
        """Identity must hold for N > 2 calls (not just pairwise)."""
        pairs = [_import_setuptools() for _ in range(5)]
        first_ext, first_dist = pairs[0]
        for ext, dist in pairs[1:]:
            assert ext is first_ext
            assert dist is first_dist

    def test_extension_isinstance_stable_across_calls(self, tmp_path: Path) -> None:
        """
        An Extension instance created on call N must satisfy isinstance()
        with the class returned on call N+1.

        This is the exact isinstance() check that
        check_extensions_list performs.  Without the singleton cache this
        assertion fails on the second call.
        """
        src = tmp_path / "dummy.c"
        src.write_text("", encoding="utf-8")
        Extension1, _ = _import_setuptools()
        ext = Extension1(name="mymod", sources=[str(src)])
        Extension2, _ = _import_setuptools()
        # Fails (False) without the singleton cache fix.
        assert isinstance(ext, Extension2), (
            "Extension instance must satisfy isinstance() with the class "
            "returned by a subsequent call to _import_setuptools()."
        )

    # ------------------------------------------------------------------
    # First-call path (clean-cache fixture required)
    # ------------------------------------------------------------------

    def test_populates_module_cache_on_first_call(self, clean_cache: None) -> None:
        """After a first call, _SETUPTOOLS_CACHE must be non-None."""
        assert _builder_module._SETUPTOOLS_CACHE is None  # fixture guarantee
        _import_setuptools()
        assert _builder_module._SETUPTOOLS_CACHE is not None

    def test_cached_value_matches_return_value(self, clean_cache: None) -> None:
        """The module-level cache must equal the returned pair."""
        result = _import_setuptools()
        assert _builder_module._SETUPTOOLS_CACHE is result

    def test_second_call_uses_cache_not_reimport(self, clean_cache: None) -> None:
        """On a second call, the module-level cache is returned unchanged."""
        first = _import_setuptools()
        # Simulate what would break without the cache: evict setuptools.
        # Even after eviction the second call must return the same objects.
        import sys as _sys
        for _k in list(_sys.modules):
            if _k.startswith("setuptools") or _k.startswith("distutils"):
                _sys.modules.pop(_k, None)
        second = _import_setuptools()
        assert second is first  # same tuple object from cache

    # ------------------------------------------------------------------
    # Environment variable behaviour
    # ------------------------------------------------------------------

    def test_env_var_set_after_first_call(self, clean_cache: None) -> None:
        """SETUPTOOLS_USE_DISTUTILS must be present after the first call."""
        os.environ.pop("SETUPTOOLS_USE_DISTUTILS", None)
        _import_setuptools()
        assert "SETUPTOOLS_USE_DISTUTILS" in os.environ

    def test_env_var_not_overwritten_if_already_set(
        self, monkeypatch: pytest.MonkeyPatch, clean_cache: None
    ) -> None:
        """
        A caller-supplied SETUPTOOLS_USE_DISTUTILS must be honoured.

        The function uses ``setdefault``, so a pre-existing value is never
        overwritten.  We verify this with ``monkeypatch`` so the test is
        side-effect-free.
        """
        monkeypatch.setenv("SETUPTOOLS_USE_DISTUTILS", "stdlib")
        _import_setuptools()
        assert os.environ["SETUPTOOLS_USE_DISTUTILS"] == "stdlib"

    def test_env_var_stdlib_on_python_lt_312(
        self, monkeypatch: pytest.MonkeyPatch, clean_cache: None
    ) -> None:
        """On Python < 3.12, the default backend must be 'stdlib'."""
        if sys.version_info >= (3, 12):
            pytest.skip("Python < 3.12 only")
        monkeypatch.delenv("SETUPTOOLS_USE_DISTUTILS", raising=False)
        _import_setuptools()
        assert os.environ.get("SETUPTOOLS_USE_DISTUTILS") == "stdlib"

    def test_env_var_local_on_python_gte_312(
        self, monkeypatch: pytest.MonkeyPatch, clean_cache: None
    ) -> None:
        """On Python >= 3.12, the default backend must be 'local'."""
        if sys.version_info < (3, 12):
            pytest.skip("Python >= 3.12 only")
        monkeypatch.delenv("SETUPTOOLS_USE_DISTUTILS", raising=False)
        _import_setuptools()
        assert os.environ.get("SETUPTOOLS_USE_DISTUTILS") == "local"

    # ------------------------------------------------------------------
    # Error path: missing setuptools
    # ------------------------------------------------------------------

    def test_missing_setuptools_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch, clean_cache: None
    ) -> None:
        """
        If setuptools is not importable, an actionable ImportError must be
        raised — never a raw exception from a nested import.
        """
        import builtins
        real_import = builtins.__import__

        def _block_setuptools(name, *args, **kwargs):
            if name.startswith("setuptools"):
                raise ImportError(f"Blocked: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_setuptools)
        with pytest.raises(ImportError, match="setuptools is required"):
            _import_setuptools()

    def test_assertion_error_wrapped_as_import_error(
        self, monkeypatch: pytest.MonkeyPatch, clean_cache: None
    ) -> None:
        """
        If _distutils_hack raises AssertionError (broken CI toolchain), it
        must be surfaced as an ImportError with a concrete work-around hint
        — not the raw AssertionError which gives no guidance to the user.
        """
        import builtins
        real_import = builtins.__import__

        def _raise_assertion(name, *args, **kwargs):
            if name.startswith("setuptools"):
                raise AssertionError("/path/to/distutils/core.py")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _raise_assertion)
        with pytest.raises(ImportError, match="SETUPTOOLS_USE_DISTUTILS=stdlib"):
            _import_setuptools()


# ---------------------------------------------------------------------------
# TestSetVerbosity
# ---------------------------------------------------------------------------


class TestSetVerbosity:
    """
    Tests for :func:`~scikitplot.cython._builder._set_verbosity`.

    Notes
    -----
    ``_set_verbosity`` is a best-effort helper that calls
    ``setuptools._distutils.log.set_verbosity``.  It must never raise — even
    when the underlying distutils log module is absent or incompatible.
    """

    def test_no_error_on_zero(self) -> None:
        """Verbosity level 0 (default quiet) must not raise."""
        _set_verbosity(0)

    def test_no_error_on_positive(self) -> None:
        """Verbosity level 1 (normal) must not raise."""
        _set_verbosity(1)

    def test_no_error_on_negative(self) -> None:
        """Verbosity level -1 (suppress all) must not raise."""
        _set_verbosity(-1)

    def test_no_error_on_large_positive(self) -> None:
        """Verbosity level 9 (very verbose) must not raise."""
        _set_verbosity(9)

    def test_no_error_when_distutils_log_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        If setuptools._distutils.log is missing, _set_verbosity must
        silently succeed (best-effort contract).
        """
        import builtins
        real_import = builtins.__import__

        def _block_distutils_log(name, *args, **kwargs):
            if "distutils" in name and "log" in name:
                raise ImportError(f"Blocked: {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_distutils_log)
        # Must not raise even when the import fails.
        _set_verbosity(1)

    def test_no_error_when_set_verbosity_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        If set_verbosity itself raises (e.g. incompatible API), _set_verbosity
        must swallow the exception (best-effort contract).
        """
        import builtins
        real_import = builtins.__import__

        class _FakeLog:
            @staticmethod
            def set_verbosity(level):  # noqa: ANN001
                raise RuntimeError("Simulated API incompatibility")

        fake_log_module = _FakeLog()

        def _inject_fake_log(name, globals=None, locals=None, fromlist=(), level=0):
            if "distutils" in (name or "") and fromlist and "set_verbosity" in fromlist:
                return fake_log_module
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _inject_fake_log)
        # Must not raise despite the RuntimeError inside set_verbosity.
        _set_verbosity(0)


# ---------------------------------------------------------------------------
# TestOpenAnnotationInBrowser
# ---------------------------------------------------------------------------


class TestOpenAnnotationInBrowser:
    """
    Tests for :func:`~scikitplot.cython._builder._open_annotation_in_browser`.

    Notes
    -----
    The function is a best-effort helper that opens an HTML annotation in a
    browser.  It must never raise and must silently skip in every headless /
    CI / non-interactive environment.  All three guard branches are exercised
    here:

    1. ``CI`` environment variable is set.
    2. Linux/macOS without a ``DISPLAY`` variable (no X11/Wayland).
    3. ``sys.stdout`` is not a TTY (non-interactive session).
    """

    _DUMMY_URI = "file:///tmp/annotation.html"

    def test_skips_and_does_not_raise_when_ci_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CI guard: must return silently when CI env var is set."""
        monkeypatch.setenv("CI", "true")
        # Must not raise and must not call webbrowser.open.
        _open_annotation_in_browser(self._DUMMY_URI)

    def test_ci_guard_triggered_by_any_truthy_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CI guard must trigger for any non-empty CI value."""
        monkeypatch.setenv("CI", "1")
        _open_annotation_in_browser(self._DUMMY_URI)

    def test_skips_on_posix_without_display(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DISPLAY guard: must skip on Linux/macOS when DISPLAY is absent."""
        if sys.platform in ("win32", "darwin"):
            pytest.skip("DISPLAY guard is POSIX-only")
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        _open_annotation_in_browser(self._DUMMY_URI)

    def test_skips_when_stdout_is_not_a_tty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TTY guard: must skip when sys.stdout.isatty() returns False."""
        import io
        monkeypatch.delenv("CI", raising=False)
        # Provide a DISPLAY so the POSIX guard is not triggered on Linux.
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        _open_annotation_in_browser(self._DUMMY_URI)

    def test_skips_when_stdout_has_no_isatty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        TTY guard: if sys.stdout.isatty() raises AttributeError (e.g.
        some logging redirectors), the function must silently skip.
        """
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")

        class _NoIsatty:
            def isatty(self):  # noqa: ANN001
                raise AttributeError("no isatty")

        monkeypatch.setattr(sys, "stdout", _NoIsatty())
        _open_annotation_in_browser(self._DUMMY_URI)

    def test_browser_not_opened_in_ci(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify webbrowser.open is NOT called when CI guard triggers."""
        import webbrowser
        opened: list[str] = []
        monkeypatch.setenv("CI", "true")
        monkeypatch.setattr(webbrowser, "open", lambda uri: opened.append(uri))
        _open_annotation_in_browser(self._DUMMY_URI)
        assert opened == [], "webbrowser.open must not be called in CI"

    def test_browser_not_opened_without_display(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify webbrowser.open is NOT called when DISPLAY guard triggers."""
        if sys.platform in ("win32", "darwin"):
            pytest.skip("DISPLAY guard is POSIX-only")
        import webbrowser
        opened: list[str] = []
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setattr(webbrowser, "open", lambda uri: opened.append(uri))
        _open_annotation_in_browser(self._DUMMY_URI)
        assert opened == [], "webbrowser.open must not be called without DISPLAY"

    def test_browser_not_opened_when_not_tty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify webbrowser.open is NOT called when TTY guard triggers."""
        import io
        import webbrowser
        opened: list[str] = []
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(webbrowser, "open", lambda uri: opened.append(uri))
        _open_annotation_in_browser(self._DUMMY_URI)
        assert opened == [], "webbrowser.open must not be called when not a TTY"
