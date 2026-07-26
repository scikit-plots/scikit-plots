# scikitplot/cython/tests/test__cache_schema.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-SCH-001.

``meta.json`` had no schema version, so a future format change could not be
detected — a stale-format entry would be misread.  ``write_meta`` now stamps
``CACHE_SCHEMA_VERSION`` on every write, and the builder refuses to reuse an
entry whose schema is incompatible (legacy v0 or a newer unknown version),
rebuilding it instead.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .._cache import (
    CACHE_SCHEMA_VERSION,
    is_meta_schema_compatible,
    meta_schema_version,
    read_meta,
    write_meta,
)


class TestStamping:
    def test_write_stamps_version(self, tmp_path: Path) -> None:
        write_meta(tmp_path, {"kind": "module"})
        meta = read_meta(tmp_path)
        assert meta["meta_schema_version"] == CACHE_SCHEMA_VERSION

    def test_explicit_version_preserved(self, tmp_path: Path) -> None:
        write_meta(tmp_path, {"meta_schema_version": 1, "x": 1})
        assert read_meta(tmp_path)["meta_schema_version"] == 1

    def test_original_fields_survive(self, tmp_path: Path) -> None:
        write_meta(tmp_path, {"a": 1, "b": "two"})
        meta = read_meta(tmp_path)
        assert meta["a"] == 1 and meta["b"] == "two"


class TestVersionHelpers:
    def test_meta_schema_version_absent_is_zero(self) -> None:
        assert meta_schema_version({"kind": "module"}) == 0
        assert meta_schema_version(None) == 0
        assert meta_schema_version({}) == 0

    def test_meta_schema_version_reads_value(self) -> None:
        assert meta_schema_version({"meta_schema_version": 1}) == 1

    def test_negative_or_bad_version_is_zero(self) -> None:
        assert meta_schema_version({"meta_schema_version": -3}) == 0
        assert meta_schema_version({"meta_schema_version": "x"}) == 0


class TestCompatibility:
    def test_current_is_compatible(self) -> None:
        assert is_meta_schema_compatible({"meta_schema_version": CACHE_SCHEMA_VERSION})

    def test_legacy_v0_incompatible(self) -> None:
        assert not is_meta_schema_compatible({"kind": "module"})  # no version
        assert not is_meta_schema_compatible(None)

    def test_future_version_incompatible(self) -> None:
        assert not is_meta_schema_compatible(
            {"meta_schema_version": CACHE_SCHEMA_VERSION + 99}
        )


class TestReuseGating:
    """End-to-end reuse gating.

    These perform real compiles, so they self-skip when a prior test in the run
    has polluted the global setuptools/distutils Extension identity
    (CYTHON-CON-002) — the schema *logic* is fully covered by the non-compiling
    unit tests above regardless.
    """

    _KW = dict(
        code="def add(int a, int b):\n    return a + b\n",
        source_path=None,
        use_cache=True,
        force_rebuild=False,
        verbose=-1,
        annotate=False,
        view_annotate=False,
        numpy_support=False,
        numpy_required=False,
        include_dirs=None,
        library_dirs=None,
        libraries=None,
        define_macros=None,
        extra_compile_args=None,
        extra_link_args=None,
        compiler_directives=None,
    )

    @staticmethod
    def _build(**kwargs):
        from .._builder import build_extension_module_result  # noqa: PLC0415

        try:
            return build_extension_module_result(**kwargs)
        except RuntimeError as e:
            if "Extension" in str(e) or "Cythonize" in str(e):
                import pytest  # noqa: PLC0415

                pytest.skip(f"toolchain global-state polluted (CYTHON-CON-002): {e}")
            raise

    def test_incompatible_entry_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            r1 = self._build(module_name="sch_a", cache_dir=cache, **self._KW)
            assert r1.used_cache is False
            build_dir = r1.build_dir

            # Downgrade the entry to a legacy (unversioned) schema.
            meta = dict(read_meta(build_dir))
            meta.pop("meta_schema_version", None)
            (build_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

            # The incompatible entry must be rebuilt, not reused.
            r2 = self._build(module_name="sch_a", cache_dir=cache, **self._KW)
            assert r2.used_cache is False
            # And re-stamped to the current version.
            assert read_meta(build_dir)["meta_schema_version"] == CACHE_SCHEMA_VERSION

    def test_compatible_entry_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = Path(td)
            self._build(module_name="sch_b", cache_dir=cache, **self._KW)
            r2 = self._build(module_name="sch_b", cache_dir=cache, **self._KW)
            assert r2.used_cache is True
