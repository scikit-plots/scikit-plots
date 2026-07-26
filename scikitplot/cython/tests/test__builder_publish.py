# scikitplot/cython/tests/test__builder_publish.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-CACHE-001.

The original single-module builder wrote the ``.pyx``, support files, and
extra sources directly into the final cache entry *before* acquiring the lock.
Concurrent same-key callers could overwrite each other's inputs, and an
interrupted build left a partially authoritative entry.

These tests pin the corrected behaviour:

- ``_publish_atomically`` swaps a completed staging directory into place and
  never leaves the final entry in a partial state;
- a build whose compile step fails leaves **no** final entry and **no**
  leftover staging directory;
- the final entry is only ever created by an atomic publish (there is no
  window in which the final entry exists but is empty of a built artifact).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from .._builder import _publish_atomically


class TestPublishAtomically:
    def test_publishes_into_absent_destination(self, tmp_path: Path) -> None:
        staging = tmp_path / ".staging-abc"
        staging.mkdir()
        (staging / "artifact.txt").write_text("built", encoding="utf-8")
        final = tmp_path / "entry"

        _publish_atomically(staging, final)

        assert final.is_dir()
        assert (final / "artifact.txt").read_text(encoding="utf-8") == "built"
        assert not staging.exists(), "staging dir should be consumed by publish"

    def test_replaces_existing_entry_and_cleans_trash(self, tmp_path: Path) -> None:
        final = tmp_path / "entry"
        final.mkdir()
        (final / "old.txt").write_text("old", encoding="utf-8")  # non-empty prior entry

        staging = tmp_path / ".staging-def"
        staging.mkdir()
        (staging / "new.txt").write_text("new", encoding="utf-8")

        _publish_atomically(staging, final)

        # New content is present, old content is gone, no trash left behind.
        assert (final / "new.txt").exists()
        assert not (final / "old.txt").exists()
        leftovers = list(tmp_path.glob(".replaced-*"))
        assert leftovers == [], f"trash not cleaned: {leftovers}"
        assert not staging.exists()

    def test_no_partial_entry_left_when_staging_missing(self, tmp_path: Path) -> None:
        # A non-existent staging dir must fail without creating a final entry.
        staging = tmp_path / "does-not-exist"
        final = tmp_path / "entry"
        with pytest.raises(OSError):
            _publish_atomically(staging, final)
        assert not final.exists()


class TestBuildTransaction:
    """Publication-contract tests that stub the compiler.

    These deliberately avoid invoking the real Cython toolchain: the goal is to
    verify the *transaction* (stage → validate → atomic publish, and cleanup on
    failure), not compilation itself.  End-to-end real-compile coverage lives in
    ``test__builder.py`` and the standalone probes.  Stubbing also sidesteps the
    unrelated global setuptools/distutils state pollution (CYTHON-CON-002) that
    can otherwise make a real compile fail depending on suite order.
    """

    _GOOD_PYX = "def add(int a, int b):\n    return a + b\n"

    @staticmethod
    def _kwargs(cache: Path) -> dict:
        return dict(
            code=TestBuildTransaction._GOOD_PYX,
            source_path=None,
            module_name=None,
            cache_dir=cache,
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

    def test_failed_build_leaves_no_final_entry_or_staging(
        self, tmp_cache: Path, monkeypatch
    ) -> None:
        """A compile failure must not publish an entry or leak a staging dir."""
        from .. import _builder as _b  # noqa: PLC0415

        def boom(**_kw):
            raise RuntimeError("simulated compile failure")

        monkeypatch.setattr(_b, "_compile", boom)

        with pytest.raises(RuntimeError, match="simulated compile failure"):
            _b.build_extension_module_result(**self._kwargs(tmp_cache))

        assert list(tmp_cache.glob(".staging-*")) == [], "leaked staging dir"
        entries = [
            p
            for p in tmp_cache.iterdir()
            if p.is_dir()
            and len(p.name) == 64
            and all(c in "0123456789abcdef" for c in p.name)
        ]
        assert entries == [], f"partial final entry published: {entries}"

    def test_invalid_artifact_is_not_published(
        self, tmp_cache: Path, monkeypatch
    ) -> None:
        """If the compiler returns a non-existent artifact, nothing publishes."""
        from .. import _builder as _b  # noqa: PLC0415

        def fake_compile(*, build_dir: Path, name: str, **_kw) -> Path:
            # Simulate a compiler that reports success but produced no file.
            return build_dir / f"{name}.does-not-exist"

        monkeypatch.setattr(_b, "_compile", fake_compile)

        with pytest.raises(RuntimeError):
            _b.build_extension_module_result(**self._kwargs(tmp_cache))

        assert list(tmp_cache.glob(".staging-*")) == []
        entries = [p for p in tmp_cache.iterdir() if p.is_dir() and len(p.name) == 64]
        assert entries == []

    def test_successful_build_publishes_complete_entry(
        self, tmp_cache: Path, monkeypatch
    ) -> None:
        """A successful (stubbed) build publishes exactly one complete entry."""
        from importlib.machinery import EXTENSION_SUFFIXES  # noqa: PLC0415
        from types import ModuleType  # noqa: PLC0415

        from .. import _builder as _b  # noqa: PLC0415

        def fake_compile(*, build_dir: Path, name: str, **_kw) -> Path:
            # Produce a plausible artifact file inside the staging dir.
            art = build_dir / f"{name}{EXTENSION_SUFFIXES[0]}"
            art.write_bytes(b"\x7fELF-stub")
            return art

        def fake_import(*, name: str, path: Path, key: str, build_dir: Path):
            mod = ModuleType(name)
            mod.__file__ = str(path)
            mod.add = lambda a, b: a + b  # noqa: E731
            return mod

        monkeypatch.setattr(_b, "_compile", fake_compile)
        monkeypatch.setattr(_b, "import_extension", fake_import)

        r = _b.build_extension_module_result(**self._kwargs(tmp_cache))

        assert r.module.add(2, 3) == 5
        assert r.build_dir.is_dir()
        assert (r.build_dir / f"{r.module_name}.pyx").exists()
        assert (r.build_dir / "meta.json").exists()
        assert list(tmp_cache.glob(".staging-*")) == []
        # Exactly one published entry.
        entries = [p for p in tmp_cache.iterdir() if p.is_dir() and len(p.name) == 64]
        assert len(entries) == 1
