# scikitplot/cython/tests/conftest.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Shared pytest fixtures and helpers for the cython submodule test suite.

These utilities are available to every test__*.py file automatically via
pytest's conftest discovery.  No imports needed in test files — fixtures
are injected by pytest; helper functions are imported explicitly where used.
"""
from __future__ import annotations

import json
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Any

import pytest

from .._cache import make_cache_key, write_meta


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FAKE_KEY: str = "a" * 64   # valid 64-hex cache key
FAKE_KEY2: str = "b" * 64  # second distinct valid key


# ---------------------------------------------------------------------------
# Helper functions  (import these explicitly in test modules that need them)
# ---------------------------------------------------------------------------


def make_valid_key() -> str:
    """Return a deterministic valid 64-hex cache key."""
    return make_cache_key({"test": "shared_conftest_helper"})


def write_fake_artifact(build_dir: Path, module_name: str = "mymod") -> Path:
    """
    Write a non-importable ELF stub as a fake compiled extension artifact.

    The filename uses ``EXTENSION_SUFFIXES[0]`` so all cache/loader helpers
    that look for valid-suffix files will find it.
    """
    suffix = EXTENSION_SUFFIXES[0]
    artifact = build_dir / f"{module_name}{suffix}"
    artifact.write_bytes(b"\x7fELF")  # minimal ELF magic
    return artifact


def write_simple_cache_entry(
    root: Path,
    key: str,
    module_name: str = "mymod",
    kind: str = "module",
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """
    Write a minimal cache entry (meta.json + bare ``.so``) into *root/key*.

    Returns the entry directory.
    """
    entry_dir = root / key
    entry_dir.mkdir(parents=True, exist_ok=True)
    so = entry_dir / f"{module_name}.so"
    so.write_bytes(b"\x7fELF")
    meta: dict[str, Any] = {
        "kind": kind,
        "module_name": module_name,
        "artifact": f"{module_name}.so",
        "created_utc": "2025-01-01T00:00:00Z",
    }
    if extra_meta:
        meta.update(extra_meta)
    write_meta(entry_dir / "meta.json", meta)
    return entry_dir


def write_full_cache_entry(
    cache_root: Path,
    key: str,
    *,
    module_name: str = "mymod",
    kind: str = "module",
    fingerprint: dict | None = None,
) -> tuple[Path, Path]:
    """
    Write a complete module cache entry using real ``EXTENSION_SUFFIXES``.

    Returns ``(build_dir, artifact_path)``.
    """
    build_dir = cache_root / key
    build_dir.mkdir(parents=True, exist_ok=True)
    artifact = write_fake_artifact(build_dir, module_name)
    meta: dict[str, Any] = {
        "kind": kind,
        "key": key,
        "module_name": module_name,
        "artifact": artifact.name,
        "artifact_filename": artifact.name,
        "created_utc": "2025-01-01T00:00:00Z",
    }
    if fingerprint is not None:
        meta["fingerprint"] = fingerprint
    (build_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return build_dir, artifact


def write_package_cache_entry(
    cache_root: Path,
    key: str,
    package_name: str = "mypkg",
    short_names: tuple[str, ...] = ("mod1",),
) -> Path:
    """
    Write a complete package cache entry.

    Returns ``build_dir``.
    """
    build_dir = cache_root / key
    pkg_dir = build_dir / package_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    module_artifacts = []
    for sn in short_names:
        art = write_fake_artifact(pkg_dir, sn)
        rel = art.relative_to(build_dir).as_posix()
        module_artifacts.append({
            "module_name": f"{package_name}.{sn}",
            "artifact": rel,
            "source_sha256": None,
        })
    meta = {
        "kind": "package",
        "key": key,
        "package_name": package_name,
        "modules": module_artifacts,
        "created_utc": "2025-01-01T00:00:00Z",
        "fingerprint": {"python": "3.x"},
    }
    (build_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return build_dir


# ---------------------------------------------------------------------------
# Shared pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> Path:
    """Return a fresh isolated temporary cache root directory."""
    root = tmp_path / "cache"
    root.mkdir()
    return root


@pytest.fixture()
def fake_module_entry(tmp_cache: Path) -> tuple[str, Path, Path]:
    """
    Populate *tmp_cache* with one realistic module cache entry.

    Returns
    -------
    tuple[str, Path, Path]
        ``(key, build_dir, artifact_path)``
    """
    key = make_cache_key({"label": "fake_module"})
    build_dir, artifact = write_full_cache_entry(tmp_cache, key, fingerprint={"python": "3.12"})
    return key, build_dir, artifact


@pytest.fixture()
def fake_package_entry(tmp_cache: Path) -> tuple[str, Path, Path]:
    """
    Populate *tmp_cache* with one realistic package cache entry.

    Returns
    -------
    tuple[str, Path, Path]
        ``(key, build_dir, first_artifact_path)``
    """
    key = make_cache_key({"label": "fake_package"})
    build_dir = write_package_cache_entry(tmp_cache, key)
    artifact = next((build_dir / "mypkg").glob("mod1*"))
    return key, build_dir, artifact
