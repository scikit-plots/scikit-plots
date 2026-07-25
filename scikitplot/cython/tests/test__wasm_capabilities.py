# scikitplot/cython/tests/test__wasm_capabilities.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-WASM-001.

Two in-package guarantees for the browser/WASM path:

1. ``platform_capabilities()`` exposes an explicit contract so callers can tell
   that a browser/WASM runtime cannot compile at runtime (prebuilt-only),
   instead of hitting an opaque build failure.
2. ``verify_template_assets()`` proves every ``support_paths`` asset referenced
   by a shipped template (notably the three ``.pxi`` files) is actually present,
   so a packaging recipe that drops ``**/*.pxi`` is caught as a test failure.

(The full prebuilt browser backend + live JupyterLite CI is infrastructure
tracked separately; these tests pin the module-level contract and guard.)
"""
from __future__ import annotations

import sys

import pytest

from .._profiles import PlatformCapabilities, platform_capabilities
from .._templates_api import verify_template_assets


class TestPlatformCapabilities:
    def test_current_process_contract(self) -> None:
        caps = platform_capabilities()
        assert isinstance(caps, PlatformCapabilities)
        # can_compile_at_runtime and prebuilt_only are complements.
        assert caps.can_compile_at_runtime is (not caps.prebuilt_only)
        assert caps.platform == sys.platform

    def test_non_browser_can_compile(self) -> None:
        caps = platform_capabilities()
        if not caps.is_browser_wasm:
            assert caps.can_compile_at_runtime is True
            assert caps.wasm_package_suffix is None

    def test_browser_detection_via_sys_platform(self, monkeypatch) -> None:
        # Simulate an Emscripten runtime.
        monkeypatch.setattr(sys, "platform", "emscripten")
        caps = platform_capabilities()
        assert caps.is_browser_wasm is True
        assert caps.can_compile_at_runtime is False
        assert caps.prebuilt_only is True
        assert caps.wasm_package_suffix == "emscripten-wasm32"


class TestTemplateAssetGuard:
    def test_all_referenced_assets_present(self) -> None:
        missing = verify_template_assets()
        assert missing == [], f"shipped templates reference missing assets: {missing}"

    def test_three_pxi_files_are_shipped(self) -> None:
        """The exact .pxi files the review flagged must be present."""
        from .._templates_api import _TEMPLATE_ROOT  # noqa: PLC0415

        pxi = {p.name for p in _TEMPLATE_ROOT.rglob("*.pxi")}
        assert {"helper_square.pxi", "common.pxi"} <= pxi

    def test_guard_detects_missing_asset(self, tmp_path, monkeypatch) -> None:
        """If a referenced support asset is missing, it is reported."""
        import json  # noqa: PLC0415

        from .. import _templates_api as _t  # noqa: PLC0415

        # Point the guard at a synthetic template root with a dangling ref.
        root = tmp_path / "_templates"
        (root / "module_cython").mkdir(parents=True)
        meta = root / "module_cython" / "x.pyx.meta.json"
        meta.write_text(json.dumps({"support_paths": ["missing.pxi"]}), encoding="utf-8")
        monkeypatch.setattr(_t, "_TEMPLATE_ROOT", root)
        missing = verify_template_assets()
        assert any("missing.pxi" in m for m in missing)
