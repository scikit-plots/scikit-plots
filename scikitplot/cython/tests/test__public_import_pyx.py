# scikitplot/cython/tests/test__public_import_pyx.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-API-001.

``cython_import_result`` resolves the ``.pyx`` path to an absolute path and
added its parent to ``include_dirs``.  Under the default strict policy
(``allow_absolute_include_dirs=False``) that absolute parent was rejected, so
importing any ``.pyx`` by its normal path raised ``SecurityError``.  It also
appended the parent twice.

The fix routes the source directory through an intrinsic ``_trusted_include_dirs``
channel that is traversal-checked but absolute-allowed, without weakening the
guard on user-supplied ``include_dirs``.

Most tests stub the compiler (fast, toolchain-independent); one end-to-end test
performs a real compile to prove the default path now works.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from .._public import compile_and_load_result, cython_import_result
from .._security import SecurityError, SecurityPolicy


SRC = "def add(int a, int b):\n    return a + b\n"


@pytest.fixture(autouse=True)
def _reset_setuptools_singleton():
    """Isolate real-compile tests from the pre-existing CYTHON-CON-002 pollution."""
    from .. import _builder as _b  # noqa: PLC0415

    original = _b._SETUPTOOLS_CACHE
    _b._SETUPTOOLS_CACHE = None
    try:
        yield
    finally:
        _b._SETUPTOOLS_CACHE = original


class TestDefaultPolicyAcceptsNormalPyx:
    def test_absolute_pyx_path_passes_validation_under_default_policy(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The headline bug: a normal absolute .pyx path must NOT raise
        SecurityError under the default policy.

        The build itself is stubbed so this test is independent of the
        pre-existing suite-wide setuptools/distutils global-state pollution
        (CYTHON-CON-002).  A real end-to-end compile of this path is covered by
        the shipped probe ``_templates/probe/compile_probe.py`` and the
        integration test below (which is skipped if the toolchain state is
        polluted).
        """
        from .. import _public as _p  # noqa: PLC0415

        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop-after-validation")

        monkeypatch.setattr(_p, "build_extension_module_result", fake_build)
        pyx = tmp_path / "mymod.pyx"
        pyx.write_text(SRC, encoding="utf-8")

        # Must reach the builder (i.e. pass security validation), not raise
        # SecurityError as it did before the fix.
        with pytest.raises(RuntimeError, match="stop-after-validation"):
            cython_import_result(
                pyx, cache_dir=tmp_path / "cache", numpy_support=False, verbose=-1
            )
        # The source directory was routed through include_dirs as a trusted dir.
        assert pyx.parent.resolve() in list(captured["include_dirs"])

    def test_absolute_pyx_path_real_compile(self, tmp_path: Path) -> None:
        """End-to-end proof the default path imports a real .pyx.

        Skipped automatically if a prior test in the run has polluted the global
        setuptools/distutils Extension identity (CYTHON-CON-002); the stubbed
        test above plus the shipped probe still cover the fix.
        """
        pyx = tmp_path / "mymod.pyx"
        pyx.write_text(SRC, encoding="utf-8")
        try:
            r = cython_import_result(
                pyx, cache_dir=tmp_path / "cache", numpy_support=False, verbose=-1
            )
        except RuntimeError as e:  # cythonize Extension-identity pollution
            if "Extension" in str(e) or "Cythonize" in str(e):
                pytest.skip(f"toolchain global-state polluted (CYTHON-CON-002): {e}")
            raise
        assert r.module.add(2, 3) == 5


class TestTrustedIncludeDirValidation:
    def test_trusted_dir_allows_absolute(self, tmp_path: Path, monkeypatch) -> None:
        """_trusted_include_dirs permits an absolute path (no SecurityError)."""
        # Stub the builder so we only exercise validation + merge, not compile.
        from .. import _public as _p  # noqa: PLC0415

        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop-after-validation")

        monkeypatch.setattr(_p, "build_extension_module_result", fake_build)
        with pytest.raises(RuntimeError, match="stop-after-validation"):
            compile_and_load_result(
                SRC,
                numpy_support=False,
                verbose=-1,
                _trusted_include_dirs=[tmp_path],  # absolute
            )
        # The trusted dir was merged into the builder's include_dirs.
        assert tmp_path in list(captured["include_dirs"])

    def test_user_absolute_include_dir_still_rejected(self, tmp_path: Path) -> None:
        pyx = tmp_path / "m.pyx"
        pyx.write_text(SRC, encoding="utf-8")
        with pytest.raises(SecurityError):
            cython_import_result(
                pyx,
                cache_dir=tmp_path / "cache",
                numpy_support=False,
                verbose=-1,
                include_dirs=["/etc"],  # user-supplied absolute → rejected
            )

    def test_traversal_in_trusted_dir_rejected(self) -> None:
        with pytest.raises(SecurityError, match="traversal"):
            compile_and_load_result(
                SRC,
                numpy_support=False,
                verbose=-1,
                _trusted_include_dirs=["../../etc"],
            )

    def test_relaxed_policy_still_allows_user_absolute(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A permissive policy still lets user-supplied absolute dirs through."""
        from .. import _public as _p  # noqa: PLC0415

        def fake_build(**kwargs):
            raise RuntimeError("stop")

        monkeypatch.setattr(_p, "build_extension_module_result", fake_build)
        # Validation should pass (allow flag set), so we reach the stubbed builder.
        with pytest.raises(RuntimeError, match="stop"):
            compile_and_load_result(
                SRC,
                numpy_support=False,
                verbose=-1,
                include_dirs=["/usr/include"],
                security_policy=SecurityPolicy(allow_absolute_include_dirs=True),
            )


class TestNoDuplicateSourceDir:
    def test_source_dir_added_once(self, tmp_path: Path, monkeypatch) -> None:
        """The source parent must be added once, not twice (old duplicate bug)."""
        from .. import _public as _p  # noqa: PLC0415

        captured = {}

        def fake_build(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop")

        monkeypatch.setattr(_p, "build_extension_module_result", fake_build)
        pyx = tmp_path / "m.pyx"
        pyx.write_text(SRC, encoding="utf-8")
        with pytest.raises(RuntimeError, match="stop"):
            cython_import_result(
                pyx, cache_dir=tmp_path / "cache", numpy_support=False, verbose=-1
            )
        inc = list(captured["include_dirs"])
        assert inc.count(pyx.parent.resolve()) == 1, f"source dir duplicated: {inc}"
