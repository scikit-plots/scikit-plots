# scikitplot/cython/tests/test__diagnostics.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-OBS-001.

Compilation captured output in an unbounded ``StringIO`` and failures raised
broad ``RuntimeError`` strings.  The capture is now a bounded buffer, and build
failures carry a typed :class:`BuildDiagnostic` (phase, module, tool versions,
status, bounded log tail) attached as ``exc.diagnostic`` — while the exception
message is preserved for backward compatibility.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from .._budget import BoundedBuffer, BuildDiagnostic

_KW = dict(
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


class TestBuildDiagnostic:
    def test_is_frozen_dataclass(self) -> None:
        d = BuildDiagnostic(phase="cythonize", module="m")
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            d.phase = "x"  # type: ignore[misc]

    def test_defaults(self) -> None:
        d = BuildDiagnostic(phase="build_ext", module="m")
        assert d.status is None
        assert d.command == ()
        assert dict(d.tool_versions) == {}
        assert d.log_tail == ""
        assert d.log_path is None


class TestBoundedCapture:
    def test_capture_buffer_is_bounded(self) -> None:
        # The builder captures into a BoundedBuffer; verify its cap semantics.
        b = BoundedBuffer(max_bytes=100)
        for i in range(1000):
            b.write(f"noise-line-{i}\n")
        assert b.truncated is True
        # Retained content stays within budget (plus the truncation marker).
        assert len(b.getvalue()) < 100 + 64


class TestDiagnosticAttachedOnFailure:
    def test_cythonize_failure_attaches_diagnostic(self) -> None:
        from .._builder import build_extension_module_result

        bad = "def f(:\n    not valid cython\n"
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(RuntimeError) as ei:
                build_extension_module_result(
                    module_name="obs_cythonize_fail",
                    code=bad,
                    cache_dir=Path(td),
                    **_KW,
                )
        diag = getattr(ei.value, "diagnostic", None)
        assert isinstance(diag, BuildDiagnostic)
        assert diag.phase == "cythonize"
        assert diag.module == "obs_cythonize_fail"
        assert "python" in diag.tool_versions
        assert "cython" in diag.tool_versions
        # Message is preserved for backward compatibility.
        assert "Cythonize failed" in str(ei.value)

    def test_diagnostic_log_tail_is_bounded(self) -> None:
        from .._builder import build_extension_module_result

        bad = "def f(:\n" + ("# padding line\n" * 5000) + "still invalid(\n"
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(RuntimeError) as ei:
                build_extension_module_result(
                    module_name="obs_big_log",
                    code=bad,
                    cache_dir=Path(td),
                    **_KW,
                )
        diag = getattr(ei.value, "diagnostic", None)
        assert isinstance(diag, BuildDiagnostic)
        # Log tail is bounded, not the full multi-thousand-line output.
        assert len(diag.log_tail) < 200_000
