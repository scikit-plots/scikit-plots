# scikitplot/cython/tests/test__resolved_toolchain.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-PORT-001.

Host/toolchain detection used a ``PATH`` heuristic (is ``cl`` present?) rather
than the compiler the build backend actually selects.  ``resolved_toolchain()``
now reports the effective compiler type + executables from distutils, it never
raises, and its identity is folded into the cache fingerprint so entries are
keyed from the resolved plan.
"""
from __future__ import annotations

from unittest import mock

from .._cache import runtime_fingerprint
from .._profiles import ResolvedToolchain, resolved_toolchain


class TestResolvedToolchain:
    def test_returns_namedtuple(self) -> None:
        rt = resolved_toolchain()
        assert isinstance(rt, ResolvedToolchain)
        assert rt._fields == ("compiler_type", "cc", "cxx", "linker")

    def test_fields_are_strings(self) -> None:
        rt = resolved_toolchain()
        assert isinstance(rt.compiler_type, str)
        assert isinstance(rt.cc, str)
        assert isinstance(rt.cxx, str)
        assert isinstance(rt.linker, str)

    def test_compiler_type_nonempty(self) -> None:
        # On any real backend this is a known type; never the empty string.
        assert resolved_toolchain().compiler_type

    def test_never_raises_on_backend_failure(self) -> None:
        # Even if the backend blows up, detection degrades to "unknown".
        with mock.patch(
            "setuptools._distutils.ccompiler.new_compiler",
            side_effect=RuntimeError("boom"),
        ):
            rt = resolved_toolchain()
        assert rt == ResolvedToolchain("unknown", "", "", "")


class TestFingerprintKeying:
    def test_fingerprint_includes_resolved_compiler(self) -> None:
        fp = runtime_fingerprint(cython_version="3.2.9", numpy_version="2.4.4")
        assert "resolved_compiler_type" in fp
        assert "resolved_cc" in fp
        assert "resolved_cxx" in fp

    def test_different_resolved_compiler_changes_fingerprint(self) -> None:
        base = dict(runtime_fingerprint(cython_version="3.2.9", numpy_version="2.4.4"))
        with mock.patch(
            "scikitplot.cython._profiles.resolved_toolchain",
            return_value=ResolvedToolchain("msvc", "cl", "cl", "link"),
        ):
            other = dict(
                runtime_fingerprint(cython_version="3.2.9", numpy_version="2.4.4")
            )
        # The resolved-compiler fields must differ, so cache keys differ.
        assert base["resolved_compiler_type"] != other["resolved_compiler_type"]

    def test_fingerprint_stable_when_toolchain_stable(self) -> None:
        a = runtime_fingerprint(cython_version="3.2.9", numpy_version="2.4.4")
        b = runtime_fingerprint(cython_version="3.2.9", numpy_version="2.4.4")
        assert a["resolved_compiler_type"] == b["resolved_compiler_type"]
        assert a["resolved_cc"] == b["resolved_cc"]
