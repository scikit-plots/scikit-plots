# scikitplot/cython/tests/test__compiler_capabilities.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-COMP-001.

Custom compilers were bare callables with no versioned capability contract, so
consumers could not reason about what a registered compiler supports without
invoking it.  ``CompilerCapabilities`` + ``compiler_capabilities()`` add a
versioned, declarative descriptor; the built-in compilers declare theirs, and
legacy bare callables get a conservative default.
"""
from __future__ import annotations

import sys

from .._custom_compiler import (
    COMPILER_SPEC_VERSION,
    CApiCompiler,
    CompilerCapabilities,
    PybindCompiler,
    compiler_capabilities,
)


class TestCapabilitiesDescriptor:
    def test_spec_version_is_int(self) -> None:
        assert isinstance(COMPILER_SPEC_VERSION, int)

    def test_defaults_are_conservative(self) -> None:
        caps = CompilerCapabilities(name="custom_x")
        assert caps.spec_version == COMPILER_SPEC_VERSION
        assert caps.supported_languages == frozenset({"c"})
        assert caps.platforms is None  # any platform
        assert caps.requires_cpp is False
        assert caps.deterministic is True

    def test_supports_platform_any(self) -> None:
        caps = CompilerCapabilities(name="x")
        assert caps.supports_platform("win32")
        assert caps.supports_platform("linux")
        assert caps.supports_platform()  # current

    def test_supports_platform_restricted(self) -> None:
        caps = CompilerCapabilities(name="x", platforms=frozenset({"linux"}))
        assert caps.supports_platform("linux")
        assert not caps.supports_platform("win32")

    def test_supports_language(self) -> None:
        caps = CompilerCapabilities(name="x", supported_languages=frozenset({"c++"}))
        assert caps.supports_language("c++")
        assert not caps.supports_language("c")

    def test_is_frozen(self) -> None:
        caps = CompilerCapabilities(name="x")
        import dataclasses

        try:
            caps.name = "y"  # type: ignore[misc]
            frozen = False
        except dataclasses.FrozenInstanceError:
            frozen = True
        assert frozen


class TestBuiltinCompilersDeclareCapabilities:
    def test_pybind_is_cpp(self) -> None:
        caps = compiler_capabilities(PybindCompiler())
        assert caps.name == "custom_pybind11"
        assert caps.requires_cpp is True
        assert caps.supports_language("c++")
        assert caps.features.get("pybind11") is True

    def test_capi_is_c(self) -> None:
        caps = compiler_capabilities(CApiCompiler())
        assert caps.name == "custom_c_api"
        assert caps.requires_cpp is False
        assert caps.supports_language("c")
        assert caps.features.get("c_api") is True


class TestLegacyCallableDefault:
    def test_bare_callable_gets_conservative_default(self) -> None:
        class Bare:
            name = "custom_bare"

            def __call__(self, *a, **k):  # pragma: no cover
                return None

        caps = compiler_capabilities(Bare())
        assert caps.name == "custom_bare"
        assert caps.supported_languages == frozenset({"c"})
        assert caps.requires_cpp is False
        assert caps.supports_platform(sys.platform)

    def test_capabilities_method_is_honoured(self) -> None:
        class WithMethod:
            name = "custom_m"

            def capabilities(self) -> CompilerCapabilities:
                return CompilerCapabilities(
                    name=self.name, supported_languages=frozenset({"c++", "cuda"})
                )

            def __call__(self, *a, **k):  # pragma: no cover
                return None

        caps = compiler_capabilities(WithMethod())
        assert caps.supports_language("cuda")

    def test_unknown_object_gets_unknown_name(self) -> None:
        caps = compiler_capabilities(object())
        assert caps.name == "unknown"
