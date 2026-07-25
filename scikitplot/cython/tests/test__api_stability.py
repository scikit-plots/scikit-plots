# scikitplot/cython/tests/test__api_stability.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CYTHON-API-002.

The facade re-exported implementation primitives with no stable/advanced/
experimental distinction.  ``_api`` attaches a machine-readable stability tier
to every public symbol.  The key guarantee is *total coverage*: every name in
the package ``__all__`` has a tier, so a newly added export cannot silently
enter the public surface untiered (that would be exactly the "unreviewable API
drift" the finding warns about).
"""
from __future__ import annotations

import pytest

import scikitplot.cython as skc
from scikitplot.cython import Stability, api_stability, list_api
from scikitplot.cython._api import API_STABILITY


class TestTotalCoverage:
    def test_every_public_symbol_has_a_tier(self) -> None:
        missing = [n for n in skc.__all__ if n not in API_STABILITY]
        assert missing == [], f"public symbols without a stability tier: {missing}"

    def test_registry_only_contains_public_symbols(self) -> None:
        public = set(skc.__all__)
        extra = [n for n in API_STABILITY if n not in public]
        assert extra == [], f"tiered names not in __all__: {extra}"

    def test_tiers_partition_the_surface(self) -> None:
        stable = set(list_api(Stability.STABLE))
        advanced = set(list_api(Stability.ADVANCED))
        experimental = set(list_api(Stability.EXPERIMENTAL))
        # Disjoint and exhaustive.
        assert stable & advanced == set()
        assert stable & experimental == set()
        assert advanced & experimental == set()
        assert stable | advanced | experimental == set(skc.__all__)


class TestKnownTiers:
    @pytest.mark.parametrize(
        "name",
        ["compile_and_load", "cython_import", "pin", "BuildResult", "SecurityPolicy"],
    )
    def test_stable(self, name: str) -> None:
        assert api_stability(name) == Stability.STABLE

    @pytest.mark.parametrize(
        "name", ["build_lock", "make_cache_key", "import_extension", "gc_cache"]
    )
    def test_advanced(self, name: str) -> None:
        assert api_stability(name) == Stability.ADVANCED

    @pytest.mark.parametrize(
        "name",
        ["platform_capabilities", "verify_template_assets", "CompilerCapabilities"],
    )
    def test_experimental(self, name: str) -> None:
        assert api_stability(name) == Stability.EXPERIMENTAL


class TestAccessors:
    def test_unknown_symbol_raises(self) -> None:
        with pytest.raises(KeyError):
            api_stability("definitely_not_a_symbol")

    def test_list_api_accepts_str_and_enum(self) -> None:
        assert list_api("stable") == list_api(Stability.STABLE)

    def test_list_api_sorted(self) -> None:
        names = list_api(Stability.STABLE)
        assert names == sorted(names)

    def test_stability_values(self) -> None:
        assert Stability.STABLE.value == "stable"
        assert Stability.ADVANCED.value == "advanced"
        assert Stability.EXPERIMENTAL.value == "experimental"
