# corpus/tests/test__capabilities.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Capability snapshot gate (CORPUS-PKG-001)
=========================================

``capability_snapshot`` must be a read-only, never-raising reproducibility
surface: Python/platform identity, ANN backend availability, and optional
distribution versions (``None`` when absent).

Run with::

    pytest scikitplot/corpus/tests/test__capabilities.py -v
"""

from __future__ import annotations

import platform

from scikitplot.corpus._capabilities import (
    capability_snapshot,
    distribution_version,
)


class TestDistributionVersion:
    def test_installed_returns_str(self):
        assert isinstance(distribution_version("numpy"), str)

    def test_absent_returns_none(self):
        assert distribution_version("no-such-distribution-xyz-123") is None


class TestCapabilitySnapshot:
    def test_top_level_shape(self):
        snap = capability_snapshot()
        assert set(snap) == {
            "python", "implementation", "platform",
            "ann_backends", "distributions",
        }

    def test_python_matches_interpreter(self):
        snap = capability_snapshot()
        assert snap["python"] == platform.python_version()
        assert snap["implementation"] == platform.python_implementation()

    def test_distributions_versions_or_none(self):
        dists = capability_snapshot()["distributions"]
        assert isinstance(dists, dict)
        assert isinstance(dists.get("numpy"), str)  # numpy is a hard dep
        for value in dists.values():
            assert value is None or isinstance(value, str)

    def test_ann_backends_shape(self):
        ann = capability_snapshot()["ann_backends"]
        assert isinstance(ann, dict)
        for entry in ann.values():
            assert set(entry) == {
                "status",
                "reason_code",
                "available",
                "version",
                "aliases",
            }
            # F-R02-05: status is the authority; `available` is derived from it.
            assert entry["available"] is (entry["status"] == "available")
            assert (entry["reason_code"] is None) is (entry["status"] == "available")
            assert isinstance(entry["available"], bool)
            assert entry["version"] is None or isinstance(entry["version"], str)
            assert isinstance(entry["aliases"], list)

    def test_backends_are_keyed_by_canonical_identity(self):
        """
        P-I0-06 / F-R02-06: one entry per implementation, aliases as a field.

        ``"brute"`` was previously a second registry entry for
        ``BruteForceBackend``, so the snapshot reported five backends for four
        classes -- with contradictory versions, because the distribution map had
        no entry for the alias.  Anything counting backends, or embedding this
        snapshot in a build manifest, recorded a phantom.
        """
        from scikitplot.corpus._similarity._backends import _BACKENDS

        ann = capability_snapshot()["ann_backends"]
        assert len(ann) == len({id(cls) for cls in _BACKENDS.values()})
        assert "brute" not in ann
        assert ann["bruteforce"]["aliases"] == ["brute"]

    def test_alias_resolves_to_the_same_backend(self):
        """An alias remains accepted at the call site even though it is not an entry."""
        from scikitplot.corpus._similarity import _backends as backends

        assert backends.canonical_backend_name("brute") == "bruteforce"
        assert backends.select_backend("brute").name == "bruteforce"

    def test_extra_distributions(self):
        snap = capability_snapshot(extra_distributions=["pytest"])
        assert "pytest" in snap["distributions"]

    def test_deterministic_and_never_raises(self):
        # Two consecutive calls agree; calling must never raise on a missing
        # optional component.
        assert (
            capability_snapshot()["ann_backends"]
            == capability_snapshot()["ann_backends"]
        )
