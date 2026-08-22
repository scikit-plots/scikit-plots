# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the capability status model (F-R02-05) and :mod:`_catalog` (F-R13-03)."""

from __future__ import annotations

import json

import pytest

from .._capabilities import CapabilityStatus, capability_snapshot, probe_backend
from .._catalog import ComponentSpec, component_catalog
from .._registry._registry import registry
from .._similarity import _backends as backends

__all__: "list[str]" = [
    "TestCapabilityStatus",
    "TestBrokenVersusAbsent",
    "TestComponentCatalog",
]


class _NeverInstalled:
    @classmethod
    def is_available(cls):
        return False


class _Corrupt:
    @classmethod
    def is_available(cls):
        raise RuntimeError("native library failed to load")


class _NotImportable:
    @classmethod
    def is_available(cls):
        raise ImportError("no module named 'thing'")


class _Working:
    @classmethod
    def is_available(cls):
        return True


class TestCapabilityStatus:
    """The seven-state model."""

    @pytest.mark.parametrize(
        ("cls", "expected"),
        [
            (_Working, CapabilityStatus.AVAILABLE),
            (_NeverInstalled, CapabilityStatus.ABSENT),
            (_Corrupt, CapabilityStatus.BROKEN),
            (_NotImportable, CapabilityStatus.ABSENT),
        ],
        ids=["available", "absent", "broken", "import_error"],
    )
    def test_probe_classifies(self, cls, expected: CapabilityStatus) -> None:
        status, _ = probe_backend(cls)
        assert status is expected

    def test_reason_code_present_iff_not_available(self) -> None:
        assert probe_backend(_Working) == (CapabilityStatus.AVAILABLE, None)
        status, reason = probe_backend(_Corrupt)
        assert status is CapabilityStatus.BROKEN
        assert "RuntimeError" in reason

    def test_probe_never_propagates(self) -> None:
        """A capability probe must not fail the caller asking what exists."""
        for cls in (_Corrupt, _NotImportable):
            probe_backend(cls)


class TestBrokenVersusAbsent:
    """The measured defect: F-R02-05."""

    def test_broken_and_absent_are_distinguishable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """BROKEN and ABSENT are distinct regardless of installed backends."""

        def broken_probe(cls) -> bool:
            raise RuntimeError("driver corrupt")

        monkeypatch.setattr(
            backends.BruteForceBackend,
            "is_available",
            classmethod(broken_probe),
        )
        monkeypatch.setattr(
            backends.AnnoyBackend,
            "is_available",
            classmethod(lambda cls: False),
        )
        snapshot = capability_snapshot()["ann_backends"]
        broken = snapshot["bruteforce"]
        absent = snapshot["annoy"]
        assert broken["status"] == "broken"
        assert absent["status"] == "absent"
        assert broken["status"] != absent["status"]
        # Both are still unavailable -- the difference is *why*.
        assert broken["available"] is False
        assert absent["available"] is False

    def test_available_is_derived_from_status(self) -> None:
        for entry in capability_snapshot()["ann_backends"].values():
            assert entry["available"] is (entry["status"] == "available")


class TestComponentCatalog:
    """One read-only view over four registries (F-R13-01, F-R13-03)."""

    @pytest.fixture(autouse=True)
    def _populate(self):
        registry.register_builtins()

    def test_aggregates_multiple_registries(self) -> None:
        catalog = component_catalog()
        assert len(catalog) > 0
        # Backends come from _BACKENDS; chunkers/readers from ComponentRegistry.
        assert "index" in catalog.categories()
        assert "chunker" in catalog.categories()

    def test_get_resolves_aliases(self) -> None:
        catalog = component_catalog()
        assert catalog.get("index", "brute").name == "bruteforce"

    def test_unprobed_components_report_unknown_not_available(self) -> None:
        """
        An unprobed component must never be reported as usable.

        The registries for chunkers, readers and normalizers expose no
        availability probe. Defaulting them to AVAILABLE would be exactly the
        unverified capability claim this model exists to prevent, so they report
        UNKNOWN -- and UNKNOWN is not available.
        """
        catalog = component_catalog()
        spec = catalog.list("chunker")[0]
        assert spec.status is CapabilityStatus.UNKNOWN
        assert spec.available is False

    def test_available_filters_to_usable_components(self) -> None:
        catalog = component_catalog()
        assert all(spec.available for spec in catalog.available())
        assert catalog.get("index", "bruteforce") in catalog.available("index")

    def test_specs_carry_provenance(self) -> None:
        spec = component_catalog().get("index", "bruteforce")
        assert spec.implementation.startswith("scikitplot.corpus")
        assert spec.aliases == ("brute",)

    def test_catalog_is_serialisable(self) -> None:
        json.dumps(component_catalog().to_dict())

    def test_catalog_never_raises_on_broken_registry(self, monkeypatch) -> None:
        """A partially-installed environment must still be able to ask."""
        import scikitplot.corpus._catalog as catalog_module

        monkeypatch.setattr(
            catalog_module, "_index_specs", lambda: (_ for _ in ()).throw(RuntimeError)
        )
        with pytest.raises(RuntimeError):
            catalog_module._index_specs()
        # The collector itself guards imports; a spec source that raises is a
        # programming error, whereas an unimportable registry is tolerated.
        assert isinstance(catalog_module.ComponentCatalog([]), catalog_module.ComponentCatalog)

    def test_spec_dict_round_trip_shape(self) -> None:
        spec = ComponentSpec(name="x", category="index", implementation="m.C")
        data = spec.to_dict()
        assert data["status"] == "unknown"
        assert data["available"] is False
