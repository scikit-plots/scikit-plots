from __future__ import annotations

import shutil

from scikitplot.corpus._capabilities import (
    CapabilityRegistry,
    CapabilitySpec,
    CapabilityStatus,
    component_capabilities,
    default_capability_registry,
)
from scikitplot.corpus._backends import (
    BackendCandidate,
    BackendPolicy,
    BackendStatus,
    run_backend_chain,
)


def test_registry_separates_installation_from_asset_readiness(monkeypatch) -> None:
    registry = CapabilityRegistry(
        [
            CapabilitySpec(
                "demo:model",
                "demo",
                module="json",
                assets_required=True,
                asset_probe=lambda: False,
                remedy="install model",
            )
        ]
    )
    report = registry.probe("demo:model", selected=True)
    assert report.installed is True
    assert report.assets_ready is False
    assert report.ready is False
    assert report.status is CapabilityStatus.MISCONFIGURED
    assert report.selected is True
    assert report.active is False


def test_unknown_model_asset_is_not_guessed_ready() -> None:
    registry = CapabilityRegistry(
        [CapabilitySpec("demo:lazy-model", "demo", module="json", assets_required=True)]
    )
    report = registry.probe("demo:lazy-model")
    assert report.installed is True
    assert report.assets_ready is None
    assert report.ready is None
    assert report.status is CapabilityStatus.UNKNOWN


def test_active_runtime_is_stronger_evidence_than_preflight_unknown() -> None:
    registry = CapabilityRegistry(
        [CapabilitySpec("demo:lazy-model", "demo", module="json", assets_required=True)]
    )
    report = registry.probe("demo:lazy-model", selected=True, active=True)
    assert report.ready is True
    assert report.active is True
    assert report.status is CapabilityStatus.AVAILABLE


def test_default_registry_isolated_copy() -> None:
    first = default_capability_registry()
    second = default_capability_registry()
    first.register(CapabilitySpec("private:test", "private"))
    assert "private:test" in first.names()
    assert "private:test" not in second.names()


def test_component_capabilities_is_json_compatible() -> None:
    snapshot = component_capabilities(["xml:lxml"])
    record = snapshot["xml:lxml"]
    assert isinstance(record["status"], str)
    assert record["role"] == "xml"
    assert "installed" in record
    assert "ready" in record


def test_pytesseract_report_checks_system_binary(monkeypatch) -> None:
    registry = default_capability_registry()
    # The package may or may not be installed in the test environment. Make
    # module discovery deterministic without importing pytesseract itself.
    import scikitplot.corpus._capabilities as capabilities

    real_module_present = capabilities._module_present

    def present(module):
        if module == "pytesseract":
            return True, None
        return real_module_present(module)

    monkeypatch.setattr(capabilities, "_module_present", present)
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "tesseract" else "/x")
    # The built-in spec captured the _binary_probe lambda, which performs the
    # shutil.which lookup at call time.
    report = registry.probe("ocr:pytesseract")
    assert report.installed is True
    assert report.ready is False
    assert report.status is CapabilityStatus.MISCONFIGURED


def test_offline_policy_skips_download_capability() -> None:
    calls: list[str] = []
    registry = CapabilityRegistry(
        [CapabilitySpec("demo:model", "demo", module="json", assets_required=True, may_download=True)]
    )
    outcome = run_backend_chain(
        [
            BackendCandidate(
                "remote-model",
                lambda: calls.append("remote") or "remote",
                capability="demo:model",
                may_download=True,
            ),
            BackendCandidate("portable", lambda: calls.append("portable") or "ok"),
        ],
        default="",
        logger=__import__("logging").getLogger(__name__),
        component="test",
        operation="demo",
        policy=BackendPolicy.offline(),
        capability_registry=registry,
    )
    assert calls == ["portable"]
    assert outcome.backend == "portable"
    assert outcome.skipped == ("remote-model",)
    assert outcome.policy == "offline"
    assert outcome.status is BackendStatus.SUCCESS


def test_custom_policy_order_is_respected() -> None:
    calls: list[str] = []
    outcome = run_backend_chain(
        [
            BackendCandidate("a", lambda: calls.append("a") or "a"),
            BackendCandidate("b", lambda: calls.append("b") or "b"),
        ],
        default="",
        logger=__import__("logging").getLogger(__name__),
        component="test",
        operation="order",
        policy=BackendPolicy().with_order("b", "a"),
    )
    assert outcome.value == "b"
    assert outcome.backend == "b"
    assert calls == ["b"]


def test_first_available_policy_does_not_hide_first_failure() -> None:
    def fail():
        raise RuntimeError("broken")

    outcome = run_backend_chain(
        [BackendCandidate("a", fail), BackendCandidate("b", lambda: "b")],
        default="",
        logger=__import__("logging").getLogger(__name__),
        component="test",
        operation="first",
        policy="first",
    )
    assert outcome.status is BackendStatus.FAILED
    assert outcome.attempted == ("a",)


def test_offline_policy_allows_offline_capable_unknown_cache_probe() -> None:
    calls: list[str] = []
    registry = CapabilityRegistry(
        [CapabilitySpec(
            "demo:model",
            "demo",
            module="json",
            assets_required=True,
            may_download=True,
        )]
    )
    outcome = run_backend_chain(
        [BackendCandidate(
            "local-only-model",
            lambda: calls.append("local") or "ok",
            capability="demo:model",
            may_download=True,
            offline_capable=True,
        )],
        default="",
        logger=__import__("logging").getLogger(__name__),
        component="test",
        operation="offline-local-cache",
        policy=BackendPolicy.offline(),
        capability_registry=registry,
    )
    assert calls == ["local"]
    assert outcome.status is BackendStatus.SUCCESS
    assert outcome.backend == "local-only-model"


def test_offline_policy_skips_may_download_backend_without_local_only_guard() -> None:
    calls: list[str] = []
    outcome = run_backend_chain(
        [BackendCandidate(
            "unguarded-model",
            lambda: calls.append("should-not-run") or "ok",
            may_download=True,
            offline_capable=False,
        )],
        default="",
        logger=__import__("logging").getLogger(__name__),
        component="test",
        operation="offline-guard",
        policy=BackendPolicy.offline(),
    )
    assert calls == []
    assert outcome.skipped == ("unguarded-model",)
    assert outcome.status is BackendStatus.UNAVAILABLE


def test_all_normal_misses_are_exhausted_not_empty() -> None:
    outcome = run_backend_chain(
        [
            BackendCandidate("a", lambda: None, accept=lambda value: value is not None),
            BackendCandidate("b", lambda: None, accept=lambda value: value is not None),
        ],
        default=None,
        logger=__import__("logging").getLogger(__name__),
        component="test",
        operation="misses",
    )
    assert outcome.status is BackendStatus.EXHAUSTED
    assert outcome.succeeded is False
    assert outcome.missed == ("a", "b")


def test_no_candidates_is_unavailable_not_empty() -> None:
    outcome = run_backend_chain(
        [],
        default=[],
        logger=__import__("logging").getLogger(__name__),
        component="test",
        operation="none",
    )
    assert outcome.status is BackendStatus.UNAVAILABLE
    assert outcome.succeeded is False


def test_backend_policy_from_config_tunes_preset_without_new_named_variant() -> None:
    policy = BackendPolicy.from_config({
        "preset": "offline",
        "order": ["local", "fallback"],
        "include_unlisted": False,
        "on_exhausted": "raise",
        "name": "offline-local-strict",
    })
    assert policy.allow_network is False
    assert policy.allow_download is False
    assert policy.order == ("local", "fallback")
    assert policy.include_unlisted is False
    assert policy.raise_on_exhausted is True
    assert policy.to_dict()["on_exhausted"] == "raise"
    assert policy.to_dict()["order"] == ["local", "fallback"]


def test_backend_policy_from_config_rejects_unknown_keys() -> None:
    try:
        BackendPolicy.from_config({"preset": "offline", "allow_netwrok": False})
    except ValueError as exc:
        assert "allow_netwrok" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("misspelled backend policy field was silently accepted")


def test_backend_policy_mapping_is_accepted_by_runner() -> None:
    calls: list[str] = []
    outcome = run_backend_chain(
        [BackendCandidate("a", lambda: calls.append("a") or "a")],
        default="",
        logger=__import__("logging").getLogger(__name__),
        component="test",
        operation="mapping-policy",
        policy={"preset": "resilient", "order": ["a"]},
    )
    assert calls == ["a"]
    assert outcome.status is BackendStatus.SUCCESS


def test_capability_registry_exposes_stable_roles() -> None:
    registry = default_capability_registry()
    roles = registry.roles()
    assert "asr" in roles
    assert "ocr" in roles
    assert "pdf" in roles
    assert "edit-distance" in roles


def test_component_capabilities_can_discover_by_role() -> None:
    ocr = component_capabilities(role="ocr")
    assert set(ocr) == {"ocr:easyocr", "ocr:pytesseract"}
    assert all(report["role"] == "ocr" for report in ocr.values())


def test_edit_distance_capability_family_includes_safe_and_explicit_backends() -> None:
    distance = component_capabilities(role="edit-distance")
    assert {
        "distance:internal",
        "distance:rapidfuzz",
        "distance:levenshtein",
        "distance:python",
    } <= set(distance)
    assert distance["distance:python"]["ready"] is True


def test_backend_policy_config_round_trip() -> None:
    original = BackendPolicy.from_config({
        "preset": "offline",
        "name": "local-ci",
        "order": ["a", "b"],
        "include_unlisted": False,
        "on_exhausted": "raise",
    })
    restored = BackendPolicy.from_config(original.to_dict())
    assert restored == original


def test_pdf_capability_role_matches_pdfreader_runtime_backends() -> None:
    from scikitplot.corpus import default_capability_registry

    registry = default_capability_registry()
    assert registry.names(role="pdf") == ("pdf:pdfminer", "pdf:pypdf")


def test_xml_capability_role_includes_always_available_stdlib_fallback() -> None:
    from scikitplot.corpus import CapabilityStatus, default_capability_registry

    registry = default_capability_registry()
    assert registry.names(role="xml") == ("xml:lxml", "xml:stdlib")
    report = registry.probe("xml:stdlib")
    assert report.status is CapabilityStatus.AVAILABLE
    assert report.ready is True
