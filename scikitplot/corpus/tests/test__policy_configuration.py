"""Focused tests for composable Corpus policy configuration."""

from __future__ import annotations

import pytest

from scikitplot.corpus import (
    BackendPolicy,
    ErrorPolicy,
    FluentCorpus,
    RuntimePolicy,
    runtime_policy,
)


def test_runtime_policy_rejects_truthy_string_security_flag() -> None:
    with pytest.raises(TypeError, match="allow_network must be bool"):
        RuntimePolicy(allow_network="false")  # type: ignore[arg-type]


def test_runtime_policy_presets_and_mapping_round_trip() -> None:
    assert RuntimePolicy.offline().to_dict() == {"allow_network": False}
    assert RuntimePolicy.networked().to_dict() == {"allow_network": True}
    assert runtime_policy("local") == RuntimePolicy.offline()
    assert runtime_policy({"allow_network": True}) == RuntimePolicy.networked()
    assert RuntimePolicy.from_config(RuntimePolicy.networked().to_dict()) == RuntimePolicy.networked()


def test_runtime_policy_rejects_unknown_config_key() -> None:
    with pytest.raises(ValueError, match="unknown RuntimePolicy"):
        RuntimePolicy.from_config({"allow_network": False, "typo": True})


def test_materialize_accepts_runtime_policy_preset_without_execution() -> None:
    fluent = FluentCorpus().storage("memory")
    with fluent.materialize(policy="offline") as runtime:
        assert runtime.policy == RuntimePolicy.offline()
        assert runtime.documents == ()


def test_backend_policy_normalizes_sequence_order_and_error_policy_string() -> None:
    policy = BackendPolicy(
        order=["one", "two"],  # type: ignore[arg-type]
        on_exhausted="raise",  # type: ignore[arg-type]
    )
    assert policy.order == ("one", "two")
    assert policy.on_exhausted is ErrorPolicy.RAISE


def test_backend_policy_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        BackendPolicy(order=("same", "same"))


def test_backend_policy_rejects_truthy_string_boolean() -> None:
    with pytest.raises(TypeError, match="allow_network must be bool"):
        BackendPolicy(allow_network="false")  # type: ignore[arg-type]


def test_backend_policy_mapping_rejects_misspelled_setting() -> None:
    with pytest.raises(ValueError, match="unknown BackendPolicy"):
        BackendPolicy.from_config({"preset": "offline", "allow_netwrok": False})
