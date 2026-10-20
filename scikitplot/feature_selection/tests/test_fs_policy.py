# scikitplot/feature_selection/tests/test_fs_policy.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""FeaturePolicy: defaults, strict loading, round trips, value-object contract."""

from __future__ import annotations

import copy
import dataclasses
import pickle

import pytest

from scikitplot.feature_selection import (
    DEFAULT_ASSESSMENTS,
    AssessmentSpec,
    BinningPolicy,
    ColumnRule,
    DriftPolicy,
    FeaturePolicy,
    IncompatibleSchemaError,
    PolicyError,
    dump_policy,
    load_policy,
)


def _rich_policy() -> FeaturePolicy:
    return FeaturePolicy(
        roles={
            "target": ["churn"],
            "identifier": {"columns": ["id"], "patterns": ["key_*"]},
        },
        kinds={"zip": "categorical_nominal", "notes": "text"},
        tags={
            "one_hot": {"columns": ["city"], "patterns": ["cat_*"]},
            "engineered": ["ratio"],
        },
        assessments={
            "skewed": {"provider": "skewed", "params": {"threshold": 0.75}},
            "very_skewed": {
                "provider": "skewed",
                "params": {"threshold": 2, "min_count": 10},
            },
            "has_outliers": {"provider": "has_outliers", "enabled": False},
        },
        overrides={"income": {"skewed": True}},
        binning={"rule": "knuth", "max_bins": 40},
        drift={"n_bins": 8, "psi_alert": 0.3},
        on_missing_column="ignore",
    )


def test_an_empty_policy_is_valid_and_lists_every_default():
    policy = FeaturePolicy()
    assert set(policy.assessments_map) == set(DEFAULT_ASSESSMENTS)
    assert policy.binning == BinningPolicy()
    assert policy.drift == DriftPolicy()
    assert FeaturePolicy.from_dict({}) == policy


def test_default_parameters_live_on_the_providers_only():
    # One threshold, one place: the policy's built-in specs carry no parameters.
    assert all(spec.params == () for spec in DEFAULT_ASSESSMENTS.values())


def test_assessments_merge_over_the_defaults():
    policy = _rich_policy()
    specs = policy.assessments_map
    assert specs["skewed"].params_dict == {"threshold": 0.75}
    assert specs["very_skewed"].provider == "skewed"
    assert specs["has_outliers"].enabled is False
    assert "constant" in specs


@pytest.mark.parametrize("fmt", ["json", "yaml", "toml"])
def test_policy_round_trips_through_every_format(fmt, tmp_path):
    pytest.importorskip("yaml") if fmt == "yaml" else None
    pytest.importorskip("tomli_w") if fmt == "toml" else None
    policy = _rich_policy()
    assert FeaturePolicy.loads(policy.dumps(fmt), fmt) == policy
    path = dump_policy(policy, tmp_path / f"policy.{fmt}")
    assert load_policy(path) == policy


def test_value_object_contract():
    policy = _rich_policy()
    assert hash(policy) == hash(copy.deepcopy(policy))
    assert pickle.loads(pickle.dumps(policy)) == policy
    assert dataclasses.replace(policy) == policy
    assert dataclasses.asdict(policy)["on_missing_column"] == "ignore"


def test_digest_is_stable_and_sensitive():
    a, b = _rich_policy(), _rich_policy()
    assert a.digest == b.digest and len(a.digest) == 64
    assert dataclasses.replace(a, on_missing_column="raise").digest != a.digest


def test_none_params_mean_provider_default():
    spec = AssessmentSpec("discrete", {"max_unique_ratio": None, "max_unique_count": 5})
    assert spec.params_dict == {"max_unique_count": 5}


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"rolez": {}}, "unknown key"),
        ({"roles": {"targett": ["y"]}}, "is not one of"),
        ({"roles": {"target": [1]}}, "must be a string"),
        ({"kinds": {"x": "numeric"}}, "is not one of"),
        ({"assessments": {"skewed": {"params": {}}}}, "needs a 'provider'"),
        (
            {"assessments": {"skewed": {"provider": "skewed", "colour": 1}}},
            "unknown key",
        ),
        (
            {"assessments": {"skewed": {"provider": "skewed", "enabled": 1}}},
            "must be a bool",
        ),
        (
            {"assessments": {"skewed": {"provider": "skewed", "params": {"t": [[1]]}}}},
            "str/int",
        ),
        ({"overrides": {"x": {"nope": True}}}, "no assessment named"),
        ({"overrides": {"x": {"skewed": 1}}}, "must be a bool"),
        ({"binning": {"rule": "magic"}}, "is not one of"),
        ({"binning": {"rule": "fixed"}}, "needs binning.n_bins"),
        ({"binning": {"n_bins": 5}}, "only used with rule='fixed'"),
        ({"binning": {"min_bins": 10, "max_bins": 5}}, "min_bins"),
        ({"binning": {"max_bins": True}}, "must be an int"),
        ({"drift": {"psi_warn": 0.5, "psi_alert": 0.1}}, "psi_warn"),
        ({"drift": {"epsilon": 0}}, "must be >"),
        ({"drift": {"colour": 1}}, "unknown key"),
        ({"on_missing_column": "skip"}, "is not one of"),
    ],
)
def test_off_contract_documents_are_refused(document, message):
    with pytest.raises(PolicyError, match=message):
        FeaturePolicy.from_dict(document)


def test_another_generation_is_refused_by_its_own_error():
    with pytest.raises(IncompatibleSchemaError):
        FeaturePolicy.from_dict({"schema_version": 2})
    with pytest.raises(PolicyError):
        FeaturePolicy.from_dict({"schema_version": True})


def test_yaml_dates_are_not_strings():
    pytest.importorskip("yaml")
    with pytest.raises(PolicyError, match="must be strings"):
        FeaturePolicy.loads("kinds:\n  2020-01-01: text\n", "yaml")
    with pytest.raises(PolicyError, match="must be a string"):
        FeaturePolicy.loads("roles:\n  target: [2020-01-01]\n", "yaml")


def test_yaml_aliases_are_refused_as_policy_errors():
    pytest.importorskip("yaml")
    with pytest.raises(PolicyError, match="aliases"):
        FeaturePolicy.loads("roles:\n  target: &t [y]\n  drop: *t\n", "yaml")


def test_a_column_with_two_roles_is_refused():
    policy = FeaturePolicy(roles={"target": ["y"], "drop": {"patterns": ["*"]}})
    with pytest.raises(PolicyError, match="exactly one role"):
        policy.resolve_roles(["x", "y"])


def test_missing_columns_raise_unless_ignored():
    with pytest.raises(PolicyError, match="not in the data"):
        FeaturePolicy(roles={"target": ["y"]}).resolve_roles(["x"])
    assert FeaturePolicy(
        roles={"target": ["y"]}, on_missing_column="ignore"
    ).resolve_roles(["x"]) == {"x": "feature"}
    with pytest.raises(PolicyError, match="kinds names"):
        FeaturePolicy(kinds={"z": "text"}).check_columns(["x"])


def test_column_rule_patterns_are_case_sensitive_everywhere():
    rule = ColumnRule(patterns=["Cat_*"])
    assert rule.resolve(["cat_a", "Cat_b"]) == ("Cat_b",)


def test_tags_are_independent_labels():
    policy = FeaturePolicy(tags={"a": ["x"], "b": {"patterns": ["x*"]}})
    assert policy.resolve_tags(["x", "xy", "z"]) == {
        "x": ("a", "b"),
        "xy": ("b",),
        "z": (),
    }


def test_jsonl_is_not_a_policy_format(tmp_path):
    with pytest.raises(PolicyError, match="one document"):
        FeaturePolicy().dumps("jsonl")
    (tmp_path / "p.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PolicyError, match="one document"):
        load_policy(tmp_path / "p.jsonl")


def test_pydantic_can_validate_the_dataclass():
    pydantic = pytest.importorskip("pydantic")
    adapter = pydantic.TypeAdapter(BinningPolicy)
    assert adapter.validate_python({"rule": "knuth"}).rule == "knuth"
    with pytest.raises((PolicyError, pydantic.ValidationError)):
        adapter.validate_python({"rule": "magic"})
