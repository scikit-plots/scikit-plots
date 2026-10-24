# scikitplot/feature_selection/tests/test_fs_assessments.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Built-in providers, the registry contract and manual overrides."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scikitplot.feature_selection import (
    AssessmentProvider,
    AssessmentRegistry,
    FeaturePolicy,
    PolicyError,
    default_registry,
    list_assessment_providers,
    profile_features,
    register_assessment,
)


def _state(series, name, policy=None):
    schema = profile_features(pd.DataFrame({"x": series}), policy)
    return schema["x"].state(name)


@pytest.mark.parametrize(
    ("values", "name", "expected"),
    [
        ([1, 1, 1], "constant", "true"),
        ([np.nan, np.nan], "constant", "true"),
        ([1, 2], "constant", "false"),
        ([[1], [2]], "constant", "false"),
        (["a", "b"] * 5, "binary", "true"),
        ([np.nan] * 3, "binary", "unknown"),
        ([0.5, 1.5, 2.5] * 10, "discrete", "false"),
        ([1, 2, 3] * 10, "discrete", "true"),
        (["a"] * 3, "discrete", "not_applicable"),
        ([1.0, 2.0, 3.0, 90.5] * 5, "skewed", "true"),
        ([1.0, 2.0, 3.0, 4.5] * 5, "skewed", "false"),
        ([1.0, 2.0, 30.5], "skewed", "unknown"),
        ([0, 1] * 10, "skewed", "not_applicable"),
        (["a", "b", "c"], "skewed", "not_applicable"),
        ([f"v{i}" for i in range(60)], "high_cardinality", "true"),
        (list(range(60)), "high_cardinality", "not_applicable"),
        ([np.nan, np.nan, 1.0], "high_missing", "true"),
        ([1.0, np.nan], "has_missing", "true"),
        ([f"id{i}" for i in range(25)], "identifier_like", "true"),
        ([f"id{i}" for i in range(5)], "identifier_like", "unknown"),
        (list(range(25)), "identifier_like", "not_applicable"),
        ([0.1 * i for i in range(25)], "identifier_like", "not_applicable"),
        ([1.0, 2.0, 2.5], "integer_valued", "false"),
        ([-1.0, 2.0], "non_negative", "false"),
        ([1.0, np.inf], "has_infinite", "true"),
        ([1.0] * 30 + [100.0], "has_outliers", "not_applicable"),
        ([1.0, 2.0, 3.0, 50.0], "has_outliers", "unknown"),
        ([1.0, 1.1, 1.2, 1.3] * 10 + [100.0], "has_outliers", "true"),
        (["a", "b"], "categorical", "true"),
        ([True, False], "categorical", "true"),
        ([0.5, 1.5, 2.5], "categorical", "false"),
        ([[1], 2], "categorical", "unknown"),
    ],
)
def test_builtin_states(values, name, expected):
    assert _state(values, name) == expected


def test_has_outliers_respects_the_minimum_ratio():
    values = [1.0, 1.1, 1.2, 1.3] * 10 + [100.0]  # one outlier in 41 values (2.4 %)
    assert _state(values, "has_outliers") == "true"
    policy = {
        "assessments": {
            "has_outliers": {
                "provider": "has_outliers",
                "params": {"min_outlier_ratio": 0.05},
            }
        }
    }
    assert _state(values, "has_outliers", policy) == "false"


def test_integer_identifiers_are_opt_in():
    policy = {
        "assessments": {
            "identifier_like": {
                "provider": "identifier_like",
                "params": {"include_integers": True},
            }
        }
    }
    assert _state(list(range(25)), "identifier_like", policy) == "true"


def test_declared_categorical_kind_changes_numeric_answers():
    policy = {"kinds": {"x": "categorical_nominal"}}
    schema = profile_features(pd.DataFrame({"x": [1.0, 2.0, 30.5] * 10}), policy)
    assert schema["x"].kind == "categorical_nominal"
    assert schema["x"].kind_origin == "declared"
    assert schema["x"].state("categorical") == "true"
    assert schema["x"].state("skewed") == "not_applicable"


def test_evidence_and_provenance_are_recorded():
    a = profile_features(pd.DataFrame({"x": [1.0, 2.0, 3.0, 90.5] * 5}))["x"].assess(
        "skewed"
    )
    assert a.provider == "skewed" and a.version == "1"
    assert dict(a.params) == {"threshold": 1.0, "min_count": 8}
    assert a.evidence_dict["direction"] == "right"
    assert a.reasons == ("abs_skewness_ge_threshold",)


def test_thresholds_come_from_the_policy():
    values = [1.0, 2.0, 3.0, 6.5] * 5
    assert _state(values, "skewed") == "false"
    lower = {
        "assessments": {"skewed": {"provider": "skewed", "params": {"threshold": 0.5}}}
    }
    assert _state(values, "skewed", lower) == "true"


def test_a_new_name_can_reuse_a_provider():
    policy = {
        "assessments": {
            "very_skewed": {"provider": "skewed", "params": {"threshold": 50.0}}
        }
    }
    schema = profile_features(pd.DataFrame({"x": [1.0, 2.0, 3.0, 90.5] * 5}), policy)
    assert schema["x"].is_("skewed") and not schema["x"].is_("very_skewed")


def test_disabled_assessments_are_not_run():
    policy = {"assessments": {"skewed": {"provider": "skewed", "enabled": False}}}
    spec = profile_features(pd.DataFrame({"x": [1.0, 2.0]}), policy)["x"]
    assert "skewed" not in spec.assessment_names
    with pytest.raises(KeyError, match="available"):
        spec.assess("skewed")


def test_overriding_a_disabled_assessment_is_refused():
    policy = {
        "assessments": {"skewed": {"provider": "skewed", "enabled": False}},
        "overrides": {"x": {"skewed": True}},
    }
    with pytest.raises(ValueError, match="disabled"):
        profile_features(pd.DataFrame({"x": [1.0, 2.0]}), policy)


def test_manual_override_keeps_the_inferred_answer():
    policy = {"overrides": {"x": {"skewed": True}}}
    a = profile_features(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.5] * 5}), policy)[
        "x"
    ].assess("skewed")
    assert (a.state, a.origin, a.inferred_state) == ("true", "manual", "false")
    assert a.reasons[0] == "manual_override"


def test_overrides_drive_kind_inference():
    policy = {"overrides": {"x": {"discrete": False}}}
    schema = profile_features(pd.DataFrame({"x": [1, 2, 3] * 10}), policy)
    assert schema["x"].kind == "numeric_continuous"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"thresh": 1.0}, "has no parameter"),
        ({"threshold": "1"}, "must be float"),
        ({"min_count": True}, "must be int"),
        ({"min_count": 1.5}, "must be int"),
        ({"threshold": -1.0}, "must be >= 0"),
        ({"min_count": 2}, "must be >= 3"),
    ],
)
def test_provider_parameters_are_validated_before_profiling(params, message):
    policy = {"assessments": {"skewed": {"provider": "skewed", "params": params}}}
    with pytest.raises(PolicyError, match=message):
        profile_features(pd.DataFrame({"x": [1.0, 2.0]}), policy)


def test_an_unknown_provider_is_refused():
    policy = {"assessments": {"mine": {"provider": "no_such_provider"}}}
    with pytest.raises(PolicyError, match="no assessment provider"):
        profile_features(pd.DataFrame({"x": [1.0]}), policy)


def test_custom_provider_through_a_private_registry():
    registry = default_registry.copy()

    @register_assessment("percentage", defaults={"tolerance": 0.0}, registry=registry)
    def percentage(context, tolerance):
        s = context.stats
        if not s.is_numeric:
            return context.result("not_applicable", "not_numeric")
        ok = s.min >= -tolerance and s.max <= 100 + tolerance
        return context.result("true" if ok else "false", "range", min=s.min, max=s.max)

    policy = {"assessments": {"percentage": {"provider": "percentage"}}}
    X = pd.DataFrame({"p": [0.0, 50.0, 99.0], "q": [0.0, 500.0, 1.0], "s": list("abc")})
    schema = profile_features(X, policy, registry=registry)
    assert [schema[c].state("percentage") for c in "pqs"] == [
        "true",
        "false",
        "not_applicable",
    ]
    with pytest.raises(PolicyError):
        profile_features(X, policy)  # not in the shared registry


def test_registry_refuses_silent_replacement_and_builtins_never_displace_users():
    registry = AssessmentRegistry()
    mine = AssessmentProvider("skewed", lambda c, **_: c.result("true"), version="mine")
    registry.register(mine)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(mine)
    registry._register_builtin(default_registry.get("skewed"))
    assert registry.get("skewed").version == "mine"
    registry.register(default_registry.get("skewed"), replace=True)
    assert registry.get("skewed").version == "1"


def test_a_provider_must_return_an_outcome():
    registry = default_registry.copy()
    register_assessment("bad", lambda context: True, registry=registry)
    with pytest.raises(TypeError, match="return context.result"):
        profile_features(
            pd.DataFrame({"x": [1]}),
            {"assessments": {"bad": {"provider": "bad"}}},
            registry=registry,
        )


def test_evidence_must_be_scalar():
    registry = default_registry.copy()
    register_assessment(
        "bad", lambda context: context.result("true", x=[1]), registry=registry
    )
    with pytest.raises(TypeError, match="evidence"):
        profile_features(
            pd.DataFrame({"x": [1]}),
            {"assessments": {"bad": {"provider": "bad"}}},
            registry=registry,
        )


def test_listing_shows_parameters_and_defaults():
    table = list_assessment_providers().set_index("name")
    assert table.loc["skewed", "parameters"] == {"min_count": 8, "threshold": 1.0}
    assert table.loc["discrete", "parameters"]["max_unique_ratio"] is None


def test_policy_object_is_accepted_directly():
    policy = FeaturePolicy(overrides={"x": {"constant": True}})
    assert profile_features(pd.DataFrame({"x": [1, 2]}), policy)["x"].is_("constant")
