# scikitplot/feature_selection/tests/test_fs_rename_analysis.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""RenamePlan rules and collisions; analysis and treatment routing tables."""

from __future__ import annotations

import pandas as pd
import pytest

from scikitplot.feature_selection import (
    DEFAULT_ANALYSIS_RULES,
    RenamePlan,
    analysis_plan,
    profile_features,
    recommend_treatments,
    treatment_groups,
)

from ._helpers import mixed_frame, target_policy

# --------------------------------------------------------------------------- #
# RenamePlan
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("rule", "kwargs", "expected"),
    [
        ("f{index:03d}", {}, ["f000", "f001", "f002"]),
        ("col-{index}", {"start": 1}, ["col-1", "col-2", "col-3"]),
        ("{name}__v2", {}, ["a__v2", "b__v2", "c__v2"]),
        ({"a": "x"}, {}, ["x", "b", "c"]),
        (("p", "q", "r"), {}, ["p", "q", "r"]),
        (str.upper, {}, ["A", "B", "C"]),
    ],
)
def test_rules(rule, kwargs, expected):
    assert RenamePlan.build(["a", "b", "c"], rule, **kwargs).names_out == expected


@pytest.mark.parametrize(
    ("rule", "error", "message"),
    [
        ("x{colour}", ValueError, "not allowed"),
        ("x{name.upper}", ValueError, "attribute or index"),
        ("same", ValueError, "no field"),
        (["p", "q"], ValueError, "one name per column"),
        ({"zz": "x"}, ValueError, "not present"),
        ({"a": "b"}, ValueError, "would both be named"),
        (lambda n: 1, TypeError, "not str"),
        (lambda n: "", ValueError, "empty"),
        (3.5, TypeError, "rule must be"),
    ],
)
def test_refusals(rule, error, message):
    with pytest.raises(error, match=message):
        RenamePlan.build(["a", "b", "c"], rule)


def test_suffix_collisions_are_deterministic_and_skip_taken_names():
    plan = RenamePlan.build(
        ["a", "b", "c", "d"], {"b": "a", "c": "a", "d": "a_1"}, on_collision="suffix"
    )
    assert plan.names_out == ["a", "a_2", "a_3", "a_1"]


def test_missing_mapping_keys_can_be_ignored_and_plans_invert():
    plan = RenamePlan.build(["a", "b"], {"a": "x", "zz": "q"}, on_missing="ignore")
    assert plan.changed == {"a": "x"}
    assert plan.inverse().names_out == ["a", "b"]
    assert RenamePlan.from_dict(plan.to_dict()) == plan


def test_a_plan_applies_only_to_its_columns():
    plan = RenamePlan.build(["a", "b"], "c{index}")
    frame = pd.DataFrame({"a": [1], "b": [2]})
    assert list(plan.apply(frame).columns) == ["c0", "c1"]
    assert list(frame.columns) == ["a", "b"]
    with pytest.raises(ValueError, match="differ"):
        plan.apply(frame[["b", "a"]])
    with pytest.raises(ValueError):
        RenamePlan.from_dict({"pairs": [["a"]]})


# --------------------------------------------------------------------------- #
# Analysis routing
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def schema():
    return profile_features(mixed_frame(), target_policy())


def test_plan_routes_by_kind_and_target(schema):
    plan = analysis_plan(schema, "churn")
    assert "churn" not in plan.index
    assert plan.loc["income", "target_tests"] == ("mannwhitneyu", "ttest_ind")
    assert plan.loc["city", "target_tests"] == ("chi2_contingency",)
    assert plan.loc["vip", "target_tests"] == ("fisher_exact", "chi2_contingency")
    assert "skewed: log_scale_axis" in plan.loc["income", "notes"]
    assert plan.loc["joined", "target_tests"] == ()


def test_tests_are_named_after_scipy_functions():
    import scipy.stats

    for entry in DEFAULT_ANALYSIS_RULES["bivariate"].values():
        for name in entry["tests"]:
            assert callable(getattr(scipy.stats, name)), name


def test_plan_target_by_kind_and_refusals(schema):
    by_kind = analysis_plan(schema, "numeric_continuous")
    assert by_kind.loc["income", "target_tests"] == ("pearsonr", "spearmanr")
    assert analysis_plan(schema).loc["income", "target_tests"] == ()
    with pytest.raises(ValueError, match="target"):
        analysis_plan(schema, "joined")


def test_custom_rules_replace_the_defaults(schema):
    rules = {
        "univariate": {"numeric_continuous": {"plots": ("violin",), "summaries": ()}},
        "bivariate": {},
        "notes": {},
    }
    plan = analysis_plan(schema, rules=rules)
    assert plan.loc["income", "plots"] == ("violin",)
    assert plan.loc["city", "plots"] == ()


def test_treatments_follow_the_assessments(schema):
    steps = recommend_treatments(schema)["steps"]
    assert steps["const"] == ("drop",)
    assert steps["income"] == ("PowerTransformer(method='yeo-johnson')",)
    assert steps["age"][0] == "SimpleImputer(strategy='median')"
    assert steps["city"] == ("OneHotEncoder(handle_unknown='infrequent_if_exist')",)
    assert steps["vip"] == ("passthrough",)
    assert steps["grade"] == ("OrdinalEncoder(categories=<fitted order>)",)
    assert "customer_id" not in steps.index  # role=identifier


def test_treatment_groups_are_column_transformer_branches(schema):
    table = recommend_treatments(schema)
    groups = treatment_groups(table)
    for name, steps in table["steps"].items():
        assert name in groups[steps]
    assert "kids" in groups[("StandardScaler()",)]
    assert sum(len(v) for v in groups.values()) == len(schema.feature_names)


def test_a_custom_rule_is_used(schema):
    table = recommend_treatments(schema, rule=lambda spec: ((spec.kind,), ("mine",)))
    assert table.loc["income", "steps"] == ("numeric_continuous",)
