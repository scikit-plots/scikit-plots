# scikitplot/feature_selection/tests/test_fs_schema.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""FeatureSchema: prototype regressions, edge cases, queries and persistence."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from scikitplot.feature_selection import (
    FeaturePolicy,
    FeatureSchema,
    IncompatibleSchemaError,
    SchemaMismatchError,
    profile_features,
)

from ._helpers import mixed_frame, target_policy


@pytest.fixture(scope="module")
def schema():
    return profile_features(mixed_frame(), target_policy())


# --------------------------------------------------------------------------- #
# Regressions for the defects of the prototype classify_variable_type
# --------------------------------------------------------------------------- #


def test_an_identifier_is_not_constant(schema):
    # FS-PROT-001: unique_ratio >= uniq_thr was filed under "const_col".
    spec = profile_features(mixed_frame())["customer_id"]
    assert spec.is_("identifier_like") and not spec.is_("constant")


def test_cardinality_thresholds_name_their_unit():
    # FS-PROT-002: a ratio was compared with a count (bin_enc_thr=25).
    X = pd.DataFrame({"c": [f"v{i % 30}" for i in range(3000)]})
    by_count = {
        "assessments": {
            "high_cardinality": {
                "provider": "high_cardinality",
                "params": {"min_unique_count": 25},
            }
        }
    }
    by_ratio = {
        "assessments": {
            "high_cardinality": {
                "provider": "high_cardinality",
                "params": {"min_unique_count": 10_000, "min_unique_ratio": 0.5},
            }
        }
    }
    assert profile_features(X, by_count)["c"].is_("high_cardinality")
    assert not profile_features(X, by_ratio)["c"].is_("high_cardinality")


def test_every_policy_option_changes_the_result():
    # FS-PROT-004: target_enc_cols was accepted and ignored.
    X = pd.DataFrame({"a": [1.0, 2.0, 3.5] * 10, "b": list("xyz") * 10})
    base = profile_features(X)
    variants = {
        "roles": {"roles": {"drop": ["a"]}},
        "kinds": {"kinds": {"a": "categorical_nominal"}},
        "tags": {"tags": {"t": ["a"]}},
        "assessments": {
            "assessments": {"skewed": {"provider": "skewed", "enabled": False}}
        },
        "overrides": {"overrides": {"a": {"skewed": True}}},
        "binning": {"binning": {"enabled": False}},
    }
    for key, policy in variants.items():
        changed = profile_features(X, policy)
        assert changed.features != base.features, key


def test_no_column_name_is_special():
    # FS-PROT-005: "tenure" was hard coded as engineered.
    X = pd.DataFrame({"tenure": [1.5, 2.5, 3.5] * 10})
    spec = profile_features(X)["tenure"]
    assert spec.role == "feature" and spec.tags == ()


def test_feature_names_exclude_other_roles(schema):
    # FS-PROT-006: "feature" was reset to every column.
    assert "churn" not in schema.feature_names
    assert "customer_id" not in schema.feature_names
    assert schema["churn"].role == "target"
    assert len(schema.names) == len(mixed_frame().columns)


def test_traits_are_independent_of_kind(schema):
    # FS-PROT-008: "skewed" replaced "continuous".
    income = schema["income"]
    assert income.kind == "numeric_continuous" and income.is_("skewed")


def test_profiling_is_deterministic():
    # FS-PROT-012: a timestamp made identical inputs differ.
    a = profile_features(mixed_frame(), target_policy())
    b = profile_features(mixed_frame(), target_policy())
    assert a == b and a.to_dict() == b.to_dict()


# --------------------------------------------------------------------------- #
# Kinds and families
# --------------------------------------------------------------------------- #


def test_kinds_of_the_mixed_frame(schema):
    kinds = {spec.name: spec.kind for spec in schema}
    assert kinds == {
        "customer_id": "categorical_nominal",
        "income": "numeric_continuous",
        "height": "numeric_continuous",
        "age": "numeric_continuous",
        "kids": "numeric_discrete",
        "city": "categorical_nominal",
        "vip": "binary",
        "const": "numeric_discrete",
        "empty": "unknown",
        "grade": "categorical_ordinal",
        "joined": "datetime",
        "churn": "binary",
    }


@pytest.mark.parametrize(
    ("series", "family", "kind"),
    [
        (pd.Series([1, 2, 3, None], dtype="Int64"), "integer", "numeric_discrete"),
        (pd.Series([1, 2, None], dtype="Int64"), "integer", "binary"),
        (pd.Series([True, None, False], dtype="boolean"), "boolean", "binary"),
        (pd.Series([True, False, True], dtype=object), "boolean", "binary"),
        (pd.Series(["1", "2", "3"]), "string", "categorical_nominal"),
        (pd.Series(["a", None, "b"], dtype="string"), "string", "binary"),
        (pd.Series(pd.to_timedelta([1, 2, 3], unit="s")), "timedelta", "timedelta"),
        (
            pd.Series(pd.date_range("2020", periods=3, tz="Europe/Istanbul")),
            "datetime",
            "datetime",
        ),
        (pd.Series([1 + 1j, 2j]), "object", "other"),
        (pd.Series([{"a": 1}, {"b": 2}]), "object", "other"),
        (pd.Series([1, "a", 2.5], dtype=object), "object", "other"),
        (pd.Series([np.nan, np.nan]), "float", "unknown"),
    ],
)
def test_family_and_kind_edge_cases(series, family, kind):
    spec = profile_features(pd.DataFrame({"x": series}))["x"]
    assert (spec.family, spec.kind) == (family, kind)


def test_pyarrow_dtypes_are_read_without_implicit_casts():
    pa = pytest.importorskip("pyarrow")
    X = pd.DataFrame(
        {
            "i": pd.Series([1, 2, None, 4] * 5, dtype=pd.ArrowDtype(pa.int64())),
            "s": pd.Series(list("abcd") * 5, dtype=pd.ArrowDtype(pa.string())),
        }
    )
    schema = profile_features(X)
    assert (schema["i"].family, schema["s"].family) == ("integer", "string")
    assert schema["i"].stats.n_missing == 5


def test_unhashable_cells_give_unknown_not_errors():
    X = pd.DataFrame({"x": [[1, 2], [3], [1, 2]]})
    spec = profile_features(X)["x"]
    assert spec.family == "object"
    assert spec.state("categorical") == "unknown"


def test_infinities_are_counted_and_excluded_from_moments():
    X = pd.DataFrame({"x": [1.0, 2.0, np.inf, -np.inf, 3.0] * 4})
    spec = profile_features(X)["x"]
    assert spec.stats.n_inf == 8 and spec.stats.n_finite == 12
    assert spec.stats.max == 3.0 and spec.is_("has_infinite")


def test_constant_data_has_no_moments():
    spec = profile_features(pd.DataFrame({"x": [2.5] * 20}))["x"]
    assert spec.stats.skewness is None and spec.state("skewed") == "not_applicable"


def test_empty_and_zero_column_frames():
    empty = profile_features(pd.DataFrame({"x": pd.Series([], dtype=float)}))
    assert empty["x"].state("constant") == "unknown" and empty.n_rows == 0
    assert len(profile_features(pd.DataFrame(index=range(3)))) == 0


@pytest.mark.parametrize(
    ("frame", "error", "message"),
    [
        (np.zeros((2, 2)), TypeError, "pandas DataFrame"),
        (pd.DataFrame({0: [1], 1: [2]}), TypeError, "must all be strings"),
        (pd.DataFrame([[1, 2]], columns=["a", "a"]), ValueError, "duplicated"),
    ],
)
def test_frames_are_checked(frame, error, message):
    with pytest.raises(error, match=message):
        profile_features(frame)


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #


def test_select_combines_conditions(schema):
    assert schema.select(kind="numeric_continuous", is_=("skewed",)) == ["income"]
    assert schema.select(
        kind=["numeric_continuous", "numeric_discrete"], is_not=("skewed", "constant")
    ) == ["height", "age", "kids"]
    assert schema.select(role=None, kind="binary") == ["vip", "churn"]
    assert schema.select(family="string") == ["city"]
    assert schema.select(role="identifier") == ["customer_id"]


def test_unknown_is_neither_true_nor_excluded_by_is_not():
    X = pd.DataFrame({"few": [1.0, 2.0, 30.5]})  # skewness needs 8 values
    s = profile_features(X)
    assert s["few"].state("skewed") == "unknown"
    assert s.select(is_=("skewed",)) == [] and s.select(is_not=("skewed",)) == ["few"]


@pytest.mark.parametrize(
    "kwargs",
    [{"kind": "numeric"}, {"family": "int"}, {"role": "label"}, {"is_": ("skew",)}],
)
def test_select_refuses_unknown_names(schema, kwargs):
    with pytest.raises(ValueError):
        schema.select(**kwargs)


def test_groups(schema):
    by_kind = schema.groups()
    assert by_kind["numeric_continuous"] == ["income", "height", "age"]
    assert "churn" not in sum(by_kind.values(), [])
    assert "skewed" in schema.groups(by="assessment")
    assert schema.groups(by="role", role=None)["target"] == ["churn"]
    with pytest.raises(ValueError):
        schema.groups(by="colour")


def test_tags_group_features():
    policy = {"tags": {"one_hot": ["city"], "money": {"patterns": ["inc*"]}}}
    s = profile_features(mixed_frame(), policy)
    assert s.groups(by="tag") == {"money": ["income"], "one_hot": ["city"]}
    assert s.select(tag="money") == ["income"]


def test_summary_and_tables(schema):
    summary = schema.summary()
    assert summary.loc["income", "kind"] == "numeric_continuous"
    assert summary.loc["income", "skewed"] == "true"
    table = schema.assessment_table()
    assert table.shape == (len(schema), len(schema.policy.assessments))
    plan = schema.binning_plan()
    assert plan.loc["city", "state"] == "not_applicable"


def test_explain_mentions_the_answers(schema):
    text = schema["income"].explain()
    assert "numeric_continuous" in text and "skewed" in text and "binning" in text


# --------------------------------------------------------------------------- #
# Later data, renaming, persistence
# --------------------------------------------------------------------------- #


def test_audit_reports_structure_without_refitting(schema):
    later = mixed_frame(seed=1).drop(columns=["kids"])
    later["extra"] = 1
    later["age"] = later["age"].astype(str)
    later = later[["income"] + [c for c in later.columns if c != "income"]]
    findings = schema.audit(later).set_index("column")["issue"].to_dict()
    assert findings["kids"] == "missing_column"
    assert findings["extra"] == "extra_column"
    assert findings["age"] == "family_changed"
    assert findings["*"] == "order_changed"
    assert schema == profile_features(mixed_frame(), target_policy())
    with pytest.raises(SchemaMismatchError, match="missing_column"):
        schema.check(later)


def test_dtype_changes_inside_a_family_are_informational(schema):
    later = mixed_frame()
    later["kids"] = later["kids"].astype("Int64")
    assert schema.audit(later)["issue"].tolist() == ["dtype_changed"]
    schema.check(later)


def test_rename_keeps_identity_and_aliases(schema):
    renamed = schema.rename({"income": "income_try"})
    spec = renamed["income_try"]
    assert spec.aliases == ("income",) and spec.stats.name == "income_try"
    assert spec.assessments == schema["income"].assessments
    assert renamed.resolve_alias("income") == "income_try"
    with pytest.raises(ValueError, match="duplicate"):
        schema.rename({"income": "height"})
    with pytest.raises(KeyError):
        schema.rename({"nope": "x"})


@pytest.mark.parametrize("fmt", ["json", "yaml", "toml"])
def test_schema_round_trips(schema, fmt, tmp_path):
    pytest.importorskip("yaml") if fmt == "yaml" else None
    pytest.importorskip("tomli_w") if fmt == "toml" else None
    assert FeatureSchema.loads(schema.dumps(fmt), fmt) == schema
    path = schema.save(tmp_path / f"schema.{fmt}")
    assert FeatureSchema.load(path) == schema


def test_schema_value_contract(schema):
    assert pickle.loads(pickle.dumps(schema)) == schema
    assert copy.deepcopy(schema) == schema
    assert hash(schema) == hash(copy.deepcopy(schema))


def test_persisted_schema_is_checked(schema):
    data = schema.to_dict()
    with pytest.raises(IncompatibleSchemaError):
        FeatureSchema.from_dict({**data, "schema_version": 2})
    with pytest.raises(IncompatibleSchemaError):
        FeatureSchema.from_dict({**data, "artifact": "other"})
    with pytest.raises(ValueError, match="digest"):
        FeatureSchema.from_dict({**data, "policy_digest": "0" * 64})
    tampered = copy.deepcopy(data)
    tampered["features"][0]["stats"]["n_rows"] = "200"
    with pytest.raises(ValueError, match="n_rows"):
        FeatureSchema.from_dict(tampered)
    tampered = copy.deepcopy(data)
    tampered["features"][0]["colour"] = 1
    with pytest.raises(ValueError, match="off-contract"):
        FeatureSchema.from_dict(tampered)


def test_a_path_is_accepted_as_policy(tmp_path):
    path = tmp_path / "p.json"
    path.write_text('{"roles": {"target": ["churn"]}}', encoding="utf-8")
    assert profile_features(mixed_frame(), str(path))["churn"].role == "target"
    assert profile_features(mixed_frame(), FeaturePolicy())["churn"].role == "feature"


def test_wide_table_is_linear_and_complete():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 300)), columns=[f"f{i}" for i in range(300)])
    schema = profile_features(X)
    assert len(schema) == 300 and set(schema.groups()) == {"numeric_continuous"}
