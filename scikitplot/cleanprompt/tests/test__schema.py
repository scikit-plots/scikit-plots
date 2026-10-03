"""
Tests for :mod:`scikitplot.cleanprompt._schema`.

Notes
-----
**Developer notes.** Three properties carry this module, and each has a silent
failure behind it.

*A stand-in must be safe to substitute.* Rewriting every ``count`` or ``type``
in a notebook changes code that has nothing to do with the dataset, and the
damage is spread over the whole document rather than raised at one place.

*A role must not be invented.* The split between ``role_for`` — declared or
observed — and ``infer_role`` — a suggestion — is the module's whole claim to
being deductive. A test that let an inferred role arrive through ``role_for``
would dissolve it.

*A stand-in must not collide*, with another stand-in or with a name the
document already uses, because a collision restores two things as one.
"""

from __future__ import annotations

import pytest

from .._exceptions import PolicyError
from .._schema import (
    NEUTRAL_ROLE,
    ROLES,
    SCHEMA_KINDS,
    infer_role,
    is_safe_to_rename,
    path_surrogate,
    role_for,
    schema_surrogate,
)


class TestSafeToRename:
    """The names that must be refused, and why."""

    @pytest.mark.parametrize(
        "name", ["acct_balance_usd", "region_code", "customerId", "x_1", "ab"]
    )
    def test_an_ordinary_column_is_accepted(self, name):
        assert is_safe_to_rename(name)[0] is True

    @pytest.mark.parametrize("name", ["type", "class", "id", "count", "sum", "index"])
    def test_a_reserved_or_common_name_is_refused(self, name):
        safe, why = is_safe_to_rename(name)
        assert safe is False
        assert why

    @pytest.mark.parametrize("name", ["x", "y", "n", "_"])
    def test_a_single_character_name_is_refused(self, name):
        assert is_safe_to_rename(name)[0] is False

    @pytest.mark.parametrize("name", ["total spend", "2024", "a-b", "", "col.name"])
    def test_a_non_identifier_is_refused(self, name):
        assert is_safe_to_rename(name)[0] is False

    def test_a_refusal_names_the_remedy(self):
        """A refusal the user cannot act on is only half a message."""
        _safe, why = is_safe_to_rename("x")
        assert "rename" in why or "hide" in why

    def test_a_pandas_attribute_is_refused(self):
        """Rewriting `count` would change every df.count() in the notebook."""
        assert is_safe_to_rename("count")[0] is False
        assert is_safe_to_rename("shape")[0] is False


class TestRoleProvenance:
    """Declared beats observed beats nothing; inference never arrives here."""

    def test_declared_wins(self):
        assert role_for("c", {"c": "date"}, {"c": "amount"}) == ("date", "declared")

    def test_observed_is_used_when_nothing_was_declared(self):
        assert role_for("c", None, {"c": "amount"}) == ("amount", "observed")

    def test_an_unknown_column_is_neutral_and_says_so(self):
        assert role_for("c") == (NEUTRAL_ROLE, "none")

    def test_an_inferable_name_is_still_neutral(self):
        """The firewall: a name is not evidence, so role_for must not use it."""
        assert infer_role("signup_date") == "date"
        assert role_for("signup_date") == (NEUTRAL_ROLE, "none")

    @pytest.mark.parametrize("source", ["declared", "observed"])
    def test_an_unknown_role_is_refused(self, source):
        mapping = {"c": "not-a-role"}
        with pytest.raises(PolicyError) as caught:
            role_for("c", mapping if source == "declared" else None,
                     mapping if source == "observed" else None)
        assert "not-a-role" in str(caught.value)


class TestInference:
    """The opt-in suggestion, and the cases it must not get wrong."""

    @pytest.mark.parametrize(
        "name,role",
        [
            ("customer_ssn", "id"),
            ("acct_balance_usd", "amount"),
            ("signup_date", "date"),
            ("created_at", "datetime"),
            ("region_code", "category"),
            ("conversion_rate", "rate"),
            ("num_orders", "count"),
            ("is_active", "flag"),
            ("comment_text", "text"),
        ],
    )
    def test_a_clear_name_is_classified(self, name, role):
        assert infer_role(name) == role

    def test_tenure_months_is_a_count_not_a_date(self):
        """A whole-token match: '_months_' must not read as '_month' in a date."""
        assert infer_role("tenure_months") == "count"

    def test_camel_case_is_split(self):
        assert infer_role("customerId") == "id"
        assert infer_role("accountBalance") == "amount"

    def test_an_unclear_name_returns_none(self):
        assert infer_role("wibble") is None
        assert infer_role("col_7") is None


class TestSchemaSurrogate:
    """Stand-ins: valid identifiers, unique, stable, collision-free."""

    def test_the_stem_reflects_the_role(self):
        assert schema_surrogate("amount", 1) == "amount_1"
        assert schema_surrogate("category", 3) == "category_3"

    def test_the_single_target_reads_without_a_suffix(self):
        assert schema_surrogate("target", 1) == "target"

    def test_a_second_target_falls_back_to_a_number(self):
        assert schema_surrogate("target", 1, avoid={"target"}) == "target_1"

    def test_every_role_produces_an_identifier(self):
        for role in ROLES:
            assert schema_surrogate(role, 1).isidentifier()

    def test_a_collision_is_stepped_past(self):
        assert schema_surrogate("amount", 1, avoid={"amount_1", "amount_2"}) == "amount_3"

    def test_it_is_deterministic(self):
        assert schema_surrogate("score", 4) == schema_surrogate("score", 4)

    def test_an_unknown_role_is_refused(self):
        with pytest.raises(PolicyError):
            schema_surrogate("nonsense", 1)

    @pytest.mark.parametrize("ordinal", [0, -1])
    def test_a_bad_ordinal_is_refused(self, ordinal):
        with pytest.raises(PolicyError):
            schema_surrogate("amount", ordinal)

    def test_many_stand_ins_stay_unique(self):
        issued = set()
        for ordinal in range(1, 200):
            one = schema_surrogate("field", ordinal, avoid=issued)
            assert one not in issued
            issued.add(one)


class TestPathSurrogate:
    """Keep the shape, discard everything that identifies."""

    def test_an_absolute_path_keeps_its_extension(self):
        assert path_surrogate("/mnt/prod/exports/q1_acme.parquet", 1) == (
            "/data/dataset_1.parquet"
        )

    def test_a_uri_keeps_its_scheme(self):
        assert path_surrogate("s3://acme/models/x.pkl", 1) == "s3://bucket/dataset_1.pkl"

    def test_a_relative_path_stays_relative(self):
        assert path_surrogate("reports/fig.png", 2) == "dataset_2.png"

    def test_a_traceback_line_reference_is_stripped(self):
        assert path_surrogate("/home/marion/work/features.py:18", 1) == (
            "/data/dataset_1.py"
        )
        assert path_surrogate("/a/b/mod.py:18:4", 1) == "/data/dataset_1.py"

    def test_nothing_of_the_original_survives(self):
        original = "/home/marion.holt/work/acme-churn/reports/fig_drivers.png"
        standin = path_surrogate(original, 1)
        for fragment in ("marion", "holt", "acme", "churn", "drivers", "home"):
            assert fragment not in standin

    def test_a_directory_without_an_extension_is_handled(self):
        assert path_surrogate("/mnt/prod/exports", 1) == "/data/dataset_1"

    def test_a_collision_is_stepped_past(self):
        first = path_surrogate("/a/b.csv", 1)
        assert path_surrogate("/c/d.csv", 1, avoid={first}) != first

    def test_a_bad_ordinal_is_refused(self):
        with pytest.raises(PolicyError):
            path_surrogate("/a/b.csv", 0)


class TestVocabulary:
    """The declared surface is the implemented one."""

    def test_the_neutral_role_exists(self):
        assert NEUTRAL_ROLE in ROLES

    def test_the_kinds_are_placeholder_safe(self):
        for kind in SCHEMA_KINDS:
            assert kind.isupper()
            assert not set(kind) & set("[]-")

    def test_doctests_pass(self):
        from ._isolated import assert_doctests_pass

        assert_doctests_pass("_schema")
