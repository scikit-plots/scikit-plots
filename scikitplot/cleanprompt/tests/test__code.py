"""
Tests for :mod:`scikitplot.cleanprompt._code`.

Notes
-----
**Developer notes.** Discovery is the half of the schema layer that must be
*conservative*, and conservatism is only testable from both sides. So each
class here pairs what must be found with what must not be, because a discovery
rule wide enough to catch every column will also rename ``df.shape``, and one
narrow enough to be safe will miss the columns a careful author declared in a
module-level list.

The third side is the one people forget: a fragment that does not parse must be
*named*. An empty result and an unread cell look identical to a caller, and
telling them apart is the whole of ``CP-023``.
"""

from __future__ import annotations

import pytest

from .._code import COLUMN_KEYWORDS, COLUMN_METHODS, Discovery, discover


def columns_of(source):
    """Return the sorted column names discovered in one fragment."""
    return sorted(discover([("fragment", source)]).columns)


class TestSubscriptDiscovery:
    """The positions where a string can only be a column label."""

    @pytest.mark.parametrize(
        "source",
        [
            "df['region_code']",
            'df["region_code"]',
            "df[['region_code']]",
            "out = frame[('region_code',)]",
            "df.loc[:, 'region_code']",
            "df.loc[mask, ['region_code']]",
        ],
    )
    def test_a_subscript_names_a_column(self, source):
        assert "region_code" in columns_of(source)

    def test_a_computed_subscript_is_not_a_discovery(self):
        """`df[f(x)]` gives no guarantee, so nothing is claimed."""
        assert columns_of("df[pick('region_code')]") == []

    def test_a_bare_slice_names_nothing(self):
        assert columns_of("df.loc[:]") == []


class TestKeywordDiscovery:
    """Parameters documented to take a column label."""

    @pytest.mark.parametrize("keyword", sorted(COLUMN_KEYWORDS)[:6])
    def test_every_declared_keyword_is_read(self, keyword):
        found = columns_of("f(x, {0}=['region_code'])".format(keyword))
        assert "region_code" in found

    def test_an_undeclared_keyword_is_ignored(self):
        assert columns_of("f(x, message='region_code')") == []

    @pytest.mark.parametrize("method", sorted(COLUMN_METHODS)[:4])
    def test_a_column_method_reads_its_first_argument(self, method):
        assert "region_code" in columns_of("df.{0}('region_code')".format(method))

    def test_both_halves_of_a_rename_are_columns(self):
        """The new name is the one the rest of the notebook goes on to use."""
        found = columns_of("df.rename(columns={'region_code': 'region'})")
        assert found == ["region", "region_code"]


class TestAttributeAccessIsNotDiscovery:
    """The deliberate asymmetry, asserted so it cannot be lost."""

    def test_attribute_access_alone_names_no_column(self):
        assert columns_of("x = df.region_code") == []

    def test_a_method_is_never_mistaken_for_a_column(self):
        assert columns_of("df.shape; df.copy(); model.coef_") == []

    def test_an_attribute_is_still_recorded_as_an_identifier(self):
        """So a stand-in cannot be generated that collides with it."""
        found = discover([("f", "x = df.region_code")])
        assert "region_code" in found.identifiers


class TestBindingResolution:
    """Constant propagation, confined to literal string sequences."""

    def test_a_list_used_as_a_selector_is_resolved(self):
        source = "ID_COLUMNS = ['customer_ssn', 'acct_ref']\ndf.drop(columns=ID_COLUMNS)"
        assert columns_of(source) == ["acct_ref", "customer_ssn"]

    def test_the_binding_may_be_in_an_earlier_fragment(self):
        found = discover(
            [
                ("cell 1", "ID_COLUMNS = ['customer_ssn']"),
                ("cell 2", "df.drop(columns=ID_COLUMNS)"),
            ]
        )
        assert "customer_ssn" in found.columns

    def test_the_site_records_which_binding_resolved_it(self):
        found = discover(
            [("f", "COLS = ['a_col']\ndf.drop(columns=COLS)")]
        )
        assert found.columns["a_col"] == ("binding:COLS",)

    def test_an_unused_binding_claims_nothing(self):
        found = discover([("f", "NUMERIC = ['acct_balance_usd']")])
        assert found.columns == {}
        assert found.bindings["NUMERIC"] == ("acct_balance_usd",)

    def test_a_computed_list_is_not_propagated(self):
        source = "COLS = [c for c in df]\ndf.drop(columns=COLS)"
        assert columns_of(source) == []

    def test_a_mixed_list_is_not_propagated(self):
        source = "COLS = ['a_col', 7]\ndf.drop(columns=COLS)"
        assert columns_of(source) == []


class TestObservedRoles:
    """Evidence in the artefact, never a guess."""

    def test_astype_states_a_role(self):
        found = discover([("f", "df = df.astype({'region_code': 'category'})")])
        assert found.observed_roles == {"region_code": "category"}

    def test_parse_dates_states_a_role(self):
        found = discover([("f", "pd.read_csv(p, parse_dates=['signup_date'])")])
        assert found.observed_roles["signup_date"] == "date"

    def test_a_dtype_mapping_states_a_role(self):
        found = discover([("f", "pd.read_csv(p, dtype={'n_orders': 'int64'})")])
        assert found.observed_roles["n_orders"] == "count"

    def test_an_unknown_dtype_states_nothing(self):
        found = discover([("f", "df.astype({'c_col': 'complex128'})")])
        assert found.observed_roles == {}


class TestUnparsedIsReported:
    """An unread cell and an empty cell must not look the same."""

    @pytest.mark.parametrize(
        "source", ["%matplotlib inline", "!pip install pandas", "def broken("]
    )
    def test_an_unparseable_fragment_is_named(self, source):
        assert discover([("cell 4", source)]).unparsed == ("cell 4",)

    def test_one_bad_cell_does_not_lose_the_others(self):
        found = discover(
            [("cell 1", "df['region_code']"), ("cell 2", "%%bash\nls")]
        )
        assert "region_code" in found.columns
        assert found.unparsed == ("cell 2",)

    def test_an_empty_fragment_is_not_reported_as_unparsed(self):
        assert discover([("cell 1", "   \n")]).unparsed == ()


class TestMerge:
    """A notebook is many fragments and one schema."""

    def test_sites_accumulate(self):
        found = discover(
            [("c1", "df['region_code']"), ("c2", "df.drop(columns=['region_code'])")]
        )
        assert found.columns["region_code"] == ("keyword:columns", "subscript")

    def test_merge_does_not_mutate_either_input(self):
        left = Discovery(columns={"a_col": ("subscript",)})
        right = Discovery(columns={"b_col": ("subscript",)})
        merged = left.merge(right)
        assert sorted(merged.columns) == ["a_col", "b_col"]
        assert sorted(left.columns) == ["a_col"]
        assert sorted(right.columns) == ["b_col"]


class TestDoctests:
    """Every documented example runs."""

    def test_doctests_pass(self):
        from ._isolated import assert_doctests_pass

        assert_doctests_pass("_code")
