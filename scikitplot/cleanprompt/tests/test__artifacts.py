"""
Tests for :mod:`scikitplot.cleanprompt._artifacts`.

Notes
-----
**Developer notes.** This layer makes one promise that is easy to state and
easy to break: **the file that comes out is the same kind of file that went
in, it discloses nothing, and it restores exactly**. All three halves have to
hold at once, and a change that satisfies any two is a plausible bug.

The fourth property is the one that separates this from a redaction that
merely works: the output has to remain *useful*. A notebook whose every column
is ``[COLUMN-7]`` is safe and worthless, so the tests below also assert that the
code still parses and that the roles a model needs survive the substitution.
"""

from __future__ import annotations

import ast
import json

import pytest

from .._artifacts import PATH_PATTERN, encode_artifact, plan_artifact
from .._engine import restore
from .._exceptions import CleanPromptError

SOURCE = """\
import pandas as pd

ID_COLUMNS = ["customer_ssn", "acct_external_ref"]

df = pd.read_parquet('/mnt/prod/exports/2026_q1_acme_pii.parquet')
df = df[['customer_ssn', 'acct_balance_usd', 'signup_date', 'region_code']]
df['balance_log'] = np.log1p(df.acct_balance_usd)
df = df.drop(columns=ID_COLUMNS)
"""


def notebook_text(cells):
    """Return a notebook as raw JSON."""
    return json.dumps(
        {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5},
        indent=1,
    )


def code_cell(source, outputs=None):
    """Return one code cell."""
    return {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": outputs or [],
        "source": [source],
    }


class TestPlan:
    """What will happen, decided before anything does."""

    def test_columns_are_discovered_from_the_module(self):
        plan = plan_artifact(SOURCE, "python")
        assert "acct_balance_usd" in plan.columns
        assert "region_code" in plan.columns

    def test_a_binding_is_resolved(self):
        plan = plan_artifact(SOURCE, "python")
        assert "acct_external_ref" in plan.columns

    def test_a_role_is_neutral_without_evidence_and_says_so(self):
        plan = plan_artifact(SOURCE, "python")
        _role, provenance, standin = plan.columns["acct_balance_usd"]
        assert provenance == "none"
        assert standin.startswith("field")

    def test_inference_is_opt_in_and_labelled(self):
        plan = plan_artifact(SOURCE, "python", infer_roles=True)
        role, provenance, standin = plan.columns["acct_balance_usd"]
        assert (role, provenance) == ("amount", "inferred")
        assert standin == "amount_1"

    def test_a_declared_role_beats_inference(self):
        plan = plan_artifact(
            SOURCE,
            "python",
            declared_roles={"acct_balance_usd": "score"},
            infer_roles=True,
        )
        assert plan.columns["acct_balance_usd"][:2] == ("score", "declared")

    def test_stand_ins_do_not_depend_on_cell_order(self):
        """I3 on the schema side: a reordered notebook must redact the same."""
        first = plan_artifact(SOURCE, "python", infer_roles=True).columns
        shuffled = "\n".join(reversed(SOURCE.splitlines()))
        second = plan_artifact(shuffled, "python", infer_roles=True).columns
        shared = set(first) & set(second)
        assert shared
        for name in shared:
            assert first[name][2] == second[name][2]

    def test_a_refusal_is_reported_rather_than_hidden(self):
        plan = plan_artifact("df[['type', 'x']]", "python")
        assert set(plan.refused) == {"type", "x"}
        assert plan.columns == {}

    def test_an_unused_column_list_becomes_a_suggestion(self):
        plan = plan_artifact("NUMERIC = ['a_col', 'b_col']\n", "python")
        assert plan.suggestions == ("NUMERIC (a_col, b_col)",)

    def test_a_partly_used_list_still_suggests_the_rest(self):
        """
        The case that matters.

        Notes
        -----
        **Developer notes.** The first rule required the whole list to be
        uncovered, which silenced exactly the shape a feature module is
        written in: ``NUMERIC = [a, b, c]`` where only ``a`` is ever used as a
        selector, so ``b`` and ``c`` were hidden from the redaction *and* from
        the report.
        """
        plan = plan_artifact(
            "NUMERIC = ['a_col', 'b_col', 'c_col']\ndf[['a_col']]\n", "python"
        )
        assert plan.suggestions == ("NUMERIC (b_col, c_col)",)
        assert "a_col" in plan.columns

    def test_a_name_bound_to_one_column_is_resolved(self):
        """`TARGET = "churned"` then `df[TARGET]` is how modules are written."""
        plan = plan_artifact('TARGET = "churned"\ny = df[TARGET]\n', "python")
        assert "churned" in plan.columns

    def test_extra_columns_answer_a_suggestion(self):
        plan = plan_artifact(
            "NUMERIC = ['a_col', 'b_col']\n", "python", extra_columns=["a_col"]
        )
        assert "a_col" in plan.columns

    def test_the_report_carries_no_original_value(self):
        payload = json.dumps(plan_artifact(SOURCE, "python").as_dict())
        assert "/mnt/prod" not in payload


class TestModuleRoundTrip:
    """The three halves of the promise, on a .py file."""

    def test_no_column_name_survives(self):
        result, _plan = encode_artifact(SOURCE, "python", infer_roles=True)
        for name in ("customer_ssn", "acct_balance_usd", "region_code", "signup_date"):
            assert name not in result.text

    def test_no_path_survives(self):
        result, _plan = encode_artifact(SOURCE, "python")
        assert "/mnt/prod" not in result.text
        assert "acme" not in result.text

    def test_attribute_access_is_rewritten_too(self):
        """Discovery ignores `df.x`; rewriting must not."""
        result, _plan = encode_artifact(SOURCE, "python", infer_roles=True)
        assert "df.acct_balance_usd" not in result.text
        assert "df.amount_1" in result.text

    def test_the_round_trip_is_exact(self):
        result, _plan = encode_artifact(SOURCE, "python", infer_roles=True)
        assert restore(result.text, result.vault).text == SOURCE

    def test_the_result_still_parses_as_python(self):
        """Safe and unusable is not the goal."""
        result, _plan = encode_artifact(SOURCE, "python", infer_roles=True)
        ast.parse(result.text)

    def test_the_roles_a_model_needs_survive(self):
        result, _plan = encode_artifact(SOURCE, "python", infer_roles=True)
        assert "amount_1" in result.text
        assert "category_1" in result.text
        assert "id_1" in result.text

    def test_a_model_rewriting_the_code_still_restores(self):
        """The reply is new code that did not exist at encode time."""
        result, _plan = encode_artifact(SOURCE, "python", infer_roles=True)
        reply = "Try df.groupby('category_1')['amount_1'].mean() instead."
        restored = restore(reply, result.vault).text
        assert "region_code" in restored
        assert "acct_balance_usd" in restored


class TestNotebook:
    """The same promise, on a notebook."""

    @staticmethod
    def _notebook():
        return notebook_text(
            [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["# Churn for Acme\n"],
                },
                code_cell(
                    "df = pd.read_parquet('/mnt/prod/acme.parquet')\n",
                    outputs=[
                        {
                            "output_type": "stream",
                            "name": "stdout",
                            "text": ["region_code\nAC-NORTH  41822\n"],
                        }
                    ],
                ),
                code_cell(
                    "df[['region_code', 'acct_balance_usd']]\n",
                    outputs=[
                        {
                            "output_type": "display_data",
                            "metadata": {},
                            "data": {"image/png": "iVBORw0KGgoAAAANSUhEUg"},
                        }
                    ],
                ),
            ]
        )

    def test_the_output_is_still_a_notebook(self):
        result, _plan = encode_artifact(self._notebook(), path="a.ipynb")
        document = json.loads(result.text)
        assert len(document["cells"]) == 3
        assert document["nbformat"] == 4

    def test_cell_structure_is_preserved(self):
        raw = self._notebook()
        result, _plan = encode_artifact(raw, path="a.ipynb")
        before, after = json.loads(raw), json.loads(result.text)
        assert [one["cell_type"] for one in before["cells"]] == [
            one["cell_type"] for one in after["cells"]
        ]
        assert [one.get("execution_count") for one in before["cells"]] == [
            one.get("execution_count") for one in after["cells"]
        ]

    def test_the_notebook_structure_survives(self):
        """
        Structure is not data.

        Notes
        -----
        **Developer notes.** The first version of the role mapping treated
        everything under ``outputs`` as a rendered result, which produced
        ``"output_type": "[OUTPUT-1]"`` — a redaction that disclosed nothing
        and left behind a file no tool could open. These are the members that
        must come back byte-identical.
        """
        raw = self._notebook()
        result, _plan = encode_artifact(raw, path="a.ipynb")
        before, after = json.loads(raw), json.loads(result.text)
        for old_cell, new_cell in zip(before["cells"], after["cells"]):
            for old_out, new_out in zip(
                old_cell.get("outputs", []), new_cell.get("outputs", [])
            ):
                for key in ("output_type", "name", "execution_count", "ename"):
                    assert old_out.get(key) == new_out.get(key), key

    def test_every_output_type_is_still_a_valid_one(self):
        valid = {"execute_result", "display_data", "stream", "error"}
        result, _plan = encode_artifact(self._notebook(), path="a.ipynb")
        for cell in json.loads(result.text)["cells"]:
            for output in cell.get("outputs", []):
                assert output["output_type"] in valid

    def test_rendered_output_is_removed_by_default(self):
        result, _plan = encode_artifact(self._notebook(), path="a.ipynb")
        assert "AC-NORTH" not in result.text
        assert "41822" not in result.text

    def test_keeping_outputs_is_possible_and_explicit(self):
        result, _plan = encode_artifact(
            self._notebook(), path="a.ipynb", drop_outputs=False
        )
        assert "AC-NORTH" in result.text

    def test_a_figure_payload_is_dropped(self):
        result, _plan = encode_artifact(self._notebook(), path="a.ipynb")
        assert "iVBORw0KGgoAAAANSUhEUg" not in result.text

    def test_the_round_trip_is_exact(self):
        raw = self._notebook()
        result, _plan = encode_artifact(raw, path="a.ipynb")
        assert restore(result.text, result.vault).text == raw

    def test_the_format_is_detected_from_the_suffix(self):
        _result, plan = encode_artifact(self._notebook(), path="a.ipynb")
        assert plan.fmt == "notebook"

    def test_a_malformed_notebook_is_refused(self):
        with pytest.raises(CleanPromptError):
            encode_artifact('{"cells": [', path="broken.ipynb")


class TestCellIsTheParseUnit:
    """
    A notebook stores lines; a statement spans them.

    Notes
    -----
    **Developer notes.** The region splitter yields one region per source
    *line*, which is correct for rewriting — each line is its own JSON string
    with its own offsets — and wrong for parsing. A selection written across
    two lines gives two fragments, neither of them valid Python, and the
    columns inside are never discovered.

    The symptom was mild enough to miss: three of six columns hidden, and a
    truthful "NOT read" line in the report that looked like a notebook quirk.
    """

    @staticmethod
    def _wrapped():
        return notebook_text(
            [
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "df = df[['customer_ssn', 'acct_balance_usd',\n",
                        "         'region_code', 'signup_date']]\n",
                    ],
                }
            ]
        )

    def test_a_statement_split_across_lines_is_parsed(self):
        plan = plan_artifact(self._wrapped(), path="a.ipynb")
        assert set(plan.columns) == {
            "customer_ssn",
            "acct_balance_usd",
            "region_code",
            "signup_date",
        }

    def test_nothing_is_reported_unread(self):
        plan = plan_artifact(self._wrapped(), path="a.ipynb")
        assert plan.unparsed == ()

    def test_the_columns_on_the_second_line_are_hidden(self):
        result, _plan = encode_artifact(self._wrapped(), path="a.ipynb")
        assert "region_code" not in result.text
        assert "signup_date" not in result.text

    def test_a_cell_that_is_genuinely_unparseable_is_still_named(self):
        raw = notebook_text([code_cell("%matplotlib inline\n")])
        assert plan_artifact(raw, path="a.ipynb").unparsed == ("cells[0]",)


class TestExtraTermsReachTheArtefactPath:
    """
    ``--hide`` must not be dropped when the input is a notebook.

    Notes
    -----
    **Developer notes.** An artefact still contains the things prose does: an
    organisation no pattern recognises, a project codename, a client. The
    first version of the artefact branch did not thread the literal terms
    through, so ``--hide Acme`` silently did nothing on exactly the files
    where the organisation name is most likely to appear.
    """

    def test_a_hidden_term_is_removed_from_a_notebook(self):
        raw = notebook_text(
            [{"cell_type": "markdown", "metadata": {}, "source": ["# Acme churn\n"]}]
        )
        result, _plan = encode_artifact(raw, path="a.ipynb", extra_terms=["Acme"])
        assert "Acme" not in result.text

    def test_a_hidden_term_is_removed_from_a_module(self):
        result, _plan = encode_artifact(
            "# Acme churn model\ndf = df[['region_code']]\n",
            "python",
            extra_terms=["Acme"],
        )
        assert "Acme" not in result.text

    def test_the_round_trip_is_still_exact(self):
        source = "# Acme churn\ndf = df[['region_code']]\n"
        result, _plan = encode_artifact(source, "python", extra_terms=["Acme"])
        assert restore(result.text, result.vault).text == source


class TestPathPattern:
    """Its own examples, and the false positives a notebook really contains."""

    @pytest.mark.parametrize("value", PATH_PATTERN.examples_yes)
    def test_every_declared_positive_matches(self, value):
        import re

        assert re.compile(PATH_PATTERN.pattern).search(value)

    @pytest.mark.parametrize("value", PATH_PATTERN.examples_no)
    def test_every_declared_negative_is_rejected(self, value):
        import re

        match = re.compile(PATH_PATTERN.pattern).search(value)
        assert match is None or match.group() != value

    @pytest.mark.parametrize(
        "value", ["text/plain", "image/png", "application/pdf", "and/or", "50/50"]
    )
    def test_a_mime_type_is_not_a_path(self, value):
        import re

        assert re.compile(PATH_PATTERN.pattern).search(value) is None


class TestDoctests:
    """Every documented example runs."""

    def test_doctests_pass(self):
        import doctest

        from .. import _artifacts

        assert doctest.testmod(_artifacts, verbose=False).failed == 0


def test_an_explicit_kinds_policy_needs_no_column_detector():
    """CP-052: a profile naming kinds, on an artefact with no columns."""
    from .._policy import profile as _named
    from .._artifacts import encode_artifact as _encode

    result, plan = _encode(
        "x = 1  # mail a@example.com\n", "python", policy=_named("minimal")
    )
    assert plan.columns == {}
    assert "a@example.com" not in result.text


class TestPathEdges:
    """CP-074 and CP-075: tags are not paths; Windows paths are, whole."""

    @pytest.mark.parametrize(
        "text", ["<doc>x</doc>", "</document>", "a </p> b", "</page >", "<br/>"]
    )
    def test_markup_is_left_alone(self, text):
        from .. import FluentCleanPrompt

        assert FluentCleanPrompt().guard().outgoing(text) == text

    @pytest.mark.parametrize(
        ("text", "user"),
        [
            ("open C:\\Users\\marion.holt\\x.txt now", "marion.holt"),
            ("C:\\Users\\ann", "ann"),
            ("sort </home/ann/in.txt", "ann"),
        ],
    )
    def test_the_user_name_is_hidden_and_restored(self, text, user):
        from .. import FluentCleanPrompt

        guard = FluentCleanPrompt().guard()
        safe = guard.outgoing(text)
        assert user not in safe
        assert guard.incoming(safe) == text
