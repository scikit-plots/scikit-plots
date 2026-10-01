"""
Tests for :mod:`scikitplot.cleanprompt._structured`.

Notes
-----
**Developer notes.** The CSV scanner reports *positions*, which :mod:`csv`
cannot; :mod:`csv` is therefore used as the oracle for *contents*. Every cell
the scanner locates must decode to exactly what :mod:`csv` reads for that row
and column, over a table of awkward inputs. The JSON scanner is held to the
parser the same way.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from .. import DEFAULT_POLICY
from .._packs import FieldSpec
from .._structured import (
    SPLITTERS,
    FieldDetector,
    _delimited_cells,
    field_regions,
    json_tokens,
    match_field,
    record_starts,
)
from .._exceptions import CleanPromptError

_CSV_CASES = [
    "a,b\n1,2\n",
    'a,b\n"x, y","say ""hi"""\n',
    'a,b\r\n"multi\nline",2\r\n',
    "a,b,c\n1,,3\n",
    "a,b\n1,2",
    "a,b\n1,\n",
    'name\n""\n',
]


class TestDelimitedAgainstCsv:
    @pytest.mark.parametrize("text", _CSV_CASES)
    def test_every_cell_matches_the_csv_module(self, text):
        expected = list(csv.reader(io.StringIO(text, newline="")))
        located = {}
        for row, column, start, end in _delimited_cells(text, ","):
            raw = text[start:end]
            quoted = start > 0 and text[start - 1] == '"'
            located[(row, column)] = raw.replace('""', '"') if quoted else raw
        for row, cells in enumerate(expected):
            for column, value in enumerate(cells):
                assert located[(row, column)] == value, (row, column)

    def test_regions_are_named_by_header(self):
        text = "name,phone\nAnn,555\n"
        regions = field_regions(text, "delimited", {"delimiter": ","})
        assert [(r.field, text[r.start : r.end]) for r in regions] == [
            ("name", "Ann"),
            ("phone", "555"),
        ]


class TestJson:
    def test_string_regions_are_the_inside_of_the_quotes(self):
        text = '{"email": "a@b.co", "n": {"mrn": 12}}'
        regions = field_regions(text, "json")
        assert [(r.field, text[r.start : r.end], r.token) for r in regions] == [
            ("email", "a@b.co", "string"),
            ("mrn", "12", "number"),
        ]

    def test_keys_null_and_empty_strings_are_not_fields(self):
        text = '{"a": null, "b": "", "c": true}'
        assert [(r.field, r.token) for r in field_regions(text, "json")] == [
            ("c", "literal")
        ]

    def test_invalid_json_is_refused(self):
        with pytest.raises(CleanPromptError, match="not valid JSON"):
            field_regions('{"a": 1,}', "json")

    def test_jsonl_offsets_are_absolute(self):
        text = '{"a": "x"}\n\n{"a": "yy"}\n'
        found = [text[r.start : r.end] for r in field_regions(text, "jsonl")]
        assert found == ["x", "yy"]

    @pytest.mark.parametrize(
        "text",
        [
            '{"k": "a\\"b", "n": -1.5e3, "t": [true, false, null]}',
            '["x", {"y": "\\u00e9"}]',
        ],
    )
    def test_tokens_cover_every_scalar_the_parser_sees(self, text):
        tokens = json_tokens(text)
        decoded = [json.loads(text[start:end]) for start, end, _ in tokens]

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    yield key
                    yield from walk(value)
            elif isinstance(node, list):
                for value in node:
                    yield from walk(value)
            else:
                yield node

        assert decoded == list(walk(json.loads(text)))


class TestKeyValue:
    def test_env_file(self):
        text = "# comment\nexport DB_PASSWORD=hunter2\nPORT = 5432  # inline\n"
        regions = field_regions(
            text, "keyvalue", {"separators": ["="], "comments": ["#"]}
        )
        assert [(r.field, text[r.start : r.end]) for r in regions] == [
            ("DB_PASSWORD", "hunter2"),
            ("PORT", "5432"),
        ]

    def test_quoted_value_region_is_inside_the_quotes(self):
        text = 'token: "abc"\n'
        (region,) = field_regions(text, "keyvalue", {"separators": [":"]})
        assert text[region.start : region.end] == "abc"

    def test_prose_leaves_sentence_punctuation_outside(self):
        text = "Patient MRN: 00412345.\n"
        (region,) = field_regions(text, "text")
        assert text[region.start : region.end] == "00412345"

    def test_mail_headers_stop_at_the_body(self):
        text = "From: a@b.co\nTo: c@d.co,\n e@f.co\n\nHi: not a header\n"
        regions = field_regions(
            text, "keyvalue", {"separators": [":"], "header_block": True}
        )
        assert [r.field for r in regions] == ["From", "To"]
        assert text[regions[1].start : regions[1].end] == "c@d.co,\n e@f.co"


class TestScriptAndPython:
    def test_shell_forms(self):
        text = (
            'export DB_PASSWORD=hunter2\nAPI_TOKEN="abc def"\nrun --password s3cret\n'
        )
        regions = field_regions(text, "script")
        assert [(r.field, text[r.start : r.end]) for r in regions] == [
            ("DB_PASSWORD", "hunter2"),
            ("API_TOKEN", "abc def"),
            ("password", "s3cret"),
        ]

    def test_python_assignments_keywords_and_dicts(self):
        text = 'password = "a"\nconnect(token="b")\ncfg = {"api_key": "c", "n": 3}\nx: str = "d"\n'
        found = {
            (r.field, text[r.start : r.end]) for r in field_regions(text, "python")
        }
        assert {
            ("password", "a"),
            ("token", "b"),
            ("api_key", "c"),
            ("n", "3"),
            ("x", "d"),
        } <= found

    def test_python_non_ascii_offsets_are_characters(self):
        text = 'note = "café"; email = "é@x.co"\n'
        regions = field_regions(text, "python")
        assert [text[r.start : r.end] for r in regions] == ["café", "é@x.co"]

    def test_implicit_concatenation_is_not_one_value(self):
        assert field_regions('password = "a" "b"\n', "python") == ()

    def test_python_that_does_not_parse_has_no_fields(self):
        assert field_regions("def (:\n", "python") == ()


class TestMatchingAndDetector:
    _INDEX = {
        "email": FieldSpec(("email",), "EMAIL", "text", True),
        "name": FieldSpec(("name",), "PERSON", "text", False),
    }

    def test_suffix_rule(self):
        assert match_field("billingEmail", self._INDEX).kind == "EMAIL"

    def test_exact_only_rule_does_not_match_as_suffix(self):
        assert match_field("model_name", self._INDEX) is None
        assert match_field("Name", self._INDEX).kind == "PERSON"

    def test_vcard_parameters_are_ignored(self):
        assert match_field("EMAIL;TYPE=work", self._INDEX).kind == "EMAIL"

    def test_detector_reports_whole_values_under_the_field_kind(self):
        text = "email,notes\nann@example.com,hello\n"
        regions = field_regions(text, "delimited", {"delimiter": ","})
        spans = list(FieldDetector(regions, self._INDEX).detect(text, DEFAULT_POLICY))
        assert [(s.kind, s.text) for s in spans] == [("EMAIL", "ann@example.com")]


def test_unknown_splitter_is_refused():
    with pytest.raises(CleanPromptError, match="unknown splitter"):
        field_regions("x", "yaml")


@pytest.mark.parametrize("splitter", ["office", "corpus", "archive"])
def test_container_splitters_locate_nothing_themselves(splitter):
    assert splitter in SPLITTERS
    assert field_regions("a: b", splitter) == ()


class TestProseSpans:
    """A prose value ends where its identifier's format says it must."""

    @pytest.mark.parametrize(
        ("text", "hidden"),
        [
            ("Summarise MRN: 00412345 for the team", "00412345"),
            ("Phone: +1 555 010 4477, after 5pm", "+1 555 010 4477"),
            ("Diagnosis: E11.9, hypertension", "E11.9, hypertension"),
            ("Address: 12 Main St, Springfield", "12 Main St, Springfield"),
        ],
    )
    def test_span_by_format(self, text, hidden):
        from .._catalog import builtin_catalog

        catalog = builtin_catalog()
        index = catalog.field_index(catalog.resolve_packs("all"))
        spans = list(
            FieldDetector(field_regions(text, "text"), index).detect(
                text, DEFAULT_POLICY
            )
        )
        assert [span.text for span in spans] == [hidden]

    def test_structured_values_are_always_whole(self):
        from .._catalog import builtin_catalog

        catalog = builtin_catalog()
        index = catalog.field_index(catalog.resolve_packs("all"))
        text = "mrn\n00412345 extra words\n"
        spans = list(
            FieldDetector(
                field_regions(text, "delimited", {"delimiter": ","}), index
            ).detect(text, DEFAULT_POLICY)
        )
        assert [span.text for span in spans] == ["00412345 extra words"]

    def test_an_unknown_span_is_refused(self):
        from .._packs import PackError, pack_from_document

        with pytest.raises(PackError, match="span"):
            pack_from_document(
                {
                    "name": "hr",
                    "version": 1,
                    "summary": "x",
                    "fields": [{"names": ["badge"], "kind": "BADGE", "span": "word"}],
                }
            )


class TestRecordStarts:
    """Where a record file may be cut: between records, never inside one."""

    @pytest.mark.parametrize(
        ("text", "starts"),
        [
            ('a,b\n"x\ny",2\n3,4\n', [0, 4, 12]),
            ('a,b\r\n"q ""x"",\r\n",2\r\n', [0, 5]),
            ("\n\n", [0, 1]),
            ("a,b", [0]),
            ("", []),
        ],
    )
    def test_delimited(self, text, starts):
        assert record_starts(text, "delimited", {"delimiter": ","}) == starts

    def test_every_record_is_a_whole_csv_row(self):
        text = 'a,b\n"x\ny",2\n3,"4\n5"\n'
        cuts = record_starts(text, "delimited", {"delimiter": ","}) + [len(text)]
        rows = [
            list(csv.reader(io.StringIO(text[low:high], newline="")))
            for low, high in zip(cuts, cuts[1:])
        ]
        assert [row for piece in rows for row in piece] == list(
            csv.reader(io.StringIO(text, newline=""))
        )

    def test_jsonl(self):
        assert record_starts('{"a": 1}\n\n{"a": 2}', "jsonl") == [0, 9, 10]

    @pytest.mark.parametrize("splitter", ["json", "text", "python"])
    def test_other_splitters_are_refused(self, splitter):
        with pytest.raises(CleanPromptError, match="record boundaries"):
            record_starts("x", splitter)


class TestProseInsideJsonStrings:
    """CP-078: a JSON string is read as prose, exactly as a text line would be."""

    @pytest.mark.parametrize("seed", range(30))
    def test_offsets_agree_with_the_parser(self, seed):
        import random

        from .._structured import _string_offsets

        rng = random.Random(seed)
        alphabet = [
            "a",
            ":",
            " ",
            '"',
            "\\",
            "\n",
            "\t",
            "é",
            "\U0001f600",
            " ",
            "\x01",
        ]
        value = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        raw = json.dumps(value, ensure_ascii=rng.random() < 0.5)[1:-1]
        starts = _string_offsets(raw)
        assert len(starts) == len(value) + 1
        for index, char in enumerate(value):
            assert (
                json.loads('"' + raw[starts[index] : starts[index + 1]] + '"') == char
            )

    @pytest.mark.parametrize(
        ("value", "hidden"),
        [
            ("password: hunter2hunter2", "hunter2hunter2"),
            ("line\npassword: s3cret-value\nnext: ok", "s3cret-value"),
            ("café \U0001f600 MRN: 00412345", "00412345"),
        ],
    )
    @pytest.mark.parametrize("ascii_only", [True, False])
    def test_values_are_hidden_and_restored(self, value, hidden, ascii_only):
        from .. import FluentCleanPrompt

        cleaner = FluentCleanPrompt().materialize()
        for fmt, text in (
            ("json", json.dumps({"note": value}, ensure_ascii=ascii_only)),
            ("jsonl", json.dumps([value], ensure_ascii=ascii_only) + "\n"),
        ):
            safe = cleaner.encode_text(text, fmt).text
            assert (
                hidden
                not in json.loads(safe if fmt == "json" else safe.strip()).__repr__()
            )
            assert cleaner.decode(safe) == text
