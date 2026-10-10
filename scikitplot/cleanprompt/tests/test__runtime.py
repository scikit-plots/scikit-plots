"""
Tests for :mod:`scikitplot.cleanprompt._runtime`.

Notes
-----
**Developer notes.** ``TestRoundTrip`` holds every built-in format to the
``round_trip`` claim it makes: each one that says ``true`` has a sample here,
is encoded and decoded, and must come back byte for byte. A new round-trip
format without a sample fails ``test_every_round_trip_format_has_a_sample``,
so the claim cannot be added untested.

``TestLeaks`` is the root cause ``CP-048`` turned into assertions: each value
a field name marks as sensitive must be absent from the encoded text.
"""

from __future__ import annotations

import io
import json
import os
import zipfile

import pytest

from .._catalog import builtin_catalog
from .._exceptions import CleanPromptError
from .._plan import FluentCleanPrompt
from .._runtime import (
    SKIPPED_DIRECTORIES,
    Cleaner,
    restore_archive,
    restore_tree,
    sentinel,
)

_NOTEBOOK = json.dumps(
    {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": ["ann@example.com\n"],
                    }
                ],
                "source": [
                    'password = "s3cr3t-pass"\n',
                    "df = pd.read_csv('/home/ann/x.csv')\n",
                ],
            }
        ],
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    },
    indent=1,
)

#: One sample per round-trip format, each holding at least one field value.
SAMPLES = {
    "text": "Patient MRN: 00412345.\nMail ann@example.com\n",
    "markdown": "# Visit\n\n- Email: ann@example.com\n",
    "rst": "Visit\n=====\n\n:Email: ann@example.com\n",
    "html": "<p>Email: ann@example.com</p>\n",
    "shell": 'export DB_PASSWORD=hunter2hunter2\nAPI_TOKEN="tok-123456"\n',
    "javascript": 'const password = "hunter2hunter2";\n',
    "sql": "UPDATE users SET password = 'hunter2hunter2' WHERE id = 7;\n",
    "rlang": 'api_key <- "abc123def456"\n',
    "python": 'API_KEY = "abc123def456"\nX = df[["mrn"]]\n',
    "notebook": _NOTEBOOK,
    "json": '{"mrn": 12345678, "email": "ann@example.com", "ok": true}\n',
    "jsonl": '{"email": "ann@example.com"}\n{"phone": "+1 555 010 4477"}\n',
    "csv": 'name,phone,note\nAnn Lee,+1 555 010 4477,"a, b"\r\n',
    "tsv": "name\tphone\nAnn Lee\t+1 555 010 4477\n",
    "env": "DB_PASSWORD=hunter2hunter2\n",
    "ini": "[db]\npassword = hunter2hunter2\n",
    "yaml": "db:\n  password: hunter2hunter2\n",
    "email": "From: Ann Lee <ann@example.com>\nTo: bob@example.org\n\nHi: body\n",
    "vcard": "BEGIN:VCARD\nFN:Ann Lee\nEMAIL;TYPE=work:ann@example.com\nEND:VCARD\n",
}


def _cleaner(*packs, **kwargs):
    builder = FluentCleanPrompt()
    if packs:
        builder = builder.packs(*packs)
    for name, value in kwargs.items():
        builder = (
            getattr(builder, name)(*value)
            if isinstance(value, tuple)
            else getattr(builder, name)(value)
        )
    return builder.materialize()


class TestRoundTrip:
    def test_every_round_trip_format_has_a_sample(self):
        claimed = {
            spec.name
            for spec in builtin_catalog().formats.values()
            if spec.round_trip and spec.splitter != "archive"
        }
        assert claimed == set(SAMPLES)

    @pytest.mark.parametrize("name", sorted(SAMPLES))
    @pytest.mark.parametrize("packs", [("auto",), ("all",), ("none",)])
    def test_decode_restores_every_byte(self, name, packs):
        cleaner = _cleaner(*packs)
        encoded = cleaner.encode_text(SAMPLES[name], name, name=f"sample.{name}")
        assert encoded.round_trip is True
        assert cleaner.decode(encoded.text) == SAMPLES[name]

    @pytest.mark.parametrize("name", sorted(SAMPLES))
    def test_something_is_hidden_in_every_sample(self, name):
        encoded = _cleaner("all").encode_text(
            SAMPLES[name], name, name=f"sample.{name}"
        )
        assert encoded.count > 0
        assert encoded.text != SAMPLES[name]

    @pytest.mark.parametrize("name", ["json", "jsonl", "notebook"])
    def test_json_stays_json(self, name):
        text = _cleaner("all").encode_text(SAMPLES[name], name, name="x").text
        for line in text.splitlines() if name == "jsonl" else [text]:
            json.loads(line)


class TestInvisibleCharactersKeepStructure:
    """
    ``CP-102``: a zero-width character in a record file leaves its structure alone.

    Notes
    -----
    **Developer notes.** Round 25's detection view first ran *every* detector
    over the view, including field, region and JSON-token detectors whose
    offsets were computed from the original document. Their spans were mapped
    a second time and cut through newlines and separators: with one
    zero-width space early in a ``.env`` file, ``DB_PASSWORD=...`` and the next
    two lines merged into one; a CSV row lost a column; JSON failed closed. The
    view is now read only by detectors that declare ``reads_view``.

    The assertion is structural equivalence: encoding the salted sample and
    removing the zero-width spaces gives exactly the encoding of the plain
    sample, and the salted sample round-trips byte for byte.
    """

    @staticmethod
    def _salted(text):
        """Insert a zero-width space after the first ASCII letter."""
        for index, char in enumerate(text):
            if char.isascii() and char.isalpha():
                return text[: index + 1] + "\u200b" + text[index + 1 :]
        return "\u200b" + text

    #: Where the salt lands on a name the format's own splitter tokenises
    #: with an identifier class, or on a structural key, the equivalence does
    #: not hold yet. Each is a known, recorded limit — never a silent skip:
    #: - ``notebook``: the salt lands in the ``cells`` key; the document is no
    #:   longer a notebook and is refused (fails closed), asserted below;
    #: - ``python``, ``rlang``: the salt lands inside an identifier; the code
    #:   splitter does not read names through the view yet
    #:   (upcoming_changes/scikitplot/cleanprompt/
    #:   invisible-characters-in-keyvalue-and-code-names.md).
    NOT_YET = {"notebook", "python", "rlang"}

    @pytest.mark.parametrize("name", sorted(set(SAMPLES) - NOT_YET))
    @pytest.mark.parametrize("packs", [("auto",), ("all",)])
    def test_structure_is_unchanged_and_the_round_trip_exact(self, name, packs):
        plain = SAMPLES[name]
        salted = self._salted(plain)
        expected = _cleaner(*packs).encode_text(plain, name, name=f"s.{name}").text
        cleaner = _cleaner(*packs)
        encoded = cleaner.encode_text(salted, name, name=f"s.{name}")
        assert encoded.text.replace("\u200b", "") == expected
        assert cleaner.decode(encoded.text) == salted

    def test_a_salted_structural_key_is_refused_not_misread(self):
        from .. import CleanPromptError

        with pytest.raises(CleanPromptError, match="not a notebook"):
            _cleaner("all").encode_text(self._salted(SAMPLES["notebook"]), "notebook", name="s")

    @pytest.mark.parametrize(
        ("fmt", "text", "value"),
        [
            ("csv", "n\u200bame,phone\nAnn Lee,+1 555 010 4477\n", "Ann Lee"),
            ("tsv", "n\u200bame\tphone\nAnn Lee\t+1 555 010 4477\n", "Ann Lee"),
            ("json", '{"m\u200brn": 12345678, "ok": true}\n', "12345678"),
            ("email", "F\u200brom: Ann Lee <ann@example.com>\n\nHi\n", "Ann Lee"),
            ("csv", "\uff4e\uff41\uff4d\uff45,x\nAnn Lee,1\n", "Ann Lee"),
        ],
    )
    def test_a_salted_field_name_still_names_the_field(self, fmt, text, value):
        """CP-103: one invisible character in a header no longer hides a column."""
        cleaner = _cleaner("all")
        encoded = cleaner.encode_text(text, fmt, name=f"s.{fmt}")
        assert value not in encoded.text
        assert cleaner.decode(encoded.text) == text

    def test_a_salted_value_in_a_record_file_is_found(self):
        """The reason the view exists still holds where it is allowed to."""
        text = "NOTE=x\nOTHER=bob\u200b@example.com\n"
        encoded = _cleaner("all").encode_text(text, "env", name="s.env")
        assert "bob" not in encoded.text
        assert encoded.text.count("\n") == text.count("\n")


class TestLeaks:
    """CP-048: a value the field name marks sensitive never reaches the output."""

    @pytest.mark.parametrize(
        ("fmt", "text", "secrets"),
        [
            (
                "csv",
                "name,phone,member_id\nMarion Holt,+1 555 010 4477,M-00412\n",
                ["Marion Holt", "M-00412"],
            ),
            (
                "json",
                '{"patient": {"mrn": "00412345", "diagnosis": "E11.9", "insurer_id": "ZX-44"}}',
                ["00412345", "E11.9", "ZX-44"],
            ),
            (
                "shell",
                'export DB_HOST=db.internal\nexport DB_PASSWORD=pw-1\nAPI_TOKEN="t-2"\n',
                ["db.internal", "pw-1", "t-2"],
            ),
            (
                "ini",
                "db_password = pw-1\nowner = marion.holt\n",
                ["pw-1", "marion.holt"],
            ),
        ],
    )
    def test_default_plan_hides_field_values(self, fmt, text, secrets):
        encoded = _cleaner().encode_text(text, fmt).text
        for secret in secrets:
            assert secret not in encoded

    def test_labelled_value_and_field_value_share_one_label(self):
        cleaner = _cleaner("patient")
        prose = cleaner.encode_text("Seen today. MRN: 00412345.\n", "text").text
        record = cleaner.encode_text('{"mrn": "00412345"}', "json").text
        assert prose == "Seen today. MRN: [MRN-1].\n"
        assert record == '{"mrn": "[MRN-1]"}'


class TestJson:
    def test_numbers_and_literals_become_sentinels(self):
        cleaner = _cleaner("patient")
        encoded = cleaner.encode_text(
            '{"mrn": 20417, "patient_id": 20417, "npi": 7}', "json"
        ).text
        assert (
            encoded
            == f'{{"mrn": {sentinel(1)}, "patient_id": {sentinel(1)}, "npi": {sentinel(2)}}}'
        )

    @pytest.mark.parametrize(
        "doc",
        [
            '{"x": 15550100123, "y": "call +1 555 010 0123 now"}',
            '{"k": "\\u00e9 ann@example.com", "k2": "a\\\\zzz"}',
            '["ann@example.com", 5, "tel"]',
            '{"u": "zzz\\u00e9"}',
        ],
    )
    def test_spans_never_break_a_token_or_an_escape(self, doc):
        cleaner = _cleaner("none", hide=("0100", "zzz"))
        encoded = cleaner.encode_text(doc, "json").text
        json.loads(encoded)
        assert cleaner.decode(encoded) == doc

    def test_invalid_json_is_refused(self):
        with pytest.raises(CleanPromptError, match="not valid JSON"):
            _cleaner().encode_text('{"a": 1,}', "json")

    def test_reencoding_is_idempotent(self):
        cleaner = _cleaner("patient")
        once = cleaner.encode_text(
            '{"mrn": 20417, "email": "a@example.com"}', "json"
        ).text
        assert _cleaner("patient").encode_text(once, "json").text == once

    def test_sentinel_bounds(self):
        assert sentinel(99_999_999) == "-9999999999"
        with pytest.raises(ValueError):
            sentinel(0)


class TestSharedVault:
    def test_one_value_one_label_across_files(self):
        cleaner = _cleaner("all")
        first = cleaner.encode_text("email\nann@example.com\n", "csv").text
        second = cleaner.encode_text('{"email": "ann@example.com"}', "json").text
        assert "[EMAIL-1]" in first and "[EMAIL-1]" in second
        assert len(cleaner) == 1

    def test_fresh_cleaners_are_deterministic(self):
        outputs = []
        for _ in range(3):
            cleaner = _cleaner("all")
            outputs.append(
                [
                    cleaner.encode_text(SAMPLES[n], n, name=n).text
                    for n in sorted(SAMPLES)
                ]
            )
        assert outputs[0] == outputs[1] == outputs[2]

    def test_prior_entries_continue_numbering(self):
        first = _cleaner("patient")
        first.encode_text('{"mrn": 1, "email": "a@example.com"}', "json")
        second = Cleaner(first.plan, prior=first.handle().entries)
        text = second.encode_text('{"mrn": 2, "email": "b@example.com"}', "json").text
        assert text == f'{{"mrn": {sentinel(2)}, "email": "[EMAIL-2]"}}'

    def test_preexisting_literal_stand_in_is_reported(self):
        cleaner = _cleaner("patient")
        cleaner.encode_text('{"mrn": 1}', "json")
        encoded = cleaner.encode_text(f"the number {sentinel(1)} again", "text")
        assert encoded.report["preexisting_labels"] == [sentinel(1)]


class TestPlanOptions:
    def test_core_off_leaves_only_packs(self):
        text = (
            _cleaner("patient", core=False)
            .encode_text("MRN: 00412345 a@example.com", "text")
            .text
        )
        assert (
            text == "MRN: [MRN-1] a@example.com"
        )  # core patterns off: only the pack acts

    def test_minimal_profile_still_runs_packs(self):
        text = (
            _cleaner("patient", profile="minimal")
            .encode_text("MRN: 00412345 ssn 078-05-1120", "text")
            .text
        )
        assert "[MRN-1]" in text

    @pytest.mark.parametrize("fmt", ["python", "notebook"])
    def test_minimal_profile_in_an_artefact_without_columns(self, fmt):
        """CP-052: an explicit-kinds profile must not refuse or drop anything."""
        cleaner = _cleaner("secrets", profile="minimal")
        source = 'password = "hunter2hunter2"\n' if fmt == "python" else _NOTEBOOK
        encoded = cleaner.encode_text(source, fmt, name="m").text
        assert "hunter2hunter2" not in encoded and "s3cr3t-pass" not in encoded
        assert cleaner.decode(encoded) == source

    def test_minimal_profile_with_core_off_in_python(self):
        cleaner = _cleaner("secrets", profile="minimal", core=False)
        assert (
            "hunter2"
            not in cleaner.encode_text(
                'password = "hunter2"\n', "python", name="m.py"
            ).text
        )

    def test_surrogate_style(self):
        cleaner = _cleaner("all", style="surrogate")
        source = '{"name": "Ann Lee", "mrn": 99}'
        encoded = cleaner.encode_text(source, "json").text
        assert "Ann Lee" not in encoded
        json.loads(encoded)
        assert cleaner.decode(encoded) == source

    def test_keep_paths(self):
        text = (
            _cleaner("none", keep=("paths",))
            .encode_text("see /home/ann/x.csv", "text")
            .text
        )
        assert "/home/ann/x.csv" in text

    def test_a_format_outside_the_plan_is_refused(self):
        with pytest.raises(CleanPromptError, match="not selected"):
            _cleaner(formats=("csv",)).encode_text("a", "json")


class TestBytesAndFiles:
    def test_office_formats_need_bytes(self):
        with pytest.raises(CleanPromptError, match="encode_file or encode_bytes"):
            _cleaner().encode_text("x", "docx")

    def test_docx_bytes_become_text(self):
        body = (
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
            "<w:p><w:r><w:t>MRN: 00412345</w:t></w:r></w:p></w:body></w:document>"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", body)
        encoded = _cleaner("patient").encode_bytes(buffer.getvalue(), "note.docx")
        assert encoded.text == "MRN: [MRN-1]\n"
        assert encoded.output_name == "note.docx.txt"
        assert encoded.round_trip is False

    def test_non_utf8_is_refused(self):
        with pytest.raises(CleanPromptError, match="not UTF-8"):
            _cleaner().encode_bytes(b"caf\xe9", "a.txt")

    def test_bom_and_crlf_survive(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_bytes("﻿email\r\nann@example.com\r\n".encode("utf-8"))
        cleaner = _cleaner()
        encoded = cleaner.encode_file(path)
        assert cleaner.decode(encoded.text).encode("utf-8") == path.read_bytes()

    def test_symlink_is_not_a_regular_file(self, tmp_path):
        target = tmp_path / "real.txt"
        target.write_text("x", encoding="utf-8")
        link = tmp_path / "link.txt"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("symlinks unavailable")
        with pytest.raises(CleanPromptError, match="regular file"):
            _cleaner().encode_file(link)


class TestTree:
    def _source(self, root):
        (root / "sub").mkdir(parents=True)
        (root / ".git").mkdir()
        (root / ".git" / "config").write_text("token = secret\n", encoding="utf-8")
        (root / "a.csv").write_text("email\nann@example.com\n", encoding="utf-8")
        (root / "sub" / "b.json").write_text(
            '{"email": "ann@example.com"}', encoding="utf-8"
        )
        (root / "sub" / "bad.json").write_text("{", encoding="utf-8")
        (root / "image.png").write_bytes(b"\x89PNG")
        return root

    def test_encode_then_decode(self, tmp_path):
        source = self._source(tmp_path / "src")
        cleaner = _cleaner()
        items = {
            item.relative: item
            for item in cleaner.encode_tree(source, tmp_path / "out")
        }
        assert items[".git/"].status == "skipped"
        assert items["image.png"].status == "skipped"
        assert items["sub/bad.json"].status == "refused"
        assert items["a.csv"].status == items["sub/b.json"].status == "encoded"
        assert not (tmp_path / "out" / ".git").exists()
        assert not (tmp_path / "out" / "image.png").exists()
        assert "ann@example.com" not in (tmp_path / "out" / "a.csv").read_text(
            encoding="utf-8"
        )
        list(cleaner.decode_tree(tmp_path / "out", tmp_path / "back"))
        assert (tmp_path / "back" / "a.csv").read_bytes() == (
            source / "a.csv"
        ).read_bytes()

    def test_restore_tree_with_a_vault(self, tmp_path):
        source = self._source(tmp_path / "src")
        cleaner = _cleaner()
        list(cleaner.encode_tree(source, tmp_path / "out"))
        list(
            restore_tree(
                tmp_path / "out", tmp_path / "back", cleaner.vault(), cleaner.policy
            )
        )
        assert (tmp_path / "back" / "sub" / "b.json").read_bytes() == (
            source / "sub" / "b.json"
        ).read_bytes()

    def test_target_inside_source_is_refused(self, tmp_path):
        source = self._source(tmp_path / "src")
        with pytest.raises(CleanPromptError, match="inside"):
            list(_cleaner().encode_tree(source, source / "out"))

    def test_skipped_directories(self):
        assert {".git", "__pycache__", ".ipynb_checkpoints"} <= SKIPPED_DIRECTORIES


class TestArchive:
    def _zip(self, path, members):
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        return path

    def test_members_are_encoded_skipped_or_refused(self, tmp_path):
        source = self._zip(
            tmp_path / "in.zip",
            {
                "a.csv": "email\nann@example.com\n",
                "../evil.txt": "x",
                "inner.zip": b"PK",
                "photo.png": b"\x89PNG",
            },
        )
        items = {
            i.relative: i
            for i in _cleaner().encode_archive(source, tmp_path / "out.zip")
        }
        assert items["a.csv"].status == "encoded"
        assert items["../evil.txt"].status == "refused"
        assert items["inner.zip"].status == "skipped"
        assert items["photo.png"].status == "skipped"
        with zipfile.ZipFile(tmp_path / "out.zip") as archive:
            assert archive.namelist() == ["a.csv"]

    def test_output_is_deterministic_and_decodes(self, tmp_path):
        source = self._zip(
            tmp_path / "in.zip",
            {"b.json": '{"email": "a@example.com"}', "a.csv": "email\nb@example.org\n"},
        )
        first, second = _cleaner(), _cleaner()
        first.encode_archive(source, tmp_path / "1.zip")
        second.encode_archive(source, tmp_path / "2.zip")
        assert (tmp_path / "1.zip").read_bytes() == (tmp_path / "2.zip").read_bytes()
        restore_archive(
            tmp_path / "1.zip", tmp_path / "back.zip", first.vault(), first.policy
        )
        with zipfile.ZipFile(tmp_path / "back.zip") as archive:
            assert archive.read("b.json") == b'{"email": "a@example.com"}'

    def test_overwriting_the_input_is_refused(self, tmp_path):
        source = self._zip(tmp_path / "in.zip", {"a.txt": "x"})
        with pytest.raises(CleanPromptError, match="overwrite"):
            _cleaner().encode_archive(source, source)

    def test_not_a_zip(self, tmp_path):
        path = tmp_path / "x.zip"
        path.write_bytes(b"nope")
        with pytest.raises(CleanPromptError, match="not a readable zip"):
            _cleaner().encode_archive(path, tmp_path / "y.zip")


class TestLifecycle:
    def test_clear_drops_values_and_closes(self):
        cleaner = _cleaner()
        cleaner.encode_text("ann@example.com", "text")
        cleaner.clear()
        assert len(cleaner) == 0
        with pytest.raises(CleanPromptError, match="cleared"):
            cleaner.decode("[EMAIL-1]")

    def test_context_manager_clears(self):
        with _cleaner() as cleaner:
            cleaner.encode_text("ann@example.com", "text")
        assert "cleared" in repr(cleaner)

    def test_nothing_secret_in_reprs(self):
        cleaner = _cleaner()
        encoded = cleaner.encode_text("ann@example.com", "text")
        assert "ann@example.com" not in repr(cleaner)
        assert "ann@example.com" not in repr(encoded)
        assert "ann@example.com" not in repr(cleaner.handle())

    def test_invalid_plan_is_refused(self):
        from .._plan import CleanPlan

        with pytest.raises(CleanPromptError, match="not valid"):
            Cleaner(CleanPlan(packs=("nope",)))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlinks")
def test_tree_does_not_follow_a_symlinked_folder(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("ann@example.com", encoding="utf-8")
    source = tmp_path / "src"
    source.mkdir()
    try:
        (source / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    items = list(_cleaner().encode_tree(source, tmp_path / "out"))
    assert [(i.relative, i.status) for i in items] == [("link/", "skipped")]


class TestFoundByFuzzing:
    """Round 12: a 4000-document fuzz of every text-native format (CP-057)."""

    def test_duplicate_json_keys_are_read_not_refused(self):
        cleaner = _cleaner("patient")
        source = '{"mrn": "1234", "mrn": "5678"}'
        encoded = cleaner.encode_text(source, "json").text
        assert "1234" not in encoded and "5678" not in encoded
        assert cleaner.decode(encoded) == source

    @pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85"])
    def test_a_unicode_line_separator_inside_a_record_is_not_a_new_line(
        self, separator
    ):
        cleaner = _cleaner("all")
        source = (
            '{"email": "ann@example.com", "note": "a' + separator + 'b"}\n{"x": 1}\n'
        )
        encoded = cleaner.encode_text(source, "jsonl").text
        for line in encoded.split("\n"):
            if line.strip():
                json.loads(line)
        assert cleaner.decode(encoded) == source

    def test_office_members_count_against_the_archive_budget(self, tmp_path):
        from .._office import OfficeLimits

        body = (
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>"
            + "x" * 4000
            + "</w:t></w:r></w:p></w:body></w:document>"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as inner:
            inner.writestr("word/document.xml", body)
        source = tmp_path / "in.zip"
        with zipfile.ZipFile(source, "w") as archive:
            for index in range(4):
                archive.writestr(f"d{index}.docx", buffer.getvalue())
        limits = OfficeLimits(max_part_bytes=1 << 20, max_total_bytes=9000)
        with pytest.raises(CleanPromptError):
            Cleaner(limits=limits).encode_archive(source, tmp_path / "out.zip")
        assert not (tmp_path / "out.zip").exists()


class TestRemember:
    """A value hidden once is hidden wherever it recurs."""

    def test_a_record_value_is_hidden_in_later_prose(self):
        cleaner = _cleaner()
        cleaner.encode_text("name,phone\nMarion Holt,+1 555 010 4477\n", "csv")
        text = cleaner.encode_text("Call Marion Holt on +1 555 010 4477.", "text").text
        assert text == "Call [PERSON-1] on [PHONE-1]."

    def test_it_can_be_turned_off(self):
        cleaner = _cleaner(remember=False)
        cleaner.encode_text("name\nMarion Holt\n", "csv")
        assert (
            cleaner.encode_text("Call Marion Holt", "text").text == "Call Marion Holt"
        )
        assert cleaner.leaks("Call Marion Holt") == ("PERSON",)

    def test_column_names_are_not_repeated_into_prose(self):
        cleaner = _cleaner()
        cleaner.encode_text("X = df[['age']]\n", "python", name="m.py")
        assert (
            cleaner.encode_text("the age of the data", "text").text
            == "the age of the data"
        )

    def test_leaks_names_kinds_never_values(self):
        cleaner = _cleaner(remember=False)
        cleaner.encode_text('{"email": "ann@example.com"}', "json")
        assert cleaner.leaks("x ann@example.com y") == ("EMAIL",)
        assert cleaner.leaks("x ann@example.comz y") == ()


class TestLoggingDiscipline:
    def test_no_value_reaches_a_log_record(self):
        from .. import configure_logging

        buffer = io.StringIO()
        configure_logging("debug", "json", stream=buffer)
        cleaner = _cleaner("all")
        for name in sorted(SAMPLES):
            cleaner.encode_text(SAMPLES[name], name, name="secret-client-file.csv")
        cleaner.decode("[EMAIL-1] and [NOPE-9]")
        log = buffer.getvalue()
        for secret in (
            "ann@example.com",
            "Ann Lee",
            "hunter2hunter2",
            "555 010 4477",
            "secret-client-file",
        ):
            assert secret not in log
        events = [
            json.loads(line)["event"] for line in log.splitlines() if '"event"' in line
        ]
        assert events.count("encoded") == len(SAMPLES) and "decoded" in events

    def test_the_cleaners_values_are_scrubbed_until_cleared(self):
        from .. import configure_logging
        from .._logging import get_logger

        buffer = io.StringIO()
        configure_logging("debug", stream=buffer)
        cleaner = _cleaner()
        cleaner.encode_text("mail ann@example.com", "text")
        get_logger("scikitplot.cleanprompt._engine").warning("oops ann@example.com")
        cleaner.clear()
        get_logger("scikitplot.cleanprompt._engine").warning("later ann@example.com")
        first, second = [
            line
            for line in buffer.getvalue().splitlines()
            if "oops" in line or "later" in line
        ]
        assert "ann@example.com" not in first
        assert "ann@example.com" in second


class TestOrderIndependence:
    """CP-063: a value is hidden in every file, whichever file names it first."""

    def _files(self, root):
        root.mkdir()
        # Bytes, not text: text mode writes CRLF on Windows, and encoding
        # keeps a file's line endings.
        (root / "a_note.txt").write_bytes(b"Call Marion Holt today.\n")
        (root / "b_people.csv").write_bytes(b"name\nMarion Holt\n")
        return root

    def test_a_folder(self, tmp_path):
        source = self._files(tmp_path / "src")
        list(_cleaner().encode_tree(source, tmp_path / "out"))
        assert (
            tmp_path / "out" / "a_note.txt"
        ).read_text(encoding="utf-8") == "Call [PERSON-1] today.\n"

    def test_a_zip(self, tmp_path):
        source = self._files(tmp_path / "src")
        archive = tmp_path / "in.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for path in sorted(source.iterdir()):
                handle.write(path, path.name)
        _cleaner().encode_archive(archive, tmp_path / "out.zip")
        with zipfile.ZipFile(tmp_path / "out.zip") as handle:
            assert handle.read("a_note.txt") == b"Call [PERSON-1] today.\n"

    def test_a_list_of_files(self, tmp_path):
        source = self._files(tmp_path / "src")
        first, _ = _cleaner().encode_files(
            [source / "a_note.txt", source / "b_people.csv"]
        )
        assert first.text == "Call [PERSON-1] today.\n"

    def test_labels_do_not_depend_on_the_second_pass(self, tmp_path):
        source = self._files(tmp_path / "src")
        outputs = []
        for index in range(2):
            list(_cleaner().encode_tree(source, tmp_path / f"out{index}"))
            outputs.append((tmp_path / f"out{index}" / "a_note.txt").read_bytes())
        assert outputs[0] == outputs[1]

    def test_without_remember_there_is_one_pass(self, tmp_path):
        source = self._files(tmp_path / "src")
        list(_cleaner(remember=False).encode_tree(source, tmp_path / "out"))
        assert (
            tmp_path / "out" / "a_note.txt"
        ).read_text(encoding="utf-8") == "Call Marion Holt today.\n"


def _chunked(chunk_chars, **kwargs):
    """A cleaner that splits record files into pieces of ``chunk_chars``."""
    builder = FluentCleanPrompt()
    for name, value in kwargs.items():
        builder = getattr(builder, name)(value)
    return Cleaner(builder.plan(), chunk_chars=chunk_chars)


class TestChunkedRecords:
    """A large CSV or JSON Lines file is encoded in record-aligned pieces."""

    _CSV = (
        "name,email,note\n"
        + "".join(
            f'Person {i % 7},p{i % 11}@example.com,"line one\nline {i}"\n'
            for i in range(60)
        )
        + "Marion Holt,ann@example.com,Marion Holt called\n"
    )
    _JSONL = "".join(
        json.dumps({"email": f"p{i % 9}@example.com", "note": f"row {i}"}) + "\n"
        for i in range(60)
    )

    @pytest.mark.parametrize(
        ("text", "fmt"), [(_CSV, "csv"), (_JSONL, "jsonl")], ids=["csv", "jsonl"]
    )
    @pytest.mark.parametrize("chunk_chars", [1, 40, 333, 10_000_000])
    def test_pieces_equal_one_pass(self, text, fmt, chunk_chars):
        whole = _chunked(None).encode_text(text, fmt)
        cleaner = _chunked(chunk_chars)
        pieces = cleaner.encode_text(text, fmt)
        assert pieces.text == whole.text
        assert cleaner.decode(pieces.text) == text

    def test_the_report_counts_the_pieces(self):
        encoded = _chunked(200).encode_text(self._JSONL, "jsonl")
        assert encoded.report["chunks"] > 1

    def test_the_header_names_fields_in_every_piece(self):
        text = "email\n" + "".join(f"u{i}@example.org\n" for i in range(30))
        safe = _chunked(30).encode_text(text, "csv").text
        assert "example.org" not in safe

    def test_a_single_record_longer_than_the_limit_is_one_piece(self):
        # A record is never cut: an over-long record forms its own piece and
        # is then held to the document ceiling like any unchunked file.
        text = "email,note\nann@example.com," + "x" * 500 + "\n"
        safe = _chunked(50).encode_text(text, "csv").text
        assert safe == "email,note\n[EMAIL-1]," + "x" * 500 + "\n"

    def test_a_record_above_the_document_limit_is_refused(self):
        from .._exceptions import LimitExceededError
        from .._policy import Limits

        cleaner = _chunked(None)
        cleaner._policy = cleaner._policy.evolve(limits=Limits(max_input_chars=100))
        many = "email\n" + "".join(f"u{i}@x.org\n" for i in range(40))
        assert cleaner.encode_text(many, "csv").report["chunks"] > 1
        with pytest.raises(LimitExceededError, match="max_input_chars"):
            cleaner.encode_text("email,n\na@x.org," + "y" * 200 + "\n", "csv")

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "10"])
    def test_chunk_chars_is_validated(self, bad):
        with pytest.raises(CleanPromptError, match="chunk_chars"):
            Cleaner(FluentCleanPrompt().plan(), chunk_chars=bad)

    def test_many_pieces_do_not_exhaust_the_entry_bound(self):
        """CP-064: the seed carried from piece to piece is not counted."""
        from .._policy import Limits

        text = "email\n" + "".join(f"u{i}@example.org\n" for i in range(40))
        cleaner = _chunked(60)
        cleaner._policy = cleaner._policy.evolve(limits=Limits(max_entries=10))
        safe = cleaner.encode_text(text, "csv").text
        assert "[EMAIL-40]" in safe


class TestSurvey:
    """A dry run reports what a folder holds and writes nothing."""

    def test_nothing_is_written_and_the_vault_is_unchanged(self, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        (source / "people.csv").write_text(
            "name,email\nMarion Holt,ann@example.com\n", encoding="utf-8"
        )
        (source / "note.txt").write_text("Call Marion Holt.\n", encoding="utf-8")
        (source / "blob.bin").write_bytes(b"\xff\xfe\x00")
        cleaner = _cleaner()
        before = sorted(p.name for p in tmp_path.rglob("*"))
        items = cleaner.survey_tree(source)
        assert sorted(p.name for p in tmp_path.rglob("*")) == before
        assert cleaner.decode("[PERSON-1] [EMAIL-1]") == "[PERSON-1] [EMAIL-1]"
        by_name = {item.relative: item for item in items}
        assert by_name["people.csv"].kinds == {"EMAIL": 1, "PERSON": 1}
        assert by_name["note.txt"].kinds == {"PERSON": 1}  # learnt, CP-063
        assert all(item.output is None for item in items)
        assert "Marion" not in repr(items) and "ann@example.com" not in repr(items)

    def test_survey_matches_what_encoding_would_do(self, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        (source / "a.txt").write_text("Mail ann@example.com\n", encoding="utf-8")
        survey = _cleaner().survey_tree(source)
        done = list(_cleaner().encode_tree(source, tmp_path / "out"))
        assert [(i.status, i.relative, i.count) for i in survey] == [
            (i.status, i.relative, i.count) for i in done
        ]

    def test_a_file_is_refused(self, tmp_path):
        (tmp_path / "x.txt").write_text("x", encoding="utf-8")
        with pytest.raises(CleanPromptError, match="not a folder"):
            _cleaner().survey_tree(tmp_path / "x.txt")


class TestKindRestrictedDecoding:
    """CP-073: the primitive under least-privilege tool arguments."""

    def test_decode_only_some_kinds(self):
        cleaner = _cleaner()
        safe = cleaner.encode_text("ann@example.com 4242 4242 4242 4242").text
        assert cleaner.decode_report(safe, kinds={"EMAIL"}).text == (
            "ann@example.com [CREDIT_CARD-1]"
        )
        assert cleaner.decode_report(safe, kinds=()).text == safe
        assert cleaner.decode_report(safe).text == "ann@example.com 4242 4242 4242 4242"

    def test_kinds_in_names_kinds_never_values(self):
        cleaner = _cleaner()
        cleaner.encode_text("ann@example.com and bob@example.org, 4242 4242 4242 4242")
        assert cleaner.kinds_in("[EMAIL-2] [email-1] [CREDIT_CARD-1] [X-9]") == (
            "CREDIT_CARD",
            "EMAIL",
            "EMAIL",
        )
        assert cleaner.kinds_in("nothing here") == ()
