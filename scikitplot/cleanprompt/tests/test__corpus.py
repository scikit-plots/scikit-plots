"""
Tests for :mod:`scikitplot.cleanprompt._corpus`.

Notes
-----
**Developer notes.** Everything that needs :mod:`scikitplot.corpus` is
skipped when it cannot be imported, because cleanprompt must work without it.
What does not need it — the module importing nothing, the error when corpus is
absent — is tested unconditionally.
"""

from __future__ import annotations

import ast
import builtins
import io
import pathlib
import zipfile

import pytest

from .. import _corpus
from .._corpus import (
    DERIVED_FIELDS,
    TEXT_FIELDS,
    read_text,
    redact_documents,
    register_corpus_readers,
)
from .._exceptions import CapabilityError
from .._plan import FluentCleanPrompt


def test_the_bridge_imports_corpus_only_inside_functions():
    tree = ast.parse(pathlib.Path(_corpus.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            assert not any(name.startswith("scikitplot") for name in names)


def test_absent_corpus_raises_a_capability_error(monkeypatch):
    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("scikitplot.corpus"):
            raise ImportError("simulated")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(CapabilityError) as caught:
        read_text("x.pdf")
    assert caught.value.tier == "corpus"
    assert caught.value.install_hint


def test_field_lists_do_not_overlap():
    assert not set(TEXT_FIELDS) & set(DERIVED_FIELDS)


@pytest.fixture()
def corpus():
    """The corpus package, or a skip saying why it is not usable here."""
    try:
        return _corpus._corpus()
    except CapabilityError as error:
        pytest.skip(str(error))


class TestRedactDocuments:
    def test_every_text_field_is_redacted_and_derived_fields_cleared(self, corpus):
        make = corpus.CorpusDocument.create
        parent = make(
            "n.txt",
            0,
            "Mail ann@example.com, MRN: 00412345.",
            metadata={"patient_name": "Ann Lee", "count": 3, "when": object()},
        )
        parent = parent.replace(
            tokens=["Mail", "ann@example.com"], source_title="Notes for ann@example.com"
        )
        child = make("n.txt", 1, "Again ann@example.com", parent_doc_id=parent.doc_id)
        cleaner = FluentCleanPrompt().packs("patient").materialize()
        safe_child, safe_parent = redact_documents([child, parent], cleaner)
        assert safe_parent.text == "Mail [EMAIL-1], MRN: [MRN-1]."
        assert safe_parent.source_title == "Notes for [EMAIL-1]"
        assert safe_parent.tokens is None
        assert safe_parent.metadata == {
            "count": 3,
            "patient_name": "[PERSON-1]",
            "cleanprompt_dropped": ["when"],
        }
        assert safe_child.text == "Again [EMAIL-1]"
        assert safe_parent.doc_id != parent.doc_id
        assert safe_child.parent_doc_id == safe_parent.doc_id
        assert safe_parent.content_hash == corpus.CorpusDocument.make_content_hash(
            text=safe_parent.text
        )
        assert cleaner.decode(safe_parent.text) == parent.text

    def test_a_parent_outside_the_batch_is_cleared(self, corpus):
        child = corpus.CorpusDocument.create("n.txt", 1, "x", parent_doc_id="0" * 16)
        (safe,) = redact_documents([child], FluentCleanPrompt().materialize())
        assert safe.parent_doc_id is None


class TestReaders:
    def _docx(self, path):
        body = (
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
            "<w:p><w:r><w:t>MRN: 00412345</w:t></w:r></w:p></w:body></w:document>"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", body)
        path.write_bytes(buffer.getvalue())
        return path

    def test_register_is_idempotent_and_corpus_reads_docx(self, corpus, tmp_path):
        register_corpus_readers()
        assert register_corpus_readers() == ()
        path = self._docx(tmp_path / "note.docx")
        assert read_text(path) == "MRN: 00412345\n"

    def test_a_reader_failure_is_attributed(self, corpus, tmp_path):
        from .._exceptions import CleanPromptError

        path = tmp_path / "broken.pdf"
        path.write_bytes(b"%PDF-1.4 not really")
        with pytest.raises(CleanPromptError):
            read_text(path)
