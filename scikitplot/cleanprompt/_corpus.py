"""
The bridge to :mod:`scikitplot.corpus`: optional in both directions.

cleanprompt and corpus both turn raw files into text. This module is the one
place they meet, and it is built so that either works without the other:

* cleanprompt uses corpus's readers for formats it does not read itself (PDF
  today) — :func:`read_text`;
* corpus uses cleanprompt to make its documents safe before they are embedded,
  indexed or sent to a model — :func:`redact_documents`;
* corpus can read Office files with cleanprompt's standard-library reader —
  :func:`register_corpus_readers`.

Notes
-----
**User notes.** A typical pipeline reads with corpus and redacts with
cleanprompt, sharing one cleaner so a name is the same stand-in in every
chunk::

    from scikitplot.corpus import DocumentReader
    from scikitplot.cleanprompt import FluentCleanPrompt, redact_documents

    cleaner = FluentCleanPrompt().packs("patient").ner("auto").materialize()
    safe = redact_documents(DocumentReader.create("notes.pdf").get_documents(), cleaner)
    # ... embed, index, prompt with `safe`; decode replies with cleaner.decode

**Developer notes — what "redacted" means for a corpus document.**

*Every text-bearing field is redacted*, not only ``text``: ``raw_text``,
``normalized_text``, ``source_title``, ``source_author``, ``url`` and every
string inside ``metadata`` carry the same content in other forms.

*Everything derived from the original text is dropped*: tokens, lemmas, stems,
keywords, morphemes, script spans, the embedding and its manifest, the raw
bytes and tensor, character offsets and counts. Each is a copy or a function
of the text that was just hidden; keeping one keeps the secret. They are
recomputed downstream from the safe text if wanted.

*Identity is recomputed.* ``content_hash`` and ``doc_id`` are digests of the
original text, and a digest of a short secret can be reversed by guessing, so
both are recomputed from the redacted text, and ``parent_doc_id`` is mapped to
the parent's new id — or cleared when the parent is not in the batch.

*Corpus's pipeline hooks are not used.* They fail open — a hook that raises is
logged and skipped — which is the right behaviour for enrichment and the wrong
one for redaction. :func:`redact_documents` raises instead.

*No import at module scope.* ``scikitplot.corpus`` is imported inside each
function, and the contract check ``CP-INDEP-001`` enforces that this file is
the only one that imports it at all.

See Also
--------
scikitplot.cleanprompt._runtime.Cleaner : The cleaner these functions use.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ._exceptions import CapabilityError, CleanPromptError
from ._office import OFFICE_EXTENSIONS, extract_office_text

__all__ = [
    "DERIVED_FIELDS",
    "TEXT_FIELDS",
    "read_text",
    "redact_documents",
    "register_corpus_readers",
]

#: Corpus document fields that carry text and are redacted.
TEXT_FIELDS = (
    "text",
    "raw_text",
    "normalized_text",
    "source_title",
    "source_author",
    "url",
)

#: Corpus document fields derived from the original text, cleared on redaction.
DERIVED_FIELDS = (
    "tokens",
    "lemmas",
    "stems",
    "keywords",
    "morphemes",
    "script_spans",
    "determinative_groups",
    "embedding",
    "embedding_manifest_id",
    "raw_bytes",
    "raw_tensor",
    "char_start",
    "char_end",
    "grapheme_count",
    "codepoint_count",
    "semanteme_count",
)

_INSTALL = "pip install scikit-plots-corpus"


def _corpus() -> Any:
    """
    Import the corpus package, or raise naming what is wrong.

    Notes
    -----
    **Developer notes.** ``ImportError`` means corpus is absent. Any other
    exception during its import means it is present but cannot run in this
    interpreter — measured: corpus evaluates ``dict[str, Any]`` at import
    time, which Python 3.8 rejects with ``TypeError`` — and is reported as
    ``BROKEN`` rather than escaping as a crash from inside a redaction call.
    """
    try:
        # Spelt ``import scikitplot.corpus`` on purpose: the architecture test
        # and CP-INDEP-001 read this statement to prove the one permitted
        # sibling import is exactly the corpus bridge. ``from scikitplot import
        # corpus`` names only the parent and is reported as a violation.
        import scikitplot.corpus  # noqa: PLC0415
    except ImportError as exc:
        msg = f"this needs scikitplot.corpus, which could not be imported: {exc}"
        raise CapabilityError(
            msg, tier="corpus", status="ABSENT", install_hint=_INSTALL
        ) from exc
    # attributed and re-raised as a capability
    except Exception as exc:  # noqa: BLE001
        msg = f"scikitplot.corpus is installed but cannot be imported here: {type(exc).__name__}: {exc}"
        raise CapabilityError(
            msg, tier="corpus", status="BROKEN", install_hint=_INSTALL
        ) from exc
    return scikitplot.corpus


def read_text(path: str | os.PathLike) -> str:
    """
    Return a file's text as read by a corpus reader.

    Parameters
    ----------
    path : path-like
        A file whose extension corpus has a reader for (PDF, for example).

    Returns
    -------
    str
        The documents' texts in reading order, separated by a blank line and
        newline-terminated.

    Raises
    ------
    CapabilityError
        If corpus, or the library its reader needs, is not installed.
    CleanPromptError
        If the reader fails on the file, or yields no text at all.

    Notes
    -----
    **Developer notes — nothing is dropped, and nothing is assumed.**

    Corpus readers apply a noise filter by default that discards short and
    digit-only chunks. That is right for building a retrieval corpus and wrong
    here: ``MRN: 00412345`` is exactly the kind of chunk it drops, and a
    redaction tool that silently loses part of a document is reporting on a
    document it did not read. So the reader runs with a filter that keeps
    every chunk.

    A reader that yields no text — a scanned PDF with no text layer, a file
    that is not what its extension says — is refused rather than returned as
    an empty string, so a folder walk reports it instead of writing an empty
    file that looks like a successful encoding. A reader's own failure is
    re-raised as a :class:`CleanPromptError` naming its exception type.
    """
    corpus = _corpus()
    source = Path(path)

    class _KeepEverything(corpus.FilterBase):
        """Keep every chunk: the caller, not a noise filter, decides."""

        def include(self, doc: Any) -> bool:
            del doc
            return True

    try:
        reader = corpus.DocumentReader.create(source, filter_=_KeepEverything())
        parts = [
            doc.text for doc in reader.get_documents() if getattr(doc, "text", None)
        ]
    except ImportError as exc:
        msg = f"reading {source.name!r} needs a library that is not installed: {exc}"
        raise CapabilityError(
            msg, tier="corpus", status="ABSENT", install_hint=_INSTALL
        ) from exc
    # attributed and re-raised, never suppressed
    except Exception as exc:  # noqa: BLE001
        msg = (
            f"the corpus reader failed on {source.name!r}: {type(exc).__name__}: {exc}"
        )
        raise CleanPromptError(msg) from exc
    if not parts:
        msg = f"the corpus reader found no text in {source.name!r}; nothing was read"
        raise CleanPromptError(msg)
    return "\n\n".join(part.rstrip("\n") for part in parts) + "\n"


def _json_native(value: Any) -> bool:
    """Return whether a value is made only of JSON types."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return True
    if isinstance(value, list):
        return all(_json_native(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _json_native(item) for key, item in value.items()
        )
    return False


def _redact_metadata(
    metadata: dict[str, Any], cleaner: Any, name: str
) -> dict[str, Any]:
    """
    Redact a metadata mapping as a JSON document.

    Notes
    -----
    **Developer notes.** Encoding the mapping *as JSON* is what lets field
    rules apply to it: ``{"patient_name": "..."}`` in metadata is hidden by the
    same key-based rule that hides it in a JSON file. A value that is not a
    JSON type cannot be encoded faithfully, so it is dropped and its key listed
    under ``cleanprompt_dropped`` — never stringified and sent.
    """
    kept = {
        key: value
        for key, value in metadata.items()
        if isinstance(key, str) and _json_native(value)
    }
    dropped = sorted(
        str(key) for key in metadata if not (isinstance(key, str) and key in kept)
    )
    if kept:
        encoded = cleaner.encode_text(
            json.dumps(kept, ensure_ascii=False, sort_keys=True), "json", name=name
        )
        kept = json.loads(encoded.text)
    if dropped:
        kept["cleanprompt_dropped"] = dropped
    return kept


def redact_documents(
    documents: Iterable[Any], cleaner: Any, format: str = "text"
) -> list[Any]:  # noqa: A002 - public keyword
    """
    Return corpus documents with every text field redacted by one cleaner.

    Parameters
    ----------
    documents : iterable of CorpusDocument
        Documents from any corpus reader or pipeline.
    cleaner : Cleaner
        The cleaner whose vault collects what is removed; decode replies with
        it.
    format : str, default='text'
        The format each text field is read as. It must be selected by the
        cleaner's plan.

    Returns
    -------
    list of CorpusDocument
        New documents, in the same order. The inputs are unchanged.

    Raises
    ------
    CleanPromptError
        If any field cannot be redacted. Nothing is returned partly redacted.

    Examples
    --------
    >>> from scikitplot.corpus._schema import CorpusDocument  # doctest: +SKIP
    >>> from scikitplot.cleanprompt._plan import FluentCleanPrompt  # doctest: +SKIP
    >>> doc = CorpusDocument.create(
    ...     "n.txt", 0, "Mail ann@example.com"
    ... )  # doctest: +SKIP
    >>> redact_documents([doc], FluentCleanPrompt().materialize())[
    ...     0
    ... ].text  # doctest: +SKIP
    'Mail [EMAIL-1]'
    """
    corpus = _corpus()
    make = corpus.CorpusDocument
    staged = []
    renamed: dict[str, str] = {}
    for document in documents:
        name = f"{document.input_path}#{document.chunk_index}"
        changes: dict[str, Any] = dict.fromkeys(DERIVED_FIELDS)
        for field in TEXT_FIELDS:
            value = getattr(document, field)
            if isinstance(value, str) and value:
                changes[field] = cleaner.encode_text(value, format, name=name).text
        changes["metadata"] = _redact_metadata(
            dict(document.metadata or {}), cleaner, name
        )
        text = changes.get("text", document.text)
        changes["content_hash"] = make.make_content_hash(text=text)
        changes["doc_id"] = make.make_doc_id(
            document.input_path, document.chunk_index, text, document.source_type
        )
        renamed[document.doc_id] = changes["doc_id"]
        staged.append((document, changes))
    out = []
    for document, changes in staged:
        parent = document.parent_doc_id
        changes["parent_doc_id"] = renamed.get(parent) if parent else None
        out.append(document.replace(**changes))
    return out


def register_corpus_readers() -> tuple[str, ...]:
    """
    Let corpus read ``.docx``, ``.xlsx`` and ``.pptx`` with cleanprompt's reader.

    Returns
    -------
    tuple of str
        The extensions registered. An extension corpus already has a reader
        for is left alone and not listed.

    Raises
    ------
    CapabilityError
        If corpus is not installed.

    Notes
    -----
    **User notes.** Call it once, before ``DocumentReader.create``. It changes
    corpus's global reader registry, which is why it is never done on import.

    **Developer notes.** The reader is the standard-library one in
    ``_office.py``, with the same size limits and DTD refusal, so corpus gains
    Office support without a new dependency.
    """
    corpus = _corpus()
    existing = set(corpus.DocumentReader.supported_types())
    registered = []
    for extension, kind in sorted(OFFICE_EXTENSIONS.items()):
        if extension in existing:
            continue

        def extractor(path: Path, _kind: str = kind, **_: Any) -> str:
            return extract_office_text(Path(path).read_bytes(), _kind)

        corpus.CustomReader.register(
            name=f"CleanPrompt{kind.capitalize()}{extension[1:].capitalize()}Reader",
            extensions=[extension],
            extractor=extractor,
        )
        registered.append(extension)
    return tuple(registered)
