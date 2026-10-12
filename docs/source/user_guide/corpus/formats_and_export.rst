.. currentmodule:: scikitplot.corpus

.. _corpus-formats-export:

Formats, structured evidence, and export
========================================

Readers normalize heterogeneous sources into :class:`CorpusDocument` rather
than making exporters understand every original file format.

Important provenance fields include the source path/URI, chunk identity,
section/type metadata and reader-specific metadata such as timecodes or archive
member paths.

Archives
--------

Archive extraction is bounded by file-count and total-size limits. Extracted
members are routed through the same builder reader seam as normal files. A
format-specific reader may be tried first for archive extensions that have a
dedicated meaning; generic extraction is a deliberate fallback, not a silent
replacement.

Export and adapters
-------------------

Use the export/adapters layer after canonical document creation rather than
re-parsing source files. Existing adapters cover downstream formats such as
LangChain/LangGraph/MCP and data exports.

When adding a new reader or exporter, keep source decoding and destination
serialization separate so a new output format cannot alter ingestion semantics.

Export and load back
--------------------

:func:`export_documents` writes a document list in one :class:`ExportFormat`.
:func:`load_documents` reads it back. Only some formats keep every field, and
only some are safe to load:

===========================  ===============  ===========  ==================================================
Format                       Embeddings       Load back    Needs
===========================  ===============  ===========  ==================================================
``JSON``, ``JSONL``          yes, with dtype  yes, safe    nothing
``YAML``                     yes, with dtype  yes, safe    PyYAML
``TOML``                     yes, with dtype  yes, safe    ``tomli-w`` (``tomli`` to read before Python 3.11)
``PICKLE``, ``JOBLIB``       yes              ``trusted``  nothing / joblib
``CSV``, ``PANDAS``          no               refused      nothing / pandas
``PARQUET``, ``POLARS``      no               refused      pandas + pyarrow / polars
``NUMPY``                    only embeddings  refused      nothing
``HUGGINGFACE``              yes              refused      datasets
``MLFLOW``                   yes              see below    mlflow
===========================  ===============  ===========  ==================================================

The four text formats share one contract with the evaluation reports:

* **Strict JSON.** ``NaN`` is written as ``null``, because JSON has no
  ``NaN``, and an infinity is refused.
* **Escaped line breaks.** The characters U+0085, U+2028 and U+2029 are
  written escaped, so a line-splitting reader never breaks a JSON Lines
  record in two.
* **No YAML aliases.** A YAML file that uses them is refused on load.
* **TOML.** TOML has no top-level array, so a TOML export keeps the
  documents under the key ``documents``.
* **Exact embeddings.** Each embedding is stored as a list of numbers with
  its ``embedding_dtype``, so it comes back with the same values and dtype,
  ``NaN`` included. Only one-dimensional ``float16``, ``float32`` and
  ``float64`` arrays can be written this way; an integer, boolean, complex
  or 2-D array is refused, naming the document. A plain Python list counts
  as ``float64``.
* **Checked records.** Every record is checked field by field on load. A
  wrong type (``chunk_index: "7"``), an unknown key, an unquoted YAML or
  TOML date where a string belongs, or an embedding value that overflows its
  dtype is refused, and the error names the file and the record. A file
  that breaks the contract never loads as plausible documents.

.. code-block:: python

   import tempfile
   from pathlib import Path

   import numpy as np
   from scikitplot.corpus import CorpusDocument, ExportFormat, export_documents, load_documents

   docs = [
       CorpusDocument.create("notes.txt", 0, "First paragraph.", metadata={"page": 1},
                             embedding=np.array([0.25, np.nan], dtype=np.float32)),
       CorpusDocument.create("notes.txt", 1, "Second paragraph.", metadata={"page": np.int64(2)}),
   ]
   out = Path(tempfile.mkdtemp())
   path = export_documents(docs, out / "corpus.jsonl", ExportFormat.JSONL)
   back = load_documents(path)              # the format comes from the suffix (or format="jsonl")
   back[0].embedding                        # array([0.25,  nan], dtype=float32)
   back[1].metadata                         # {'page': 2}

Metadata must hold JSON-like values: strings, numbers, booleans, ``None``,
lists, and mappings with string keys. NumPy scalars are converted. Anything
else, such as a ``datetime`` or a ``set``, is refused, and the error names
the document. Tuples come back as lists.

Each format has limits of its own:

* **JSON.** A ``NaN`` in metadata reads back as ``None``. A ``NaN`` in
  any other field (``bbox``, a timecode) is refused, because it would not
  come back.
* **TOML.** ``None`` inside metadata or inside a list is refused, because
  TOML would silently drop it.
* **JSON and TOML.** A string with a lone surrogate is refused, because
  these formats cannot encode it. YAML keeps it.

Loading a CSV, Parquet, NumPy or HuggingFace export raises
:class:`ValueError` and names the formats to use instead. Those formats
flatten metadata or drop embeddings, so they cannot rebuild every field, and
an empty or partial result would look like real data. An MLflow export is a
JSON Lines artifact of a run: download its ``corpus_documents.jsonl`` and
load that file. A pickle or joblib file
can run code when it is loaded, so it needs ``trusted=True``. Every format
accepts ``expected_sha256``, which checks the file before it is parsed.
