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
