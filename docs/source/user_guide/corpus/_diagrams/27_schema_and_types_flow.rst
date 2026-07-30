:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TB
        R[Reader raw chunks]
        C[Chunk and ChunkResult records]
        D[Canonical CorpusDocument]
        E[Canonical enums<br/>source, section, modality, export, match, error]
        P[PipelineResult and BuildResult]
        S[StorageQuery and SearchResult]
        O[Export and adapter schemas]
        L[Legacy compatibility records in _types]

        R --> C
        C --> D
        E --> D
        D --> P
        D --> S
        D --> O
        L -. compatibility boundary .-> D
