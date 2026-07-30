:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TB
        B[CorpusBuilder.build]
        X[Expand files, directories, URLs, and archives]
        I[Ingest sources serially or with workers]
        R[Reader, chunker, and filter]
        N[Optional normalization]
        E[Optional enrichment]
        V[Optional embedding]
        S[Optional similarity index]
        O[BuildResult]
        A[Search, export, or adapt]

        B --> X
        X --> I
        I --> R
        R --> N
        N --> E
        E --> V
        V --> S
        S --> O
        O --> A
