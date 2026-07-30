:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TB
        I[import scikitplot.corpus]
        F[Package facade __init__]
        M[Import public submodules]
        R[Reader registration side effect]
        E[Aggregate module __all__ lists]
        P[Top-level public names]

        I --> F
        F --> M
        M --> R
        M --> E
        E --> P

        P --> B[CorpusBuilder]
        P --> L[CorpusPipeline]
        P --> DR[DocumentReader]
        P --> C[Components and data contracts]
