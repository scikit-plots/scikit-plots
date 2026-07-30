:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TB
        I[Input or environment failure]
        R[Reader and parser boundary]
        T[Transform stage boundary]
        E[Embedding or model boundary]
        S[Storage or export boundary]
        P{Configured handling}
        X[Raise typed exception]
        G[Record source error in BuildResult]
        W[Log optional-stage warning and continue]
        K[PipelineGuard skip, log, or retry]
        C[Cleanup and close resources]

        I --> R
        R --> T
        T --> E
        E --> S
        R --> P
        T --> P
        E --> P
        S --> P
        P --> X
        P --> G
        P --> W
        P --> K
        X --> C
        G --> C
        W --> C
        K --> C
