:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TB
        D[CorpusDocument collection]
        M{Primary modality}
        T[Text embedding backend]
        I[Image embedding backend]
        A[Audio or video backend]
        K[Build model and configuration cache key]
        H{Valid cache hit?}
        L[Load cached vectors]
        C[Compute vectors]
        V[Validate shape and numeric values]
        S[Publish cache entry]
        O[Attach embeddings to documents]

        D --> M
        M --> T
        M --> I
        M --> A
        T --> K
        I --> K
        A --> K
        K --> H
        H -->|Yes| L
        H -->|No| C
        C --> V
        V --> S
        L --> O
        S --> O
