:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TB
        D[CorpusDocument objects]
        B[StorageBase contract]
        M[InMemoryStorage]
        J[JSONLStorage]
        S[SQLiteStorage with FTS5]
        W[save or save_batch]
        G[get by document ID]
        Q[query and pagination]
        C[count]
        R[QueryResult or CorpusDocument]

        D --> B
        B --> M
        B --> J
        B --> S
        M --> W
        J --> W
        S --> W
        M --> G
        J --> G
        S --> G
        M --> Q
        J --> Q
        S --> Q
        Q --> R
        G --> R
        W --> C
