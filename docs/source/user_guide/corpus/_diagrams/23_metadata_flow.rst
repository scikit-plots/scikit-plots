:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TB
        F[Source filename and path]
        P[provenance_from_filename]
        D[CorpusDocument collection]
        S[compute_stats]
        M[CollectionManifest]
        C[CorpusStats]
        O[Document provenance and collection metadata]

        F --> P
        P --> O
        D --> S
        S --> C
        D --> M
        M --> O
        C --> O
