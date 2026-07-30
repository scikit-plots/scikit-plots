:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TB
        D[Normalized or original documents]
        C[EnricherConfig]
        T[Tokenize]
        F[Filter stopwords and punctuation]
        L[Optional lemmatization]
        S[Optional stemming]
        K[Keyword and score extraction]
        M[Counts, POS, NER, and metadata]
        O[Enriched CorpusDocument objects]

        D --> C
        C --> T
        T --> F
        F --> L
        L --> S
        S --> K
        K --> M
        M --> O
