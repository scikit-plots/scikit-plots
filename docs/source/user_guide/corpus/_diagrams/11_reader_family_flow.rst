:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TB
        B[DocumentReader contract]

        B --> T[Text family<br/>TXT, Markdown, ReST]
        B --> X[Structured family<br/>PDF, XML, TEI, ALTO]
        B --> M[Media family<br/>Image, audio, video]
        B --> N[Network family<br/>Web and YouTube]
        B --> A[Archive and custom<br/>ZIP and CustomReader]

        T --> R[Raw chunk dictionaries]
        X --> R
        M --> R
        N --> R
        A --> R

        R --> C[Optional chunker]
        C --> F[Filter]
        F --> D[CorpusDocument stream]
