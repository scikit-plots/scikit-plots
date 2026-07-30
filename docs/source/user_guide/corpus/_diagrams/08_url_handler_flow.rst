:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TB
        U[Input URL]
        C[classify_url]
        Q{Extensionless web URL?}
        P[Optional Content-Type probe]
        K{Resolved URL kind}
        Y[YouTube reader]
        W[Web reader]
        R[Provider-specific resolve_url]
        D[Bounded download]
        I[Infer safe local extension]
        L[Local path for reader dispatch]

        U --> C
        C --> Q
        Q -->|Yes and enabled| P
        Q -->|No| K
        P --> K
        K -->|YouTube| Y
        K -->|HTML page| W
        K -->|Downloadable, Drive, GitHub| R
        R --> D
        D --> I
        I --> L
