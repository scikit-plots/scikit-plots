:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TD
        A[Cache maintenance starts] --> B[List cache entries]
        B --> C[Acquire entry lock]
        C --> D[Read current pin registry]
        D --> E[Evaluate age, size, and policy]
        E --> F{Entry pinned?}
        F -->|Yes| G[Keep entry]
        F -->|No| H{Entry active or protected?}
        H -->|Yes| I[Skip entry]
        H -->|No| J{Eligible for removal?}
        J -->|No| G
        J -->|Yes| K[Delete cached entry]
        K --> L[Record maintenance result]
        G --> M[Release entry lock]
        I --> M
        L --> M
        M --> N{More entries?}
        N -->|Yes| C
        N -->|No| O[Finish GC report]
