:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TB
        I[Import built-in components]
        R[ComponentRegistry]
        C[Chunker registrations]
        F[Filter registrations]
        D[Reader registrations]
        N[Normalizer registrations]
        L[Lookup by stable name]
        B[Build configured component]
        S[Snapshot registry]
        O[Restore or extend registry]

        I --> R
        R --> C
        R --> F
        R --> D
        R --> N
        C --> L
        F --> L
        D --> L
        N --> L
        L --> B
        R --> S
        S --> O
