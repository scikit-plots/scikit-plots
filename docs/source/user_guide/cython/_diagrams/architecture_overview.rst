:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart LR
        U[User code] --> API[Public API]
        API --> PUB[_public.py / _api.py]
        PUB --> SEC[SecurityPolicy]
        PUB --> BUD[Budget and profiles]
        PUB --> BLD[Builder]
        BLD --> TMP[Templates API]
        BLD --> LOCK[Per-key lock]
        BLD --> CACHE[Cache]
        BLD --> COMP[Compiler backend]
        COMP --> ART[Built artifacts]
        ART --> META[Metadata and manifest]
        META --> LDR[Loader]
        CACHE --> LDR
        LDR --> RES[Result objects]
        CACHE --> GC[GC and purge]
        CACHE --> PIN[Pins]
