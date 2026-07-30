:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TD
        A[Request single-module build] --> B[Normalize source path]
        B --> C[Resolve compiler options]
        C --> D[Validate includes, arguments, and roots]
        D --> E[Compute cache key]
        E --> F[Acquire per-key lock]
        F --> G{Valid cached build available?}

        G -->|Yes| H[Read and validate metadata]
        H --> I[Verify artifact containment and integrity]
        I --> J[Load compiled extension]
        J --> K[Return result]

        G -->|No| L[Create private staging directory]
        L --> M[Compile extension]
        M --> N{Compilation succeeded?}
        N -->|No| O[Capture diagnostics and clean staging]
        N -->|Yes| P[Validate produced artifacts]
        P --> Q[Write metadata and manifest]
        Q --> R[Publish build atomically]
        R --> S[Load extension]
        S --> K
