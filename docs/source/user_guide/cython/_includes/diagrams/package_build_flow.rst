.. mermaid::

   flowchart TD

       A[Request package build] --> B[Resolve example or package source]
       B --> C[Validate package root and containment]
       C --> D[Collect package files and dependencies]
       D --> E[Apply SecurityPolicy]
       E --> F[Compute package cache key]
       F --> G[Acquire package lock]
       G --> H{Valid cached package available?}

       H -->|Yes| I[Validate cached package manifest]
       I --> J[Import package or module]
       J --> K[Return package build result]

       H -->|No| L[Create private staging area]
       L --> M[Copy or render sources]
       M --> N[Compile package extensions]
       N --> O{Build succeeded?}
       O -->|No| P[Abort and remove staging]
       O -->|Yes| Q[Validate all package outputs]
       Q --> R[Write metadata and manifest]
       R --> S[Publish package cache atomically]
       S --> T[Import built package]
       T --> K
