:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    flowchart TD
        A[Input request] --> B[Normalize paths]
        B --> C[Validate roots and containment]
        C --> D[Validate include directories]
        D --> E[Validate compiler and linker arguments]
        E --> F[Validate template or example source]
        F --> G[Validate cache keys and destinations]
        G --> H[Apply resource budgets]
        H --> I{Approved?}

        I -->|Yes| J[Proceed to build, cache, or import]
        I -->|No| K[Reject with structured error]
