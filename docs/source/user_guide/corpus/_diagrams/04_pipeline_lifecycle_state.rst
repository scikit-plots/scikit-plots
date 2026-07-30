:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    stateDiagram-v2
        direction TB

        [*] --> Configured
        Configured --> Dispatching: run / run_url / run_batch
        Dispatching --> Reading: reader selected
        Reading --> Transforming: documents collected
        Transforming --> Embedding: engine configured
        Transforming --> Exporting: embedding skipped
        Embedding --> Exporting
        Exporting --> Completed
        Completed --> Configured: next independent run

        Reading --> Failed: source or reader error
        Exporting --> Failed: export error
        Transforming --> TransformWarning: optional stage error
        TransformWarning --> Exporting: continue with unchanged documents
        Failed --> Configured: caller handles exception
