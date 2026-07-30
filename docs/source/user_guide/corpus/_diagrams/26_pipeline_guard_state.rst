:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    stateDiagram-v2
        direction TB

        [*] --> Created
        Created --> CheckpointLoaded: checkpoint configured
        Created --> Iterating: no checkpoint
        CheckpointLoaded --> Iterating

        Iterating --> DedupCheck: document received
        DedupCheck --> Iterating: duplicate or already checkpointed
        DedupCheck --> Yielded: accepted
        Yielded --> CheckpointAppend: checkpoint interval reached
        CheckpointAppend --> Iterating
        Yielded --> Iterating: no checkpoint write

        Iterating --> ErrorPolicy: source raises
        ErrorPolicy --> Iterating: skip or log
        ErrorPolicy --> Retrying: retry policy
        Retrying --> Iterating: retry succeeds
        Retrying --> Iterating: retries exhausted and skipped
        ErrorPolicy --> Failed: raise policy

        Iterating --> Closed: close or context exit
        Failed --> Closed
        Closed --> [*]
