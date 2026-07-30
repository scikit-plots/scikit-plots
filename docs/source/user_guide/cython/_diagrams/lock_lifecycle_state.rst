:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    stateDiagram-v2
        [*] --> Idle
        Idle --> Requested : build, GC, pin, or purge requests lock
        Requested --> Acquired : exclusive ownership granted
        Requested --> Denied : contention or timeout
        Denied --> Idle : caller retries or exits

        Acquired --> Active : protected operation runs
        Active --> Releasing : operation completes
        Active --> Failed : operation raises
        Failed --> Releasing : cleanup begins

        Releasing --> Idle : ownership released
