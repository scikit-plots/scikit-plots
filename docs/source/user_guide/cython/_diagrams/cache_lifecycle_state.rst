:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    stateDiagram-v2
        [*] --> Absent

        Absent --> Staging : build starts
        Staging --> Published : validated atomic publish
        Staging --> Failed : build or validation fails
        Failed --> Absent : staging cleanup

        Published --> Loaded : artifact imported
        Loaded --> Published : import reference released

        Published --> Pinned : pin added
        Pinned --> Published : pin removed

        Published --> Rejected : invalid metadata or artifact
        Rejected --> Absent : purge or quarantine cleanup

        Published --> Reclaimed : eligible GC removal
        Pinned --> Reclaimed : explicit forced purge
        Reclaimed --> Absent
