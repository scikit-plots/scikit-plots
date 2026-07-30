:orphan:

..
  # https://github.com/mgaitan/sphinxcontrib-mermaid

.. mermaid::

    stateDiagram-v2
        direction TB

        [*] --> Created
        Created --> Ready: configuration accepted
        Ready --> Building: build
        Building --> Built: BuildResult stored
        Building --> BuiltWithErrors: some sources fail
        BuiltWithErrors --> Searchable: index available
        Built --> Searchable: index available
        Built --> Exportable
        Searchable --> Exportable
        Built --> Building: add or rebuild
        BuiltWithErrors --> Building: add or rebuild
        Exportable --> Building: add or rebuild
        Ready --> Closed: close
        Built --> Closed: close
        BuiltWithErrors --> Closed: close
        Exportable --> Closed: close
        Closed --> [*]
