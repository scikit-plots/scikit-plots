:orphan:


```mermaid

stateDiagram-v2
    direction TB

    [*] --> Uninitialized
    Uninitialized --> Connected: construct SQLiteStorage
    Connected --> SchemaReady: initialize schema and FTS
    SchemaReady --> Reading: get / query / count
    SchemaReady --> Writing: save / save_batch
    Reading --> SchemaReady
    Writing --> SchemaReady: commit succeeds
    Writing --> Rollback: operation fails
    Rollback --> SchemaReady
    SchemaReady --> Closed: close
    Reading --> Closed: close after operation
    Closed --> [*]

```
