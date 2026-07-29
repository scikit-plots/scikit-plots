:orphan:


```mermaid

flowchart TB
    A[Archive path]
    D[Detect supported archive]
    L[Apply member and byte limits]
    T[Create extraction destination]
    M[Inspect each member]
    S{Safe path and supported type?}
    X[Extract accepted member]
    R[Dispatch member to reader]
    C[Collect CorpusDocument objects]
    J[Reject or skip member]

    A --> D
    D --> L
    L --> T
    T --> M
    M --> S
    S -->|Yes| X
    X --> R
    R --> C
    S -->|No| J

```
