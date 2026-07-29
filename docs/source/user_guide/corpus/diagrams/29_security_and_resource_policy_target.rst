:orphan:


```mermaid

flowchart TB
    U[Untrusted file, URL, archive, media, or plugin]
    C[Recommended CorpusSession context]
    P[Central PolicyEngine]
    B[Hierarchical ResourceBudget]
    N[Network and redirect policy]
    A[Archive and path policy]
    R[Parser and sandbox policy]
    X[Plugin and serialization policy]
    Q[Validated bounded processing]
    D[Diagnostics and provenance]
    O[Trusted output sinks]

    U --> C
    C --> P
    C --> B
    P --> N
    P --> A
    P --> R
    P --> X
    B --> N
    B --> A
    B --> R
    B --> X
    N --> Q
    A --> Q
    R --> Q
    X --> Q
    Q --> D
    D --> O

```
