:orphan:


```mermaid

flowchart TB
    A[run, run_url, or run_batch]
    D[Dispatch one source]
    Q{URL source?}
    RF[DocumentReader.create or from_url]
    RD[Read, chunk, and filter]
    N[Optional normalization]
    E[Optional NLP enrichment]
    V[Optional embedding]
    X[Optional export]
    P[PipelineResult]

    A --> D
    D --> Q
    Q -->|Local path| RF
    Q -->|HTTP or HTTPS| RF
    RF --> RD
    RD --> N
    N --> E
    E --> V
    V --> X
    X --> P

```
