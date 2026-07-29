:orphan:


```mermaid

flowchart TB
    U[User or application]
    A[CorpusBuilder or CorpusPipeline]
    S[Source resolution]
    R[Format-specific reader]
    C[Chunking and filtering]
    T[Normalize and enrich]
    E[Optional embedding]
    D[CorpusDocument collection]
    O[Search, storage, export, adapters]

    U --> A
    A --> S
    S --> R
    R --> C
    C --> T
    T --> E
    E --> D
    D --> O

```
