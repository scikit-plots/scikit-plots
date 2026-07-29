:orphan:


```mermaid

flowchart TB
    D[CorpusDocument collection]
    T[TextNormalizer or NormalizationPipeline]
    O[Select original text]
    S[Apply configured steps in order]
    U[Unicode normalization]
    W[Whitespace cleanup]
    H[HTML stripping]
    L[Lowercasing]
    R[Duplicate-line removal]
    N[New document with normalized_text]

    D --> T
    T --> O
    O --> S
    S --> U
    U --> W
    W --> H
    H --> L
    L --> R
    R --> N

```
