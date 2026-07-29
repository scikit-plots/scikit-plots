:orphan:


```mermaid

flowchart TB
    D[CorpusDocument collection]
    B[SimilarityIndex.build]
    M{Search mode}
    S[Strict text matching]
    K[Keyword or BM25]
    V[Semantic vector search]
    H[Hybrid rank fusion]
    Q[Search query]
    C[Candidate scoring]
    R[Sort, threshold, and limit]
    O[SearchResult list]

    D --> B
    B --> M
    M --> S
    M --> K
    M --> V
    M --> H
    Q --> C
    S --> C
    K --> C
    V --> C
    H --> C
    C --> R
    R --> O

```
