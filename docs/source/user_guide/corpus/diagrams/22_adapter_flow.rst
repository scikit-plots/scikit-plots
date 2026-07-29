:orphan:


```mermaid

flowchart TB
    D[CorpusDocument or SearchResult objects]
    A[Adapter functions]
    L[LangChain documents or retriever]
    G[LangGraph state]
    M[MCP resources, tool results, or server]
    H[HuggingFace Dataset]
    R[RAG tuples]
    J[JSONL iterator]
    N[NumPy arrays]
    T[TensorFlow or Torch datasets]

    D --> A
    A --> L
    A --> G
    A --> M
    A --> H
    A --> R
    A --> J
    A --> N
    A --> T

```
