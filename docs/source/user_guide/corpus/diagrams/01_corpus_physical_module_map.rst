:orphan:


```mermaid

flowchart TB
    F[Public facade<br/>scikitplot.corpus]

    subgraph ORCH[Orchestration]
        B[_corpus_builder]
        P[_pipeline]
        H[_custom_hooks]
    end

    subgraph INPUT[Input and dispatch]
        S[_sources]
        U[_url_handler]
        D[_downloader]
        A[_archive_handler]
        R[_readers and _base]
    end

    subgraph PROCESS[Processing]
        C[_chunkers]
        N[_normalizers]
        E[_enrichers]
        M[_embeddings]
    end

    subgraph DATA[Contracts and outputs]
        SC[_schema and _types]
        X[_similarity]
        ST[_storage]
        EX[_export]
        AD[_adapters]
        MD[_metadata]
        RG[_registry]
    end

    F --> B
    F --> P
    F --> H
    B --> S
    P --> S
    H --> P
    S --> U
    U --> D
    D --> A
    A --> R
    R --> C
    C --> N
    N --> E
    E --> M
    M --> SC
    SC --> X
    SC --> ST
    SC --> EX
    SC --> AD
    SC --> MD
    RG -. component lookup .-> B
    RG -. component lookup .-> P

```
