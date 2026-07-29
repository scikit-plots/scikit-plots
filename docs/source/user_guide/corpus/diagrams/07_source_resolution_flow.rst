:orphan:


```mermaid

flowchart TB
    C[CorpusSource constructor]
    K{Source kind}
    F[Single file]
    D[Directory or glob]
    U[URL list]
    M[Manifest]
    V[Validate source configuration]
    I[iter_entries generator]
    E[SourceEntry with provenance]
    R[Reader or builder ingestion]

    C --> K
    K --> F
    K --> D
    K --> U
    K --> M
    F --> V
    D --> V
    U --> V
    M --> V
    V --> I
    I --> E
    E --> R

```
