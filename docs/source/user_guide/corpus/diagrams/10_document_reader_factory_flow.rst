:orphan:


```mermaid

flowchart TB
    I[Path, URL, sequence, or manifest]
    C[DocumentReader.create]
    Q{Input shape}
    U[from_url]
    M[from_manifest]
    S[_MultiSourceReader]
    L[Local path validation]
    X[Extension lookup in reader registry]
    R[Concrete DocumentReader]
    D[get_documents]

    I --> C
    C --> Q
    Q -->|URL| U
    Q -->|Manifest| M
    Q -->|Sequence| S
    Q -->|Local path| L
    U --> R
    M --> S
    S --> D
    L --> X
    X --> R
    R --> D

```
