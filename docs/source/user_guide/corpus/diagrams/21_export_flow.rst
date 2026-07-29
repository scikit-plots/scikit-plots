:orphan:


```mermaid

flowchart TB
    D[CorpusDocument collection]
    F[ExportFormat]
    X[export_documents dispatcher]
    S[Safe interchange formats<br/>CSV, JSON, JSONL, NumPy, Parquet]
    T[Trusted-only formats<br/>pickle and joblib]
    I[Integration formats<br/>Pandas, Polars, HuggingFace, MLflow]
    W[Write staging content]
    P[Atomic publish where supported]
    O[Output artifact]

    D --> F
    F --> X
    X --> S
    X --> T
    X --> I
    S --> W
    T --> W
    I --> W
    W --> P
    P --> O

```
