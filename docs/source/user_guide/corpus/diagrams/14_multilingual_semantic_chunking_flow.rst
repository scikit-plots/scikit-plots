:orphan:


```mermaid

flowchart TB
    T[Raw multilingual text]
    G[Grapheme-safe normalization]
    S[ScriptSegmenter]
    P[ScriptSpan sequence]
    W[WritingSystemAdapter]
    R[Script-specific segmentation strategy]
    M[SemanticChunker boundary decisions]
    C[Chunks with language and script metadata]

    T --> G
    G --> S
    S --> P
    P --> W
    W --> R
    R --> M
    M --> C

    L[Language data and stopwords] -. assists .-> R
    X[Custom tokenizer registry] -. extends .-> R

```
