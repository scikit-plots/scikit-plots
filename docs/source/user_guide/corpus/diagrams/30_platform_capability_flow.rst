:orphan:


```mermaid

flowchart TB
    A[Application request]
    C[Capability preflight]
    P{Runtime profile}
    F[Full CPython environment]
    M[Minimal CPython environment]
    W[Pyodide, Emscripten, or JupyterLite]
    O[Optional native, model, and subprocess backends]
    S[Pure-Python and browser-safe stages]
    D[Actionable unavailable-capability diagnostic]
    R[Resolved execution plan]

    A --> C
    C --> P
    P --> F
    P --> M
    P --> W
    F --> O
    F --> S
    M --> S
    M --> D
    W --> S
    W --> D
    O --> R
    S --> R
    D --> R

```
