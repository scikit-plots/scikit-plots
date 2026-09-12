# Impute family and ownership boundaries

```
scikitplot.impute
  ├─ __init__.py       public sklearn re-exports + experimental ANNImputer gate
  ├─ _ann.py           ANNImputer semantics and backend dispatch
  ├─ _privacy.py       public/private/external index policy
  └─ _base.py          sklearn-derived compatibility mirror (role currently ambiguous)

backend owners
  ├─ scikitplot.annoy._annoy   Cython Annoy implementation
  └─ voyager                   optional external HNSW implementation

supporting owners
  ├─ scikitplot.utils._path / _time
  ├─ scikitplot.experimental
  └─ scikitplot root/version/package wiring
```

Impute may adapt those components, but must not duplicate their implementation
or silently reinterpret a backend failure as permission to switch ownership.
