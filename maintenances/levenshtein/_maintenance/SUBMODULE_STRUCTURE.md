# Submodule structure

```text
scikitplot/levenshtein/
├── __init__.py
│   └── public facade; re-exports `_core.__all__`
├── _core.py
│   ├── backend discovery and selection
│   ├── pure-Python Wagner-Fischer fallback
│   ├── distance/similarity metrics
│   ├── deterministic ranking
│   └── lazy Corpus scorer adapter
├── meson.build
│   └── package install/build registration
└── tests/test_levenshtein.py
    └── focused runtime contract
```

Related but separately owned:

```text
scikitplot/cexternals/_editdistance/   bundled native implementation
scikitplot/corpus/                     optional retrieval consumer
```
