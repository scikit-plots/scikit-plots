# Submodule structure

```text
scikitplot.seaborn
├── _auc.py               aucplot / _AucPlotter
├── _confusion_matrix.py  evalplot / _ConfusionMatrixPlotter
├── _decile.py            decileplot / _DecilePlotter / label metadata
├── _model.py             modelplot / _ModelPlotter
└── tests/                 four focused regression modules
```

Dependency direction: user data/model -> scikitplot.seaborn semantic wrapper -> seaborn VectorPlotter compatibility surface -> matplotlib artists; numeric metrics are computed through sklearn/pandas/numpy. The vendored seaborn subset, when present in a complete project tree, is fallback infrastructure and has separate ownership.
