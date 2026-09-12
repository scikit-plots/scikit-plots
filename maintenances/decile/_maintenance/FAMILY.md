# Decile family and ownership boundaries

```text
scikitplot.decile
  ├─ __init__.py
  │    ├─ exports current `_decile_modelplotpy` API
  │    ├─ exposes `kds` namespace
  │    └─ exposes `modelplotpy` legacy namespace
  ├─ _decile_modelplotpy.py
  │    └─ current ModelPlotPy + response/lift/gain/financial plots
  ├─ kds/
  │    └─ vendored KDS-style decile table + lift/gain/KS/report adapter
  └─ modelplotpy/
       └─ legacy ModelPlotPy compatibility implementation

shared owners outside this subsystem
  ├─ scikitplot._preprocess
  ├─ scikitplot.api._utils.validation
  ├─ scikitplot.utils._matplotlib
  ├─ scikitplot._docstrings
  ├─ scikitplot._testing
  └─ scikitplot.externals._seaborn._compat
```

Do not collapse current ModelPlotPy and legacy ModelPlotPy into one evidence lane. The legacy namespace remains user-visible and can have independent side effects/regressions. KDS parameters that select a positive class or score column are part of the KDS adapter's contract even though validation helpers live elsewhere.
