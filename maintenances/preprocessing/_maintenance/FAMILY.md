# Preprocessing family and ownership boundaries

```text
scikitplot.preprocessing
  ├─ __init__.py
  │    └─ exports _encoders.__all__
  └─ _encoders.py
       ├─ GetDummies
       │    ├─ pandas-backed token normalization/get_dummies
       │    ├─ selective source-column expansion
       │    ├─ dense DataFrame passthrough
       │    └─ dummy-only SciPy CSR sparse mode
       └─ DummyCodeEncoder
            ├─ sklearn BaseEstimator/TransformerMixin contract
            ├─ per-feature categories + multi-label token expansion
            ├─ dense/CSR output
            ├─ unknown/drop/infrequent policies
            ├─ inverse_transform
            └─ get_feature_names_out / set_output

external owners
  ├─ pandas: Series.str.get_dummies/DataFrame behavior
  ├─ sklearn: estimator, validation and output-container infrastructure
  ├─ scipy: CSR implementation
  └─ root scikitplot: __version__ + vendored packaging/version path
```

The most important invariant is that **category identity is `(input_feature, category)`, never category value alone**. The same string appearing in two columns must create two independent encoded dimensions.
