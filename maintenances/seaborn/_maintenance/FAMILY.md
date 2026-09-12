# Family boundaries

`scikitplot.seaborn` is the high-level semantic plotting layer. It may consume public sklearn metrics, pandas/numpy containers, matplotlib artists, and either an installed seaborn or the project vendored seaborn compatibility layer. It does not own those dependencies.

The four owned functional families are ROC/PR (`aucplot`), evaluation matrices/reports (`evalplot`), decile analytics (`decileplot`), and estimator/model attributes (`modelplot`). A passing family does not bless another family.

`scikitplot.decile` is a separate API domain. Similar lift/gain concepts do not make `scikitplot.seaborn._decile` a maintenance extension of that submodule.
