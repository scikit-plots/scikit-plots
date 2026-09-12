# Maintaining `scikitplot.impute`

`scikitplot.impute` is an estimator/composition layer. It owns imputation
semantics and index-access policy, but it does **not** own the Annoy or Voyager
implementations it queries.

Start with `_maintenance/FRESH_CHAT_HANDOFF.md`, then run the contract, review,
and maintenance regression suite from any working directory.

Do not make missing surrounding-package files look like impute defects. The
supplied review snapshot is partial: root package wiring, `utils._path`,
`utils._time`, the packaging shim, experimental activation, and a built Annoy
extension are absent. Those are integration/release evidence gaps.

Conversely, do not hide impute-owned findings behind that partial snapshot. The
current broad Annoy fallback and unused module-scope pandas import are real local
review findings. `_base.py` also needs an explicit ownership decision because
its tests currently test sklearn rather than the local mirror.
