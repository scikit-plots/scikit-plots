# Maintenance changelog

## 2026-09-12

- Re-reviewed `scikitplot.annoy` separately from `cexternals/_annoy`.
- Modeled both compiled relationships: native backend inheritance and direct
  Cython/header compilation.
- Replaced the path-broken legacy tracker with repository-root-aware contract
  tooling and regression tests.
- Added fail-closed generation checks and identified the missing
  `scikitplot/_build_utils/tempita.py` build prerequisite.
- Classified `_annoy/annoymodule.cpp` as inactive legacy source under the current
  Meson graph, not as Cython-generated authority.
