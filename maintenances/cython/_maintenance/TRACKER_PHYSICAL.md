# Physical Tracker — `scikitplot.cython`

Re-derived from the tree. Do not hand-edit:

```console
$ python scikitplot/cython/_maintenance/check_trackers.py
```

## Totals

```text
source      23 files /  12039 LOC
tests       45 files /  11492 LOC
templates  306 files /   5709 LOC
markdown    33
```

test : source LOC = **0.95** — the project's highest.

## Areas

| Area | files | LOC |
|---|---:|---:|
| `(root)` | 23 | 12039 |
| `_templates` | 306 | 5709 |
| `tests` | 45 | 11492 |

## Largest source modules

| LOC | Module |
|---:|---|
|  2097 | `_builder.py` |
|  1450 | `_public.py` |
|  1253 | `_templates_api.py` |
|  1241 | `_custom_compiler.py` |
|  1031 | `_cache.py` |
|   785 | `__init__.pyi` |
|   728 | `_profiles.py` |
|   697 | `_security.py` |

## Template families

22 families, 306 files. They are **test inputs**, not
documentation — `test__templates_containment.py` and
`test__template_validation.py` treat them as such.

| Family | files |
|---|---:|
| `basic_cython` | 13 |
| `basic_python` | 11 |
| `complex_cython` | 13 |
| `complex_python` | 11 |
| `devel_cython` | 11 |
| `devel_numcpp_cpp_api_cython` | 11 |
| `devel_numcpp_cpp_api_python` | 11 |
| `devel_numpy_c_api_cython` | 11 |
| `devel_numpy_c_api_python` | 11 |
| `devel_python` | 11 |
| `easy_cython` | 17 |
| `easy_python` | 11 |
| `hard_cython` | 13 |
| `hard_python` | 11 |
| `medium_cython` | 15 |
| `medium_python` | 11 |
| `mixed` | 22 |
| `module_cython` | 17 |
| `module_python` | 11 |
| `package_examples` | 27 |
| `probe` | 30 |
| `workflow` | 6 |

The `probe` (30) and `package_examples` (27) families are the largest. `probe`
in particular is worth reading before changing capability detection.

## Tripwires

| Metric | Now | Tripwire |
|---|---|---|
| test : source LOC | 0.95 | < 0.80 |
| `__pycache__` | **present** | any |
| template families without a containment test | 0 (assumed) | ≥ 1 |
| largest module | 2097 | > 2 500 |

## Known physical debt

### 1. `__pycache__/` ships in the source tree

Observation **O-6**, recorded during the Corpus campaign and found again in
`_cli`. Present here too. Byte-compiled files are stale by definition.

### 2. `_builder.py` is 2097 lines

The largest module, and the one that constructs subprocess arguments — a trust
boundary. Not over the tripwire, but the file where size and risk coincide.

### 3. 33 markdown files

Four at the root (`MAINTAINING`, `ADR-0001`, `DEV_NOTES`, `OPERATIONS`) and one
per template family. That is a defensible ratio — unlike `annoy`'s 48, these are
not finished campaign checkpoints. `test__maintainer_docs.py` and
`test__operations_docs.py` already test that the docs match the code, which is
rare and worth preserving.
