# Verification

## Focused runtime

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -c /dev/null \
  scikitplot/levenshtein/tests/test_levenshtein.py \
  -q -p no:cacheprovider --confcutdir=scikitplot/levenshtein
```

## Maintenance

```sh
python -B maintenances/levenshtein/_maintenance/tools/check_contract.py --json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -c /dev/null \
  maintenances/levenshtein/_maintenance/tests \
  -q -p no:cacheprovider --confcutdir=maintenances/levenshtein
```

## User guide

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -c /dev/null \
  docs/source/user_guide/levenshtein \
  -q -p no:cacheprovider --confcutdir=docs/source/user_guide/levenshtein
```

## Gallery

```sh
python galleries/examples/levenshtein/plot_levenshtein_basics_script.py
python galleries/examples/levenshtein/plot_levenshtein_backends_script.py
python galleries/examples/levenshtein/plot_levenshtein_ranking_script.py
python galleries/examples/levenshtein/plot_levenshtein_sequences_script.py
python galleries/examples/levenshtein/plot_levenshtein_corpus_script.py
```

The gallery lane is intentionally local and deterministic: no network,
downloads or external service credentials.

## Import isolation

Use a subprocess and assert that importing the facade does not newly import
`rapidfuzz` or `Levenshtein`.

## Release lane

Repository-wide tests, Sphinx/Sphinx-Gallery, wheel/build and supported Python
matrix belong to release verification. If unavailable, record them as such.
