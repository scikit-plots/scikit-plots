#!/usr/bin/env bash

sed -n '330,390p' \
  scikitplot/corpus/_custom_hooks.py

sed -n '775,855p' \
  scikitplot/corpus/_custom_hooks.py

grep -Rni \
  -E "CustomFilter|CustomNLPEnricher|post_filter_hook.*rais|hook.*rais" \
  scikitplot/corpus/tests/test__custom_hooks.py \
  scikitplot/corpus/*/tests 2>/dev/null

# And, after changing CustomFilter, run:
python -m pytest \
  scikitplot/corpus/tests/test__custom_hooks.py \
  -k "filter" \
  -vv --maxfail=0
