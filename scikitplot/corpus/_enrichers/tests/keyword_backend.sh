#!/usr/bin/env bash

# inspect current implementation first
grep -Rni \
  -E "YAKE|KeyBERT|keybert|yake|frequency keywords|falling back|fallback" \
  scikitplot/corpus/_enrichers \
  | head -n 160

# Then locate the keyword extraction functions/classes and show the relevant source.
grep -n \
  -E "def .*keyword|def .*yake|def .*keybert|Keyword|YAKE|KeyBERT" \
  scikitplot/corpus/_enrichers/_nlp*.py

# Then print the relevant region, for example:
# Adjust filename/range according to the grep result.
sed -n '1,260p' \
  scikitplot/corpus/_enrichers/_nlp_enricher.py

# Also inspect the existing fallback tests:
grep -Rni \
  -E "yake|keybert|frequency|fallback|missing.*dependency|ImportError" \
  scikitplot/corpus/_enrichers/tests

# Then run only the NLP enricher tests before editing:
python -m pytest \
  scikitplot/corpus/_enrichers/tests/test__nlp_enricher.py \
  scikitplot/corpus/_enrichers/tests/test__nlp_enricher_advanced.py \
  -vv --maxfail=0

# What we want to determine
# A. missing YAKE/KeyBERT
#    → WARNING
#    → genuinely run frequency extractor
# B. missing YAKE/KeyBERT
#    → WARNING
#    → return None / no keywords
# C. missing YAKE/KeyBERT
#    → raise
