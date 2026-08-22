#!/usr/bin/env bash

# CustomChunker
# → strict/wrapped propagation ✅

# CustomNormalizer
# → strict/wrapped propagation ✅

# core NLP custom callbacks
# → propagation ✅

# Custom retrieval scorer
# → wrapped propagation ✅

# CustomFilter
# → callback failure
# → warning + KEEP document
# → real bug being fixed

# CustomNLPEnricher
# → callback failure
# → warning + genuine built-in fallback
# → intentional fail-soft behavior ✅

# PipelineHooks
# → warning + continue
# → intentional lifecycle fail-soft behavior ✅
# → structured ErrorRecord enhancement deferred

# locate every custom callback boundary
echo "locate every custom callback boundary"
grep -Rni \
  -E "custom_pipeline|custom_tokenizer|custom_stemmer|custom_lemmatizer|custom_projection|custom_.*fn|callback|hook" \
  scikitplot/corpus \
  --exclude-dir=tests

echo "Then specifically find broad exception swallowing near those paths:"
grep -Rni \
  -E "except Exception|except BaseException|logger\.(warning|exception)|try:" \
  scikitplot/corpus/_normalizers \
  scikitplot/corpus/_chunkers \
  scikitplot/corpus/_enrichers \
  scikitplot/corpus/_embeddings

echo "inspect the existing error architecture:"
grep -Rni \
  -E "class ErrorRecord|ErrorRecord|ErrorCategory|ErrorPolicy|error_policy|COLLECT|SKIP|RAISE" \
  scikitplot/corpus \
  --exclude-dir=tests

echo "And show the diagnostic definitions:"
sed -n '1,260p' scikitplot/corpus/_diagnostics.py

echo "inspect existing callback failure tests:"
grep -Rni \
  -E "custom.*raises|custom.*exception|callback.*raises|hook.*raises|side_effect=.*Error|boom|custom_pipeline" \
  scikitplot/corpus/*/tests \
  scikitplot/corpus/**/*/tests 2>/dev/null

echo "baseline focused tests:"
python -m pytest \
  scikitplot/corpus/_normalizers/tests \
  scikitplot/corpus/_chunkers/tests \
  scikitplot/corpus/_enrichers/tests \
  scikitplot/corpus/_embeddings/tests \
  -k "custom and (raise or exception or error or pipeline or callback)" \
  -vv --maxfail=0
