#!/usr/bin/env bash

# looking for cross-component regressions.
python -m pytest \
  scikitplot/corpus/_readers/tests/test__xml.py \
  scikitplot/corpus/_readers/tests/test__xml_advanced.py \
  scikitplot/corpus/_normalizers/tests/test__text_normalizer.py \
  scikitplot/corpus/_chunkers/tests/test__custom_tokenizer.py \
  scikitplot/corpus/_chunkers/tests/test__sentence_multilang.py \
  scikitplot/corpus/_chunkers/tests/test__word_multilang.py \
  scikitplot/corpus/_embeddings/tests/test__multimodal_embedding.py \
  scikitplot/corpus/_enrichers/tests/test__nlp_enricher.py \
  scikitplot/corpus/_enrichers/tests/test__nlp_enricher_advanced.py \
  scikitplot/corpus/tests/test__custom_hooks.py \
  -ra --maxfail=0
