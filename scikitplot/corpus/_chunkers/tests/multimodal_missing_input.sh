#!/usr/bin/env bash

# inspect current ownership first
grep -Rni \
  -E "raw_tensor|zeros|zero.*tensor|modality|multimodal|image|audio|video" \
  scikitplot/corpus/_embeddings \
  | head -n 120

# Then identify the main multimodal implementation file and run its relevant section. Most likely:
grep -n \
  -E "def .*embed|raw_tensor|zeros|modality" \
  scikitplot/corpus/_embeddings/_multimodal*.py

# Also show the tests currently encoding the fallback:
grep -Rni \
  -E "missing_raw_tensor|unknown_modality|zeros|fallback" \
  scikitplot/corpus/_embeddings/tests

# And run the current multimodal test file once before changing anything:
python -m pytest \
  scikitplot/corpus/_embeddings/tests/test__multimodal_embedding.py \
  -vv --maxfail=0

# sed -n '835,975p' \
#   scikitplot/corpus/_embeddings/_multimodal_embedding.py
# sed -n '1225,1335p' \
#   scikitplot/corpus/_embeddings/_multimodal_embedding.py
# sed -n '540,610p' \
#   scikitplot/corpus/_embeddings/tests/test__multimodal_embedding.py
grep -n \
  -E "replace\\(|with_updates|embedding=|copy_with|model_copy|dataclasses.replace" \
  scikitplot/corpus/_embeddings/_multimodal_embedding.py \
  | head -n 60
