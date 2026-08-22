#!/usr/bin/env bash

echo "Show both failing tests"
sed -n '70,115p' \
  scikitplot/corpus/tests/test__catalog.py

sed -n '115,160p' \
  scikitplot/corpus/tests/test__import_hygiene.py

echo "Locate Voyager and all availability probes"
grep -Rni \
  -E "voyager|Voyager|def is_available|capability_snapshot|CapabilityStatus" \
  scikitplot/corpus/_capabilities.py \
  scikitplot/corpus/_catalog.py \
  scikitplot/corpus/_similarity \
  | head -n 180

echo "Show capability-probe implementation"
sed -n '80,210p' \
  scikitplot/corpus/_capabilities.py

sed -n '150,270p' \
  scikitplot/corpus/_catalog.py

echo "Locate the exact Voyager backend"
# After grep gives the line, show roughly ±50 lines. If it is in _similarity/_backends.py, likely:
grep -n \
  -E "class .*Voyager|voyager|Voyager" \
  scikitplot/corpus/_similarity/_backends.py

sed -n '<START>,<END>p' \
  scikitplot/corpus/_similarity/_backends.py

echo "useful direct environment diagnostic"
python - <<'PY'
import importlib.util
import sys

for name in ("voyager", "faiss", "lxml", "spacy", "tensorflow", "transformers"):
    spec = importlib.util.find_spec(name)
    print(
        f"{name:12} "
        f"find_spec={'YES' if spec else 'NO ':3} "
        f"loaded={'YES' if name in sys.modules else 'NO'}"
    )
PY

echo "fix Voyager import hygiene"
# Before editing, check whether _backends.py already has a find_spec helper:
grep -n \
  -E "find_spec|module_available|importlib\\.util" \
  scikitplot/corpus/_similarity/_backends.py

# Re-run only the affected set
# python -m pytest \
#   scikitplot/corpus/tests/test__catalog.py \
#   scikitplot/corpus/tests/test__import_hygiene.py \
#   scikitplot/corpus/_similarity/tests/test__backends.py \
#   -ra --maxfail=0
