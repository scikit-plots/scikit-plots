#!/usr/bin/env bash

python - <<'PY'
import nltk
from nltk.stem import SnowballStemmer

print("NLTK:", nltk.__version__)
print("languages:")
for language in SnowballStemmer.languages:
    print(" ", language)

for language in [
    "arabic",
    "english",
    "russian",
    "chinese",
    "japanese",
    "korean",
]:
    try:
        stemmer = SnowballStemmer(language)
        print(f"{language:<10} -> SUPPORTED: {type(stemmer).__name__}")
    except Exception as exc:
        print(
            f"{language:<10} -> UNSUPPORTED: "
            f"{type(exc).__name__}: {exc}"
        )
PY

# Official NLTK currently lists:
# arabic
# danish
# dutch
# english
# finnish
# french
# german
# hungarian
# italian
# norwegian
# porter
# portuguese
# romanian
# russian
# spanish
# swedish

# inspect scikit-plots validation
python - <<'PY'
import inspect

from scikitplot.corpus._chunkers._word import WordChunkerConfig

print(inspect.getsource(WordChunkerConfig.__post_init__))
PY

# If the validation is elsewhere, also run:
grep -Rni \
  -E "SNOWBALL|SnowballStemmer|unsupported.*language|snowball.*language" \
  scikitplot/corpus/_chunkers
