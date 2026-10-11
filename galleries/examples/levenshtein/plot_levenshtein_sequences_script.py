"""
Sequences and explicit normalization
====================================

Levenshtein distance is not restricted to prose strings, and preprocessing is
an application choice rather than hidden facade behavior.
"""

import unicodedata

from scikitplot.levenshtein import distance, rank

assert distance(
    ["spam", "egg"],
    ["spam", "ham"],
    backend="python",
) == 1

assert distance(b"abc", b"axc", backend="python") == 1

decomposed = "cafe\u0301"
composed = "café"

raw_distance = distance(decomposed, composed, backend="python")
assert raw_distance > 0

normalize = lambda text: unicodedata.normalize("NFC", text)
assert distance(normalize(decomposed), normalize(composed), backend="python") == 0

records = [
    {"name": "Scikit Plots"},
    {"name": "Scikit Learn"},
]
matches = rank(
    "scikit plots",
    records,
    key=lambda item: item["name"].casefold(),
    backend="python",
)
assert matches[0].choice["name"] == "Scikit Plots"

print("raw Unicode distance:", raw_distance)
print("normalized distance:", 0)
