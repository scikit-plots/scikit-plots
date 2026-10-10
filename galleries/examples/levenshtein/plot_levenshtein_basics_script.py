"""
Levenshtein basics
==================

Compute raw and normalized unit-cost edit distance.
"""

from scikitplot.levenshtein import (
    distance,
    normalized_distance,
    normalized_similarity,
    similarity,
)

left = "kitten"
right = "sitting"

d = distance(left, right, backend="python")
sim = similarity(left, right, backend="python")
norm_d = normalized_distance(left, right, backend="python")
norm_sim = normalized_similarity(left, right, backend="python")

assert d == 3
assert sim == 4
assert abs(norm_d - 3 / 7) < 1e-12
assert abs(norm_sim - 4 / 7) < 1e-12

print(f"{left!r} -> {right!r}")
print("distance:", d)
print("similarity:", sim)
print("normalized distance:", round(norm_d, 3))
print("normalized similarity:", round(norm_sim, 3))
