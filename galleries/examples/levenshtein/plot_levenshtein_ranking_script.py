"""
Rank fuzzy matches
==================

Use stable normalized-similarity ranking with a semantic cutoff.
"""

from scikitplot.levenshtein import closest, rank

choices = ["kitten", "bitten", "sitting", "written", "completely different"]

matches = rank(
    "kitten",
    choices,
    backend="python",
    limit=4,
    score_cutoff=0.5,
)

assert [item.choice for item in matches] == [
    "kitten",
    "bitten",
    "written",
    "sitting",
]
assert all(item.backend == "python" for item in matches)

for item in matches:
    print(
        f"{item.choice:22s} "
        f"distance={item.distance} "
        f"similarity={item.similarity:.3f}"
    )

best = closest("levnshtein", ["Levenshtein", "Hamming", "Jaccard"], backend="python")
assert best is not None
print("closest:", best.choice)
