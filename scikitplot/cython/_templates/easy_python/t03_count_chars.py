"""Count characters in a string."""

from collections import Counter


def count_chars(s: str) -> dict[str, int]:
    return dict(Counter(s))
