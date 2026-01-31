"""Repeat a string (pure Python)."""


def repeat(s: str, n: int) -> str:
    if n < 0:
        raise ValueError("n must be >= 0")
    return s * n
