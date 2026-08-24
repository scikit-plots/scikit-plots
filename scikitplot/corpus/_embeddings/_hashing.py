# scikitplot/corpus/_embeddings/_hashing.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Deterministic dependency-light hashing embeddings.

This module provides a small local embedding callable for examples, tests,
offline workflows, and environments where downloading a model is undesirable.
It is not intended to replace semantic model embeddings: token hashing preserves
lexical overlap, not learned meaning.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import numpy.typing as npt

__all__ = [
    "HashEmbedder",
]

# Unicode letters, optionally containing an apostrophe. ``[^\\W\\d_]`` means
# "word character that is neither a digit nor underscore" under Unicode rules.
_DEFAULT_TOKEN_PATTERN = r"[^\W\d_]+(?:['’][^\W\d_]+)*"  # ruff: ignore[ambiguous-unicode-character-string, hardcoded-password-string]


@dataclass(frozen=True)
class HashEmbedder:
    """
    Embed text with deterministic signed feature hashing.

    Parameters
    ----------
    dimension : int, default=1024
        Number of output features. Must be positive.
    token_pattern : str, optional
        Regular expression used to extract tokens. The default recognizes
        Unicode words and internal straight/curly apostrophes.
    lowercase : bool, default=True
        Lowercase text before tokenization.

    Notes
    -----
    This embedder is deliberately local and deterministic. Each token is hashed
    with BLAKE2b into one output column and assigned a deterministic sign. Rows
    are L2-normalized. The result is suitable for reproducible lexical/dense
    teaching paths, smoke tests, and offline retrieval baselines.

    It does **not** provide learned semantic similarity and should not be
    presented as a substitute for sentence-transformer or API embeddings.

    Examples
    --------
    >>> embedder = HashEmbedder(dimension=32)
    >>> vectors = embedder(["ghost father", "sleep dream"])
    >>> vectors.shape
    (2, 32)
    >>> vectors.dtype == np.float32
    True
    """

    dimension: int = 1024
    token_pattern: str = field(default=_DEFAULT_TOKEN_PATTERN, repr=False)
    lowercase: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise TypeError("dimension must be an integer")
        if self.dimension <= 0:
            raise ValueError("dimension must be > 0")
        if not isinstance(self.token_pattern, str) or not self.token_pattern:
            raise ValueError("token_pattern must be a non-empty regular expression")
        try:
            pattern = re.compile(self.token_pattern)
        except re.error as exc:
            raise ValueError(f"invalid token_pattern: {exc}") from exc
        if pattern.search("") is not None:
            raise ValueError("token_pattern must not match the empty string")
        if not isinstance(self.lowercase, bool):
            raise TypeError("lowercase must be a bool")

    def __call__(self, texts: Iterable[str]) -> npt.NDArray[np.float32]:
        """
        Return a normalized ``float32`` matrix for ``texts``.

        Parameters
        ----------
        texts : iterable of str
            Input texts. The iterable is consumed exactly once.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(n_texts, dimension)`` and dtype ``float32``.
        """
        text_list = list(texts)
        for index, text in enumerate(text_list):
            if not isinstance(text, str):
                raise TypeError(
                    f"texts[{index}] must be str, got {type(text).__name__}"
                )

        matrix = np.zeros((len(text_list), self.dimension), dtype=np.float32)
        pattern = re.compile(self.token_pattern)

        for row, text in enumerate(text_list):
            for match in pattern.finditer(text):
                token = match.group(0)
                if self.lowercase:
                    token = token.casefold()
                digest = hashlib.blake2b(
                    token.encode("utf-8"),
                    digest_size=8,
                ).digest()
                column = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, column] += sign

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, norms, out=matrix, where=norms != 0)
        return matrix
