# scikitplot/corpus/_notices.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Once-per-process notices (R3-7).

A missing optional segmenter (jieba, MeCab, PyThaiNLP, ...) is a fact about
the environment, not about the text being processed. Logging it on every
call buried real warnings: constructing one ``SemanticChunker`` and chunking
one text logged the Japanese-strategy warning several times. A notice is
logged once per process per key; :func:`~scikitplot.corpus.component_capabilities`
reports every optional segmenter on demand.

Notes
-----
**Developer.** Keys are stable strings (``"segmenter:jieba"``), so two code
paths that report the same missing dependency share one notice.
:func:`reset_notices` exists for tests.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

__all__: list[str] = []

#: Where the full picture is, appended to every segmenter notice.
CAPABILITIES_HINT = (
    "scikitplot.corpus.component_capabilities(role='segmentation') lists every"
    " optional segmenter."
)

_seen: set[str] = set()
_lock = threading.Lock()


def notice_once(
    logger: logging.Logger,
    key: str,
    message: str,
    *args: Any,
    level: int = logging.WARNING,
) -> bool:
    """Log ``message`` the first time ``key`` is seen in this process.

    Parameters
    ----------
    logger : logging.Logger
        The logger to use.
    key : str
        Stable identity of the notice.
    message : str
        A ``%``-style format string.
    *args : Any
        Its arguments.
    level : int, default=logging.WARNING
        Log level.

    Returns
    -------
    bool
        ``True`` when the notice was logged now; ``False`` when it already
        was, or when ``logger`` does not log ``level`` now (the notice is
        then not used up, so it appears once logging allows it).
    """
    if not logger.isEnabledFor(level):
        return False
    with _lock:
        if key in _seen:
            return False
        _seen.add(key)
    logger.log(level, message, *args)
    return True


def reset_notices() -> None:
    """Forget every notice, so each is logged again (for tests)."""
    with _lock:
        _seen.clear()
