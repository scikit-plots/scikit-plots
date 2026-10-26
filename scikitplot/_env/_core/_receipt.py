# scikitplot/_env/_core/_receipt.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Receipts: what built an environment, stored inside it.

Notes
-----
**User notes.** After a successful build the builder writes
``<prefix>/.scikitplot-env.json``: the request, the plan digest, the state
digest, the backend, the dependency-file hashes and the step results. The
next ``ensure`` with the same state digest does nothing; ``scikitplot env
run`` uses it to know which backend runs commands. Deleting the receipt
only makes the next ``ensure`` update the environment again.

**Developer notes.** The receipt is written to a temporary file and renamed,
so a reader never sees half of it. A receipt that is not valid JSON or has
another schema is reported and treated as absent, never trusted.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = [
    "ENV_MARKERS",
    "RECEIPT_NAME",
    "RECEIPT_SCHEMA",
    "is_environment",
    "read_receipt",
    "write_receipt",
]

logger = logging.getLogger(__name__)

#: File name of the receipt inside the prefix.
RECEIPT_NAME = ".scikitplot-env.json"

#: Schema identifier of a receipt.
RECEIPT_SCHEMA = "scikitplot.env.receipt/v1"

#: Entries only an environment has; a ``bin/python`` alone (``/usr``) is not one.
ENV_MARKERS = ("pyvenv.cfg", "conda-meta", RECEIPT_NAME)


def is_environment(prefix: str | os.PathLike) -> bool:
    """
    Return whether ``prefix`` is an environment the builder may change or delete.

    Parameters
    ----------
    prefix : path-like
        Directory.

    Returns
    -------
    bool
        True when it holds ``pyvenv.cfg``, ``conda-meta`` or a receipt.
    """
    return any((Path(prefix) / marker).exists() for marker in ENV_MARKERS)


def read_receipt(prefix: str | os.PathLike) -> dict[str, Any] | None:
    """
    Return the receipt of the environment at ``prefix``.

    Parameters
    ----------
    prefix : path-like
        Environment directory.

    Returns
    -------
    dict or None
        The receipt, or None when there is none or it is unusable (logged).
    """
    path = Path(prefix) / RECEIPT_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        logger.warning("ignoring unreadable receipt %s: %s", path, exc)
        return None
    if not isinstance(data, dict) or data.get("schema") != RECEIPT_SCHEMA:
        logger.warning("ignoring receipt %s: schema is not %s", path, RECEIPT_SCHEMA)
        return None
    return data


def write_receipt(prefix: str | os.PathLike, data: dict[str, Any]) -> Path:
    """
    Write ``data`` as the receipt of the environment at ``prefix``.

    Parameters
    ----------
    prefix : path-like
        Environment directory (must exist).
    data : dict
        Receipt content; ``schema`` is set here.

    Returns
    -------
    pathlib.Path
        The receipt file.
    """
    path = Path(prefix) / RECEIPT_NAME
    text = json.dumps(
        {**data, "schema": RECEIPT_SCHEMA}, indent=2, sort_keys=True, default=str
    )
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".receipt-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        os.chmod(tmp, 0o644)  # mkstemp makes 0600; others sharing the env read it
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path
