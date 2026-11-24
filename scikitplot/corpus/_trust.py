"""
Trust gate for loading formats whose deserialisation runs code.

Loading a pickle or joblib file executes whatever the file contains, so such
a load must be asked for explicitly (``trusted=True``) and, when the caller
knows the file's SHA-256, the bytes are checked before anything reads them.

Notes
-----
**User notes.** A refusal says which function to call with ``trusted=True``.
Pass it only for files you made or whose source you trust; pass
``expected_sha256`` as well to refuse a file that changed since.

**Developer notes.** This module is a duplicated contract: a byte-identical
copy lives in ``scikitplot/corpus`` and ``scikitplot/_decorates``, because no
submodule imports another at module level (see
``scikitplot/tests/test_submodule_independence.py``, which compares the
copies by digest). Change every copy at once. Standard library only.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

#: Formats whose loading executes code from the file: pickle and every
#: format built on it (joblib, cloudpickle, dill, pandas ``read_pickle``).
#: Names are compared case-insensitively.
CODE_EXECUTING_FORMATS = frozenset(
    {"pickle", "joblib", "cloudpickle", "dill", "pandas-pickle"}
)

_CHUNK = 1024 * 1024  # 1 MiB
_SHA256 = re.compile(r"[0-9a-f]{64}")


def require_trusted(fmt: str, trusted: bool, *, caller: str) -> None:
    """
    Refuse to load a code-executing format unless ``trusted`` is ``True``.

    Parameters
    ----------
    fmt : str
        Format name (``"pickle"``, ``"joblib"``, ``"json"``, ...).
    trusted : bool
        The caller's explicit opt-in. Only the value ``True`` opts in; a
        truthy string such as ``"false"`` does not.
    caller : str
        Public function the user called, named in the error.

    Raises
    ------
    ValueError
        If *fmt* is in :data:`CODE_EXECUTING_FORMATS` and *trusted* is not
        ``True``.
    """
    if str(fmt).lower() in CODE_EXECUTING_FORMATS and trusted is not True:
        raise ValueError(
            f"{caller}: loading {fmt} files is disabled by default: a {fmt} "
            "file can run arbitrary Python code when it is loaded. Pass "
            f"trusted=True to {caller}() only for a file whose source you "
            "trust, and expected_sha256= to refuse a changed file."
        )


def verify_sha256(
    path: str | Path, expected_sha256: str | None, *, caller: str
) -> None:
    """
    Check a file's SHA-256 before anything deserialises it.

    Parameters
    ----------
    path : str or pathlib.Path
        File to hash, read in 1 MiB blocks (constant memory).
    expected_sha256 : str or None
        Expected hex digest, case and surrounding whitespace ignored.
        ``None`` skips the check.
    caller : str
        Public function the user called, named in the error.

    Raises
    ------
    ValueError
        If *expected_sha256* is not 64 hex digits, or the file's digest
        differs.
    """
    if expected_sha256 is None:
        return
    expected = str(expected_sha256).strip().lower()
    if not _SHA256.fullmatch(expected):
        raise ValueError(
            f"{caller}: expected_sha256 must be 64 hexadecimal digits, "
            f"got {expected_sha256!r}."
        )
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"{caller}: {str(path)!r} SHA-256 {actual} does not match the "
            f"expected {expected}. Refusing to load a tampered or wrong file."
        )
