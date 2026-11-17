# scikitplot/_brand/_kit/_io.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Writing brand assets: atomic, idempotent, confined to one folder.

See Also
--------
scikitplot._brand._eukrina.build : writes a whole asset set with these.

Notes
-----
**User notes:** running a build twice changes nothing the second time: a file
is rewritten only when its bytes differ, and the manifest lists every file
with its SHA-256 so a set can be checked later with :func:`verify_manifest`.

**Developer notes:**

- A file is written to a temporary name in the same folder and moved into
  place with :func:`os.replace`, so a reader never sees half a file.
- Asset names are relative POSIX paths of safe characters; ``..``, absolute
  paths, backslashes and names that resolve outside the output folder are
  refused, symbolic links included.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "MANIFEST_NAME",
    "WriteResult",
    "safe_target",
    "verify_manifest",
    "write_bytes",
    "write_manifest",
]

MANIFEST_NAME = "manifest.json"
_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class WriteResult:
    """
    The outcome of writing one asset.

    Attributes
    ----------
    name : str
        Relative POSIX path inside the output folder.
    sha256 : str
        Hash of the bytes.
    size : int
        Number of bytes.
    changed : bool
        False when the file already had these bytes and was left alone.
    """

    name: str
    sha256: str
    size: int
    changed: bool


def safe_target(root: str | Path, name: str) -> Path:
    """
    Return the path of an asset inside ``root``, refusing anything outside it.

    Parameters
    ----------
    root : str or pathlib.Path
        Output folder.
    name : str
        Relative POSIX path, for example ``"svg/eukrina-symbol-light.svg"``.
        Each part starts with a letter or digit and holds only letters,
        digits, ``.``, ``_`` and ``-``.

    Returns
    -------
    pathlib.Path
        The resolved path.

    Raises
    ------
    ValueError
        If ``name`` is empty, absolute, has a backslash or an unsafe part, or
        resolves outside ``root`` (through a symbolic link, for example).
    """
    if not isinstance(name, str) or not name or name.startswith("/") or "\\" in name:
        raise ValueError(f"asset names are relative POSIX paths, got {name!r}")
    parts = name.split("/")
    for part in parts:
        if not _PART.match(part) or part in (".", ".."):
            raise ValueError(f"unsafe part {part!r} in asset name {name!r}")
    base = Path(root).resolve()
    target = base.joinpath(*parts).resolve()
    if base != target and base not in target.parents:
        raise ValueError(f"asset {name!r} resolves outside the output folder {base}")
    return target


def write_bytes(root: str | Path, name: str, data: bytes) -> WriteResult:
    """
    Write an asset atomically, and only if its bytes changed.

    Parameters
    ----------
    root : str or pathlib.Path
        Output folder; created if missing.
    name : str
        Relative POSIX path (see :func:`safe_target`).
    data : bytes
        Content.

    Returns
    -------
    WriteResult
        Name, hash, size, and whether the file changed.

    Raises
    ------
    TypeError
        If ``data`` is not bytes.
    ValueError
        For an unsafe name.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"data must be bytes, got {type(data).__name__}")
    data = bytes(data)
    target = safe_target(root, name)
    digest = hashlib.sha256(data).hexdigest()
    if (
        target.is_file()
        and target.stat().st_size == len(data)
        and target.read_bytes() == data
    ):
        return WriteResult(name, digest, len(data), False)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return WriteResult(name, digest, len(data), True)


def write_manifest(
    root: str | Path, results: list[WriteResult], meta: Mapping[str, Any]
) -> WriteResult:
    """
    Write ``manifest.json``: the build's settings and every file with its hash.

    Parameters
    ----------
    root : str or pathlib.Path
        Output folder.
    results : list of WriteResult
        The files of the build.
    meta : mapping
        JSON-serialisable description of the build (generator, settings).

    Returns
    -------
    WriteResult
        The manifest's own write result.

    Raises
    ------
    ValueError
        If two results have the same name.

    Notes
    -----
    **Developer notes:** files are sorted by name and keys are sorted, so the
    manifest is identical for identical builds. It holds no time stamp for
    the same reason.
    """
    names = [r.name for r in results]
    if len(names) != len(set(names)):
        raise ValueError("two assets of one build have the same name")
    body: dict[str, Any] = {
        "meta": dict(meta),
        "files": [
            {"name": r.name, "sha256": r.sha256, "bytes": r.size}
            for r in sorted(results, key=lambda r: r.name)
        ],
    }
    text = json.dumps(body, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    return write_bytes(root, MANIFEST_NAME, text.encode("ascii"))


def verify_manifest(root: str | Path) -> list[str]:
    """
    Check an asset folder against its manifest.

    Parameters
    ----------
    root : str or pathlib.Path
        Output folder that holds ``manifest.json``.

    Returns
    -------
    list of str
        One line per problem (missing file, wrong hash); empty when the folder
        matches.

    Raises
    ------
    FileNotFoundError
        If there is no manifest.
    """
    manifest = safe_target(root, MANIFEST_NAME)
    data = json.loads(manifest.read_text(encoding="ascii"))
    problems: list[str] = []
    for entry in data.get("files", []):
        target = safe_target(root, entry["name"])
        if not target.is_file():
            problems.append(f"missing: {entry['name']}")
        elif hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
            problems.append(f"changed: {entry['name']}")
    return problems
