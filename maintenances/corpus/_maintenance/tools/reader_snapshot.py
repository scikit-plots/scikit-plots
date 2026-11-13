#!/usr/bin/env python3
"""
Snapshot what the corpus readers make of a folder of files, and compare two snapshots.

A change to a reader can alter the text of files no test covers: line
endings, an encoding, a page order. Unit tests check the cases someone
thought of; this tool checks every file of a real folder. Take a snapshot
with the tree before the change and one with the tree after it, then compare
them: every file whose chunks, error or warnings changed is listed. Each
file is read more than once, so a reader whose output varies between two
reads of the same bytes is reported as unstable.

Usage
-----
::

    PYTHONPATH=<tree before> python reader_snapshot.py snapshot FOLDER --out before.json
    PYTHONPATH=<tree after>  python reader_snapshot.py snapshot FOLDER --out after.json
    python reader_snapshot.py compare before.json after.json

``snapshot`` options: ``--include GLOB`` and ``--exclude GLOB`` (repeatable;
which files to read, matched against the path relative to FOLDER, case
ignored: ``--include '*.pdf'`` reads the PDFs only, ``--exclude '*.jpg'``
skips the slow OCR of photos), ``--repeat N`` (reads per file, default 2),
``--reader-kwargs JSON`` (passed to ``DocumentReader.create``, for example
``'{"ocr": "auto"}'``), ``--documents`` (snapshot the documents after the
reader's filter instead of the raw chunks). ``compare`` exits 1 when the
snapshots differ or a file is unstable, 0 otherwise; ``--ignore-warnings``
leaves warnings out of the comparison, ``--text-only`` compares the texts
and errors only (for a change that adds metadata).

Notes
-----
**What is recorded per file** (path relative to FOLDER): ``result`` (``ok``
or ``error``), the number of chunks and characters, a SHA-256 digest of the
texts and one of every chunk (all keys, so ``page_number`` or ``encoding``
count too), the error as ``Type: message``, the warnings logged under
``scikitplot``, and ``stable``: whether every read gave the same data.
The folder's own path is replaced by ``<root>`` in messages, so two
snapshots of copies of a folder compare equal.

**Developer notes.** :func:`snapshot` takes the reading function as an
argument, so this module imports nothing from ``scikitplot`` until the
command line runs, and its tests need no build. Only the standard library
is used; Python 3.8 and newer.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence

#: Version of the snapshot format; :func:`compare` refuses another.
FORMAT = 1

#: Longest error or warning text kept.
_MAX_MESSAGE = 300

#: Fields that describe what a reader produced; ``stable`` compares these.
DATA_FIELDS = ("result", "chunks", "chars", "text_digest", "digest", "error")

#: Fields compared by ``compare(..., text_only=True)``: the text, not metadata.
TEXT_FIELDS = ("result", "chunks", "chars", "text_digest", "error")

Reader = Callable[[Path], Iterable[Dict[str, Any]]]


class _Collect(logging.Handler):
    """Keep the messages of the records it handles."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception as exc:  # noqa: BLE001
            message = f"<unformattable log record: {type(exc).__name__}>"
        self.messages.append(message)


def _clean(message: str, root: Path) -> str:
    """Return *message* with the folder's path replaced and its length capped."""
    text = message.replace(str(root), "<root>").replace(root.as_posix(), "<root>")
    return text if len(text) <= _MAX_MESSAGE else text[:_MAX_MESSAGE] + "..."


def _digest(chunks: Sequence[dict[str, Any]]) -> str:
    """Return the SHA-256 of *chunks*, every key included, in a stable form."""
    blob = json.dumps(list(chunks), sort_keys=True, default=repr, ensure_ascii=True)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def _text_digest(chunks: Sequence[dict[str, Any]]) -> str:
    """Return the SHA-256 of the chunks' texts in order, nothing else."""
    blob = json.dumps([str(c.get("text", "")) for c in chunks], ensure_ascii=True)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def read_once(path: Path, read: Reader, root: Path, logger: str) -> dict[str, Any]:
    """
    Read one file once and describe the outcome.

    Parameters
    ----------
    path : pathlib.Path
        The file.
    read : callable
        ``read(path)`` returning the chunks as dicts.
    root : pathlib.Path
        The folder, replaced by ``<root>`` in messages.
    logger : str
        Name of the logger whose warnings are kept.

    Returns
    -------
    dict
        ``result``, ``chunks``, ``chars``, ``text_digest`` (the texts),
        ``digest`` (every key of every chunk), ``error``, ``warnings``.
    """
    handler = _Collect()
    log = logging.getLogger(logger)
    log.addHandler(handler)
    try:
        chunks = [dict(chunk) for chunk in read(path)]
        outcome: dict[str, Any] = {
            "result": "ok",
            "chunks": len(chunks),
            "chars": sum(len(str(chunk.get("text", ""))) for chunk in chunks),
            "text_digest": _text_digest(chunks),
            "digest": _digest(chunks),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        # Broad catch: any failure of a reader is an outcome to record.
        outcome = {
            "result": "error",
            "chunks": 0,
            "chars": 0,
            "text_digest": None,
            "digest": None,
            "error": _clean(f"{type(exc).__name__}: {exc}", root),
        }
    finally:
        log.removeHandler(handler)
    outcome["warnings"] = sorted({_clean(m, root) for m in handler.messages})
    return outcome


def selected(name: str, include: Sequence[str], exclude: Sequence[str]) -> bool:
    """
    Whether the file *name* is read, given the glob patterns.

    Parameters
    ----------
    name : str
        Path relative to the folder, with ``/`` separators.
    include : sequence of str
        Patterns of the files to read; empty reads every file.
    exclude : sequence of str
        Patterns of the files to skip; they win over *include*.

    Returns
    -------
    bool
        ``True`` when *name* matches an *include* pattern (or there is
        none) and no *exclude* pattern.

    Notes
    -----
    ``fnmatch`` patterns, matched case-insensitively against the whole
    relative path; ``*`` also matches ``/``, so ``*.pdf`` takes PDFs in
    every subfolder and ``video/*`` everything under ``video``.
    """
    lowered = name.lower()

    def matches(patterns: Sequence[str]) -> bool:
        return any(fnmatch.fnmatchcase(lowered, p.lower()) for p in patterns)

    return (not include or matches(include)) and not matches(exclude)


def snapshot(
    root: Path,
    read: Reader,
    *,
    repeat: int = 2,
    logger: str = "scikitplot",
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> dict[str, Any]:
    """
    Read every file under *root* *repeat* times and describe each.

    Parameters
    ----------
    root : pathlib.Path
        Folder to read, recursively. Files are taken in sorted order of
        their path relative to *root*.
    read : callable
        ``read(path)`` returning the chunks of one file as dicts.
    repeat : int, optional
        Reads per file, at least 1. Default: 2.
    logger : str, optional
        Logger whose warnings are recorded. Default: ``"scikitplot"``.
    include, exclude : sequence of str, optional
        Glob patterns of the files to read and to skip (:func:`selected`).
        Default: every file.

    Returns
    -------
    dict
        ``{"format": 1, "repeat": repeat, "files": {relative path: record}}``,
        plus ``"include"`` and ``"exclude"`` when either is given,
        each record as :func:`read_once` returns it from the first read,
        plus ``stable``: whether every read gave the same data (warnings
        aside: a message may name a temporary path).

    Raises
    ------
    ValueError
        If *root* is not a folder or *repeat* is less than 1.
    TypeError
        If *include* or *exclude* is a string or holds a non-string.
    """
    for label, patterns in (("include", include), ("exclude", exclude)):
        if isinstance(patterns, str) or not all(isinstance(p, str) for p in patterns):
            raise TypeError(
                f"reader_snapshot: {label} must be a sequence of glob strings;"
                f" got {patterns!r}."
            )
    include, exclude = tuple(include), tuple(exclude)
    if not root.is_dir():
        raise ValueError(f"reader_snapshot: {root} is not a folder.")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError(
            f"reader_snapshot: repeat must be an int >= 1; got {repeat!r}."
        )
    root = root.resolve()
    files: dict[str, Any] = {}
    for path in sorted(
        (
            p
            for p in root.rglob("*")
            if p.is_file()
            and selected(p.relative_to(root).as_posix(), include, exclude)
        ),
        key=lambda p: p.relative_to(root).as_posix(),
    ):
        reads = [read_once(path, read, root, logger) for _ in range(repeat)]
        record = dict(reads[0])
        record["stable"] = all(
            all(other[f] == reads[0][f] for f in DATA_FIELDS) for other in reads[1:]
        )
        files[path.relative_to(root).as_posix()] = record
    result: dict[str, Any] = {"format": FORMAT, "repeat": repeat, "files": files}
    if include or exclude:
        result["include"] = list(include)
        result["exclude"] = list(exclude)
    return result


def compare(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    ignore_warnings: bool = False,
    text_only: bool = False,
) -> list[str]:
    """
    Return one line per difference between two snapshots.

    Parameters
    ----------
    before, after : dict
        Snapshots from :func:`snapshot` (or their JSON).
    ignore_warnings : bool, optional
        Leave warnings out. Default: ``False``.
    text_only : bool, optional
        Compare the text and errors only (``TEXT_FIELDS``), not metadata or
        warnings: for a change that adds a metadata key. Default: ``False``.

    Returns
    -------
    list of str
        ``only before: f``, ``only after: f``, ``changed: f: field a -> b``
        and ``unstable: f`` (in *after*) lines, sorted by file, after a
        ``selection:`` line when the two were taken with other
        ``include``/``exclude`` patterns (a file only in one may then be
        a filter, not a reader, difference). Empty when the snapshots agree.

    Raises
    ------
    ValueError
        If either snapshot has another format version.
    """
    for name, snap in (("before", before), ("after", after)):
        if snap.get("format") != FORMAT:
            raise ValueError(
                f"reader_snapshot: the {name} snapshot has format"
                f" {snap.get('format')!r}; this tool reads format {FORMAT}."
            )
    fields = list(TEXT_FIELDS if text_only else DATA_FIELDS)
    if not (ignore_warnings or text_only):
        fields.append("warnings")
    lines: list[str] = [
        f"selection: {key} {before.get(key, [])!r} -> {after.get(key, [])!r}"
        for key in ("include", "exclude")
        if before.get(key, []) != after.get(key, [])
    ]
    old, new = before["files"], after["files"]
    for name in sorted(set(old) | set(new)):
        if name not in new:
            lines.append(f"only before: {name}")
            continue
        if name not in old:
            lines.append(f"only after: {name}")
        else:
            lines.extend(
                f"changed: {name}: {field} {old[name].get(field)!r}"
                f" -> {new[name].get(field)!r}"
                for field in fields
                if old[name].get(field) != new[name].get(field)
            )
        if not new[name].get("stable", True):
            lines.append(f"unstable: {name}: two reads of the same bytes differ")
    return lines


def _scikitplot_reader(reader_kwargs: dict[str, Any], documents: bool) -> Reader:
    """Return a reading function backed by ``scikitplot.corpus`` readers."""
    from scikitplot.corpus._base import DocumentReader  # noqa: PLC0415

    def read(path: Path) -> Iterable[dict[str, Any]]:
        reader = DocumentReader.create(path, **reader_kwargs)
        if not documents:
            return reader.get_raw_chunks()
        return (
            {"text": doc.text, "doc_id": doc.doc_id, "metadata": doc.metadata}
            for doc in reader.get_documents()
        )

    return read


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    take = sub.add_parser("snapshot", help="read every file of a folder")
    take.add_argument("folder", type=Path)
    take.add_argument("--out", type=Path, required=True)
    take.add_argument("--repeat", type=int, default=2)
    take.add_argument("--reader-kwargs", default="{}", help="JSON object")
    take.add_argument("--documents", action="store_true")
    take.add_argument("--include", action="append", default=[], metavar="GLOB")
    take.add_argument("--exclude", action="append", default=[], metavar="GLOB")
    diff = sub.add_parser("compare", help="compare two snapshots")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    diff.add_argument("--ignore-warnings", action="store_true")
    diff.add_argument("--text-only", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "snapshot":
        kwargs = json.loads(args.reader_kwargs)
        if not isinstance(kwargs, dict):
            parser.error("--reader-kwargs must be a JSON object")
        result = snapshot(
            args.folder,
            _scikitplot_reader(kwargs, args.documents),
            repeat=args.repeat,
            include=args.include,
            exclude=args.exclude,
        )
        args.out.write_text(
            json.dumps(result, indent=1, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
        files = result["files"]
        print(  # noqa: T201
            f"{len(files)} file(s): {sum(r['result'] == 'error' for r in files.values())}"
            f" error(s), {sum(not r['stable'] for r in files.values())} unstable;"
            f" written to {args.out}"
        )
        return 0

    lines = compare(
        json.loads(args.before.read_text(encoding="utf-8")),
        json.loads(args.after.read_text(encoding="utf-8")),
        ignore_warnings=args.ignore_warnings,
        text_only=args.text_only,
    )
    for line in lines:
        print(line)  # noqa: T201
    print(f"{len(lines)} difference(s).")  # noqa: T201
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main())
