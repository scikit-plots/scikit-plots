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
    python reader_snapshot.py show FILE             # the chunks of one file

``snapshot`` options: ``--include GLOB`` and ``--exclude GLOB`` (repeatable;
which files to read, matched against the path relative to FOLDER, case
ignored: ``--include '*.pdf'`` reads the PDFs only, ``--exclude '*.jpg'``
skips the slow OCR of photos), ``--max-per-type N`` (at most N files of each
suffix, the first in sorted order: a fixed sample of a large media folder),
``--repeat N`` (reads per file, default 2), ``--timeout SECONDS`` (read each
file in its own child process and give up on it after SECONDS: one slow or
crashing file cannot stall the run), ``--reader-kwargs JSON`` (passed to
``DocumentReader.create``, for example ``'{"ocr": "auto"}'``), ``--reader
FILE.py:FUNCTION`` (snapshot another reading function, ``FUNCTION(path,
**reader_kwargs)`` returning chunk dicts, instead of the corpus readers),
``--documents`` (snapshot the documents after the reader's filter instead of
the raw chunks), ``--lines`` (for readers whose chunks carry ``line_start``
and ``line_end``, such as the code and Lean readers, record whether the
chunks account for every line: see :func:`line_coverage`). ``compare`` exits 1 when the snapshots differ or a file is
unstable, 0 otherwise; ``--ignore-warnings`` leaves warnings out of the
comparison, ``--text-only`` compares the texts and errors only (for a change
that adds metadata), ``--timing`` adds ``timing:`` and ``slower:`` lines
(total time, peak memory, files at least twice and one second slower) that
never change the exit status. ``show`` prints the chunks of one file (text,
or ``--json`` for every key) with the same reader options as ``snapshot``.

Notes
-----
**What is recorded per file** (path relative to FOLDER): ``result`` (``ok``
or ``error``), the number of chunks and characters, a SHA-256 digest of the
texts and one of every chunk (all keys, so ``page_number`` or ``encoding``
count too), the error as ``Type: message``, the warnings logged under
``scikitplot``, and ``stable``: whether every read gave the same data.
The folder's own path is replaced by ``<root>`` in messages, so two
snapshots of copies of a folder compare equal. ``seconds`` (the fastest of
the reads of the file) and, written by the command line, ``run`` (total
seconds, peak resident memory, Python and platform) describe cost; they are
never compared as differences, only reported by ``compare --timing``.
``lines`` (with ``--lines``; ``None`` otherwise): non-blank lines no chunk
covers, chunks that overlap the one before, and chunks whose text is not
the file's own lines; compared like the other fields.

**Isolated reads.** With ``--timeout`` each read runs in a child process
(``read-one``), which sends back the chunks and the warnings; the record is
the same as an in-process read, and the error keeps the reader's exception
name. A read that passes the time limit is recorded as
``TimeoutError: no result within N s`` (on POSIX the child's whole process
group is ended, OCR engines included); one that ends the child without a
result as ``ChildProcessError``. Two reads in two processes also differ when
a reader depends on the interpreter's hash seed (set order), which one
process cannot show. Each isolated read pays for a new interpreter and its
imports (about 2 to 3 s with ``scikitplot``), so use it for media and large
or untrusted folders, not for a folder of small text files.

**Developer notes.** :func:`snapshot` takes the reading function as an
argument, so this module imports nothing from ``scikitplot`` until the
command line runs, and its tests need no build. Only the standard library
is used; Python 3.8 and newer.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import logging
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence

#: Version of the snapshot format; :func:`compare` refuses another.
FORMAT = 1

#: Longest error or warning text kept.
_MAX_MESSAGE = 300

#: ``compare --timing`` lists a file as slower when its time grew by at
#: least this factor and by at least ``_SLOWER_SECONDS``.
_SLOWER_FACTOR = 2.0
_SLOWER_SECONDS = 1.0

#: First token of the line that carries a child's result (``read-one``):
#: anything else a library prints on standard output is ignored.
_RESULT_MARK = "@@reader_snapshot@@"

#: Fields that describe what a reader produced; ``stable`` compares these.
DATA_FIELDS = ("result", "chunks", "chars", "text_digest", "digest", "error", "lines")

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


def line_coverage(path: Path, chunks: Sequence[dict[str, Any]]) -> dict[str, int] | None:
    """
    Check that chunks with line ranges account for every line of *path*.

    Parameters
    ----------
    path : pathlib.Path
        The file the chunks were read from.
    chunks : sequence of dict
        The chunks, in the order the reader yielded them.

    Returns
    -------
    dict or None
        ``None`` when no chunk carries ``line_start`` and ``line_end``.
        Otherwise counts: ``nonblank`` (non-blank lines of the file),
        ``uncovered`` (non-blank lines in no chunk's range), ``overlaps``
        (chunks that start at or before the end of an earlier one) and
        ``inexact`` (chunks whose text is not the file's lines over their
        range, or whose range runs past the file).

    Notes
    -----
    The file is decoded with the first chunk's ``encoding`` (UTF-8 when
    no chunk names one; undecodable bytes replaced), with universal
    newlines and a leading byte order mark dropped, as the code readers
    read it: line numbers are an editor's. A reader that leaves lines out
    on purpose (``LeanReader(proofs="drop")``) reports them as
    ``uncovered``; that is a count to explain, not an error.
    """
    ranges = [
        (chunk["line_start"], chunk["line_end"], chunk.get("text"))
        for chunk in chunks
        if isinstance(chunk.get("line_start"), int)
        and isinstance(chunk.get("line_end"), int)
    ]
    if not ranges:
        return None
    encoding = next((c["encoding"] for c in chunks if c.get("encoding")), "utf-8")
    with open(path, encoding=encoding, errors="replace", newline=None) as stream:
        lines = [line[:-1] if line.endswith("\n") else line for line in stream]
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0][1:]
    covered = bytearray(len(lines) + 2)
    overlaps = inexact = 0
    furthest = 0
    for start, end, text in ranges:
        if start <= furthest:
            overlaps += 1
        furthest = max(furthest, end)
        if start < 1 or end > len(lines) or end < start:
            inexact += 1
            continue
        covered[start : end + 1] = b"\x01" * (end - start + 1)
        if text is not None and text != "\n".join(lines[start - 1 : end]):
            inexact += 1
    nonblank = [number for number, line in enumerate(lines, 1) if line.strip()]
    return {
        "nonblank": len(nonblank),
        "uncovered": sum(1 for number in nonblank if not covered[number]),
        "overlaps": overlaps,
        "inexact": inexact,
    }


def read_once(
    path: Path, read: Reader, root: Path, logger: str, *, lines: bool = False
) -> dict[str, Any]:
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
    lines : bool, optional
        Also check the chunks' line ranges (:func:`line_coverage`).
        Default: ``False``.

    Returns
    -------
    dict
        ``result``, ``chunks``, ``chars``, ``text_digest`` (the texts),
        ``digest`` (every key of every chunk), ``error``, ``lines`` (``None``
        unless *lines*), ``warnings``.
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
            "lines": line_coverage(path, chunks) if lines else None,
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
            "lines": None,
        }
    finally:
        log.removeHandler(handler)
    outcome["warnings"] = sorted({_clean(m, root) for m in handler.messages})
    return outcome


def peak_rss_mib() -> float | None:
    """
    Return the peak resident memory of this process and its waited children.

    Returns
    -------
    float or None
        Mebibytes, the larger of this process's peak and the largest peak
        of a child it has waited for (an isolated read); ``None`` where the
        standard library cannot measure it (Windows).

    Notes
    -----
    ``ru_maxrss`` is in kibibytes on Linux and in bytes on macOS.
    """
    try:
        import resource  # noqa: PLC0415 - POSIX only
    except ImportError:
        return None
    peak = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(peak / divisor, 1)


def _exception(name: str, message: str) -> Exception:
    """Return an exception whose class name is *name*: a child's error, kept."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,99}", name or ""):
        name = "ReaderError"
    return type(name, (Exception,), {})(message)


def isolated(
    argv: Sequence[str], *, timeout: float, logger: str = "scikitplot"
) -> Reader:
    """
    Return a reading function that reads each file in a child process.

    Parameters
    ----------
    argv : sequence of str
        The child's command line without the file, for example
        ``[sys.executable, "-B", "reader_snapshot.py", "read-one"]``; the
        file's path is appended. The child prints one line
        ``_RESULT_MARK {"chunks": [...], "warnings": [...]}`` or
        ``_RESULT_MARK {"error": [type, message], "warnings": [...]}``.
    timeout : float
        Seconds a read may take, greater than 0.
    logger : str, optional
        Logger the child's warnings are logged on again, so the caller's
        handlers see them as if the read were in process. Default:
        ``"scikitplot"``.

    Returns
    -------
    callable
        ``read(path)`` returning the chunks; it raises the child's error
        (an exception with the child's class name), ``TimeoutError`` after
        *timeout*, or ``ChildProcessError`` when the child ends without a
        result.

    Raises
    ------
    ValueError
        If *timeout* is not a number greater than 0 or *argv* is empty.
    """
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError(f"reader_snapshot: timeout must be a number; got {timeout!r}.")
    if not timeout > 0:
        raise ValueError(f"reader_snapshot: timeout must be > 0; got {timeout!r}.")
    command = [str(a) for a in argv]
    if not command:
        raise ValueError("reader_snapshot: argv must name the child command.")
    log = logging.getLogger(logger)

    def read(path: Path) -> list[dict[str, Any]]:
        posix = os.name == "posix"
        proc = subprocess.Popen(  # noqa: S603 - the caller's own command line
            [*command, str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=posix,
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if posix:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
            else:
                proc.kill()
            proc.communicate()
            raise TimeoutError(f"no result within {timeout:g} s") from None
        lines = [
            line[len(_RESULT_MARK) :]
            for line in out.decode("utf-8", errors="replace").splitlines()
            if line.startswith(_RESULT_MARK)
        ]
        if not lines:
            raise ChildProcessError(
                f"reader process ended with status {proc.returncode} and no result"
            )
        payload = json.loads(lines[-1])
        for message in payload.get("warnings", []):
            log.warning("%s", message)
        if "error" in payload:
            name, message = payload["error"]
            raise _exception(name, message)
        return payload["chunks"]

    return read


def check_reader_spec(spec: str, *, documents: bool = False) -> tuple[Path, str]:
    """
    Check a ``FILE.py:FUNCTION`` reader without running it.

    Parameters
    ----------
    spec : str
        The ``--reader`` value other than ``"scikitplot"``.
    documents : bool, optional
        Whether ``--documents`` was given (only the corpus readers have
        documents). Default: ``False``.

    Returns
    -------
    tuple of (pathlib.Path, str)
        The file and the function name.

    Raises
    ------
    ValueError
        If *spec* is not ``FILE.py:FUNCTION``, FILE does not exist, or
        *documents* is set.
    """
    file_name, sep, function = spec.rpartition(":")
    if not sep or not file_name.endswith(".py") or not function.isidentifier():
        raise ValueError(
            f"reader_snapshot: --reader must be 'scikitplot' or"
            f" 'FILE.py:FUNCTION'; got {spec!r}."
        )
    if documents:
        raise ValueError("reader_snapshot: --documents needs the corpus readers.")
    path = Path(file_name)
    if not path.is_file():
        raise ValueError(f"reader_snapshot: reader file {path} does not exist.")
    return path, function


def make_reader(
    spec: str = "scikitplot",
    reader_kwargs: dict[str, Any] | None = None,
    *,
    documents: bool = False,
) -> Reader:
    """
    Return the reading function a command line names.

    Parameters
    ----------
    spec : str, optional
        ``"scikitplot"`` (the corpus readers, ``DocumentReader.create``) or
        ``"FILE.py:FUNCTION"``: a function ``FUNCTION(path, **reader_kwargs)``
        returning chunk dicts, loaded from FILE by its path (the file runs:
        name only your own code). Default: ``"scikitplot"``.
    reader_kwargs : dict or None, optional
        Keyword arguments for the reader. Default: none.
    documents : bool, optional
        With the corpus readers, return the documents after the reader's
        filter (``text``, ``doc_id``, ``metadata``) instead of the raw
        chunks. Default: ``False``.

    Returns
    -------
    callable
        ``read(path)`` returning an iterable of chunk dicts.

    Raises
    ------
    ValueError
        If *spec* is neither form, FILE does not exist or has no FUNCTION,
        or *documents* is used with another reader.
    """
    kwargs = dict(reader_kwargs or {})
    if spec == "scikitplot":
        return _scikitplot_reader(kwargs, documents)
    path, function = check_reader_spec(spec, documents=documents)
    module_spec = importlib.util.spec_from_file_location(
        f"_reader_snapshot_{path.stem}", path
    )
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"reader_snapshot: cannot load {path}.")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    func = getattr(module, function, None)
    if not callable(func):
        raise ValueError(f"reader_snapshot: {path} has no function {function!r}.")

    def read(file: Path) -> Iterable[dict[str, Any]]:
        return func(file, **kwargs)

    return read


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
    max_per_type: int | None = None,
    lines: bool = False,
) -> dict[str, Any]:
    """
    Read every file under *root* *repeat* times and describe each.

    Parameters
    ----------
    root : pathlib.Path
        Folder to read, recursively. Files are taken in sorted order of
        their path relative to *root*.
    read : callable
        ``read(path)`` returning the chunks of one file as dicts (in this
        process, or in a child: :func:`isolated`).
    repeat : int, optional
        Reads per file, at least 1. Default: 2.
    logger : str, optional
        Logger whose warnings are recorded. Default: ``"scikitplot"``.
    include, exclude : sequence of str, optional
        Glob patterns of the files to read and to skip (:func:`selected`).
        Default: every file.
    max_per_type : int or None, optional
        Read at most this many files of each suffix (case ignored; a file
        without one counts as its own type), the first in sorted order, after
        *include* and *exclude*: the same sample on every run. ``None``
        reads all. Default: ``None``.
    lines : bool, optional
        Record :func:`line_coverage` for each file. Default: ``False``.

    Returns
    -------
    dict
        ``{"format": 1, "repeat": repeat, "files": {relative path: record}}``,
        plus ``"include"`` and ``"exclude"`` when either is given and
        ``"max_per_type"`` when given, each record as :func:`read_once`
        returns it from the first read, plus ``stable``: whether every read
        gave the same data (warnings aside: a message may name a temporary
        path), and ``seconds``: the fastest read (the first one also pays
        for imports).

    Raises
    ------
    ValueError
        If *root* is not a folder, *repeat* is less than 1 or
        *max_per_type* is not ``None`` or an int of at least 1.
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
    if max_per_type is not None and (
        isinstance(max_per_type, bool)
        or not isinstance(max_per_type, int)
        or max_per_type < 1
    ):
        raise ValueError(
            "reader_snapshot: max_per_type must be None or an int >= 1;"
            f" got {max_per_type!r}."
        )
    root = root.resolve()
    files: dict[str, Any] = {}
    taken: dict[str, int] = {}
    for path in sorted(
        (
            p
            for p in root.rglob("*")
            if p.is_file()
            and selected(p.relative_to(root).as_posix(), include, exclude)
        ),
        key=lambda p: p.relative_to(root).as_posix(),
    ):
        kind = path.suffix.lower()
        if max_per_type is not None:
            if taken.get(kind, 0) >= max_per_type:
                continue
            taken[kind] = taken.get(kind, 0) + 1
        reads, seconds = [], []
        for _ in range(repeat):
            start = time.perf_counter()
            reads.append(read_once(path, read, root, logger, lines=lines))
            seconds.append(time.perf_counter() - start)
        record = dict(reads[0])
        record["stable"] = all(
            all(other[f] == reads[0][f] for f in DATA_FIELDS) for other in reads[1:]
        )
        record["seconds"] = round(min(seconds), 4)
        files[path.relative_to(root).as_posix()] = record
    result: dict[str, Any] = {"format": FORMAT, "repeat": repeat, "files": files}
    if include or exclude:
        result["include"] = list(include)
        result["exclude"] = list(exclude)
    if max_per_type is not None:
        result["max_per_type"] = max_per_type
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
        f"selection: {key} {before.get(key, default)!r} -> {after.get(key, default)!r}"
        for key, default in (("include", []), ("exclude", []), ("max_per_type", None))
        if before.get(key, default) != after.get(key, default)
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


def timing_lines(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """
    Return the cost of *after* against *before*, for information.

    Parameters
    ----------
    before, after : dict
        Snapshots; ``run`` (written by the command line) and each file's
        ``seconds`` are used when present.

    Returns
    -------
    list of str
        ``timing: seconds a -> b (xF)`` and ``timing: peak_rss_mib a -> b``
        when both snapshots have them, then ``slower: f: a s -> b s (xF)``
        for each file read by both whose time grew by at least
        ``_SLOWER_FACTOR`` and ``_SLOWER_SECONDS``, sorted by file. Times
        vary between runs and machines: these lines are not differences.
    """
    lines: list[str] = []
    old_run, new_run = before.get("run") or {}, after.get("run") or {}
    for key in ("seconds", "peak_rss_mib"):
        a, b = old_run.get(key), new_run.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            factor = f" (x{b / a:.2f})" if a > 0 else ""
            lines.append(f"timing: {key} {a:g} -> {b:g}{factor}")
    old, new = before.get("files", {}), after.get("files", {})
    for name in sorted(set(old) & set(new)):
        a, b = old[name].get("seconds"), new[name].get("seconds")
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            continue
        if b - a >= _SLOWER_SECONDS and b >= _SLOWER_FACTOR * a:
            factor = f" (x{b / a:.1f})" if a > 0 else ""
            lines.append(f"slower: {name}: {a:g} s -> {b:g} s{factor}")
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


def _child_payload(path: Path, read: Reader, logger: str) -> dict[str, Any]:
    """Read *path* once in this process; return the child's result payload."""
    handler = _Collect()
    log = logging.getLogger(logger)
    log.addHandler(handler)
    try:
        payload: dict[str, Any] = {"chunks": [dict(c) for c in read(path)]}
    except Exception as exc:  # noqa: BLE001 - sent to the parent as data
        payload = {"error": [type(exc).__name__, str(exc)]}
    finally:
        log.removeHandler(handler)
    payload["warnings"] = handler.messages
    return payload


def _reader_options(sub: argparse.ArgumentParser) -> None:
    """Add the options that choose and configure the reader."""
    sub.add_argument("--reader", default="scikitplot", metavar="SPEC")
    sub.add_argument("--reader-kwargs", default="{}", help="JSON object")
    sub.add_argument("--documents", action="store_true")
    sub.add_argument("--logger", default="scikitplot", help=argparse.SUPPRESS)


def _reader_kwargs(parser: argparse.ArgumentParser, text: str) -> dict[str, Any]:
    """Parse ``--reader-kwargs``; exit with a usage error unless an object."""
    try:
        kwargs = json.loads(text)
    except ValueError as exc:
        parser.error(f"--reader-kwargs is not JSON: {exc}")
    if not isinstance(kwargs, dict):
        parser.error("--reader-kwargs must be a JSON object")
    return kwargs


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0911, PLR0915
    """Run the command line; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    take = sub.add_parser("snapshot", help="read every file of a folder")
    take.add_argument("folder", type=Path)
    take.add_argument("--out", type=Path, required=True)
    take.add_argument("--repeat", type=int, default=2)
    take.add_argument("--include", action="append", default=[], metavar="GLOB")
    take.add_argument("--exclude", action="append", default=[], metavar="GLOB")
    take.add_argument("--max-per-type", type=int, default=None, metavar="N")
    take.add_argument("--timeout", type=float, default=None, metavar="SECONDS")
    take.add_argument("--lines", action="store_true")
    _reader_options(take)
    diff = sub.add_parser("compare", help="compare two snapshots")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    diff.add_argument("--ignore-warnings", action="store_true")
    diff.add_argument("--text-only", action="store_true")
    diff.add_argument("--timing", action="store_true")
    show = sub.add_parser("show", help="print the chunks of one file")
    show.add_argument("file", type=Path)
    show.add_argument("--json", action="store_true")
    _reader_options(show)
    one = sub.add_parser("read-one", help=argparse.SUPPRESS)
    one.add_argument("file", type=Path)
    _reader_options(one)
    args = parser.parse_args(argv)

    if args.command in ("snapshot", "show", "read-one"):
        kwargs = _reader_kwargs(parser, args.reader_kwargs)
        try:
            if args.command == "snapshot" and args.timeout is not None:
                # Only the children read: check the reader, do not load it here.
                if args.reader != "scikitplot":
                    check_reader_spec(args.reader, documents=args.documents)
            else:
                read = make_reader(args.reader, kwargs, documents=args.documents)
        except ValueError as exc:
            parser.error(str(exc))

    if args.command == "read-one":
        payload = _child_payload(args.file, read, args.logger)
        print(  # noqa: T201
            _RESULT_MARK
            + json.dumps(payload, default=repr, ensure_ascii=True, sort_keys=True)
        )
        return 0

    if args.command == "show":
        for number, chunk in enumerate(read(args.file)):
            if args.json:
                print(json.dumps(dict(chunk), default=repr, ensure_ascii=False))  # noqa: T201
            else:
                print(f"--- chunk {number} ---\n{dict(chunk).get('text', '')}")  # noqa: T201
        return 0

    if args.command == "snapshot":
        if args.timeout is not None:
            child = [sys.executable, "-B", str(Path(__file__).resolve()), "read-one"]
            child += ["--reader", args.reader, "--reader-kwargs", args.reader_kwargs]
            child += ["--logger", args.logger] + (["--documents"] if args.documents else [])
            try:
                read = isolated(child, timeout=args.timeout, logger=args.logger)
            except ValueError as exc:
                parser.error(str(exc))
        start = time.perf_counter()
        try:
            result = snapshot(
                args.folder,
                read,
                repeat=args.repeat,
                logger=args.logger,
                include=args.include,
                exclude=args.exclude,
                max_per_type=args.max_per_type,
                lines=args.lines,
            )
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        result["run"] = {
            "seconds": round(time.perf_counter() - start, 2),
            "peak_rss_mib": peak_rss_mib(),
            "python": platform.python_version(),
            "platform": sys.platform,
            "timeout": args.timeout,
        }
        args.out.write_text(
            json.dumps(result, indent=1, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
        files = result["files"]
        run = result["run"]
        peak = "n/a" if run["peak_rss_mib"] is None else f"{run['peak_rss_mib']:g} MiB"
        print(  # noqa: T201
            f"{len(files)} file(s): {sum(r['result'] == 'error' for r in files.values())}"
            f" error(s), {sum(not r['stable'] for r in files.values())} unstable;"
            f" {run['seconds']:g} s, peak {peak}; written to {args.out}"
        )
        if args.lines:
            checked = [r["lines"] for r in files.values() if r.get("lines")]
            print(  # noqa: T201
                f"lines: {len(checked)} file(s) with line ranges;"
                f" {sum(c['uncovered'] for c in checked)} uncovered,"
                f" {sum(c['overlaps'] for c in checked)} overlapping,"
                f" {sum(c['inexact'] for c in checked)} inexact"
                f" of {sum(c['nonblank'] for c in checked)} non-blank line(s)"
            )
        return 0

    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    lines = compare(
        before,
        after,
        ignore_warnings=args.ignore_warnings,
        text_only=args.text_only,
    )
    for line in lines:
        print(line)  # noqa: T201
    print(f"{len(lines)} difference(s).")  # noqa: T201
    if args.timing:
        for line in timing_lines(before, after):
            print(line)  # noqa: T201
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main())
