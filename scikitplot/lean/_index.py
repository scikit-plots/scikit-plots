# scikitplot/lean/_index.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Evidence index over a Lean project: what a claim states, and what it rests on.

:class:`LeanIndex` reads every ``.lean`` file of a project with the rules of
:class:`~scikitplot.corpus.LeanReader` and keeps, for each declaration, its
file and lines, kind, namespace, audit flags and the names its code uses.
From that it answers, quoting the file and lines of every item:

- **What a theorem states.** The project definitions its statement unfolds
  to, with their text, and the names it takes from dependencies.
- **What it rests on.** Every declaration its proof names, directly or
  through others, with the path to each that carries an audit flag.
- **What cannot be traced by name.** Instances, attribute-tagged lemmas and
  syntax extensions are used without being named; any flagged one in the
  project is listed as an implicit risk.
- **What the project holds.** Flags, duplicate names, imported modules the
  project lacks, syntax extensions, files it could not read.

The answers are plain data (JSON) with a source location on every item,
for a reviewer, a script or a language model to check claim by claim, and a
Markdown rendering of the same.

What this is not: a proof checker. It reads text. Whether a theorem is
proved, from which axioms, is what Lean's kernel decides (``lake build``,
``#print axioms``, an independent replay such as ``lean4checker``); every
dossier ends with those steps, and its verdict never says more than the text
shows.
"""  # noqa: D205, D400

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import zlib
from array import array
from collections import Counter, OrderedDict, deque
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from . import _parallel
from ._shapes import _bound_names, fingerprint, states_a_claim, vacuity
from ._syntax import (
    _MODIFIERS,
    _NOT_REFERENCES,
    _REFERENCE,
    LEAN_FLAGS,
    _is_name,
    _Lexer,
    _Splitter,
)

__all__ = ["LeanIndex"]

#: Identity of the rules that build the index, recorded on every report.
PRODUCER = "lean.index/1"

#: Edge vocabulary of :mod:`scikitplot.corpus._graph` (``RelationType.REFERENCES``
#: and ``EdgeTrust.EXTRACTED``), as plain strings: this package imports no
#: corpus code; a corpus test pins the two equal.
RELATION = "references"
TRUST = "extracted"

#: Chunks that declare nothing: no node of the graph.
_NOT_NODES = frozenset({"import", "context", "moduledoc", "comment"})

#: Kinds whose body is their meaning: a statement unfolds through them.
_DEFINITIONS = frozenset(
    {
        "def",
        "abbrev",
        "structure",
        "class",
        "inductive",
        "coinductive",
        "opaque",
        "axiom",
        "instance",
        "irreducible_def",
        "notation",
        "notation3",
        "infix",
        "infixl",
        "infixr",
        "prefix",
        "postfix",
        "alias",
    }
)

#: Kinds a proof can use without naming them.
_IMPLICIT = frozenset(
    {
        "instance",
        "macro",
        "macro_rules",
        "syntax",
        "elab",
        "elab_rules",
        "notation",
        "notation3",
        "infix",
        "infixl",
        "infixr",
        "prefix",
        "postfix",
        "binder_predicate",
        "declare_syntax_cat",
        "simproc",
        "dsimproc",
        "simproc_decl",
        "dsimproc_decl",
        "initialize",
        "builtin_initialize",
        "run_cmd",
        "run_elab",
        "run_meta",
        "#eval",
    }
)

#: Commands that run code at build time; with a learned project command,
#: these are where declarations can be added that no source line spells.
_RUNS_CODE = frozenset(
    {"run_cmd", "run_elab", "run_meta", "#eval", "initialize", "builtin_initialize"}
)

#: Flags that mean code adds declarations to the environment.
_ADDS = frozenset({"add_decl", "axiom"})

#: Every audit flag a chunk can carry: the token flags of ``LEAN_FLAGS``,
#: ``axiom`` on an ``axiom`` declaration and ``after_exit`` past ``#exit``.
AUDIT_FLAGS = frozenset(LEAN_FLAGS.values()) | {"axiom", "after_exit"}

#: Graph backends: the standard library, NumPy, and NumPy with SciPy's searches.
_GRAPHS = ("python", "numpy", "scipy")

#: Past this many files, import bitsets are kept zlib-compressed (see
#: ``LeanIndex._sees``): below it they take little memory either way.
_COMPRESS_FILES = 20_000

#: Import bitsets kept expanded at once (declarations come file by file).
_EXPANDED = 64

#: Name-resolution contexts (namespace and opens) kept resolved at once.
_CONTEXT_CACHE = 256

#: Files whose lines are kept in memory for quoting at once.
_LINES_CACHE = 64


def _commands_of(path: Path) -> dict[str, int]:
    """Return the commands *path* defines; none when it cannot be read (it is skipped later)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as stream:
            return _defined_commands(stream)
    except OSError:
        return {}


def _lean_files(
    root: Path, *, include_hidden: bool
) -> tuple[list[tuple[Path, str]], list[tuple[str, str]]]:
    """
    Return the ``.lean`` files under *root* with their labels, and those refused.

    ``os.walk`` without following links to folders, so the same files are
    found on every Python (``Path.rglob`` changed how it treats links in
    3.13). A link to a file is kept only when it resolves to a file inside
    *root*: a project folder never makes the index read outside it. Labels
    are POSIX paths relative to *root*; both lists are sorted by label.
    """
    base = root.resolve()
    found: list[tuple[Path, str]] = []
    refused: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        dirnames.sort()
        here = Path(dirpath)
        for name in filenames:
            if not name.endswith(".lean"):
                continue
            path = here / name
            label = path.relative_to(root).as_posix()
            if path.is_symlink():
                try:
                    target = path.resolve(strict=True)
                except (OSError, RuntimeError):
                    refused.append((label, "broken symbolic link"))
                    continue
                try:  # different drives (Windows) raise: outside
                    inside = os.path.commonpath([str(base), str(target)]) == str(base)
                except ValueError:
                    inside = False
                if not inside:
                    refused.append(
                        (label, "symbolic link to a file outside the folder")
                    )
                    continue
            if path.is_file():
                found.append((path, label))
    found.sort(key=lambda item: item[1])
    refused.sort()
    return found, refused


def _defined_commands(lines: Iterable[str]) -> dict[str, int]:
    """
    Return the command keywords a file defines, with the line of each definition.

    A line that, after attributes and modifiers, starts with ``elab``,
    ``macro`` or ``syntax`` and holds ``: command`` defines the first word
    of its first string literal (``elab "copy_all" : command => ...``).
    ``#`` commands need no learning. Plain string operations: linear.
    """
    found: dict[str, int] = {}
    for number, line in enumerate(lines, start=1):
        if "command" not in line:
            continue
        words = line.split()
        i = 0
        while i < len(words) and (words[i] in _MODIFIERS or words[i].startswith("@[")):
            i += 1
        if i == len(words) or words[i] not in ("elab", "macro", "syntax"):
            continue
        tail = line.replace(" ", "")
        if ":command" not in tail:
            continue
        first = line.find('"')
        second = line.find('"', first + 1) if first >= 0 else -1
        if second < 0:
            continue
        parts = line[first + 1 : second].split()
        if parts and not parts[0].startswith("#") and _is_name(parts[0]):
            found.setdefault(parts[0], number)
    return found


#: How a name was found, strongest first.
_HOW = ("exact", "prefix", "suffix")

_ROOT = "_root_."


#: Binder names a statement introduces: `(x y : T)`, `{x}`, `[i : C]` ...
def _names_the_project(module: str, modules: Any, roots: Any) -> bool:
    """
    Return whether an import no file provides still names the project.

    Its first component is that of a provided module (a missing module), or
    it is a provided module with leading components added (the index was
    built from a subfolder: ``Mathlib/`` provides ``Algebra.X`` and the
    files import ``Mathlib.Algebra.X``).
    """
    parts = module.split(".")
    return parts[0] in roots or any(
        ".".join(parts[i:]) in modules for i in range(1, len(parts))
    )


def _compact(chunk: dict[str, Any]) -> tuple[Any, ...]:
    """
    Return what the index keeps of one command, as a plain tuple.

    Imports: ``("import", modules)``; other commands outside the graph:
    ``(kind,)``; a declaration: ``(kind, name, line_start, line_end,
    namespace, opens, references, flags, attributes, private, shape,
    claim, vacuous)``. The statement-shape digest, whether it is a
    definition of type ``Prop`` and why such a definition says nothing
    (:func:`._shapes.vacuity`) are computed here, so a worker process does
    that work too.
    """
    kind = chunk["lean_kind"]
    if kind == "import":
        return (kind, tuple(chunk.get("imports", ())))
    if kind in _NOT_NODES:
        return (kind,)
    name = chunk.get("declaration")
    shape = None
    claim = False
    vacuous = None
    if kind in ("theorem", "lemma"):
        statement = chunk.get("statement")
        shape = None if statement is None else fingerprint(statement, kind, name)
    elif kind in ("def", "abbrev"):
        claim = states_a_claim(chunk["text"])
        vacuous = vacuity(chunk["text"]) if claim else None
    return (
        kind,
        name,
        chunk["line_start"],
        chunk["line_end"],
        chunk.get("namespace", ""),
        tuple(chunk.get("opens", ())),
        tuple(chunk.get("references", ())),
        tuple(chunk.get("flags", ())),
        tuple(chunk.get("attributes", ())),
        "private" in chunk.get("modifiers", ()),
        shape,
        claim,
        vacuous,
    )


def _read_file(
    path: str | Path, label: str, settings: tuple[str, frozenset[str], int, int]
) -> tuple[Any, ...]:
    """
    Read one ``.lean`` file into records; the work a worker process does.

    Returns ``("ok", records, lines, problems, digest)`` (records from
    :func:`_compact`; problems as ``(line, text)`` for text Lean refuses to
    read, such as an unmatched ``end``) or ``("skip", reason)``: larger than ``max_file_bytes``, unreadable, or
    nesting deeper than ``max_scope_depth``. A file is read whole before
    any of it is added, so a skipped file adds nothing.
    """
    dialect, commands, max_depth, max_bytes = settings
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        return ("skip", f"unreadable: {exc.strerror or exc}")
    if size > max_bytes:
        return ("skip", f"larger than max_file_bytes={max_bytes:,}")
    splitter = _Splitter(
        file_name=label,
        meta={},
        step=None,
        cap=None,
        proofs="keep",
        dialect=dialect,
        kinds=None,
        watch=(),
        commands=commands,
        max_scope_depth=max_depth,
        flag_tokens=LEAN_FLAGS,
        references=True,
        statements=True,
    )
    records: list[tuple[Any, ...]] = []
    number = 0
    try:
        with open(path, "rb") as stream:
            data = stream.read()
    except OSError as exc:
        return ("skip", f"unreadable: {exc.strerror or exc}")
    text, invalid = _decode(data)
    try:
        # Universal newlines, as text mode reads them: \r\n, \r and \n.
        for number, read in enumerate(io.StringIO(text, newline=None), start=1):
            line = read[:-1] if read.endswith("\n") else read
            if number == 1 and line.startswith("\ufeff"):
                line = line[1:]
            records.extend(_compact(chunk) for chunk in splitter.feed(number, line))
        records.extend(_compact(chunk) for chunk in splitter.finish())
    except ValueError as exc:  # nesting past max_scope_depth
        return ("skip", str(exc))
    problems = tuple(splitter.problems)
    if invalid is not None:
        problems = tuple(sorted(((invalid, _NOT_UTF8), *problems)))
    return ("ok", records, number, problems, _digest(data))


#: Bytes of a file's digest: 128 bits, so two different files never meet.
_DIGEST_BYTES = 16


def _digest(data: bytes) -> bytes:
    """Return the digest that tells whether a file changed since it was read."""
    return hashlib.blake2b(data, digest_size=_DIGEST_BYTES).digest()


#: The source problem of a file with bytes that are not UTF-8.
_NOT_UTF8 = (
    "not valid UTF-8: the first invalid byte is on this line"
    " (invalid bytes are read as U+FFFD)"
)


def _decode(data: bytes) -> tuple[str, int | None]:
    """
    Return *data* decoded as UTF-8, and the line of its first invalid byte.

    Lean reads UTF-8 only. Invalid bytes are replaced by U+FFFD so the rest
    of the file is still indexed. The line counts a CR LF pair, a lone CR
    and a lone LF as one break each, as the reader does; it is ``None``
    when the file is valid.
    """
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        head = data[: exc.start]
        line = head.count(b"\n") + head.count(b"\r") - head.count(b"\r\n") + 1
        return data.decode("utf-8", errors="replace"), line


def _workers(value: Any) -> int:
    """Validate ``workers``: an int >= 1, or ``None`` for every CPU."""
    if value is None:
        return max(1, os.cpu_count() or 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"LeanIndex: workers must be an int >= 1 or None; got {value!r}."
        )
    return value


def _layer_modules(
    files: list[tuple[int, str]], source_root: str | None, imported: Counter[str]
) -> dict[str, int]:
    """
    Map the modules one layer's files provide to their file numbers.

    *files* are (file number, label within the layer). With *source_root*,
    module names start below it; otherwise at the depth under which the
    most *imported* modules name a file exactly (ties: the fewest folders).
    """
    paths = [
        (file_id, label[: -len(".lean")].split("/"))
        for file_id, label in files
        if label.endswith(".lean")
    ]
    if source_root is not None:
        prefix = source_root.split("/") if source_root else []
        fixed: dict[str, int] = {}
        for file_id, parts in paths:
            if len(parts) > len(prefix) and parts[: len(prefix)] == prefix:
                fixed.setdefault(".".join(parts[len(prefix) :]), file_id)
        return fixed
    deepest = max((len(parts) - 1 for _f, parts in paths), default=0)
    best: tuple[int, dict[str, int]] = (-1, {})
    for depth in range(deepest + 1):
        modules: dict[str, int] = {}
        for file_id, parts in paths:
            if len(parts) > depth:
                modules.setdefault(".".join(parts[depth:]), file_id)
        resolved = sum(count for module, count in imported.items() if module in modules)
        if resolved > best[0]:
            best = (resolved, modules)
    return best[1]


#: A layer's name: what may precede the ``:`` of its files' labels.
_LAYER_NAME = re.compile(r"[A-Za-z0-9_.\-]+")


@contextlib.contextmanager
def _reading_pool(workers: int, files: int) -> Iterator[Any]:
    """Yield a process pool to read *files* with, or ``None`` for one process."""
    pool = None
    if workers > 1 and files > 1:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
    try:
        yield pool
    finally:
        if pool is not None:
            pool.shutdown()  # Python 3.8: no cancel_futures


def _learn(
    files: list[tuple[Path, str]], pool: Any, workers: int
) -> dict[str, tuple[str, int]]:
    """Return the commands *files* define: keyword to (label, line), first wins."""
    paths = [path for path, _label in files]
    defined = (
        pool.map(_commands_of, paths, chunksize=_batch_size(len(files), workers))
        if pool is not None
        else map(_commands_of, paths)
    )
    learned: dict[str, tuple[str, int]] = {}
    for (_path, label), found in zip(files, defined):
        for keyword, number in found.items():
            learned.setdefault(keyword, (label, number))
    return learned


#: Most files a worker reads before it sends their records back.
_BATCH_FILES = 64


def _batch_size(files: int, workers: int) -> int:
    """
    Return how many files a worker reads per batch.

    About sixteen batches per worker, so the work stays spread, and never
    more than :data:`_BATCH_FILES`: a worker holds a whole batch's records
    until it sends them, and the parent unpickles a batch at once, so an
    uncapped batch grew with the project (5,474 files a batch and 420 MB a
    worker on a project of 175,187 files; 64 files and 140 MB capped, read
    as fast).
    """
    return max(1, min(files // (workers * 16), _BATCH_FILES))


def _read_file_star(arguments: tuple[Any, ...]) -> tuple[Any, ...]:
    """:func:`_read_file` for ``Executor.map`` (one picklable argument)."""
    return _read_file(*arguments)


def _scan_file(arguments: tuple[Path, bool]) -> tuple[bytes | None, dict[str, int]]:
    """
    Return a file's digest and, when asked, the commands it defines.

    What :meth:`LeanIndex.refresh` reads of every file of a layer: the
    bytes once, decoded as :func:`_commands_of` decodes them. ``None`` and
    no commands when the file cannot be read (it is then read again, and
    listed as skipped).
    """
    path, learn = arguments
    try:
        with open(path, "rb") as stream:
            data = stream.read()
    except OSError:
        return None, {}
    defined: dict[str, int] = {}
    if learn:
        text = data.decode("utf-8", errors="replace")
        defined = _defined_commands(io.StringIO(text, newline=None))
    return _digest(data), defined


class LeanIndex:
    r"""
    Declarations of a Lean project, their audit flags and the names they use.

    Build one with :meth:`from_folder` (or :meth:`add_file` per file), then
    ask :meth:`summary` for the project and :meth:`dossier` for a claim.
    Both return JSON-ready dicts; :meth:`render` turns either into Markdown.

    Parameters
    ----------
    dialect : {"lean4", "lean3"}, optional
        As for :class:`~scikitplot.corpus.LeanReader`. Default: ``"lean4"``.
    commands : collection of str, optional
        Command keywords a project defines, as for ``LeanReader``.
    max_scope_depth : int, optional
        As for ``LeanReader``; a file nesting deeper is listed as skipped.
        Default: 256.
    max_file_bytes : int, optional
        Larger files are listed as skipped. Default: 50 MB.
    suffix : {"unique", "none"}, optional
        How a dotted name that matches no full name is still linked:
        ``"unique"`` (default) links ``h.myLemma`` to the one declaration
        whose last component is ``myLemma``, the usual reading of dot
        notation; ``"none"`` links full names only.
    visibility : {"imports", "all"}, optional
        Which declarations a name can link to. ``"imports"`` (default):
        those of the same file and of the files it imports, directly or
        through others, as Lean sees them; a file with a project import
        no file provides sees every declaration, so an incomplete index
        links too much, never too little. ``"all"``: every declaration of
        the index, as text alone suggests.
    source_root : str or None, optional
        The folder, relative to the labels, that module names start from
        (``"src"``; ``""`` for the index root). Files outside it provide
        no module. ``None`` (default) finds it from the imports; see
        :meth:`_module_files`. Set it when a small project's only imports
        could name either a dependency or a project file.
    graph : {"python", "numpy", "scipy"}, optional
        How the whole-graph steps run. ``"python"`` (default): the standard
        library. ``"numpy"``: once a step needs every edge (``summary``,
        ``discover``, ``export``), the edges are packed into two NumPy
        arrays (about six times less memory than one array per
        declaration) and use counts and breadth-first searches run
        vectorised. ``"scipy"``: packed as for ``"numpy"``, the
        breadth-first searches in :mod:`scipy.sparse.csgraph` (a fixed cost
        of milliseconds per search: for benchmarking, and for closures of
        hundreds of thousands of declarations). Reports are identical for
        all three; name resolution, most of the time on real projects, is
        the same in all of them.
    workers : int or None, optional
        Processes that resolve names when a step needs every edge
        (``summary``, ``discover``, ``export``): ``1`` (default) resolves
        them here, ``None`` uses every CPU. Used on Linux, where forked
        workers start from the index in memory (nothing is sent to them),
        for projects of 20,000 declarations or more; elsewhere names are
        resolved here. Every report is the same for any number of workers.
        :meth:`from_folder` also reads files with its ``workers``.

        Memory grows with the workers: each copies the parts of the index
        it reads (Python's reference counts write to them) and keeps its
        own lookup cache. On a project of a million declarations a summary
        took 187 s and 2.6 GB in all with two workers, against 318 s and
        1.7 GB with one; set ``workers`` to what the machine's memory
        allows, not only its CPUs.

    See Also
    --------
    scikitplot.corpus.LeanReader : The chunks this index is built from.

    Notes
    -----
    **How a name is resolved**, as Lean looks a name up: in the
    declaration's namespace and each enclosing one, innermost first, then
    in each namespace open for it (``_root_.`` skips all of that). A name
    that matches nothing is tried again without its last components, for
    names Lean derives (``Foo.mk``, ``foo.eq_1``, the ``h`` of
    ``h.symm``); then, with ``suffix="unique"`` and a dotted name, each
    component after the first is matched to the one declaration with that
    last component, as dot notation on a hypothesis (``h.myLemma``) names
    ``MyProp.myLemma``.
    Every edge records which of the three (``exact``, ``prefix``,
    ``suffix``) found it. A ``private`` declaration is found only from its
    own file. All matches are kept, so the closure of a theorem includes
    every declaration it could be naming: it may hold too much, never too
    little of what the text names.

    **What a text index cannot see.** Lean finds instances, ``@[simp]``
    lemmas, coercions and notation without their names appearing in the
    proof, and macros and metaprograms can create declarations under names
    no source line spells. A dossier lists every flagged declaration of
    those kinds in the project as an implicit risk rather than assume it
    unused.

    **Statement versus proof.** A theorem's statement is read with
    ``LeanReader(proofs="drop")``; the project definitions it names, and the
    definitions those name in turn, are what the claim means, and a
    dossier quotes each. Names a statement binds (``(s : Real)``, ``∀ n``)
    are left out; names found nowhere in the project come from its
    dependencies and are listed as external.

    **Memory and time.** Per declaration the index keeps a few integers and
    the names its code uses (interned strings); names are resolved only for
    the declarations a dossier reaches, once each. A file is read twice at
    most: once to index it, once more to quote it.

    **Developer note.** Edges use the vocabulary of
    :mod:`scikitplot.corpus._graph` (:data:`RELATION`, :data:`TRUST`): relation ``references``, trust
    ``extracted`` (a deterministic extractor), producer :data:`PRODUCER`.

    Examples
    --------
    >>> from scikitplot.lean import LeanIndex
    >>> index = LeanIndex.from_folder("MyProject")  # doctest: +SKIP
    >>> report = index.dossier("MyProject.mainTheorem")  # doctest: +SKIP
    >>> report["verdict"]  # doctest: +SKIP
    'no lexical red flags; kernel check pending'
    >>> print(LeanIndex.render(report))  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        dialect: str = "lean4",
        commands: Iterable[str] = (),
        max_scope_depth: int = 256,
        max_file_bytes: int = 50 * 1024 * 1024,
        suffix: str = "unique",
        visibility: str = "imports",
        source_root: str | None = None,
        graph: str = "python",
        workers: int | None = 1,
    ) -> None:
        if dialect not in ("lean4", "lean3"):
            raise ValueError(
                f"LeanIndex: dialect must be 'lean4' or 'lean3'; got {dialect!r}."
            )
        if visibility not in ("imports", "all"):
            raise ValueError(
                f"LeanIndex: visibility must be 'imports' or 'all'; got {visibility!r}."
            )
        if source_root is not None and (
            not isinstance(source_root, str)
            or source_root.startswith("/")
            or "\\" in source_root
            or ".." in source_root.split("/")
        ):
            raise ValueError(
                "LeanIndex: source_root must be a relative POSIX folder such as"
                f" 'src' ('' for the index root); got {source_root!r}."
            )
        if suffix not in ("unique", "none"):
            raise ValueError(
                f"LeanIndex: suffix must be 'unique' or 'none'; got {suffix!r}."
            )
        for option, value in (
            ("max_scope_depth", max_scope_depth),
            ("max_file_bytes", max_file_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"LeanIndex: {option} must be an int > 0; got {value!r}."
                )
        if isinstance(commands, (str, bytes)):
            raise TypeError(
                f"LeanIndex: commands must be a collection of str; got {commands!r}."
            )
        self._dialect = dialect
        # Commands given here, and those each layer's files define (learned):
        # a layer is read with the given ones and its own and lower layers'.
        self._given_commands = frozenset(commands)
        self._layer_commands: list[frozenset[str]] = [frozenset()]
        self._commands = self._given_commands
        self._max_depth = max_scope_depth
        self._max_bytes = max_file_bytes
        self._suffix = suffix
        self._visibility = visibility
        self._source_root = None if source_root is None else source_root.strip("/")
        # Layers (add_folder): name and source root; layer 0 is the unnamed
        # root of from_folder, add_file and add_chunks. One layer per file.
        self._layers: list[tuple[str, str | None]] = [("", self._source_root)]
        # Where each layer's files are (to refresh them); not saved.
        self._layer_roots: list[Path | None] = [None]
        self._layer_of: list[int] = []
        # Per file: the BLAKE2b digest of its bytes (None: not read from a
        # file, or skipped), and its commands counted by kind, packed as
        # (kind number, count) pairs (``_kinds`` is their sum).
        self._digests: list[bytes | None] = []
        self._fk_names: list[str] = []
        self._fk_ids: dict[str, int] = {}
        self._fk_ptr: array[int] = array("q", [0])
        self._fk_data: array[int] = array("i")
        if graph not in _GRAPHS:
            raise ValueError(
                f"LeanIndex: graph must be one of {', '.join(map(repr, _GRAPHS))}; got {graph!r}."
            )
        if graph != "python":
            try:  # import them: a broken install fails here, not mid-report
                import numpy as np  # noqa: F401, PLC0415

                if graph == "scipy":
                    import scipy.sparse.csgraph  # noqa: F401, PLC0415
            except ImportError as exc:
                needs = (
                    "NumPy and SciPy: pip install numpy scipy"
                    if graph == "scipy"
                    else "NumPy: pip install numpy"
                )
                raise ImportError(
                    f"LeanIndex(graph={graph!r}) needs {needs}; graph='python' needs nothing."
                ) from exc
        self._graph = graph
        self._workers = _workers(workers)
        self._packed: tuple[Any, Any] | None = None
        # Per expanded-kind set, the mask of nodes whose edges a packed
        # closure follows; dropped with the packed edges.
        self._follow: dict[frozenset[str], Any] = {}
        self._reach: dict[int, int | bytes] = {}
        self._expanded: OrderedDict[int, int] = OrderedDict()
        # files
        self._labels: list[str] = []
        self._label_ids: dict[str, int] = {}
        self._paths: list[Path | None] = []
        self._first_node: list[int] = []  # each file's first node
        self._learned: dict[str, tuple[str, int]] = {}
        self._file_lines: list[int] = []
        self._imports: list[tuple[str, ...]] = []
        self._skipped: list[tuple[str, str]] = []
        # file id -> ((line, problem), ...): text Lean refuses to read.
        self._problems: dict[int, tuple[tuple[int, str], ...]] = {}
        # declarations (one entry per node, in reading order)
        self._name: list[str | None] = []
        self._kind: list[str] = []
        self._file: array[int] = array("l")
        self._start: array[int] = array("l")
        self._end: array[int] = array("l")
        self._namespace: list[str] = []
        self._opens: list[tuple[str, ...]] = []
        self._private: set[int] = set()
        # Read once, while indexing: each result's statement-shape digest
        # (16 bytes) and the definitions of type Prop; discovery groups them.
        self._shape_of: dict[int, bytes] = {}
        self._claims: set[int] = set()
        # Prop definitions that say nothing about their subject: the reason.
        self._vacuous: dict[int, str] = {}
        self._flags: dict[int, tuple[str, ...]] = {}
        self._attributes: dict[int, tuple[str, ...]] = {}
        # The names each declaration's code uses, packed: each distinct
        # reference once, numbered (_ref_names, _ref_ids), and per declaration
        # its reference numbers, data[ptr[n]:ptr[n + 1]]. Arrays rather than
        # a tuple per declaration: smaller, and a forked worker reads them
        # without writing a reference count, so it copies no page of them.
        self._ref_names: list[str] = []
        self._ref_ids: dict[str, int] = {}
        # Each declaration's context (namespace and opens), numbered: the
        # resolution cache is keyed by context number.
        self._ctx_keys: list[tuple[str, tuple[str, ...]]] = []
        self._ctx_ids: dict[tuple[str, tuple[str, ...]], int] = {}
        self._ctx_of: array[int] = array("i")
        # In a forked worker only (_parallel): the references and contexts
        # as UTF-8 text in shared buffers, decoded privately, so the worker
        # never reads, and so never copies, the shared string objects.
        self._flat: _parallel.FlatNames | None = None
        self._ref_ptr: array[int] = array("q", [0])
        self._ref_data: array[int] = array("i")
        self._edges: dict[int, array[int]] = {}
        self._by_name: dict[str, Any] = {}
        self._by_short: dict[str, Any] | None = None
        self._kinds: Counter[str] = Counter()
        self._intern: dict[Any, Any] = {}
        # Resolutions cached per context number: reference number -> codes.
        self._contexts: OrderedDict[int, dict[int, tuple[int, ...]]] = OrderedDict()
        self._lines_cache: OrderedDict[int, list[str]] = OrderedDict()
        # Bumped by every new file: the key of the whole-project caches.
        self._generation = 0
        # False once a file is added out of label order; the next query
        # renumbers files and declarations (see _canonical).
        self._in_order = True

    # -- building ---------------------------------------------------------------

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        *,
        include_hidden: bool = False,
        learn_commands: bool = True,
        workers: int | None = 1,
        **options: Any,
    ) -> LeanIndex:
        """
        Index every ``.lean`` file under *folder*, in sorted order.

        Parameters
        ----------
        folder : str or pathlib.Path
            The project, or the part of it to index.
        include_hidden : bool, optional
            Also read files under a folder whose name starts with ``.``,
            such as Lake's ``.lake/packages`` (the project's dependencies,
            often far larger than the project). Default: ``False``.
            Symbolic links to folders are never followed; a link to a file
            is read only when it resolves inside *folder*, and is otherwise
            listed as skipped. The same files are read on every Python.
        workers : int or None, optional
            Processes that read the files (``concurrent.futures``, standard
            library). ``1`` (default) reads them here, one at a time;
            ``None`` uses every CPU. Files are read independently and added
            in sorted order, so the index, and every report, is the same for
            any number of workers. Worth it for large projects: starting
            the processes and sending the records back cost a little.
        learn_commands : bool, optional
            Before indexing, read every file for the commands the project
            defines (``elab``, ``macro`` or ``syntax "kw" ... : command``)
            and treat each keyword as a command (``commands``), so an
            invocation is a chunk of its own and the code it runs can be
            followed. Default: ``True``.
        **options
            As for :class:`LeanIndex`.

        Returns
        -------
        LeanIndex

        Raises
        ------
        ValueError
            If *folder* is not a folder, or an option is invalid.
        """
        root = Path(folder)
        if not root.is_dir():
            raise ValueError(f"LeanIndex: {folder!s} is not a folder.")
        workers = _workers(workers)
        files, refused = _lean_files(root, include_hidden=include_hidden)
        with _reading_pool(workers, len(files)) as pool:
            learned = _learn(files, pool, workers) if learn_commands else {}
            index = cls(workers=workers, **options)
            index._learned = learned
            index._set_layer_commands(0, frozenset(learned))
            index._layer_roots[0] = root
            index._skipped.extend(refused)
            index._read_files(files, pool, workers, layer=0)
        return index

    def add_folder(
        self,
        folder: str | Path,
        *,
        name: str,
        include_hidden: bool = False,
        learn_commands: bool = True,
        source_root: str | None = None,
    ) -> None:
        """
        Add a project's files as a layer over what the index holds.

        Load a dependency's saved index once (mathlib, say), then add each
        project that imports it: the project's names resolve into the
        dependency as into its own files, and its theorems' dossiers follow
        their proofs through both, without reading the dependency again.

        Parameters
        ----------
        folder : str or pathlib.Path
            The project's root, as for :meth:`from_folder`.
        name : str
            The layer's name: letters, digits, ``_``, ``-`` and ``.``. Its
            files are labelled ``name:path`` (``pfr:PFR/Main.lean``), so two
            projects' files never share a label.
        include_hidden : bool, optional
            As for :meth:`from_folder`. Default: ``False``.
        learn_commands : bool, optional
            Learn the commands the project's own files define (they apply to
            the project's files, never to the layers below, which cannot
            import it). Default: ``True``.
        source_root : str or None, optional
            The folder, under *folder*, that the project's module names start
            from; ``None`` (default) finds it from the imports, per layer.

        Raises
        ------
        ValueError
            If *folder* is not a folder, *name* is not a valid layer name or
            is taken, or reading a file fails as in :meth:`from_folder`.

        See Also
        --------
        load : Read a saved index, to add layers over.

        Notes
        -----
        **User.** Lean gives every module one name across all packages: a
        module two layers provide is resolved to the lower layer's file and
        listed in the summary (``imports.module_clashes``), since Lean would
        refuse to build them together. Files are read with the index's
        ``workers``.
        """
        if not isinstance(name, str) or not _LAYER_NAME.fullmatch(name):
            raise ValueError(
                f"LeanIndex.add_folder: name must be letters, digits, '_', '-' or '.';"
                f" got {name!r}."
            )
        if name in (layer for layer, _root in self._layers):
            raise ValueError(
                f"LeanIndex.add_folder: a layer is already named {name!r}."
            )
        # A label tells its layer by its prefix: no file already indexed or
        # skipped may start with this one (a base path "p:x.lean", say).
        prefix = f"{name}:"
        held = [label for label in self._labels if label.startswith(prefix)]
        held += [label for label, _ in self._skipped if label.startswith(prefix)]
        if held:
            raise ValueError(
                f"LeanIndex.add_folder: {held[0]!r} already starts with {prefix!r};"
                " choose another name."
            )
        root = Path(folder)
        if not root.is_dir():
            raise ValueError(f"LeanIndex.add_folder: {folder!s} is not a folder.")
        found, refused = _lean_files(root, include_hidden=include_hidden)
        files = [(path, f"{name}:{label}") for path, label in found]
        taken = [label for _path, label in files if label in self._label_ids]
        if taken:
            raise ValueError(f"LeanIndex.add_folder: {taken[0]!r} is already indexed.")
        clean = None if source_root is None else source_root.strip("/")
        self._layers.append((name, clean))
        self._layer_roots.append(root)
        self._layer_commands.append(frozenset())
        with _reading_pool(self._workers, len(files)) as pool:
            if learn_commands:
                learned = _learn(files, pool, self._workers)
                for keyword, where in learned.items():
                    self._learned.setdefault(keyword, where)
                self._set_layer_commands(len(self._layers) - 1, frozenset(learned))
            self._skipped.extend(
                (f"{name}:{label}", reason) for label, reason in refused
            )
            self._read_files(files, pool, self._workers, layer=len(self._layers) - 1)

    def refresh(
        self,
        layer: str = "",
        *,
        folder: str | Path | None = None,
        include_hidden: bool = False,
        learn_commands: bool = True,
        max_items: int = 50,
    ) -> dict[str, Any]:
        """
        Read again only what changed in one layer's folder: new, changed and gone files.

        Parameters
        ----------
        layer : str, optional
            The layer (:meth:`add_folder`; ``""``, the default, is the first).
        folder : str or pathlib.Path or None, optional
            Where the layer's files are now. ``None`` (default): the folder
            it was read from, or given as ``root`` to :meth:`load`.
        include_hidden : bool, optional
            As for :meth:`from_folder`; give what the layer was read with.
            Default: ``False``.
        learn_commands : bool, optional
            As for :meth:`from_folder`; give what the layer was read with.
            Default: ``True``.
        max_items : int, optional
            Most file labels listed per change. Default: 50.

        Returns
        -------
        dict
            JSON-ready; ``report`` is ``"refresh"``: ``added``, ``changed``
            and ``removed`` (listings of labels), ``unchanged`` (a count),
            ``files_read`` and ``commands_changed`` with
            ``layers_read_again``.

        Raises
        ------
        ValueError
            If *layer* names no layer, it has no known folder and *folder*
            is not given, the folder is not a folder, a new file's label
            would name another layer, or the layers above must be read
            again and one of their files has no known path.

        See Also
        --------
        from_folder, add_folder, load

        Notes
        -----
        **User.** The index after a refresh is the index a new read of the
        same files would give, down to the bytes :meth:`save` writes. Every
        file of the layer is read once, for its digest (BLAKE2b, 128 bits)
        and the commands it defines; only new files and those whose digest
        changed are parsed again. A file the index skipped is always read
        again.

        **Developer.** When the commands the layer defines change, every
        file of the layer and of the layers above is read again: a command
        changes how the files that can use it are split. Otherwise the
        stale files are dropped (:meth:`_drop_files`, one renumbering) and
        the changed and new ones added, as :meth:`add_folder` adds files.
        The names dropped declarations used stay in the name tables until
        the index is saved and loaded (a save keeps only names in use).

        On mathlib4 (9,425 files with PFR), a refresh with nothing changed
        takes 0.8 s and with 20 files changed 1.4 s, against 27 s to read
        the folder again; on 175,187 files, 13 s with nothing changed
        against 222 s.
        """
        self._canonical()
        _positive(max_items, "max_items")
        scope = self._scope(layer)
        at = [name for name, _root in self._layers].index(layer)
        root = Path(folder) if folder is not None else self._layer_roots[at]
        if root is None:
            raise ValueError(
                f"LeanIndex.refresh: layer {layer!r} has no known folder; give folder=."
            )
        if not root.is_dir():
            raise ValueError(f"LeanIndex.refresh: {root!s} is not a folder.")
        found, listed_refused = _lean_files(root, include_hidden=include_hidden)
        prefix = f"{layer}:" if layer else ""
        files = [(path, prefix + label) for path, label in found]
        refused = [(prefix + label, reason) for label, reason in listed_refused]
        if at == 0:
            for _path, label in files:
                if self._layer_of_label(label):
                    raise ValueError(
                        f"LeanIndex.refresh: {label!r} would name another layer."
                    )
        current = {self._labels[f]: f for f in sorted(scope)}
        with _reading_pool(self._workers, len(files)) as pool:
            jobs = [(path, learn_commands) for path, _label in files]
            scans = list(
                pool.map(
                    _scan_file, jobs, chunksize=_batch_size(len(jobs), self._workers)
                )
                if pool is not None
                else map(_scan_file, jobs)
            )
            learned: dict[str, tuple[str, int]] = {}
            for (_path, label), (_found, defined) in zip(files, scans):
                for keyword, number in defined.items():
                    learned.setdefault(keyword, (label, number))
            seen = {label for _path, label in files}
            removed = sorted(set(current) - seen)
            added = [label for _path, label in files if label not in current]
            changed = [
                label
                for (_path, label), (digest, _defined) in zip(files, scans)
                if label in current
                and (digest is None or self._digests[current[label]] != digest)
            ]
            full = frozenset(learned) != self._layer_commands[at]
            if full:
                read = self._read_again_from(at, files, refused, learned, pool)
            else:
                stale = {current[label] for label in (*removed, *changed)}
                read = self._read_changed(
                    at,
                    files,
                    refused=refused,
                    learned=learned,
                    stale=stale,
                    fresh=set(changed) | set(added),
                    pool=pool,
                )
        self._layer_roots[at] = root
        return {
            "report": "refresh",
            "producer": PRODUCER,
            "layer": layer,
            "files": len(files),
            "unchanged": len(current) - len(removed) - len(changed),
            "added": _listing(added, max_items),
            "changed": _listing(changed, max_items),
            "removed": _listing(removed, max_items),
            "files_read": read,
            "commands_changed": full,
            "layers_read_again": (
                [name for name, _root in self._layers[at:]] if full else []
            ),
        }

    def _read_changed(  # noqa: PLR0913
        self,
        at: int,
        files: list[tuple[Path, str]],
        *,
        refused: list[tuple[str, str]],
        learned: dict[str, tuple[str, int]],
        stale: set[int],
        fresh: set[str],
        pool: Any,
    ) -> int:
        """
        Drop the *stale* files of layer *at*, then read the *fresh* labels.

        The layer's commands are unchanged: only where each is defined may
        move, within the layer. Returns the number of files read.
        """
        self._skipped = [
            item for item in self._skipped if self._layer_of_label(item[0]) != at
        ]
        self._drop_files(stale)
        self._skipped.extend(refused)
        for keyword, where in learned.items():
            old = self._learned.get(keyword)
            if old is None or self._layer_of_label(old[0]) == at:
                self._learned[keyword] = where
        for path, label in files:
            if label not in fresh:
                self._paths[self._label_ids[label]] = path
        again = [(path, label) for path, label in files if label in fresh]
        self._read_files(again, pool, self._workers, layer=at)
        return len(again)

    def _read_again_from(
        self,
        at: int,
        files: list[tuple[Path, str]],
        refused: list[tuple[str, str]],
        learned: dict[str, tuple[str, int]],
        pool: Any,
    ) -> int:
        """
        Read layer *at* (its *files*, as listed now) and every layer above it again.

        The layers above keep their files and the commands they define;
        only where each command is defined is learned again. Returns the
        number of files read.
        """
        upper: list[list[tuple[Path, str]]] = []
        for j in range(at + 1, len(self._layers)):
            held = [f for f, layer in enumerate(self._layer_of) if layer == j]
            missing = [self._labels[f] for f in held if self._paths[f] is None]
            if missing:
                raise ValueError(
                    f"LeanIndex.refresh: the commands of layer {self._layers[at][0]!r}"
                    f" changed, so the layers above are read again, and"
                    f" {missing[0]!r} has no known file; load with root= for every"
                    " layer."
                )
            upper.append([(self._paths[f], self._labels[f]) for f in held])
        registered = set(self._labels)
        self._skipped = [
            (label, reason)
            for label, reason in self._skipped
            if self._layer_of_label(label) < at
            or (self._layer_of_label(label) > at and label not in registered)
        ]
        self._learned = {
            keyword: where
            for keyword, where in self._learned.items()
            if self._layer_of_label(where[0]) < at
        }
        self._drop_files({f for f, layer in enumerate(self._layer_of) if layer >= at})
        self._skipped.extend(refused)
        for keyword, where in learned.items():
            self._learned.setdefault(keyword, where)
        self._set_layer_commands(at, frozenset(learned))
        self._read_files(files, pool, self._workers, layer=at)
        read = len(files)
        for j, held in enumerate(upper, start=at + 1):
            keep = self._layer_commands[j]
            for keyword, where in _learn(held, pool, self._workers).items():
                if keyword in keep:
                    self._learned.setdefault(keyword, where)
            self._read_files(held, pool, self._workers, layer=j)
            read += len(held)
        return read

    def _read_files(
        self, files: list[tuple[Path, str]], pool: Any, workers: int, *, layer: int
    ) -> None:
        """Read *files* (path, label) into the index, in order, as layer *layer*."""
        settings = self._settings(layer)
        jobs = [(path, label, settings) for path, label in files]
        outcomes = (
            pool.map(_read_file_star, jobs, chunksize=_batch_size(len(files), workers))
            if pool is not None
            else map(_read_file_star, jobs)
        )
        for (path, label), outcome in zip(files, outcomes):
            self._apply(self._new_file(label, path, layer=layer), label, outcome)

    def add_file(self, path: str | Path, *, label: str | None = None) -> None:
        """
        Index one ``.lean`` file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file.
        label : str or None, optional
            How reports name it (default: *path* as given); a relative
            path makes reports the same on every machine.

        Notes
        -----
        A file larger than ``max_file_bytes``, or nesting deeper than
        ``max_scope_depth``, is listed in the summary as skipped, with the
        reason, and indexing goes on.
        """
        path = Path(path)
        label = str(path) if label is None else label
        file_id = self._new_file(label, path)
        self._apply(file_id, label, _read_file(path, label, self._settings()))

    def _settings(
        self, layer: int | None = None
    ) -> tuple[str, frozenset[str], int, int]:
        """
        Return the options a file is read with, as :func:`_read_file` takes them.

        A file of layer *layer* is read with the given commands and those
        its layer and the layers below define; ``None`` gives every command.
        """
        commands = self._commands if layer is None else self._commands_for(layer)
        return (self._dialect, commands, self._max_depth, self._max_bytes)

    def _commands_for(self, layer: int) -> frozenset[str]:
        """Return the commands layer *layer* is read with."""
        found = set(self._given_commands)
        for learned in self._layer_commands[: layer + 1]:
            found |= learned
        return frozenset(found)

    def _set_layer_commands(self, layer: int, learned: frozenset[str]) -> None:
        """Record the commands layer *layer*'s files define, and the union."""
        self._layer_commands[layer] = learned
        self._commands = self._given_commands.union(*self._layer_commands)

    def _apply(self, file_id: int, label: str, outcome: tuple[Any, ...]) -> None:
        """Add what :func:`_read_file` read of one file, or list it as skipped."""
        if outcome[0] == "skip":
            self._skipped.append((label, outcome[1]))
            return
        _, records, lines, problems, digest = outcome
        for record in records:
            self._add_record(record, file_id)
        self._count_kinds(file_id, Counter(record[0] for record in records))
        self._file_lines[file_id] = lines
        self._digests[file_id] = digest
        if problems:
            self._problems[file_id] = problems

    def _count_kinds(self, file_id: int, counted: Counter[str]) -> None:
        """Record the last file's commands by kind (files are counted in order)."""
        if file_id != len(self._fk_ptr) - 2 or self._fk_ptr[-1] != len(self._fk_data):
            raise RuntimeError("LeanIndex: kinds are counted once, for the last file.")
        for kind in sorted(counted):
            found = self._fk_ids.get(kind)
            if found is None:
                found = self._fk_ids[kind] = len(self._fk_names)
                self._fk_names.append(kind)
            self._fk_data.extend((found, counted[kind]))
        self._fk_ptr[-1] = len(self._fk_data)
        self._kinds.update(counted)

    def _file_kinds(self, file_ids: Iterable[int]) -> Counter[str]:
        """Return the commands of *file_ids* counted by kind."""
        counted: Counter[str] = Counter()
        ptr, data, names = self._fk_ptr, self._fk_data, self._fk_names
        for f in file_ids:
            for i in range(ptr[f], ptr[f + 1], 2):
                counted[names[data[i]]] += data[i + 1]
        return counted

    def add_chunks(self, chunks: Iterable[dict[str, Any]], *, label: str) -> None:
        """
        Index the chunks :class:`~scikitplot.corpus.LeanReader` made of one file.

        Parameters
        ----------
        chunks : iterable of dict
            Raw chunks read with ``references=True`` and ``proofs="keep"``,
            for example from a file inside an archive.
        label : str
            How reports name the file.

        Notes
        -----
        Parts of one long declaration (``part`` 2, 3, ...) join their first
        part. Without a file on disk, reports give locations but no text.
        """
        file_id = self._new_file(label, None)
        last_end = 0
        counted: Counter[str] = Counter()
        for chunk in chunks:
            if chunk.get("part", 0) > 1 and self._name and self._file[-1] == file_id:
                node = len(self._name) - 1
                refs = set(self._refs_of(node)) | set(chunk.get("references", ()))
                del self._ref_data[self._ref_ptr[node] :]  # the last node's
                self._ref_data.extend(self._ref_id(ref) for ref in sorted(refs))
                self._ref_ptr[node + 1] = len(self._ref_data)
                self._end[node] = chunk["line_end"]
                flags = set(self._flags.get(node, ())) | set(chunk.get("flags", ()))
                if flags:
                    self._flags[node] = tuple(sorted(flags))
            else:
                record = _compact(chunk)
                self._add_record(record, file_id)
                counted[record[0]] += 1
            last_end = max(last_end, chunk.get("line_end", 0))
        self._count_kinds(file_id, counted)
        self._file_lines[file_id] = last_end

    def _new_file(self, label: str, path: Path | None, *, layer: int = 0) -> int:
        """
        Register a file and drop every cache that a new declaration can change.

        Name resolution and edges are cached per query; a file added later
        may declare a name an earlier query looked up, so the caches go and
        the next query resolves against the whole index. The names each
        declaration uses are kept, so nothing has to be read again: adding
        files in any order, with queries in between, gives the same index.
        """
        if label in self._label_ids:
            raise ValueError(
                f"LeanIndex: {label!r} is already indexed; give each file one label."
            )
        file_id = len(self._labels)
        if self._labels and label < self._labels[-1]:
            self._in_order = False
        self._generation += 1
        self._label_ids[label] = file_id
        self._labels.append(label)
        self._paths.append(path)
        self._imports.append(())
        self._file_lines.append(0)
        self._layer_of.append(layer)
        self._digests.append(None)
        self._fk_ptr.append(len(self._fk_data))
        self._first_node.append(len(self._name))
        self._by_short = None
        self._edges.clear()
        self._reach.clear()
        self._expanded.clear()
        self._packed = None
        self._follow.clear()
        self._contexts.clear()
        return file_id

    def _canonical(self) -> None:
        """
        Renumber files by label and declarations by file, once, before a query.

        Reports list declarations, break ties and choose among equally short
        paths by their numbers, which follow the order files were added.
        Files added out of label order are renumbered here, so every report
        equals the one :meth:`from_folder` gives: the same files give the
        same reports in any order of adding them. Files added in order (as
        :meth:`from_folder` adds them) are never renumbered.
        """
        if self._in_order:
            return
        self._renumber(sorted(range(len(self._labels)), key=self._labels.__getitem__))

    def _drop_files(self, dropped: set[int]) -> None:
        """
        Remove the files *dropped* and their declarations; renumber the rest.

        What else names a dropped file (``_skipped``, ``_learned``) is the
        caller's to update.
        """
        if not dropped:
            return
        self._kinds.subtract(self._file_kinds(dropped))
        self._kinds = +self._kinds  # no zero counts
        keep = [f for f in range(len(self._labels)) if f not in dropped]
        self._renumber(sorted(keep, key=self._labels.__getitem__))

    def _renumber(self, order: list[int]) -> None:
        """
        Keep the files *order* (sorted by label) and renumber files and declarations.

        Declarations follow their files; a file left out goes with its
        declarations. Every cache is dropped.
        """
        file_of = [-1] * len(self._labels)
        for new, old in enumerate(order):
            file_of[old] = new
        bounds = [*self._first_node, len(self._name)]
        nodes = [n for f in order for n in range(bounds[f], bounds[f + 1])]
        node_of = array("q", [-1]) * len(self._name)
        for new, old in enumerate(nodes):
            node_of[old] = new

        def remap(table: dict[int, Any]) -> dict[int, Any]:
            return {node_of[n]: table[n] for n in nodes if n in table}

        self._name = [self._name[n] for n in nodes]
        self._kind = [self._kind[n] for n in nodes]
        self._namespace = [self._namespace[n] for n in nodes]
        self._opens = [self._opens[n] for n in nodes]
        ptr, data = self._ref_ptr, self._ref_data
        self._ref_ptr, self._ref_data = array("q", [0]), array("i")
        for n in nodes:
            self._ref_data.extend(data[ptr[n] : ptr[n + 1]])
            self._ref_ptr.append(len(self._ref_data))
        for field in ("_start", "_end"):
            column = getattr(self, field)
            setattr(self, field, array(column.typecode, (column[n] for n in nodes)))
        self._file = array(self._file.typecode, (file_of[self._file[n]] for n in nodes))
        self._ctx_of = array(self._ctx_of.typecode, (self._ctx_of[n] for n in nodes))
        self._flags = remap(self._flags)
        self._attributes = remap(self._attributes)
        self._shape_of = remap(self._shape_of)
        self._private = {node_of[n] for n in self._private if node_of[n] >= 0}
        self._claims = {node_of[n] for n in self._claims if node_of[n] >= 0}
        self._vacuous = remap(self._vacuous)
        by_name: dict[str, Any] = {}
        for node, name in enumerate(self._name):  # rebuilt as reading builds it
            if name is None:
                continue
            held = by_name.get(name)
            if held is None:
                by_name[name] = node
            elif isinstance(held, int):
                by_name[name] = [held, node]
            else:
                held.append(node)
        self._by_name = by_name
        first, self._first_node = 0, []
        for f in order:
            self._first_node.append(first)
            first += bounds[f + 1] - bounds[f]
        self._labels = [self._labels[f] for f in order]
        self._paths = [self._paths[f] for f in order]
        self._imports = [self._imports[f] for f in order]
        self._file_lines = [self._file_lines[f] for f in order]
        self._layer_of = [self._layer_of[f] for f in order]
        self._digests = [self._digests[f] for f in order]
        ptr, data = self._fk_ptr, self._fk_data
        self._fk_ptr, self._fk_data = array("q", [0]), array("i")
        for f in order:
            self._fk_data.extend(data[ptr[f] : ptr[f + 1]])
            self._fk_ptr.append(len(self._fk_data))
        self._label_ids = {label: i for i, label in enumerate(self._labels)}
        self._problems = {
            file_of[f]: self._problems[f] for f in order if f in self._problems
        }
        # Every cache holds numbers: dropped.
        self._generation += 1
        self._by_short = None
        self._edges.clear()
        self._reach.clear()
        self._expanded.clear()
        self._packed = None
        self._follow.clear()
        self._contexts.clear()
        self._lines_cache.clear()
        self._in_order = True

    def _ref_id(self, ref: str) -> int:
        """Return *ref*'s number, giving it the next one the first time."""
        found = self._ref_ids.get(ref)
        if found is None:
            found = self._ref_ids[ref] = len(self._ref_names)
            self._ref_names.append(ref)
        return found

    def _context_id(self, namespace: str, opens: tuple[str, ...]) -> int:
        """Return the number of the context *namespace* and *opens*, new the first time."""
        key = (namespace, opens)
        found = self._ctx_ids.get(key)
        if found is None:
            found = self._ctx_ids[key] = len(self._ctx_keys)
            self._ctx_keys.append(key)
        return found

    def _context(self, ctx_id: int) -> tuple[str, tuple[str, ...]]:
        """Return context *ctx_id* as ``(namespace, opens)``."""
        if self._flat is not None:  # a forked worker: from the shared text
            return self._flat.context(ctx_id)
        return self._ctx_keys[ctx_id]

    def _ref_name(self, ref_id: int) -> str:
        """Return reference *ref_id* as text."""
        if self._flat is not None:  # a forked worker: from the shared text
            return self._flat.name(ref_id)
        return self._ref_names[ref_id]

    def _refs_of(self, node: int) -> tuple[str, ...]:
        """Return the names *node*'s code uses, in the order they were recorded."""
        start, stop = self._ref_ptr[node], self._ref_ptr[node + 1]
        return tuple(self._ref_name(i) for i in self._ref_data[start:stop])

    def _interned(self, value: Any) -> Any:
        return self._intern.setdefault(value, value)

    def _add_record(self, record: tuple[Any, ...], file_id: int) -> None:
        """Add one command, as :func:`_compact` recorded it."""
        kind = record[0]
        if kind == "import":
            self._imports[file_id] = tuple(
                dict.fromkeys(self._imports[file_id] + record[1])
            )
            return
        if kind in _NOT_NODES:
            return
        (
            _,
            name,
            start,
            end,
            namespace,
            opens,
            refs,
            flags,
            attrs,
            private,
            shape,
            claim,
            vacuous,
        ) = record
        node = len(self._name)
        self._name.append(name)
        self._kind.append(sys.intern(kind))
        self._file.append(file_id)
        self._start.append(start)
        self._end.append(end)
        self._namespace.append(sys.intern(namespace))
        self._opens.append(self._interned(opens))
        self._ctx_of.append(self._context_id(self._namespace[-1], self._opens[-1]))
        self._ref_data.extend(self._ref_id(ref) for ref in refs)
        self._ref_ptr.append(len(self._ref_data))
        if flags:
            self._flags[node] = flags
        if attrs:
            self._attributes[node] = attrs
        if private:
            self._private.add(node)
        if shape is not None:
            self._shape_of[node] = shape
        if claim:
            self._claims.add(node)
        if vacuous is not None:
            self._vacuous[node] = vacuous
        if name is not None:
            held = self._by_name.get(name)
            if held is None:
                self._by_name[name] = node
            elif isinstance(held, int):
                self._by_name[name] = [held, node]
            else:
                held.append(node)

    # -- resolving names ----------------------------------------------------

    def _clashes(self) -> list[tuple[str, list[int]]]:
        """
        Return the names declared more than once in a way Lean would refuse together.

        A ``private`` name is local to its file, so it clashes only within
        one file; public names clash wherever two of them meet. Two public
        declarations in modules that are never imported together build
        fine apart: they are listed, as parallel declarations of one name.
        """
        key = self._generation
        cached = getattr(self, "_clash_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        clashes = []
        for name, held in self._by_name.items():
            if isinstance(held, int):
                continue
            public = [n for n in held if n not in self._private]
            per_file = Counter(self._file[n] for n in held)
            if len(public) > 1 or max(per_file.values()) > 1:
                clashes.append((name, list(held)))
        clashes.sort()
        self._clash_cache = (key, clashes)
        return clashes

    def _node_at(self, file_id: int, line: int) -> int | None:
        """Return the node of *file_id* whose lines hold *line*, or ``None``."""
        first = self._first_node[file_id]
        last = (
            self._first_node[file_id + 1]
            if file_id + 1 < len(self._first_node)
            else len(self._name)
        )
        starts = self._start[first:last]
        i = bisect.bisect_right(starts, line) - 1
        if i >= 0 and self._start[first + i] <= line <= self._end[first + i]:
            return first + i
        return None

    def _scope(self, layer: str | None) -> frozenset[int] | None:
        """
        Return the files of the layer named *layer*, or ``None`` for every file.

        Raises
        ------
        ValueError
            If *layer* is not ``None`` and names no layer (``""`` is the first).
        """
        if layer is None:
            return None
        names = [name for name, _root in self._layers]
        if not isinstance(layer, str) or layer not in names:
            raise ValueError(
                f"LeanIndex: layer must be None or one of {names!r}; got {layer!r}."
            )
        at = names.index(layer)
        return frozenset(f for f, held in enumerate(self._layer_of) if held == at)

    def _layer_of_label(self, label: str) -> int:
        """Return the layer of a file label, indexed or skipped (by its prefix)."""
        for at in range(len(self._layers) - 1, 0, -1):
            if label.startswith(f"{self._layers[at][0]}:"):
                return at
        return 0

    def _module_files(self) -> dict[str, int]:
        """
        Map each module the files provide to its file, at each layer's source root.

        Lean names a module by its path from the package's source root:
        ``Mathlib/Data/Nat.lean`` is ``Mathlib.Data.Nat`` from the
        repository root, ``src/A/B.lean`` is ``A.B`` under ``src/``. The
        root is found once per layer (:meth:`add_folder`), as the number of
        leading folders to strip (0, 1, ...) under which the most imports
        name a file exactly; ties go to the fewest. Only imports from the
        layer itself and the layers above it count, and none a lower layer
        provides: a project's copy of ``Mathlib/...`` under ``PFR/`` is not
        taken for its source root because Mathlib imports Mathlib. A module is never
        matched by a tail of its path alone, so a core ``import
        Lean.Elab.Term`` is not taken for ``Mathlib/Lean/Elab/Term.lean``.
        Of two files giving one module, the first in label order provides
        it within a layer, and the lower layer across layers (the others
        are listed by :meth:`_module_clashes`).
        """
        key = self._generation
        cached = getattr(self, "_module_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        modules: dict[str, int] = {}
        clashes: list[tuple[str, int, int]] = []
        for layer, (name, source_root) in enumerate(self._layers):
            strip = len(name) + 1 if name else 0
            files = [f for f, at in enumerate(self._layer_of) if at == layer]
            # A layer is imported by itself and the layers above it, never by
            # one below; imports a lower layer already provides do not count.
            imported = Counter(
                module
                for f, at in enumerate(self._layer_of)
                if at >= layer
                for module in self._imports[f]
                if module not in modules
            )
            found = _layer_modules(
                [(f, self._labels[f][strip:]) for f in files], source_root, imported
            )
            for module, file_id in found.items():
                first = modules.setdefault(module, file_id)
                if first != file_id:
                    clashes.append((module, first, file_id))
        self._module_cache = (key, modules)
        self._clash_modules = (key, clashes)
        return modules

    def _module_clashes(self) -> list[tuple[str, int, int]]:
        """Return ``(module, file used, file also giving it)`` across layers."""
        self._module_files()
        return self._clash_modules[1]

    def _sees(self, file_id: int) -> int:
        """
        Return the files *file_id* can see, as a bitset (bit ``g`` for file ``g``).

        Its own file and every project file it imports, directly or through
        others. ``-1`` (every bit) with ``visibility="all"``, or when the file
        or one it imports has a project import no file provides: what it
        sees is then unknown, and nothing is hidden. Computed on first use
        for the files a query reaches, iteratively (no recursion limit),
        and kept until a file is added.

        Each bitset is one bit per project file, so all of them together
        grow with the square of the file count (3.85 GB for 175,187 files).
        Past :data:`_COMPRESS_FILES` files they are kept zlib-compressed
        (18 times smaller on that project; sparse bitsets compress well)
        and expanded on use, the last :data:`_EXPANDED` of them kept
        expanded: declarations are resolved file by file.
        """
        if self._visibility == "all":
            return -1
        expanded = self._expanded
        bits = expanded.get(file_id)
        if bits is not None:
            expanded.move_to_end(file_id)
            return bits
        reach = self._reach
        if file_id not in reach:
            self._close_imports(file_id)
        return self._expand(file_id)

    def _expand(self, file_id: int) -> int:
        """Return the stored bitset of *file_id* as an int, through the small cache."""
        expanded = self._expanded
        bits = expanded.get(file_id)
        if bits is not None:
            expanded.move_to_end(file_id)
            return bits
        stored = self._reach[file_id]
        bits = (
            int.from_bytes(zlib.decompress(stored), "little")
            if isinstance(stored, bytes)
            else stored
        )
        expanded[file_id] = bits
        if len(expanded) > _EXPANDED:
            expanded.popitem(last=False)
        return bits

    def _store(self, file_id: int, bits: int) -> None:
        """Keep a finished bitset, compressed for a large project."""
        if len(self._labels) > _COMPRESS_FILES and bits > 0:
            raw = bits.to_bytes((bits.bit_length() + 7) // 8, "little")
            self._reach[file_id] = zlib.compress(raw, 1)
        else:
            self._reach[file_id] = bits

    def _close_imports(self, file_id: int) -> None:
        """Compute and store the bitsets of *file_id* and every file it imports."""
        reach = self._reach
        direct, unknown = self._import_files()
        stack = [(file_id, 0)]
        reach[file_id] = (
            0  # in progress: an import cycle (refused by Lean) adds nothing
        )
        while stack:
            current, i = stack.pop()
            children = direct[current]
            if i < len(children):
                stack.append((current, i + 1))
                child = children[i]
                if child not in reach:
                    reach[child] = 0
                    stack.append((child, 0))
                continue
            bits = 1 << current
            if current in unknown:
                bits = -1
            else:
                for child in children:
                    if reach[child] == 0:  # in progress (a cycle)
                        bits |= 1 << child
                        continue
                    bits |= (1 << child) | self._expand(child)
            self._store(current, bits)

    def _import_files(self) -> tuple[list[list[int]], set[int]]:
        """
        Return each file's directly imported project files, and the files whose imports are unknown.

        An import is unknown when no file provides it yet it names the
        project: its first component is that of a provided module (a
        missing module), or it is a provided module with leading components
        added (the index was built from a subfolder: ``Mathlib/`` provides
        ``Algebra.X``, the files import ``Mathlib.Algebra.X``).
        """
        key = self._generation
        cached = getattr(self, "_import_files_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        modules = self._module_files()
        _, roots = self._modules()
        direct: list[list[int]] = []
        unknown: set[int] = set()
        for file_id, imports in enumerate(self._imports):
            found = []
            for module in imports:
                provider = modules.get(module)
                if provider is not None:
                    found.append(provider)
                    continue
                if _names_the_project(module, modules, roots):
                    unknown.add(file_id)
            direct.append(found)
        self._import_files_cache = (key, direct, unknown)
        return direct, unknown

    def _imported(self, file_id: int) -> dict[int, int]:
        """Return the project files *file_id* imports, transitively: each to the one importing it."""
        modules = self._module_files()
        seen = {file_id: -1}
        queue = deque([file_id])
        while queue:
            current = queue.popleft()
            for module in self._imports[current]:
                imported = modules.get(module)
                if imported is not None and imported not in seen:
                    seen[imported] = current
                    queue.append(imported)
        return seen

    def _toward(self, targets: Sequence[int]) -> dict[int, tuple[int, int]]:
        """
        Return every node that reaches one of *targets*, to its next step toward one.

        Breadth-first along the edges reversed, so each node maps to
        ``(next, how)``: the node it names on a shortest path to a target,
        and how that name was found; a target maps to ``(-1, -1)``. The
        reversed edges are built for the call, in two passes over the edges
        (compressed rows of 4-byte node numbers), then dropped: time and
        memory linear in the edges.
        """
        if self._graph != "python":
            from ._graph_numpy import reverse  # noqa: PLC0415

            indptr, codes = self._pack()
            rev_indptr, rev_codes = reverse(indptr, codes, len(self._name))
            return self._backend().breadth_first(
                rev_indptr, rev_codes, targets, len(self._name)
            )
        n = len(self._name)
        self._resolve_all()
        offsets = array("q", bytes(8 * (n + 1)))
        for node in range(n):
            for code in self._edges_of(node):
                offsets[code // 4 + 1] += 1
        for i in range(n):
            offsets[i + 1] += offsets[i]
        sources = array("i", bytes(4 * offsets[n]))
        fill = array("q", offsets[:n])
        for node in range(n):
            for code in self._edges_of(node):
                target = code // 4
                sources[fill[target]] = node * 4 + code % 4
                fill[target] += 1
        del fill
        toward: dict[int, tuple[int, int]] = dict.fromkeys(targets, (-1, -1))
        queue = deque(targets)
        while queue:
            target = queue.popleft()
            for i in range(offsets[target], offsets[target + 1]):
                source, how = divmod(sources[i], 4)
                if source not in toward:
                    toward[source] = (target, how)
                    queue.append(source)
        return toward

    def _steps(
        self, start: int, toward: dict[int, tuple[int, int]]
    ) -> tuple[list[dict[str, Any]], int]:
        """Return the hops from *start* to a target along :meth:`_toward`'s map, and the target."""
        hops = []
        node = start
        while True:
            following, how = toward[node]
            if following < 0:
                return hops, node
            hops.append(
                {
                    "declaration": self._name[following],
                    "file": self._labels[self._file[following]],
                    "line_start": self._start[following],
                    "found_by": _HOW[how],
                    "from": self._name[node],
                }
            )
            node = following

    def _generators(self) -> list[dict[str, Any]]:
        """
        Return the commands that add declarations no source line spells.

        An invocation of a learned project command, or a ``run_cmd``,
        ``#eval`` or ``initialize``, whose code reaches a declaration
        flagged ``add_decl`` (or one building an axiom with ``axiomDecl``):
        what it adds exists only when Lean builds the file. Free when no
        declaration is so flagged; otherwise one search back from the
        flagged ones, so the cost does not grow with the invocations.
        """
        key = self._generation
        cached = getattr(self, "_generator_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        adders = [
            n
            for n in sorted(self._flags)
            if set(self._flags[n]) & _ADDS and self._kind[n] != "axiom"
        ]
        found: list[dict[str, Any]] = []
        if adders:
            definitions: dict[str, int] = {}
            label_ids = {label: i for i, label in enumerate(self._labels)}
            for keyword, (label, line) in self._learned.items():
                file_id = label_ids.get(label)
                node = self._node_at(file_id, line) if file_id is not None else None
                if node is not None:
                    definitions[keyword] = node
            toward = self._toward(adders)
            for node, kind in enumerate(self._kind):
                if kind in definitions:
                    start = definitions[kind]
                elif kind in _RUNS_CODE:
                    start = node
                else:
                    continue
                if start not in toward:
                    continue
                hops, adder = self._steps(start, toward)
                found.append(
                    {
                        "invocation": self._location(node),
                        "definition": self._location(start) if start != node else None,
                        "adds_declarations_at": {**self._location(adder), "path": hops},
                    }
                )
        self._generator_cache = (key, found)
        return found

    def _ids(self, name: str) -> tuple[int, ...]:
        held = self._by_name.get(name)
        if held is None:
            return ()
        return (held,) if isinstance(held, int) else tuple(held)

    def _short(self) -> dict[str, Any]:
        if self._by_short is None:
            short: dict[str, Any] = {}
            for name, held in self._by_name.items():
                last = name.rsplit(".", 1)[-1]
                ids = (held,) if isinstance(held, int) else tuple(held)
                short[last] = short.get(last, ()) + ids
            self._by_short = short
        return self._by_short

    def _lookup(
        self, ref: str, prefixes: Sequence[str], opens: Sequence[str]
    ) -> tuple[int, ...]:
        if ref.startswith(_ROOT):
            return self._ids(ref[len(_ROOT) :])
        hits: list[int] = []
        for prefix in prefixes:
            hits.extend(self._ids(f"{prefix}.{ref}"))
        hits.extend(self._ids(ref))
        for opened in opens:
            hits.extend(self._ids(f"{opened}.{ref}"))
        return tuple(dict.fromkeys(hits))

    def _resolve(
        self, ref: str, namespace: str, opens: tuple[str, ...]
    ) -> tuple[int, ...]:
        """Return ``node * 4 + how`` codes for the declarations *ref* may name."""
        ref_id = self._ref_ids.get(ref)
        ctx_id = self._ctx_ids.get((namespace, opens))
        if ref_id is not None and ctx_id is not None:
            return self._resolve_id(ref_id, ctx_id)
        return self._resolve_name(
            ref, namespace, opens
        )  # a name or context no code used

    def _resolve_id(self, ref_id: int, ctx_id: int) -> tuple[int, ...]:
        """:meth:`_resolve` of a reference number in a context number, cached."""
        key = ctx_id
        cache = self._contexts.get(key)
        if cache is None:
            cache = self._contexts[key] = {}
            if len(self._contexts) > _CONTEXT_CACHE:
                self._contexts.popitem(last=False)
        else:
            self._contexts.move_to_end(key)
        hit = cache.get(ref_id)  # an int key: a hit reads no string
        if hit is None:
            namespace, opens = self._context(ctx_id)
            hit = cache[ref_id] = self._resolve_name(
                self._ref_name(ref_id), namespace, opens
            )
        return hit

    def _resolve_name(
        self, ref: str, namespace: str, opens: tuple[str, ...]
    ) -> tuple[int, ...]:
        """Resolve *ref* in a context, uncached: the lookups :meth:`_resolve` names."""
        parts = namespace.split(".") if namespace else []
        prefixes = [".".join(parts[:i]) for i in range(len(parts), 0, -1)]
        codes: tuple[int, ...] = ()
        # `.card`, `.inl`: a field or leading-dot name, its head unknown: no
        # lookup can find it, only its components' short names.
        found = () if ref[0] == "." else self._lookup(ref, prefixes, opens)
        if found:
            codes = tuple(node * 4 for node in found)
        elif ref[0] != ".":
            head = ref
            while "." in head and not codes:
                head = head.rsplit(".", 1)[0]
                found = self._lookup(head, prefixes, opens)
                codes = tuple(node * 4 + 1 for node in found)
        if not codes and self._suffix == "unique" and "." in ref:
            # Dot notation, `h.lemma.trans`: each component after the
            # first may name a lemma; link those one declaration has.
            short = self._short()
            found = [
                ids[0]
                for component in ref.split(".")[1:]
                for ids in (short.get(component, ()),)
                if len(ids) == 1
            ]
            codes = tuple(node * 4 + 2 for node in dict.fromkeys(found))
        return codes

    def _edges_of(self, node: int) -> array[int]:
        """Return *node*'s edges as ``target * 4 + how``, resolving them once."""
        if self._packed is not None:
            indptr, codes = self._packed
            return codes[indptr[node] : indptr[node + 1]].tolist()
        edges = self._edges.get(node)
        if edges is not None:
            return edges
        edges = self._compute_edges(node)
        self._edges[node] = edges
        return edges

    def _compute_edges(self, node: int) -> array[int]:
        """Resolve *node*'s edges as ``target * 4 + how``, sorted; nothing is cached."""
        ids = self._ref_data[self._ref_ptr[node] : self._ref_ptr[node + 1]]
        ctx_id = self._ctx_of[node]
        file_id = self._file[node]
        sees = self._sees(file_id)
        best: dict[int, int] = {}
        for ref_id in ids:
            for code in self._resolve_id(ref_id, ctx_id):
                target, how = divmod(code, 4)
                if target == node:
                    continue
                if not (sees >> self._file[target]) & 1:
                    continue  # in a file this one does not import
                if target in self._private and self._file[target] != file_id:
                    continue  # a private declaration is seen from its own file only
                if how < best.get(target, 3):
                    best[target] = how
        return array("l", sorted(target * 4 + how for target, how in best.items()))

    def _resolve_all(self) -> None:
        """
        Resolve every edge not resolved yet, in worker processes when they help.

        Called by each step that needs every edge. With ``workers`` of 1, or
        where :func:`._parallel.usable` says no, this does nothing and the
        step resolves edges as it reads them; the edges are the same.
        """
        if self._packed is not None:
            return
        missing = [n for n in range(len(self._name)) if n not in self._edges]
        if not _parallel.usable(self._workers, len(missing)):
            return
        for file_id in range(len(self._labels)):  # what every worker reads,
            self._sees(file_id)  # built once here, not once per worker
        self._short()
        flat = _parallel.FlatNames(self._ref_names, self._ctx_keys)
        self._edges.update(_parallel.resolve(self, missing, self._workers, flat))

    def _pack(self) -> tuple[Any, Any]:
        """
        Resolve every edge and pack them into CSR arrays (``graph="numpy"``).

        The per-declaration arrays are dropped once packed; a file added
        later drops the packing, as it drops every edge.
        """
        if self._packed is None:
            from ._graph_numpy import pack  # noqa: PLC0415

            self._resolve_all()
            # Edges already resolved are reused; the rest go straight into
            # the packed arrays without being kept one array per node, so
            # the peak holds the packed edges, not both copies.
            cached = self._edges

            def edges_of(node: int) -> Any:
                found = cached.pop(node, None)
                return found if found is not None else self._compute_edges(node)

            self._packed = pack(edges_of, len(self._name))
            self._edges = {}
        return self._packed

    def _backend(self) -> Any:
        """Return the module of the packed backend (``numpy`` or ``scipy``)."""
        if self._graph == "scipy":
            from . import _graph_scipy  # noqa: PLC0415

            return _graph_scipy
        from . import _graph_numpy  # noqa: PLC0415

        return _graph_numpy

    def _closure(
        self, roots: Sequence[int], *, expand: frozenset[str] | None = None
    ) -> dict[int, tuple[int, int]]:
        """Breadth-first: each reached node to ``(parent, how)``; roots to ``(-1, -1)``."""
        if self._packed is not None:
            # Edges already packed by a whole-graph step: search vectorised.
            # (A dossier alone keeps the lazy search: packing resolves every
            # edge of the project, far more than one closure needs.)
            import numpy as np  # noqa: PLC0415

            breadth_first = self._backend().breadth_first
            follow = None
            if expand is not None:
                follow = self._follow.get(expand)
                if follow is None:  # one pass over the kinds per kind set
                    follow = np.fromiter(
                        (kind in expand for kind in self._kind),
                        dtype=bool,
                        count=len(self._kind),
                    )
                    self._follow[expand] = follow
            indptr, codes = self._packed
            return breadth_first(indptr, codes, roots, len(self._name), follow=follow)
        reached: dict[int, tuple[int, int]] = dict.fromkeys(roots, (-1, -1))
        queue = deque(roots)
        while queue:
            node = queue.popleft()
            if expand is not None and self._kind[node] not in expand:
                continue  # reached, but its body is not followed
            for code in self._edges_of(node):
                target, how = divmod(code, 4)
                if target not in reached:
                    reached[target] = (node, how)
                    queue.append(target)
        return reached

    # -- reading text back --------------------------------------------------

    def _lines(self, file_id: int) -> list[str] | None:
        lines = self._lines_cache.get(file_id)
        if lines is not None:
            self._lines_cache.move_to_end(file_id)
            return lines
        path = self._paths[file_id]
        if path is None:
            return None
        try:
            with open(path, encoding="utf-8", errors="replace", newline=None) as stream:
                lines = [line[:-1] if line.endswith("\n") else line for line in stream]
        except OSError:
            return None  # gone since indexing: locations without quoted text
        if lines and lines[0].startswith("﻿"):
            lines[0] = lines[0][1:]
        self._lines_cache[file_id] = lines
        if len(self._lines_cache) > _LINES_CACHE:
            self._lines_cache.popitem(last=False)
        return lines

    def _excerpt(self, node: int, max_lines: int = 12) -> tuple[str | None, bool]:
        lines = self._lines(self._file[node])
        if lines is None:
            return None, False
        start, end = self._start[node], self._end[node]
        stop = min(end, start + max_lines - 1)
        return "\n".join(lines[start - 1 : stop]), stop < end

    def _location(self, node: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "declaration": self._name[node],
            "kind": self._kind[node],
            "file": self._labels[self._file[node]],
            "line_start": self._start[node],
            "line_end": self._end[node],
        }
        if node in self._flags:
            item["flags"] = list(self._flags[node])
        return item

    # -- questions ----------------------------------------------------------

    def _nodes(self, name: str) -> list[int]:
        """Return the nodes named *name*, or whose name ends with ``.name``."""
        nodes = list(self._ids(name))
        if not nodes:
            ending = "." + name
            nodes = [
                node
                for node, declared in enumerate(self._name)
                if declared is not None and declared.endswith(ending)
            ]
        return sorted(nodes)

    def find(self, name: str) -> list[dict[str, Any]]:
        """
        Return the declarations named *name*, or whose name ends with ``.name``.

        Parameters
        ----------
        name : str
            A full name (``Foo.bar``) or its last components (``bar``).

        Returns
        -------
        list of dict
            Their locations, in reading order.
        """
        self._canonical()
        return [self._location(node) for node in self._nodes(name)]

    def summary(  # noqa: PLR0912
        self, *, max_items: int = 50, layer: str | None = None
    ) -> dict[str, Any]:
        """
        Describe the project: sizes, kinds, flags, duplicates, imports, extensions, problems.

        Parameters
        ----------
        max_items : int, optional
            Most examples listed per finding; every list says how many there
            are in all, and whether it was cut. Default: 50.
        layer : str or None, optional
            Describe only the files of this layer (:meth:`add_folder`; ``""``
            is the first): a project without the dependencies below it.
            ``None`` (default) describes every file.

        Returns
        -------
        dict
            JSON-ready; ``report`` is ``"summary"``, ``layer`` the layer
            described (``None``: all).

        Raises
        ------
        ValueError
            If *max_items* is not a positive integer or *layer* names no
            layer.

        Notes
        -----
        **User.** A layer's summary counts and lists what its own files
        hold. Names still resolve through every layer, so a duplicate
        name or a module clash is listed when one of its files is in the
        layer, with all of its places, and a missing module when a file of
        the layer imports it.
        """
        self._canonical()
        _positive(max_items, "max_items")
        scope = self._scope(layer)

        def inside(file_id: int) -> bool:
            return scope is None or file_id in scope

        flagged: dict[str, list[int]] = {}
        for node in sorted(self._flags):
            if inside(self._file[node]):
                for flag in self._flags[node]:
                    flagged.setdefault(flag, []).append(node)
        duplicates = [
            (name, held)
            for name, held in self._clashes()
            if any(inside(self._file[n]) for n in held)
        ]
        extensions = [
            node
            for node, kind in enumerate(self._kind)
            if kind in _IMPLICIT and kind != "instance" and inside(self._file[node])
        ]
        missing, external = self._import_gaps(scope)
        if scope is None:
            files = range(len(self._labels))
            kinds = self._kinds
            declarations = len(self._name)
            at = None
        else:
            files = sorted(scope)
            at = [name for name, _root in self._layers].index(layer)
            kinds = self._file_kinds(files)
            declarations = sum(1 for f in self._file if f in scope)
        labels = {self._labels[f] for f in files}
        return {
            "report": "summary",
            "producer": PRODUCER,
            "visibility": self._visibility,
            "layer": layer,
            "files": len(files),
            "lines": sum(self._file_lines[f] for f in files),
            "declarations": declarations,
            "kinds": dict(sorted(kinds.items())),
            "flags": {
                flag: _listing([self._location(n) for n in nodes], max_items)
                for flag, nodes in sorted(flagged.items())
            },
            "duplicate_names": _listing(
                [
                    {"declaration": name, "at": [self._location(n) for n in held]}
                    for name, held in duplicates
                ],
                max_items,
            ),
            "imports": {
                "external_roots": dict(sorted(external.items())),
                "missing_modules": _listing(
                    [
                        {"module": module, "imported_by": sorted(set(files))}
                        for module, files in sorted(missing.items())
                    ],
                    max_items,
                ),
                "module_clashes": _listing(
                    [
                        {
                            "module": module,
                            "used": self._labels[used],
                            "also_in": self._labels[other],
                        }
                        for module, used, other in self._module_clashes()
                        if inside(used) or inside(other)
                    ],
                    max_items,
                ),
            },
            "syntax_extensions": self._extension_report(extensions, max_items),
            "learned_commands": {
                keyword: {"file": label, "line": line}
                for keyword, (label, line) in sorted(self._learned.items())
                if at is None or self._layer_of_label(label) == at
            },
            "generators": _listing(
                [
                    item
                    for item in self._generators()
                    if item["invocation"]["file"] in labels
                ],
                max_items,
            ),
            "skipped_files": [
                {"file": label, "reason": reason}
                for label, reason in sorted(self._skipped)
                if at is None or self._layer_of_label(label) == at
            ],
            "source_problems": _listing(
                self._problem_items(f for f in self._problems if inside(f)), max_items
            ),
        }

    def _extension_report(self, nodes: list[int], max_items: int) -> dict[str, Any]:
        """
        Describe the syntax extensions: every one, how many of each kind, the flagged.

        Lean applies an extension (notation, macro, elaborator) without its
        name appearing where it is used, so a flagged one can bring ``sorry``
        into a proof that never names it: those are listed apart, each with
        its flags. A location is built only for an item that is listed.
        """
        flagged = [n for n in nodes if n in self._flags]
        return {
            "count": len(nodes),
            "items": [self._location(n) for n in nodes[:max_items]],
            "cut": len(nodes) > max_items,
            "by_kind": dict(sorted(Counter(self._kind[n] for n in nodes).items())),
            "flagged": {
                "count": len(flagged),
                "items": [self._location(n) for n in flagged[:max_items]],
                "cut": len(flagged) > max_items,
            },
        }

    def _problem_items(self, file_ids: Iterable[int]) -> list[dict[str, Any]]:
        """Return the source problems of *file_ids*, by file and line."""
        return [
            {"file": self._labels[file_id], "line": line, "problem": problem}
            for file_id in sorted(file_ids)
            if file_id in self._problems
            for line, problem in self._problems[file_id]
        ]

    def dossier(  # noqa: PLR0912, PLR0915
        self,
        target: str,
        *,
        max_items: int = 50,
    ) -> dict[str, Any]:
        """
        Gather the evidence on one declaration: its statement, its dependencies, its risks.

        Parameters
        ----------
        target : str
            The declaration: its full name, or its last components when
            those name exactly one declaration.
        max_items : int, optional
            Most items listed per finding (counts are always complete).
            Default: 50.

        Returns
        -------
        dict
            JSON-ready; ``report`` is ``"dossier"``. ``target`` (location,
            statement), ``statement`` (``local_definitions`` with their
            text, ``external_names``), ``dependencies`` (counts and the
            ``red_flags`` with their paths), ``implicit_risks``,
            ``checks`` (each with ``status`` ``pass``, ``review``,
            ``fail`` or ``not_checked``), ``verdict`` and ``kernel_steps``.
            When *target* names no declaration, or several, ``matches``
            lists them and nothing else is filled in.
        """
        self._canonical()
        _positive(max_items, "max_items")
        nodes = self._nodes(target)
        report: dict[str, Any] = {
            "report": "dossier",
            "producer": PRODUCER,
            "visibility": self._visibility,
            "query": target,
            "edge": {
                "relation": RELATION,
                "trust": TRUST,
            },
        }
        if len(nodes) != 1:
            report["matches"] = [self._location(n) for n in nodes]
            report["verdict"] = "target not found" if not nodes else "target ambiguous"
            report["flags_found"] = []
            return report
        node = nodes[0]
        statement, statement_refs = self._statement(node)
        report["target"] = {**self._location(node), "statement": statement}

        # What the statement means: the project definitions it unfolds to.
        bound = _bound_names(statement or "")
        local: dict[int, int] = {}
        external: set[str] = set()
        for ref in sorted(statement_refs):
            if ref.split(".", 1)[0] in bound:
                continue
            codes = self._resolve(ref, self._namespace[node], self._opens[node])
            if not codes:
                external.add(ref)
            for code in codes:
                target_node, how = divmod(code, 4)
                if target_node != node and how < local.get(target_node, 3):
                    local[target_node] = how
        meaning = self._closure(sorted(local), expand=_DEFINITIONS)
        definitions = []
        for item in sorted(meaning, key=lambda n: (self._file[n], self._start[n])):
            entry = self._location(item)
            entry["found_by"] = (
                _HOW[local[item]] if item in local else _HOW[meaning[item][1]]
            )
            entry["text"], entry["text_cut"] = self._excerpt(item)
            if item in self._vacuous:
                entry["vacuous"] = self._vacuous[item]
            definitions.append(entry)
        report["statement"] = {
            "local_definitions": _listing(definitions, max_items),
            "external_names": _listing(sorted(external), max_items),
        }

        # What the proof rests on: every declaration it names, transitively.
        reached = self._closure([node])
        hows: Counter[str] = Counter(
            _HOW[how] for _parent, how in reached.values() if how >= 0
        )
        flagged = [n for n in reached if n in self._flags and n != node]
        red_flags = []
        for item in sorted(flagged, key=lambda n: (self._file[n], self._start[n])):
            entry = self._location(item)
            entry["path"] = self._path(item, reached)
            red_flags.append(entry)
        if node in self._flags:
            red_flags.insert(0, {**self._location(node), "path": []})
        clashing = {name for name, _held in self._clashes()}
        duplicates = sorted(
            {self._name[n] for n in reached if self._name[n] in clashing}
        )
        report["dependencies"] = {
            "declarations": len(reached) - 1,
            "files": len({self._file[n] for n in reached if n != node}),
            "found_by": {how: hows.get(how, 0) for how in _HOW},
            "red_flags": _listing(red_flags, max_items),
            "duplicate_names": _listing(duplicates, max_items),
        }

        # What could be used without being named: flagged instances, tagged
        # lemmas and syntax extensions, and declarations generated by code
        # that runs, in what the target's environment holds: its own file
        # before it, and the files it can see (``visibility``).
        home = self._file[node]
        sees = self._sees(home)

        def in_scope(file_id: int, line: int) -> bool:
            if file_id == home:
                return (
                    line < self._start[node]
                )  # later in its own file: not yet declared
            return bool((sees >> file_id) & 1)

        implicit = [
            n
            for n in sorted(self._flags)
            if n not in reached
            and (self._kind[n] in _IMPLICIT or n in self._attributes)
            and in_scope(self._file[n], self._start[n])
        ]
        risks: list[dict[str, Any]] = [
            {"risk": "used_without_name", **self._location(n)} for n in implicit
        ]
        imported = self._imported(home)
        generated = []
        for generator in self._generators():
            invocation = generator["invocation"]
            file_id = self._label_ids[invocation["file"]]
            if not in_scope(file_id, invocation["line_start"]):
                continue
            chain = []  # empty when the imports that reach it are unknown
            current = file_id if file_id in imported else -1
            while current != -1:
                chain.append(self._labels[current])
                current = imported[current]
            generated.append(
                {
                    "risk": "generated_declarations",
                    **generator,
                    "via_imports": chain[::-1],
                }
            )
        risks += generated
        report["implicit_risks"] = _listing(risks, max_items)

        missing = len(self._import_gaps()[0])
        # Each detail is chosen in a statement: a conditional expression as
        # an argument is laid out differently by black and ruff format.
        if definitions:
            on_definitions = (
                f"{len(definitions)} project definition(s) give the statement its"
                " meaning: read each"
            )
        else:
            on_definitions = "the statement uses no project definition"
        vacuous = [d for d in definitions if "vacuous" in d]
        if vacuous:
            on_vacuous = (
                f"{len(vacuous)} definition(s) the statement unfolds to say nothing"
                f" about their subject ({', '.join(d['declaration'] for d in vacuous)}):"
                " the statement may hold whatever it is about"
            )
        else:
            on_vacuous = "no definition the statement unfolds to is vacuous"
        if red_flags:
            on_named = (
                f"{len(red_flags)} flagged declaration(s) among those the proof names"
            )
        else:
            on_named = "no flagged declaration among those the proof names"
        if risks:
            on_implicit = (
                f"{len(implicit)} flagged instance(s), tagged lemma(s) or syntax"
                f" extension(s) could be used without being named; {len(generated)}"
                " command(s) in imported modules add declarations no source spells"
            )
        else:
            on_implicit = "no flagged declaration of a kind used without its name"
        if duplicates:
            on_duplicates = f"{len(duplicates)} name(s) declared more than once among the dependencies"
        else:
            on_duplicates = "every dependency's name is declared once"
        if missing:
            imports_status = "fail"
            on_imports = f"{missing} imported module(s) missing from the project"
        elif self._skipped:
            imports_status = "review"
            on_imports = f"{len(self._skipped)} file(s) could not be read"
        else:
            imports_status = "pass"
            on_imports = "every imported project module is present"
        problems = self._problem_items({home} | {self._file[n] for n in reached})
        report["source_problems"] = _listing(problems, max_items)
        if problems:
            on_source = (
                f"{len(problems)} place(s) Lean refuses to read in the files the"
                " proof depends on (see source_problems): names after them may be"
                " placed in the wrong scope"
            )
        else:
            on_source = "the files the proof depends on read as Lean reads them"
        checks = [
            _check(
                "statement_definitions",
                "review" if definitions else "pass",
                on_definitions,
            ),
            _check("vacuous_definitions", "review" if vacuous else "pass", on_vacuous),
            _check("named_dependencies", "fail" if red_flags else "pass", on_named),
            _check("implicit_dependencies", "review" if risks else "pass", on_implicit),
            _check(
                "duplicate_names", "review" if duplicates else "pass", on_duplicates
            ),
            _check("imported_modules_present", imports_status, on_imports),
            _check("source_text", "review" if problems else "pass", on_source),
            _check(
                "kernel",
                "not_checked",
                "only Lean's kernel decides whether the proof holds",
            ),
        ]
        report["checks"] = checks
        report["flags_found"] = sorted(
            {flag for n in flagged + implicit for flag in self._flags[n]}
            | set(self._flags.get(node, ()))
            | {
                flag
                for item in generated
                for flag in item["adds_declarations_at"].get("flags", ())
            }
        )
        failed = [c["check"] for c in checks if c["status"] == "fail"]
        report["verdict"] = (
            "lexical red flags: " + ", ".join(failed)
            if failed
            else "no lexical red flags; kernel check pending"
        )
        name = self._name[node]
        report["kernel_steps"] = [
            "lake build",
            f"#print axioms {name}  -- expect only propext, Classical.choice, Quot.sound",
            "lean4checker (replay every declaration through the kernel)",
        ]
        return report

    def _statement(self, node: int) -> tuple[str | None, set[str]]:
        """Return the target's statement (proof left out) and the names it uses."""
        path = self._paths[self._file[node]]
        if path is None:
            return None, set(self._refs_of(node))
        splitter = _Splitter(
            file_name=self._labels[self._file[node]],
            meta={},
            step=None,
            cap=None,
            proofs="drop",
            dialect=self._dialect,
            kinds=None,
            watch=(),
            commands=self._commands,
            max_scope_depth=self._max_depth,
            flag_tokens={},
        )
        text = None
        try:
            with open(path, encoding="utf-8", errors="replace", newline=None) as stream:
                for number, read in enumerate(stream, start=1):
                    line = read[:-1] if read.endswith("\n") else read
                    if number == 1 and line.startswith("﻿"):
                        line = line[1:]
                    for chunk in splitter.feed(number, line):
                        if chunk["line_start"] == self._start[node]:
                            text = chunk["text"]
                    if text is not None:
                        break
                else:
                    for chunk in splitter.finish():
                        if chunk["line_start"] == self._start[node]:
                            text = chunk["text"]
        except (OSError, ValueError):
            text = None  # gone or changed since indexing
        if text is None:
            # The file is gone or changed since it was indexed: no quoted
            # statement; the names its code used, as for chunks without a file.
            return None, set(self._refs_of(node))
        lexer = _Lexer()
        refs: set[str] = set()
        for line in text.split("\n"):
            refs.update(_REFERENCE.findall(lexer.mask(line)))
        own = {self._name[node], (self._name[node] or "").rsplit(".", 1)[-1]}
        return text, {r for r in refs if r not in _NOT_REFERENCES and r not in own}

    def _path(
        self, node: int, reached: dict[int, tuple[int, int]]
    ) -> list[dict[str, Any]]:
        """Return the hops from the target to *node*, each with where it is and how it was found."""
        hops = []
        while True:
            parent, how = reached[node]
            if parent < 0:
                break
            hops.append(
                {
                    "declaration": self._name[node],
                    "file": self._labels[self._file[node]],
                    "line_start": self._start[node],
                    "found_by": _HOW[how],
                    "from": self._name[parent],
                }
            )
            node = parent
        return hops[::-1]

    def _import_gaps(
        self, scope: frozenset[int] | None = None
    ) -> tuple[dict[str, list[str]], Counter[str]]:
        """
        Return the imported project modules no file provides, and external roots.

        Of the imports of the files in *scope* (``None``: every file).

        A module is provided by the file at its path from the project's
        source root (:meth:`_module_files`). An import no file provides
        is missing when it names the project (:func:`_names_the_project`:
        the same rule that makes a file's imports unknown, so a missing
        module and a file that sees everything always come together); any
        other is a dependency's, counted by its first component.
        """
        key = self._generation
        cached = getattr(self, "_gaps", None)
        if scope is None and cached is not None and cached[0] == key:
            return cached[1], cached[2]
        provided, roots = self._modules()
        modules = self._module_files()
        missing: dict[str, list[str]] = {}
        external: Counter[str] = Counter()
        for file_id, imports in enumerate(self._imports):
            if scope is not None and file_id not in scope:
                continue
            for module in imports:
                if module in provided:
                    continue
                if _names_the_project(module, modules, roots):
                    missing.setdefault(module, []).append(self._labels[file_id])
                else:
                    external[module.split(".", 1)[0]] += 1
        if scope is None:
            self._gaps = (key, missing, external)
        return missing, external

    def _modules(self) -> tuple[set[str], set[str]]:
        """Return the modules the files provide (see :meth:`_module_files`), and their first components."""
        provided = set(self._module_files())
        roots = {module.split(".", 1)[0] for module in provided}
        return provided, roots

    # -- discovery ----------------------------------------------------------

    def discover(
        self,
        *,
        top: int = 50,
        min_instances: int = 2,
        patterns: bool = True,
        layer: str | None = None,
    ) -> dict[str, Any]:
        """
        Return the candidates a development implies: rules, patterns, claims, anomalies.

        Parameters
        ----------
        top : int, optional
            Most items listed per section (each section keeps its full
            count). Default: 50.
        min_instances : int, optional
            Fewest theorems of one statement shape that make a pattern.
            Default: 2.
        patterns : bool, optional
            Group results by statement shape and list the ``Prop``
            definitions, both recorded while the files were indexed (no
            file is read again). ``False`` leaves them out: no statement
            shapes, no claim definitions, no ``same_statement`` on parallel
            declarations. Default: ``True``.
        layer : str or None, optional
            List only the declarations of this layer's files
            (:meth:`add_folder`; ``""`` is the first). Uses are still counted
            from every layer: a dependency's theorem the project names is
            load-bearing for it, but listed only in the dependency's layer.
            ``None`` (default) lists every file's.

        Returns
        -------
        dict
            JSON-ready; ``report`` is ``"discovery"``, ``layer`` the layer
            listed (``None``: all). Sections:

            - ``rules``: ``load_bearing_theorems`` and ``central_concepts``,
              ranked by the files whose declarations name them, then by
              the declarations;
            - ``patterns``: ``statement_shapes`` (theorems whose statements
              are equal once bound names are renamed in order and numerals
              replaced) and ``numbered_series`` (names differing only by a
              trailing number);
            - ``claims``: ``claim_definitions`` (definitions of type
              ``Prop``), ``axioms``, ``proof_wanted``, ``unproven``
              (``sorry`` or ``admit``), ``vacuous_predicates`` (``Prop``
              definitions whose body is ``True``, ``False`` and connectives
              only, or uses none of their parameters; each with its reason);
            - ``anomalies``: ``parallel_declarations`` (public names declared
              in several files), ``generators`` (code adding declarations
              at build time), ``isolated_results`` (theorems no declaration
              names, outside the headline modules);
            - ``headline_results``: theorems of the modules no project file
              imports that nothing names.

        Raises
        ------
        ValueError
            If *top* or *min_instances* is not a positive integer, or
            *layer* names no layer.

        See Also
        --------
        export : The same, written as documents for people and models.

        Notes
        -----
        **User.** Every item is a candidate chosen by a stated rule, with
        its ``file`` and lines: a pattern is not proved general by having
        instances, an anomaly is a question, not a fault. Lean's kernel
        decides proofs.

        **Developer.** Lexical and deterministic: the same files give the
        same report. One pass over the edges counts uses. Shapes are
        16-byte digests taken while indexing, so discovery reads no file
        except to quote the statements of the items it lists; a file gone
        since indexing leaves its items listed without a quote.
        """
        self._canonical()
        from ._discovery import discover  # noqa: PLC0415  (imports this module)

        return discover(
            self, top=top, min_instances=min_instances, patterns=patterns, layer=layer
        )

    def near_duplicates(  # noqa: PLR0913
        self,
        *,
        threshold: float = 0.8,
        layer: str | None = None,
        against: str | None = None,
        min_pairs: int = 4,
        top: int = 50,
    ) -> dict[str, Any]:
        """
        Find results whose statements are nearly the same, but not of one shape.

        Parameters
        ----------
        threshold : float, optional
            Least Jaccard similarity of two statements' sets of adjacent
            token pairs, in (0, 1]. Statements are compared by their shapes
            (bound names renamed in order, numerals replaced), as
            :meth:`discover` groups them. Default: 0.8.
        layer, against : str or None, optional
            Pairs with one result in layer *layer* and the other in layer
            *against* (``""`` is the first; ``None``: any). A project
            restating its dependency: ``layer="project", against=""``.
            Default: ``None`` for both.
        min_pairs : int, optional
            Fewest token pairs a statement needs to be compared: in a tiny
            statement one token decides. Default: 4.
        top : int, optional
            Most pairs listed; ``pairs`` keeps the full count. Default: 50.

        Returns
        -------
        dict
            JSON-ready; ``report`` is ``"near_duplicates"``. ``pairs``,
            most similar first, each with ``similarity`` (rounded),
            ``shared_pairs`` and ``pairs_in_either`` (exact), and both
            results (location and statement); ``statements_compared``;
            ``files_not_read`` (no known path).

        Raises
        ------
        ValueError
            If *threshold* is not in (0, 1], *min_pairs* or *top* is not a
            positive integer, or *layer* or *against* names no layer.

        See Also
        --------
        discover : Results of one statement shape (``statement_shapes``).

        Notes
        -----
        **User.** Every pair at or above *threshold* is found and none
        below: the search is exact, not sampled. A pair is a candidate,
        two statements to read side by side: one may generalize the other,
        restate it, or differ where it matters.

        **Developer.** The statements are read again from the files (with
        the index's ``workers``), so the index keeps no extra data. Pairs
        are found by prefix filtering over token pairs ordered rarest
        first (:func:`._similar.similar_pairs`). On mathlib4 with PFR it
        compares about 200,000 statements; see the user guide for times.
        """
        from ._similar import near_duplicates  # noqa: PLC0415  (imports this module)

        return near_duplicates(
            self,
            threshold=threshold,
            layer=layer,
            against=against,
            min_pairs=min_pairs,
            top=top,
        )

    def export(
        self,
        folder: str | Path,
        *,
        targets: Sequence[str] = (),
        top: int = 50,
        patterns: bool = True,
        layer: str | None = None,
    ) -> list[str]:
        """
        Write helper documents on the project, for people, models and corpora.

        Parameters
        ----------
        folder : str or pathlib.Path
            Where to write; created if missing. Files already there under
            the same names are replaced, each atomically. The export records
            what it wrote in ``.lean-export.json``; the next export into the
            same folder removes the files that record lists and it no longer
            writes (an old target's dossier), and never any other file, so
            exporting twice leaves the folder as one export would.
        targets : sequence of str, optional
            Declarations to write a dossier on, besides the headline
            results. Default: none.
        top : int, optional
            As for :meth:`discover`. Default: 50.
        patterns : bool, optional
            As for :meth:`discover`. Default: ``True``.
        layer : str or None, optional
            As for :meth:`discover` and :meth:`summary`: documents on one
            layer's files. Default: ``None`` (every file).

        Returns
        -------
        list of str
            The files written, relative to *folder*, in writing order:
            ``README.md`` (what each file holds and how items are chosen),
            ``summary.md``/``.json``, ``discovery.json``, ``rules.md``,
            ``patterns.md``, ``claims.md``, ``anomalies.md``,
            ``dossiers/<name>.md``/``.json`` and ``records.jsonl`` (one
            record per item, for :class:`~scikitplot.corpus.JSONReader`
            records mode or an MCP server).

        Raises
        ------
        ValueError
            If *folder* exists and is not a folder, or an option is invalid.

        See Also
        --------
        discover, dossier, summary
        """
        self._canonical()
        from ._discovery import export  # noqa: PLC0415  (imports this module)

        return export(
            self, folder, targets=targets, top=top, patterns=patterns, layer=layer
        )

    # -- output -------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """
        Write the index to one file, to load back with :meth:`load`.

        Parameters
        ----------
        path : str or pathlib.Path
            The file to write (replaced atomically if it exists); its folder
            is made if needed.

        Returns
        -------
        pathlib.Path
            *path*.

        Raises
        ------
        ValueError
            If *path* is a folder.

        See Also
        --------
        load : Read a saved index.

        Notes
        -----
        **User.** A snapshot holds the index, not its caches, in the format
        ``lean.index.snapshot/1``: numbers and text only, nothing that runs
        when loaded, with a checksum. The same files give the same bytes,
        whatever order they were added in. Files are named by label, not
        by path, so a snapshot can be moved and shared.

        Examples
        --------
        >>> import tempfile, pathlib
        >>> from scikitplot.lean import LeanIndex
        >>> root = pathlib.Path(tempfile.mkdtemp())
        >>> _ = (root / "A.lean").write_text("theorem t : True := trivial")
        >>> index = LeanIndex.from_folder(root)
        >>> saved = index.save(root / "index.leanidx")
        >>> LeanIndex.load(saved).find("t")[0]["declaration"]
        't'
        """
        from ._snapshot import save  # noqa: PLC0415

        return save(self, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        root: str | Path | dict[str, str | Path] | None = None,
        graph: str = "python",
        workers: int | None = 1,
    ) -> LeanIndex:
        """
        Read an index written by :meth:`save`.

        Parameters
        ----------
        path : str or pathlib.Path
            The snapshot.
        root : str, pathlib.Path, dict or None, optional
            The project folder the snapshot was made from. With it, reports
            quote statements and definitions from the files inside it; a
            label that would lead outside it is not read. With layers
            (:meth:`add_folder`), a dict from layer name to folder (``""``
            for the first). ``None`` (default): no file is read, and reports
            give each item's file and lines without its text.
        graph : {"python", "numpy", "scipy"}, optional
            As for :class:`LeanIndex`. Default: ``"python"``.
        workers : int or None, optional
            As for :class:`LeanIndex`. Default: ``1``.

        Returns
        -------
        LeanIndex
            The index saved, with the options it was built with; every
            report is the one the saved index gives.

        Raises
        ------
        ValueError
            If the file is not a snapshot of this format, or is damaged or
            altered: a wrong checksum, a section missing or out of place,
            a number out of range.
        OSError
            If the file cannot be read.

        See Also
        --------
        save : Write a snapshot.
        """
        from ._snapshot import load  # noqa: PLC0415

        return load(path, cls, root=root, graph=graph, workers=workers)

    @staticmethod
    def render(report: dict[str, Any]) -> str:
        """
        Render a report of :meth:`summary`, :meth:`dossier`, :meth:`discover`, :meth:`refresh` or :meth:`near_duplicates` as Markdown.

        Every item keeps its ``file:line``, so the text is as checkable as
        the data.
        """
        if report.get("report") == "summary":
            return _render_summary(report)
        if report.get("report") == "refresh":
            return _render_refresh(report)
        if report.get("report") == "near_duplicates":
            from ._similar import render  # noqa: PLC0415  (imports this module)

            return render(report)
        if report.get("report") == "discovery":
            from ._discovery import render  # noqa: PLC0415  (imports this module)

            return render(report)
        return _render_dossier(report)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _positive(value: Any, option: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"LeanIndex: {option} must be an int > 0; got {value!r}.")


def _listing(items: list[Any], max_items: int) -> dict[str, Any]:
    """Return ``{"count", "items", "cut"}``: never a silently shortened list."""
    return {
        "count": len(items),
        "items": items[:max_items],
        "cut": len(items) > max_items,
    }


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"check": name, "status": status, "detail": detail}


def _where(item: dict[str, Any]) -> str:
    return f"`{item['file']}:{item['line_start']}`"


def _render_refresh(report: dict[str, Any]) -> str:
    out = ["# Lean index refresh", "", _refresh_line(report) + "."]
    for key in ("added", "changed", "removed"):
        listing = report[key]
        if listing["count"]:
            out += ["", f"## {key.capitalize()} ({listing['count']})"]
            out += [f"- {label}" for label in listing["items"]]
            if listing["cut"]:
                out.append("- (list cut)")
    return "\n".join(out) + "\n"


def _of_layer(report: dict[str, Any]) -> str:
    """Return ``" in layer ..."`` for a report on one layer, else ``""``."""
    layer = report.get("layer")
    if layer is None:
        return ""
    return f" in layer `{layer}`" if layer else " in the first layer"


def _render_summary(report: dict[str, Any]) -> str:
    out = [
        "# Lean project summary",
        "",
        (
            f"{report['files']:,} files, {report['lines']:,} lines,"
            f" {report['declarations']:,} declarations" + _of_layer(report) + "."
        ),
        "",
        "## Audit flags",
    ]
    if not report["flags"]:
        out.append("None found in code.")
    for flag, listing in report["flags"].items():
        out.append(f"- **{flag}**: {listing['count']}")
        out.extend(
            f"  - {item['declaration'] or item['kind']} at {_where(item)}"
            for item in listing["items"]
        )
        if listing["cut"]:
            out.append("  - (list cut)")
    out += ["", "## Imports"]
    roots = report["imports"]["external_roots"]
    out.append(
        "External: "
        + (", ".join(f"{r} ({n})" for r, n in roots.items()) or "none")
        + "."
    )
    missing = report["imports"]["missing_modules"]
    out.append(f"Missing project modules: {missing['count']}.")
    out.extend(
        f"- `{m['module']}` (imported by {', '.join(m['imported_by'][:3])})"
        for m in missing["items"]
    )
    clashes = report["imports"].get("module_clashes", {"count": 0, "items": []})
    if clashes["count"]:
        out.append(
            f"Modules two layers provide (Lean refuses them together): {clashes['count']}."
        )
        out.extend(
            f"- `{c['module']}`: {c['used']}, also {c['also_in']}"
            for c in clashes["items"]
        )
    out += ["", "## Code that adds declarations at build time"]
    generators = report["generators"]
    if not generators["count"]:
        out.append("None found.")
    for item in generators["items"]:
        adder = item["adds_declarations_at"]
        out.append(
            f"- `{item['invocation']['kind']}` at {_where(item['invocation'])}:"
            f" `{adder['declaration']}` at {_where(adder)}"
        )
    learned = report["learned_commands"]
    out += [
        "",
        (
            f"Names declared more than once (as Lean counts): {report['duplicate_names']['count']}."
            f" Project commands learned: {len(learned)}."
            f" Skipped files: {len(report['skipped_files'])}."
        ),
    ]
    extensions = report["syntax_extensions"]
    kinds = ", ".join(f"{kind} {n:,}" for kind, n in extensions["by_kind"].items())
    out += [
        "",
        f"## Syntax extensions: {extensions['count']:,}"
        + (f" ({kinds})" if kinds else ""),
    ]
    risky = extensions["flagged"]
    out.append(
        f"Flagged (used without their names, so a dossier lists them as risks): {risky['count']}."
    )
    out.extend(
        f"- {item['kind']} at {_where(item)}: {', '.join(item['flags'])}"
        for item in risky["items"]
    )
    if risky["cut"]:
        out.append("  - (list cut)")
    problems = report["source_problems"]
    if problems["count"]:
        out += ["", f"## Text Lean refuses to read: {problems['count']}"]
        out.extend(
            f"- {p['file']}:{p['line']}: {p['problem']}" for p in problems["items"]
        )
        if problems["cut"]:
            out.append("  - (list cut)")
    return "\n".join(out) + "\n"


def _render_dossier(report: dict[str, Any]) -> str:  # noqa: PLR0912
    out = [f"# Evidence dossier: `{report['query']}`", ""]
    if "matches" in report:
        out.append(f"**{report['verdict']}**.")
        out.extend(f"- `{m['declaration']}` at {_where(m)}" for m in report["matches"])
        return "\n".join(out) + "\n"
    target = report["target"]
    out += [
        f"**Verdict:** {report['verdict']}.",
        "",
        f"## Statement ({target['kind']} at {_where(target)})",
        "",
        "```lean",
        target["statement"] or "(no text: indexed without its file)",
        "```",
        "",
        "## What the statement means",
    ]
    definitions = report["statement"]["local_definitions"]
    if not definitions["count"]:
        out.append("It uses no project definition.")
    for item in definitions["items"]:
        why = f" **vacuous**: {item['vacuous']}" if "vacuous" in item else ""
        out += [
            f"- `{item['declaration']}` ({item['kind']}, {_where(item)}){why}",
            "",
            "```lean",
            item["text"] or "",
            "```" + (" (cut)" if item["text_cut"] else ""),
            "",
        ]
    external = report["statement"]["external_names"]
    out.append(
        "From dependencies: "
        + (", ".join(f"`{n}`" for n in external["items"]) or "none")
        + (" (list cut)" if external["cut"] else "")
        + "."
    )
    deps = report["dependencies"]
    out += [
        "",
        "## What the proof names",
        (
            f"{deps['declarations']:,} declarations in {deps['files']:,} files"
            f" (found by name: {deps['found_by']['exact']:,} exact,"
            f" {deps['found_by']['prefix']:,} by prefix,"
            f" {deps['found_by']['suffix']:,} by suffix)."
        ),
    ]
    for item in deps["red_flags"]["items"]:
        hops = item["path"]
        chain = (
            " → ".join(
                [f"`{hops[0]['from']}`"] + [f"`{h['declaration']}`" for h in hops]
            )
            if hops
            else "the target itself"
        )
        out.append(
            f"- **{', '.join(item['flags'])}** in `{item['declaration']}` at {_where(item)}: {chain}"
        )
    implicit = report["implicit_risks"]
    out += ["", f"## Used without a name: {implicit['count']}"]
    for item in implicit["items"]:
        if item["risk"] == "generated_declarations":
            invocation, adder = item["invocation"], item["adds_declarations_at"]
            out.append(
                f"- `{invocation['kind']}` at {_where(invocation)} adds declarations"
                f" no source spells (`{adder['declaration']}` at {_where(adder)});"
                f" imported via {' → '.join(f'`{f}`' for f in item['via_imports'])}"
            )
        else:
            out.append(
                f"- `{item['declaration'] or item['kind']}` ({', '.join(item['flags'])})"
                f" at {_where(item)}"
            )
    out += ["", "## Checks"]
    out.extend(
        f"- {c['check']}: **{c['status']}** ({c['detail']})" for c in report["checks"]
    )
    out += ["", "## To settle it", *[f"- `{step}`" for step in report["kernel_steps"]]]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def _index_for(args: argparse.Namespace) -> LeanIndex:
    """Return the index the command line asks for: read from a snapshot, or built (and saved)."""
    workers = None if args.workers == 0 else args.workers
    if args.refresh and args.load is None:
        raise ValueError("--refresh needs --load FILE")
    if args.load is not None:
        index = LeanIndex.load(
            args.load, root=args.folder, graph=args.graph, workers=workers
        )
        if args.refresh:
            done = index.refresh(
                include_hidden=args.include_hidden,
                learn_commands=not args.no_learn_commands,
            )
            print(_refresh_line(done), file=sys.stderr)  # noqa: T201
            if args.save is not None:
                index.save(args.save)
        return index
    if args.base is not None:
        index = LeanIndex.load(
            args.base, root=args.base_root, graph=args.graph, workers=workers
        )
        index.add_folder(
            args.folder,
            name=args.name or args.folder.resolve().name,
            include_hidden=args.include_hidden,
            learn_commands=not args.no_learn_commands,
            source_root=args.source_root,
        )
        if args.save is not None:
            index.save(args.save)
        return index
    index = LeanIndex.from_folder(
        args.folder,
        include_hidden=args.include_hidden,
        learn_commands=not args.no_learn_commands,
        dialect=args.dialect,
        commands=args.command,
        visibility=args.visibility,
        source_root=args.source_root,
        workers=workers,
        graph=args.graph,
    )
    if args.save is not None:
        index.save(args.save)
    return index


def _reports_for(index: LeanIndex, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return the reports the command line asks for."""
    if args.target:
        return [index.dossier(t, max_items=args.max_items) for t in args.target]
    if args.discover:
        return [index.discover(top=args.max_items, layer=args.layer)]
    if args.near_duplicates:
        return [
            index.near_duplicates(
                threshold=args.threshold,
                layer=args.layer,
                against=args.against,
                top=args.max_items,
            )
        ]
    return [index.summary(max_items=args.max_items, layer=args.layer)]


def _refresh_line(report: dict[str, Any]) -> str:
    """Return one line saying what a :meth:`LeanIndex.refresh` read again."""
    line = (
        f"refreshed: {report['added']['count']} added,"
        f" {report['changed']['count']} changed, {report['removed']['count']} removed,"
        f" {report['unchanged']} unchanged; {report['files_read']} file(s) read"
    )
    if report["commands_changed"]:
        line += (
            " (commands changed: layers "
            + ", ".join(repr(name) for name in report["layers_read_again"])
            + " read again)"
        )
    return line


def _report_unresolved(reports: list[dict[str, Any]], prog: str) -> bool:
    """
    Name on standard error each target that is not one declaration.

    Returns ``True`` when there is one: a ``--fail-on`` gate must never
    pass on a name it could not check.
    """
    unresolved = [r for r in reports if r["report"] == "dossier" and "matches" in r]
    for report in unresolved:
        names = ", ".join(m["declaration"] for m in report["matches"])
        print(  # noqa: T201
            f"{prog}: --target {report['query']}: {report['verdict']}"
            + (f" (candidates: {names}; give the full name)" if names else ""),
            file=sys.stderr,
        )
    return bool(unresolved)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Index a Lean project and print its summary or a dossier per target.

    Returns 2 when a ``--target`` is not found or is ambiguous (with the
    candidates on standard error), so a gate never passes on a name it
    could not check; 1 when ``--fail-on`` names a flag found (in a
    target's dependencies or implicit risks; in the whole project without
    targets); 0 otherwise. Usage errors also exit 2.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scikitplot.lean",
        description="Evidence on a Lean project: statements, dependencies, audit flags.",
    )
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        metavar="NAME",
        help="print the dossier of NAME (repeatable); exit 2 if NAME is not one declaration",
    )
    parser.add_argument("--json", action="store_true", help="JSON instead of Markdown")
    parser.add_argument("--dialect", default="lean4", choices=("lean4", "lean3"))
    parser.add_argument(
        "--command",
        action="append",
        default=[],
        metavar="KEYWORD",
        help="a keyword that starts a project command (repeatable)",
    )
    parser.add_argument(
        "--max-items", type=int, default=50, help="most items listed per finding"
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="also read hidden folders, such as .lake/packages",
    )
    parser.add_argument(
        "--fail-on",
        action="append",
        default=[],
        metavar="FLAG",
        help="exit 1 if FLAG is found (repeatable); see AUDIT_FLAGS",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="print rules, patterns, claims and anomalies",
    )
    parser.add_argument(
        "--near-duplicates",
        action="store_true",
        help=(
            "print pairs of results whose statements are nearly the same"
            " (--threshold; --layer and --against choose the two sides)"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="least similarity for --near-duplicates, in (0, 1] (default 0.8)",
    )
    parser.add_argument(
        "--against",
        default=None,
        metavar="NAME",
        help="with --near-duplicates: the layer the other result is in",
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="DIR",
        help="also write the helper documents to DIR",
    )
    parser.add_argument(
        "--no-learn-commands",
        action="store_true",
        help="do not learn the commands the project defines",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "processes reading the files and, on Linux, resolving names"
            " (0: every CPU); the reports are the same"
        ),
    )
    parser.add_argument(
        "--graph",
        default="python",
        choices=_GRAPHS,
        help="whole-graph steps in the standard library, NumPy or SciPy; the reports are the same",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        metavar="FOLDER",
        help="folder module names start from (default: found from the imports)",
    )
    parser.add_argument(
        "--visibility",
        default="imports",
        choices=("imports", "all"),
        help="link names only into imported files (default), or anywhere",
    )
    parser.add_argument(
        "--save",
        type=Path,
        metavar="FILE",
        help="also write the index to FILE (see LeanIndex.save)",
    )
    parser.add_argument(
        "--load",
        type=Path,
        metavar="FILE",
        help="read the index from FILE instead of indexing; FOLDER is where its files are",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "with --load: read again the files of FOLDER that are new or changed"
            " since FILE was saved, and drop those gone (with --save, save it)"
        ),
    )
    parser.add_argument(
        "--base",
        type=Path,
        metavar="FILE",
        help="read a saved index (a dependency, such as mathlib) and add FOLDER over it",
    )
    parser.add_argument(
        "--base-root",
        type=Path,
        default=None,
        metavar="FOLDER",
        help="where the --base index's files are, to quote them (default: not read)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="FOLDER's layer name over --base (default: the folder's name)",
    )
    parser.add_argument(
        "--layer",
        default=None,
        metavar="NAME",
        help=(
            "summarize, discover and export one layer's files only"
            " ('' for the first; with --base, FOLDER's layer is its --name)"
        ),
    )
    args = parser.parse_args(argv)
    unknown = sorted(set(args.fail_on) - AUDIT_FLAGS)
    if unknown:
        parser.error(
            f"--fail-on: unknown flag(s) {', '.join(unknown)};"
            f" choose from {', '.join(sorted(AUDIT_FLAGS))}"
        )
    try:
        index = _index_for(args)
        reports = _reports_for(index, args)
        if args.export is not None:
            index.export(
                args.export,
                targets=args.target,
                top=args.max_items,
                layer=args.layer,
            )
    except (TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.json:
        print(  # noqa: T201
            json.dumps(
                reports if args.target else reports[0], indent=1, ensure_ascii=False
            )
        )  # noqa: T201
    else:
        print("\n".join(LeanIndex.render(r) for r in reports))  # noqa: T201
    if _report_unresolved(reports, parser.prog):
        return 2
    found: set[str] = set()
    for report in reports:
        if report["report"] == "dossier":
            found.update(report["flags_found"])
        else:  # the flags of the files reported on (--layer)
            scope = index._scope(args.layer)
            found.update(
                flag
                for node, flags in index._flags.items()
                if scope is None or index._file[node] in scope
                for flag in flags
            )
    return 1 if found & set(args.fail_on) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
