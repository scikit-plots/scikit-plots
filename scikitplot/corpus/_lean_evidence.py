# scikitplot/corpus/_lean_evidence.py
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
import json
import sys
from array import array
from collections import Counter, OrderedDict, deque
from pathlib import Path
from typing import Any, Iterable, Sequence

from ._graph import EdgeTrust, RelationType
from ._readers._lean import (
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
PRODUCER = "corpus.lean-index/1"

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
_BINDER = None  # compiled on first use (keeps import cheap)
_QUANTIFIED = None


def _binders() -> tuple[Any, Any]:
    global _BINDER, _QUANTIFIED  # noqa: PLW0603
    if _BINDER is None:
        import re  # noqa: PLC0415

        name = r"[^\W\d][\w'!?]*"
        _BINDER = re.compile(rf"[(\[{{⦃⟨]\s*((?:{name}\s+)*{name})\s*:")
        _QUANTIFIED = re.compile(rf"(?:∀|∃!?|λ|\bfun\b|Π|Σ)\s*((?:{name}\s*)+)")
    return _BINDER, _QUANTIFIED


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
    dossier quotes each. Names a statement binds (``(s : ℂ)``, ``∀ n``)
    are left out; names found nowhere in the project come from its
    dependencies and are listed as external.

    **Memory and time.** Per declaration the index keeps a few integers and
    the names its code uses (interned strings); names are resolved only for
    the declarations a dossier reaches, once each. A file is read twice at
    most: once to index it, once more to quote it.

    **Developer note.** Edges use the vocabulary of
    :mod:`scikitplot.corpus._graph`: relation ``references``, trust
    ``extracted`` (a deterministic extractor), producer :data:`PRODUCER`.

    Examples
    --------
    >>> from scikitplot.corpus import LeanIndex
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
    ) -> None:
        if dialect not in ("lean4", "lean3"):
            raise ValueError(
                f"LeanIndex: dialect must be 'lean4' or 'lean3'; got {dialect!r}."
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
        self._commands = frozenset(commands)
        self._max_depth = max_scope_depth
        self._max_bytes = max_file_bytes
        self._suffix = suffix
        # files
        self._labels: list[str] = []
        self._paths: list[Path | None] = []
        self._first_node: list[int] = []  # each file's first node
        self._learned: dict[str, tuple[str, int]] = {}
        self._file_lines: list[int] = []
        self._imports: list[tuple[str, ...]] = []
        self._skipped: list[tuple[str, str]] = []
        # declarations (one entry per node, in reading order)
        self._name: list[str | None] = []
        self._kind: list[str] = []
        self._file: array[int] = array("l")
        self._start: array[int] = array("l")
        self._end: array[int] = array("l")
        self._namespace: list[str] = []
        self._opens: list[tuple[str, ...]] = []
        self._private: set[int] = set()
        self._flags: dict[int, tuple[str, ...]] = {}
        self._attributes: dict[int, tuple[str, ...]] = {}
        self._refs: list[tuple[str, ...] | None] = []
        self._edges: dict[int, array[int]] = {}
        self._by_name: dict[str, Any] = {}
        self._by_short: dict[str, Any] | None = None
        self._kinds: Counter[str] = Counter()
        self._intern: dict[Any, Any] = {}
        self._contexts: OrderedDict[tuple[str, tuple[str, ...]], dict[str, tuple[int, ...]]] = (
            OrderedDict()
        )
        self._lines_cache: OrderedDict[int, list[str]] = OrderedDict()

    # -- building ---------------------------------------------------------------

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        *,
        include_hidden: bool = False,
        learn_commands: bool = True,
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
        files = []
        for path in sorted(root.rglob("*.lean"), key=lambda p: p.as_posix()):
            relative = path.relative_to(root)
            if not include_hidden and any(
                part.startswith(".") for part in relative.parts[:-1]
            ):
                continue
            if path.is_file():
                files.append((path, relative.as_posix()))
        learned: dict[str, tuple[str, int]] = {}
        if learn_commands:
            for path, label in files:
                with open(path, encoding="utf-8", errors="replace") as stream:
                    for keyword, number in _defined_commands(stream).items():
                        learned.setdefault(keyword, (label, number))
        if learned:
            options["commands"] = sorted(set(options.get("commands", ())) | set(learned))
        index = cls(**options)
        index._learned = learned
        for path, label in files:
            index.add_file(path, label=label)
        return index

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
        file_id = len(self._labels)
        self._labels.append(label)
        self._paths.append(path)
        self._imports.append(())
        self._file_lines.append(0)
        self._first_node.append(len(self._name))
        if path.stat().st_size > self._max_bytes:
            self._skipped.append((label, f"larger than max_file_bytes={self._max_bytes:,}"))
            return
        splitter = self._splitter(label)
        nodes_before = len(self._name)
        number = 0
        try:
            with open(path, encoding="utf-8", errors="replace", newline=None) as stream:
                for number, read in enumerate(stream, start=1):
                    line = read[:-1] if read.endswith("\n") else read
                    if number == 1 and line.startswith("﻿"):
                        line = line[1:]
                    for chunk in splitter.feed(number, line):
                        self._add_chunk(chunk, file_id)
            for chunk in splitter.finish():
                self._add_chunk(chunk, file_id)
        except ValueError as exc:  # nesting past max_scope_depth
            self._drop_nodes(nodes_before)
            self._imports[file_id] = ()
            self._skipped.append((label, str(exc)))
            return
        self._file_lines[file_id] = number
        self._by_short = None  # rebuilt on the next lookup

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
        file_id = len(self._labels)
        self._labels.append(label)
        self._paths.append(None)
        self._imports.append(())
        self._file_lines.append(0)
        self._first_node.append(len(self._name))
        last_end = 0
        for chunk in chunks:
            if chunk.get("part", 0) > 1 and self._name and self._file[-1] == file_id:
                node = len(self._name) - 1
                refs = set(self._refs[node] or ()) | set(chunk.get("references", ()))
                self._refs[node] = tuple(sys.intern(ref) for ref in sorted(refs))
                self._end[node] = chunk["line_end"]
                flags = set(self._flags.get(node, ())) | set(chunk.get("flags", ()))
                if flags:
                    self._flags[node] = tuple(sorted(flags))
            else:
                self._add_chunk(chunk, file_id)
            last_end = max(last_end, chunk.get("line_end", 0))
        self._file_lines[file_id] = last_end
        self._by_short = None

    def _splitter(self, label: str) -> _Splitter:
        return _Splitter(
            file_name=label,
            meta={},
            step=None,
            cap=None,
            proofs="keep",
            dialect=self._dialect,
            kinds=None,
            watch=(),
            commands=self._commands,
            max_scope_depth=self._max_depth,
            flag_tokens=LEAN_FLAGS,
            references=True,
        )

    def _interned(self, value: Any) -> Any:
        return self._intern.setdefault(value, value)

    def _add_chunk(self, chunk: dict[str, Any], file_id: int) -> None:
        kind = chunk["lean_kind"]
        self._kinds[kind] += 1
        if kind == "import":
            self._imports[file_id] = tuple(
                dict.fromkeys(self._imports[file_id] + tuple(chunk.get("imports", ())))
            )
            return
        if kind in _NOT_NODES:
            return
        node = len(self._name)
        name = chunk.get("declaration")
        self._name.append(name)
        self._kind.append(sys.intern(kind))
        self._file.append(file_id)
        self._start.append(chunk["line_start"])
        self._end.append(chunk["line_end"])
        self._namespace.append(sys.intern(chunk.get("namespace", "")))
        self._opens.append(self._interned(tuple(chunk.get("opens", ()))))
        self._refs.append(tuple(sys.intern(ref) for ref in chunk.get("references", ())))
        if chunk.get("flags"):
            self._flags[node] = tuple(chunk["flags"])
        if chunk.get("attributes"):
            self._attributes[node] = tuple(chunk["attributes"])
        if "private" in chunk.get("modifiers", ()):
            self._private.add(node)
        if name is not None:
            held = self._by_name.get(name)
            if held is None:
                self._by_name[name] = node
            elif isinstance(held, int):
                self._by_name[name] = [held, node]
            else:
                held.append(node)

    def _drop_nodes(self, keep: int) -> None:
        """Forget the nodes from *keep* on (a file that could not be read whole)."""
        for node in range(keep, len(self._name)):
            name = self._name[node]
            held = self._by_name.get(name) if name is not None else None
            if isinstance(held, int):
                del self._by_name[name]
            elif held is not None:
                held.remove(node)
                if len(held) == 1:
                    self._by_name[name] = held[0]
            self._flags.pop(node, None)
            self._attributes.pop(node, None)
            self._private.discard(node)
        for column in (self._name, self._kind, self._namespace, self._opens, self._refs):
            del column[keep:]
        for column in (self._file, self._start, self._end):
            del column[keep:]

    # -- resolving names ----------------------------------------------------

    def _clashes(self) -> list[tuple[str, list[int]]]:
        """
        Return the names declared more than once in a way Lean would refuse together.

        A ``private`` name is local to its file, so it clashes only within
        one file; public names clash wherever two of them meet. Two public
        declarations in modules that are never imported together build
        fine apart: they are listed, as parallel declarations of one name.
        """
        key = len(self._name)
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

    def _module_files(self) -> dict[str, int]:
        """Map each module name a file can provide to the first such file."""
        key = len(self._labels)
        cached = getattr(self, "_module_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        modules: dict[str, int] = {}
        for file_id, label in enumerate(self._labels):
            if not label.endswith(".lean"):
                continue
            parts = label[: -len(".lean")].split("/")
            for i in range(len(parts)):
                modules.setdefault(".".join(parts[i:]), file_id)
        self._module_cache = (key, modules)
        return modules

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
        n = len(self._name)
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
        toward: dict[int, tuple[int, int]] = {t: (-1, -1) for t in targets}
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
        key = len(self._name)
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

    def _lookup(self, ref: str, prefixes: Sequence[str], opens: Sequence[str]) -> tuple[int, ...]:
        if ref.startswith(_ROOT):
            return self._ids(ref[len(_ROOT) :])
        hits: list[int] = []
        for prefix in prefixes:
            hits.extend(self._ids(f"{prefix}.{ref}"))
        hits.extend(self._ids(ref))
        for opened in opens:
            hits.extend(self._ids(f"{opened}.{ref}"))
        return tuple(dict.fromkeys(hits))

    def _resolve(self, ref: str, namespace: str, opens: tuple[str, ...]) -> tuple[int, ...]:
        """Return ``node * 4 + how`` codes for the declarations *ref* may name."""
        key = (namespace, opens)
        cache = self._contexts.get(key)
        if cache is None:
            cache = self._contexts[key] = {}
            if len(self._contexts) > 256:
                self._contexts.popitem(last=False)
        else:
            self._contexts.move_to_end(key)
        hit = cache.get(ref)
        if hit is not None:
            return hit
        parts = namespace.split(".") if namespace else []
        prefixes = [".".join(parts[:i]) for i in range(len(parts), 0, -1)]
        codes: tuple[int, ...] = ()
        found = self._lookup(ref, prefixes, opens)
        if found:
            codes = tuple(node * 4 for node in found)
        else:
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
        cache[ref] = codes
        return codes

    def _edges_of(self, node: int) -> array[int]:
        """Return *node*'s edges as ``target * 4 + how``, resolving them once."""
        edges = self._edges.get(node)
        if edges is not None:
            return edges
        refs = self._refs[node] or ()
        namespace, opens = self._namespace[node], self._opens[node]
        file_id = self._file[node]
        best: dict[int, int] = {}
        for ref in refs:
            for code in self._resolve(ref, namespace, opens):
                target, how = divmod(code, 4)
                if target == node:
                    continue
                if target in self._private and self._file[target] != file_id:
                    continue  # a private declaration is seen from its own file only
                if how < best.get(target, 3):
                    best[target] = how
        edges = array("l", sorted(target * 4 + how for target, how in best.items()))
        self._edges[node] = edges
        self._refs[node] = None
        return edges

    def _closure(
        self, roots: Sequence[int], *, expand: frozenset[str] | None = None
    ) -> dict[int, tuple[int, int]]:
        """Breadth-first: each reached node to ``(parent, how)``; roots to ``(-1, -1)``."""
        reached: dict[int, tuple[int, int]] = {root: (-1, -1) for root in roots}
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
        with open(path, encoding="utf-8", errors="replace", newline=None) as stream:
            lines = [line[:-1] if line.endswith("\n") else line for line in stream]
        if lines and lines[0].startswith("﻿"):
            lines[0] = lines[0][1:]
        self._lines_cache[file_id] = lines
        if len(self._lines_cache) > 64:
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
        return [self._location(node) for node in self._nodes(name)]

    def summary(self, *, max_items: int = 50) -> dict[str, Any]:  # noqa: PLR0912
        """
        Describe the project: sizes, kinds, flags, duplicates, imports, extensions.

        Parameters
        ----------
        max_items : int, optional
            Most examples listed per finding; every list says how many there
            are in all, and whether it was cut. Default: 50.

        Returns
        -------
        dict
            JSON-ready; ``report`` is ``"summary"``.
        """
        _positive(max_items, "max_items")
        flagged: dict[str, list[int]] = {}
        for node in sorted(self._flags):
            for flag in self._flags[node]:
                flagged.setdefault(flag, []).append(node)
        duplicates = self._clashes()
        extensions = [
            node
            for node, kind in enumerate(self._kind)
            if kind in _IMPLICIT and kind != "instance"
        ]
        missing, external = self._import_gaps()
        return {
            "report": "summary",
            "producer": PRODUCER,
            "files": len(self._labels),
            "lines": sum(self._file_lines),
            "declarations": len(self._name),
            "kinds": dict(sorted(self._kinds.items())),
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
            },
            "syntax_extensions": _listing(
                [self._location(n) for n in extensions], max_items
            ),
            "learned_commands": {
                keyword: {"file": label, "line": line}
                for keyword, (label, line) in sorted(self._learned.items())
            },
            "generators": _listing(self._generators(), max_items),
            "skipped_files": [
                {"file": label, "reason": reason} for label, reason in self._skipped
            ],
        }

    def dossier(self, target: str, *, max_items: int = 50) -> dict[str, Any]:  # noqa: PLR0912, PLR0915
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
        _positive(max_items, "max_items")
        nodes = self._nodes(target)
        report: dict[str, Any] = {
            "report": "dossier",
            "producer": PRODUCER,
            "query": target,
            "edge": {
                "relation": RelationType.REFERENCES.value,
                "trust": EdgeTrust.EXTRACTED.value,
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
            entry["found_by"] = _HOW[local[item]] if item in local else _HOW[meaning[item][1]]
            entry["text"], entry["text_cut"] = self._excerpt(item)
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
        # lemmas and syntax extensions anywhere; and declarations generated
        # by code that runs in a module the target's module imports.
        implicit = [
            n
            for n in sorted(self._flags)
            if n not in reached
            and (self._kind[n] in _IMPLICIT or n in self._attributes)
        ]
        risks: list[dict[str, Any]] = [
            {"risk": "used_without_name", **self._location(n)} for n in implicit
        ]
        imported = self._imported(self._file[node])
        label_ids = {label: i for i, label in enumerate(self._labels)}
        generated = []
        for generator in self._generators():
            invocation = generator["invocation"]
            file_id = label_ids[invocation["file"]]
            if file_id not in imported:
                continue
            if file_id == self._file[node] and invocation["line_start"] > self._start[node]:
                continue  # later in the target's own file: not in its environment
            chain = []
            current = file_id
            while current != -1:
                chain.append(self._labels[current])
                current = imported[current]
            generated.append(
                {"risk": "generated_declarations", **generator, "via_imports": chain[::-1]}
            )
        risks += generated
        report["implicit_risks"] = _listing(risks, max_items)

        missing = len(self._import_gaps()[0])
        checks = [
            _check(
                "statement_definitions",
                "review" if definitions else "pass",
                f"{len(definitions)} project definition(s) give the statement its"
                " meaning: read each" if definitions else
                "the statement uses no project definition",
            ),
            _check(
                "named_dependencies",
                "fail" if red_flags else "pass",
                f"{len(red_flags)} flagged declaration(s) among those the proof names"
                if red_flags else "no flagged declaration among those the proof names",
            ),
            _check(
                "implicit_dependencies",
                "review" if risks else "pass",
                f"{len(implicit)} flagged instance(s), tagged lemma(s) or syntax"
                f" extension(s) could be used without being named; {len(generated)}"
                " command(s) in imported modules add declarations no source spells"
                if risks else "no flagged declaration of a kind used without its name",
            ),
            _check(
                "duplicate_names",
                "review" if duplicates else "pass",
                f"{len(duplicates)} name(s) declared more than once among the dependencies"
                if duplicates else "every dependency's name is declared once",
            ),
            _check(
                "imported_modules_present",
                "fail" if missing else ("review" if self._skipped else "pass"),
                f"{missing} imported module(s) missing from the project"
                if missing else (
                    f"{len(self._skipped)} file(s) could not be read"
                    if self._skipped else "every imported project module is present"
                ),
            ),
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
            return None, set(self._refs[node] or ())
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
        if text is None:  # pragma: no cover - the index read the same file
            return None, set()
        lexer = _Lexer()
        refs: set[str] = set()
        for line in text.split("\n"):
            refs.update(_REFERENCE.findall(lexer.mask(line)))
        own = {self._name[node], (self._name[node] or "").rsplit(".", 1)[-1]}
        return text, {r for r in refs if r not in _NOT_REFERENCES and r not in own}

    def _path(self, node: int, reached: dict[int, tuple[int, int]]) -> list[dict[str, Any]]:
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

    def _import_gaps(self) -> tuple[dict[str, list[str]], Counter[str]]:
        """
        Return the imported project modules no file provides, and external roots.

        A module is provided by a file whose path ends with it
        (``src/A/B.lean`` provides ``A.B``); an import whose first
        component is a folder or file of the project but that no file
        provides is missing; any other is a dependency's, counted by its
        first component.
        """
        key = len(self._labels)
        cached = getattr(self, "_gaps", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        provided, roots = self._modules()
        missing: dict[str, list[str]] = {}
        external: Counter[str] = Counter()
        for file_id, imports in enumerate(self._imports):
            for module in imports:
                if module in provided:
                    continue
                if module.split(".", 1)[0] in roots:
                    missing.setdefault(module, []).append(self._labels[file_id])
                else:
                    external[module.split(".", 1)[0]] += 1
        self._gaps = (key, missing, external)
        return missing, external

    def _modules(self) -> tuple[set[str], set[str]]:
        """Return the module names the files can provide, and their first components."""
        provided: set[str] = set()
        roots: set[str] = set()
        for label in self._labels:
            parts = label[: -len(".lean")].split("/") if label.endswith(".lean") else []
            for i in range(len(parts)):
                provided.add(".".join(parts[i:]))
            if parts:
                roots.update(parts[:-1] or parts)
        return provided, roots

    # -- discovery ----------------------------------------------------------

    def discover(
        self, *, top: int = 50, min_instances: int = 2, patterns: bool = True
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
            Read every file again for statement shapes and ``Prop``
            definitions (linear in the project's size). ``False`` skips
            that pass: no statement shapes, no claim definitions, no
            ``same_statement`` on parallel declarations. Default: ``True``.

        Returns
        -------
        dict
            JSON-ready; ``report`` is ``"discovery"``. Sections:

            - ``rules``: ``load_bearing_theorems`` and ``central_concepts``,
              ranked by the files whose declarations name them, then by
              the declarations;
            - ``patterns``: ``statement_shapes`` (theorems whose statements
              are equal once bound names are renamed in order and numerals
              replaced) and ``numbered_series`` (names differing only by a
              trailing number);
            - ``claims``: ``claim_definitions`` (definitions of type
              ``Prop``), ``axioms``, ``proof_wanted``, ``unproven``
              (``sorry`` or ``admit``);
            - ``anomalies``: ``parallel_declarations`` (public names declared
              in several files), ``generators`` (code adding declarations
              at build time), ``isolated_results`` (theorems no declaration
              names, outside the headline modules);
            - ``headline_results``: theorems of the modules no project file
              imports that nothing names.

        Raises
        ------
        ValueError
            If *top* or *min_instances* is not a positive integer.

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
        same report. One pass over the edges counts uses; with
        *patterns*, one more pass reads each file's statements with
        proofs left out.
        """
        from ._lean_discovery import discover  # noqa: PLC0415  (imports this module)

        return discover(self, top=top, min_instances=min_instances, patterns=patterns)

    def export(
        self,
        folder: str | Path,
        *,
        targets: Sequence[str] = (),
        top: int = 50,
        patterns: bool = True,
    ) -> list[str]:
        """
        Write helper documents on the project, for people, models and corpora.

        Parameters
        ----------
        folder : str or pathlib.Path
            Where to write; created if missing. Files already there under
            the same names are replaced, each atomically.
        targets : sequence of str, optional
            Declarations to write a dossier on, besides the headline
            results. Default: none.
        top : int, optional
            As for :meth:`discover`. Default: 50.
        patterns : bool, optional
            As for :meth:`discover`. Default: ``True``.

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
        from ._lean_discovery import export  # noqa: PLC0415  (imports this module)

        return export(self, folder, targets=targets, top=top, patterns=patterns)

    # -- output -------------------------------------------------------------

    @staticmethod
    def render(report: dict[str, Any]) -> str:
        """
        Render a :meth:`summary`, :meth:`dossier` or :meth:`discover` report as Markdown.

        Every item keeps its ``file:line``, so the text is as checkable as
        the data.
        """
        if report.get("report") == "summary":
            return _render_summary(report)
        if report.get("report") == "discovery":
            from ._lean_discovery import render  # noqa: PLC0415  (imports this module)

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
    return {"count": len(items), "items": items[:max_items], "cut": len(items) > max_items}


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"check": name, "status": status, "detail": detail}


def _bound_names(statement: str) -> set[str]:
    """Return the names a statement binds: ``(s : ℂ)``, ``{x y}``, ``∀ n``, ``fun x``."""
    binder, quantified = _binders()
    lexer = _Lexer()
    masked = "\n".join(lexer.mask(line) for line in statement.split("\n"))
    names: set[str] = set()
    for found in binder.finditer(masked):
        names.update(found.group(1).split())
    for found in quantified.finditer(masked):
        names.update(found.group(1).split())
    return names


def _where(item: dict[str, Any]) -> str:
    return f"`{item['file']}:{item['line_start']}`"


def _render_summary(report: dict[str, Any]) -> str:
    out = [
        "# Lean project summary",
        "",
        f"{report['files']:,} files, {report['lines']:,} lines,"
        f" {report['declarations']:,} declarations.",
        "",
        "## Audit flags",
    ]
    if not report["flags"]:
        out.append("None found in code.")
    for flag, listing in report["flags"].items():
        out.append(f"- **{flag}**: {listing['count']}")
        out.extend(f"  - {item['declaration'] or item['kind']} at {_where(item)}" for item in listing["items"])
        if listing["cut"]:
            out.append("  - (list cut)")
    out += ["", "## Imports"]
    roots = report["imports"]["external_roots"]
    out.append(
        "External: " + (", ".join(f"{r} ({n})" for r, n in roots.items()) or "none") + "."
    )
    missing = report["imports"]["missing_modules"]
    out.append(f"Missing project modules: {missing['count']}.")
    out.extend(f"- `{m['module']}` (imported by {', '.join(m['imported_by'][:3])})" for m in missing["items"])
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
        f"Names declared more than once (as Lean counts): {report['duplicate_names']['count']}."
        f" Syntax extensions: {report['syntax_extensions']['count']}."
        f" Project commands learned: {len(learned)}."
        f" Skipped files: {len(report['skipped_files'])}.",
    ]
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
        out += [
            f"- `{item['declaration']}` ({item['kind']}, {_where(item)})",
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
        f"{deps['declarations']:,} declarations in {deps['files']:,} files"
        f" (found by name: {deps['found_by']['exact']:,} exact,"
        f" {deps['found_by']['prefix']:,} by prefix, {deps['found_by']['suffix']:,} by suffix).",
    ]
    for item in deps["red_flags"]["items"]:
        hops = item["path"]
        chain = (
            " → ".join([f"`{hops[0]['from']}`"] + [f"`{h['declaration']}`" for h in hops])
            if hops
            else "the target itself"
        )
        out.append(f"- **{', '.join(item['flags'])}** in `{item['declaration']}` at {_where(item)}: {chain}")
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
    out.extend(f"- {c['check']}: **{c['status']}** ({c['detail']})" for c in report["checks"])
    out += ["", "## To settle it", *[f"- `{step}`" for step in report["kernel_steps"]]]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """
    Index a Lean project and print its summary or a dossier per target.

    Returns 1 when ``--fail-on`` names a flag found (in a target's
    dependencies or implicit risks; in the whole project without targets),
    0 otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scikitplot.corpus._lean_evidence",
        description="Evidence on a Lean project: statements, dependencies, audit flags.",
    )
    parser.add_argument("folder", type=Path)
    parser.add_argument("--target", action="append", default=[], metavar="NAME")
    parser.add_argument("--json", action="store_true", help="JSON instead of Markdown")
    parser.add_argument("--dialect", default="lean4", choices=("lean4", "lean3"))
    parser.add_argument("--command", action="append", default=[], metavar="KEYWORD")
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument("--fail-on", action="append", default=[], metavar="FLAG")
    parser.add_argument(
        "--discover", action="store_true", help="print rules, patterns, claims and anomalies"
    )
    parser.add_argument(
        "--export", type=Path, metavar="DIR", help="also write the helper documents to DIR"
    )
    parser.add_argument("--no-learn-commands", action="store_true")
    args = parser.parse_args(argv)
    try:
        index = LeanIndex.from_folder(
            args.folder,
            include_hidden=args.include_hidden,
            learn_commands=not args.no_learn_commands,
            dialect=args.dialect,
            commands=args.command,
        )
        if args.target:
            reports = [index.dossier(t, max_items=args.max_items) for t in args.target]
        elif args.discover:
            reports = [index.discover(top=args.max_items)]
        else:
            reports = [index.summary(max_items=args.max_items)]
        if args.export is not None:
            index.export(args.export, targets=args.target, top=args.max_items)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(reports if args.target else reports[0], indent=1, ensure_ascii=False))  # noqa: T201
    else:
        print("\n".join(LeanIndex.render(r) for r in reports))  # noqa: T201
    found: set[str] = set()
    for report in reports:
        if report["report"] == "dossier":
            found.update(report["flags_found"])
        else:
            found.update(index.summary(max_items=1)["flags"])
    return 1 if found & set(args.fail_on) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
