# scikitplot/corpus/_readers/_lean.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Reader for Lean source files (Lean 4, and Lean 3 with ``dialect="lean3"``).

:class:`LeanReader` splits a ``.lean`` file into its commands. Each
declaration (``theorem``, ``lemma``, ``def``, ``structure``, ``instance``,
...), together with its doc comment and attributes, becomes one chunk that
names its kind, its fully qualified name, its namespace and the lines it
came from. Every non-blank line of the file lands in exactly one chunk, so
any retrieved text points back to its file and lines, in a project of any
size.

The reader also marks, per chunk, what a reviewer of a formal proof checks
first: ``sorry`` and ``admit``, declared axioms, ``native_decide``,
``unsafe`` code, ``implemented_by`` and ``extern``, a disabled kernel check
and text after ``#exit``. These marks say what the source text contains;
whether a theorem depends on them is what ``#print axioms`` in Lean
answers.
"""  # noqa: D205, D400

from __future__ import annotations

from collections.abc import Collection as _Collection
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (  # noqa: F401
    Any,
    ClassVar,
    Collection,
    Generator,
    Mapping,
    Optional,
)

try:  # the Lean language core
    from ...lean._syntax import LEAN_FLAGS, _is_name, _Splitter
except ModuleNotFoundError as _missing:
    # The partial scikit-plots-corpus without its ``lean`` extra. The reader
    # stays registered for ``.lean`` so using it says what to install; any
    # other missing module is a bug and is raised.
    if not (_missing.name or "").split(".")[:2] == ["scikitplot", "lean"]:
        raise
    from ..._distributions import install_hint

    LEAN_FLAGS = MappingProxyType({})
    _is_name = _Splitter = None  # type: ignore[assignment]
    _LEAN_MISSING: str | None = (
        "LeanReader needs scikitplot.lean, which is not installed: "
        f"{install_hint('scikitplot.lean')} (or the corpus extra: "
        "scikit-plots-corpus[lean])."
    )
else:
    _LEAN_MISSING = None
from ._code import CodeReader

__all__ = ["LEAN_FLAGS", "LeanReader"]


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _strings(value: Any, option: str, *, words: bool = False) -> tuple[str, ...]:
    """Validate a collection of non-empty strings for *option*."""
    if isinstance(value, (str, bytes)) or not isinstance(value, _Collection):
        raise TypeError(
            f"LeanReader: {option} must be a collection of str (a list, a set);"
            f" got {value!r}."
        )
    items = tuple(value)
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"LeanReader: {option} must hold str; got {item!r}.")
        if not item.strip():
            raise ValueError(
                f"LeanReader: {option} must hold non-empty str; got {item!r}."
            )
        if words and (item != item.strip() or any(ch.isspace() for ch in item)):
            raise ValueError(
                f"LeanReader: {option} items are single tokens without spaces;"
                f" got {item!r}."
            )
    return items


@dataclass
class LeanReader(CodeReader):
    r"""
    Read a Lean source file one command at a time, each chunk naming its declaration.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the ``.lean`` file.
    proofs : {"keep", "drop"}, optional
        ``"keep"`` (default) keeps every line. ``"drop"`` leaves out the
        proofs of ``theorem``, ``lemma`` and ``example``: the chunk holds
        the statement (doc comment and attributes included) and names the
        proof's lines in ``proof_line_start`` and ``proof_line_end``. A
        statement index of a large development is a fraction of its size.
    kinds : collection of str or None, optional
        Only chunks whose ``lean_kind`` is in this collection are yielded,
        for example ``{"theorem", "lemma"}``. ``None`` (default): all.
    watch : collection of str, optional
        Identifiers to find: each chunk lists in ``watched`` those it uses
        in code. A name matches whole, or as the last components of a
        dotted name (``keyLemma`` matches ``Main.keyLemma`` and the
        ``keyLemma`` of ``keyLemma.symm``). Default: none.
    commands : collection of str, optional
        Further command keywords that start a chunk of their own at column
        0, for commands a project defines. Default: none.
    references : bool, optional
        Also record, per chunk, the identifiers its code uses
        (``references``) and the namespaces open for it (``opens``): what
        :class:`~scikitplot.lean.LeanIndex` resolves into a dependency
        graph. Default: ``False``.
    dialect : {"lean4", "lean3"}, optional
        ``"lean3"`` also reads Lean 3's ``begin ... end`` and
        ``match ... end`` blocks, whose ``end`` closes no namespace, and
        its ``| pattern := proof`` equations. Default: ``"lean4"``.
    max_scope_depth : int, optional
        Most namespaces and sections open at once; deeper nesting is
        refused. Default: 256.
    lines_per_chunk : int or None, optional
        A command longer than this is yielded in parts of at most this many
        lines, each with ``part`` (1, 2, ...). ``None``: one chunk per
        command. Default: 200.
    encoding, max_file_bytes, language
        As for :class:`~scikitplot.corpus._readers.CodeReader`.

    Attributes
    ----------
    flag_tokens : Mapping of str to str
        Class variable: code token to audit flag, :data:`LEAN_FLAGS`.
        Subclass to change it.

    Raises
    ------
    TypeError
        If ``kinds``, ``watch`` or ``commands`` is not a collection of str,
        or ``references`` is not a bool.
    ValueError
        If an option's value is invalid, the file is too large or binary,
        or it nests more than ``max_scope_depth`` scopes.

    See Also
    --------
    scikitplot.corpus._readers.CodeReader : Other source code, in line blocks.

    Notes
    -----
    **User note.** Each chunk carries ``language`` (``"lean"``),
    ``line_start`` and ``line_end`` (1-based, inclusive, the file's own
    line numbers), ``encoding``, and:

    - ``lean_kind``: the command keyword (``"theorem"``, ``"def"``,
      ``"instance"``, ``"#eval"``, ...); ``"import"`` for the module
      header, ``"context"`` for a run of ``open``, ``variable``,
      ``namespace``, ``section``, ``end``, ``set_option`` ...;
      ``"moduledoc"`` for a ``/-! -/`` comment; ``"comment"`` for comments
      apart from any command; ``"command"`` for code before any known
      command.
    - ``declaration``: the fully qualified name, namespaces included
      (``_root_.`` removed); absent for unnamed declarations
      (``example``, an unnamed ``instance``).
    - ``namespace``, ``modifiers`` (``private``, ``noncomputable``, ...),
      ``attributes`` (``simp``, ``implemented_by``, ...) and ``imports``,
      when present.
    - ``flags``: audit marks found in the chunk's code (see
      :data:`LEAN_FLAGS`); with ``proofs="drop"`` the dropped proof's
      marks are on its statement.
    - ``watched``, ``part``, ``proof_line_start``, ``proof_line_end``:
      see the options.
    - With ``references=True``: ``references``, the identifiers the
      chunk's code uses, sorted, without Lean's own words and the chunk's
      own name (with ``proofs="drop"``, the dropped proof's too); and
      ``opens``, the namespaces open for it (its file's, its scopes', its
      own ``open .. in``). Which declaration an identifier names is for
      :class:`~scikitplot.lean.LeanIndex` to resolve.

    With ``proofs="keep"`` and ``kinds=None`` every non-blank line of the
    file is in exactly one chunk, in order: the line ranges never overlap.

    **What starts a command.** A known command keyword at column 0, after
    any attributes and modifiers (``@[simp] private theorem``); a
    declaration keyword (``theorem``, ``def``, ``instance``, ...) or a
    scope keyword at any indentation, since those never begin a line of a
    proof; inside ``mutual``, the indentation of its first declaration. A
    doc comment, an attribute line and comments directly above a command
    belong to it; ``open Foo in`` and ``set_option o v in`` join the
    command they prefix, so its flags include the option. Any other line
    continues the command before it: a command a project defines with
    ``syntax`` is read as part of the preceding chunk unless it is named
    in ``commands``. A declaration's name is read from its keyword's
    line.

    **Lexical audit, not proof checking.** ``flags`` say what the text
    contains, outside comments and strings. A macro that expands to
    ``sorry`` is flagged where it is defined, not where it is used, and an
    axiom is flagged where it is declared, not in the theorems that use
    it. Whether a theorem is proved from Lean's standard axioms alone is
    what ``#print axioms`` (and an independent kernel replay) answers.

    **Statement and proof.** With ``proofs="drop"`` the statement ends at
    the first ``:=`` outside brackets that is not a ``let``'s or a
    ``have``'s, at a ``where``, or at a line that starts with ``|`` and
    holds ``=>`` (``:=`` in Lean 3): the equations of a proof by pattern
    matching. Definitions keep their bodies; their values are their
    meaning.

    **Developer note.** The file is decoded as
    :class:`~scikitplot.corpus._readers.TextReader` decodes it, then read
    line by line through a lexer that blanks comments, strings and
    character literals (Lean's rules: nested ``/- -/``, ``/--`` and
    ``/-!`` docs, escapes and gaps, ``s!`` interpolation, ``r#"..."#`` raw
    strings, ``'`` primes, «names» over lines). Every pattern moves
    forward, so time is linear in the file's size, and memory is one
    command (a long one is emitted in parts as it is read) plus at most
    ``max(lines_per_chunk, 256)`` comment lines waiting for the next
    command.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = LeanReader(input_path=Path("Main.lean"))  # doctest: +SKIP
    >>> [
    ...     c.get("declaration")
    ...     for c in reader.get_raw_chunks()
    ...     if c["lean_kind"] == "theorem"
    ... ]  # doctest: +SKIP
    ['mainTheorem', ...]

    Statements only, with every use of a name located:

    >>> reader = LeanReader(  # doctest: +SKIP
    ...     input_path=Path("Main.lean"), proofs="drop", watch=["keyLemma"]
    ... )
    """

    file_type: ClassVar[str] = ".lean"
    file_types: ClassVar[list[str] | None] = [".lean"]
    languages: ClassVar[Mapping[str, str]] = MappingProxyType({".lean": "lean"})
    flag_tokens: ClassVar[Mapping[str, str]] = LEAN_FLAGS

    proofs: str = field(default="keep")
    """``"keep"`` every line, or ``"drop"`` the proofs of theorems."""

    kinds: Optional[Collection[str]] = field(default=None)  # noqa: UP045
    """Only these ``lean_kind`` values are yielded; ``None``: all."""

    watch: Collection[str] = field(default=())
    """Identifiers to locate; each chunk lists those it uses in ``watched``."""

    commands: Collection[str] = field(default=())
    """Further command keywords that start a chunk at column 0."""

    dialect: str = field(default="lean4")
    """``"lean4"`` or ``"lean3"``."""

    max_scope_depth: int = field(default=256)
    """Most namespaces and sections open at once."""

    references: bool = field(default=False)
    """Record the identifiers each chunk uses and the namespaces open for it."""

    def __post_init__(self) -> None:
        """Validate the options."""
        if _LEAN_MISSING is not None:
            raise ImportError(_LEAN_MISSING)
        super().__post_init__()
        if self.proofs not in ("keep", "drop"):
            raise ValueError(
                f"LeanReader: proofs must be 'keep' or 'drop'; got {self.proofs!r}."
            )
        if self.dialect not in ("lean4", "lean3"):
            raise ValueError(
                f"LeanReader: dialect must be 'lean4' or 'lean3'; got {self.dialect!r}."
            )
        depth = self.max_scope_depth
        if isinstance(depth, bool) or not isinstance(depth, int) or depth <= 0:
            raise ValueError(
                f"LeanReader: max_scope_depth must be an int > 0; got {depth!r}."
            )
        if not isinstance(self.references, bool):
            raise TypeError(
                f"LeanReader: references must be a bool; got {self.references!r}."
            )
        if self.kinds is not None:
            self.kinds = frozenset(_strings(self.kinds, "kinds"))
        self.watch = _strings(self.watch, "watch", words=True)
        commands = _strings(self.commands, "commands", words=True)
        for command in commands:
            if not _is_name(command.rstrip("%")):
                raise ValueError(
                    f"LeanReader: commands holds identifiers; got {command!r}."
                )
        self.commands = frozenset(commands)

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Yield the file's commands, each with its line range and declaration.

        Yields
        ------
        dict
            ``text``, ``section_type``, ``source_type``, ``language``,
            ``encoding``, ``line_start``, ``line_end``, ``lean_kind`` and,
            when they apply, ``declaration``, ``namespace``, ``modifiers``,
            ``attributes``, ``imports``, ``flags``, ``watched``, ``part``,
            ``proof_line_start`` and ``proof_line_end``.

        Raises
        ------
        ValueError
            If the file is too large or binary, or nests more than
            ``max_scope_depth`` scopes.
        """
        codec = self._codec_of_file()
        if codec is None:
            return
        splitter = _Splitter(
            file_name=self.file_name,
            meta={"language": self._language(), "encoding": codec},
            step=self.lines_per_chunk,
            cap=self._text_limit()[0],
            proofs=self.proofs,
            dialect=self.dialect,
            kinds=self.kinds,
            watch=tuple(self.watch),
            commands=frozenset(self.commands),
            max_scope_depth=self.max_scope_depth,
            flag_tokens=self.flag_tokens,
            references=self.references,
        )
        # Universal newlines, as CodeReader reads: line numbers match an
        # editor's, and a "\r\n" file reads as Lean reads it.
        with open(self.input_path, encoding=codec, newline=None) as stream:
            for number, read in enumerate(stream, start=1):
                line = read[:-1] if read.endswith("\n") else read
                if number == 1 and line.startswith("\ufeff"):
                    line = line[1:]
                chunks = splitter.feed(number, line)
                if chunks:
                    yield from chunks
        yield from splitter.finish()
