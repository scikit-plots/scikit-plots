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

import bisect
import logging
import re
from collections.abc import Collection as _Collection
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (  # noqa: F401
    Any,
    ClassVar,
    Collection,
    Generator,
    Iterator,
    Mapping,
    Optional,
)

from .._schema import SectionType, SourceType
from ._code import CodeReader

logger = logging.getLogger(__name__)

__all__ = ["LEAN_FLAGS", "LeanReader"]

# ---------------------------------------------------------------------------
# Audit flags
# ---------------------------------------------------------------------------

_LEAN_FLAGS = {
    "sorry": "sorry",
    "sorryAx": "sorry",
    "admit": "admit",
    "native_decide": "native",
    "ofReduceBool": "native",
    "+native": "native",
    "unsafe": "unsafe",
    "implemented_by": "implemented_by",
    "extern": "extern",
    "skipKernelTC": "skip_kernel_tc",
    "addDecl": "add_decl",
    "addAndCompile": "add_decl",
}

#: Code token to the audit flag it raises on its chunk. Tokens are matched
#: as whole identifiers (or as the last components of a dotted one, so
#: ``debug.skipKernelTC`` and ``Lean.ofReduceBool`` count) in code only,
#: never in comments or strings. Two flags come from the structure, not a
#: token: ``"axiom"`` on an ``axiom`` declaration, and ``"after_exit"`` on
#: every chunk after ``#exit`` (Lean does not read that text at all).
LEAN_FLAGS: Mapping[str, str] = MappingProxyType(_LEAN_FLAGS)

# ---------------------------------------------------------------------------
# Command keywords
# ---------------------------------------------------------------------------

#: Commands that declare something: each is a chunk of its own.
_DECLARATIONS = frozenset(
    {
        "theorem",
        "lemma",
        "def",
        "abbrev",
        "instance",
        "example",
        "axiom",
        "opaque",
        "structure",
        "class",
        "inductive",
        "coinductive",
        "macro",
        "macro_rules",
        "syntax",
        "elab",
        "elab_rules",
        "notation",
        "infix",
        "infixl",
        "infixr",
        "prefix",
        "postfix",
        "binder_predicate",
        "declare_syntax_cat",
        "initialize",
        "builtin_initialize",
        "alias",
        "irreducible_def",
        "simproc",
        "dsimproc",
        "simproc_decl",
        "dsimproc_decl",
        "builtin_simproc",
        "builtin_dsimproc",
        "builtin_simproc_decl",
        "builtin_dsimproc_decl",
        "notation3",
        "proof_wanted",  # a stated claim without a proof (Batteries)
        "recall",
    }
)

#: Declarations whose first identifier after the keyword is their name.
_NAMED = frozenset(
    {
        "theorem",
        "lemma",
        "def",
        "abbrev",
        "instance",
        "axiom",
        "opaque",
        "structure",
        "class",
        "inductive",
        "coinductive",
        "declare_syntax_cat",
        "initialize",
        "builtin_initialize",
        "alias",
        "irreducible_def",
        "simproc",
        "dsimproc",
        "simproc_decl",
        "dsimproc_decl",
        "builtin_simproc",
        "builtin_dsimproc",
        "builtin_simproc_decl",
        "builtin_dsimproc_decl",
        "proof_wanted",
        "recall",
    }
)

#: Declarations that start a command even when indented. These words are
#: keywords, so they never begin a line of a proof or a term; projects that
#: indent their commands inside a namespace are read correctly.
_INDENTED = frozenset(
    {
        "theorem",
        "lemma",
        "def",
        "abbrev",
        "instance",
        "example",
        "axiom",
        "opaque",
        "structure",
        "class",
        "inductive",
        "coinductive",
        "macro",
        "macro_rules",
        "syntax",
        "elab",
        "elab_rules",
        "notation",
        "infix",
        "infixl",
        "infixr",
        "prefix",
        "postfix",
        "declare_syntax_cat",
        "irreducible_def",
    }
)

#: Commands that open or close a scope.
_SCOPES = frozenset({"namespace", "section", "end", "mutual"})

#: Commands that set context for what follows; a run of them is one chunk.
_CONTEXT = frozenset(
    {
        "open",
        "export",
        "variable",
        "variable?",
        "universe",
        "set_option",
        "attribute",
        "include",
        "omit",
        # Lean 3
        "variables",
        "universes",
        "parameter",
        "parameters",
        "open_locale",
        "theory",  # `noncomputable theory`
    }
)

#: The module header: ``module``, ``prelude`` and imports.
_IMPORTS = frozenset({"import", "module", "prelude"})

#: Other commands with a chunk of their own (``#``-commands are found by
#: their ``#``).
_OTHER = frozenset(
    {
        "run_cmd",
        "run_elab",
        "run_meta",
        "run_tac",
        "add_decl_doc",
        "assert_not_exists",
        "assert_not_imported",
        "library_note",
        "register_option",
        "register_builtin_option",
        "register_simp_attr",
        "register_label_attr",
        "initialize_simps_projections",
        "compile_inductive%",
        "compile_def%",
        "seal",
        "unseal",
        "suppress_compilation",
        "grind_pattern",
        "recommended_spelling",
        "gen_injective_theorems%",
        "docs_to_verso",
        "declare_config_elab",
        "declare_simp_like_tactic",
        "register_error_explanation",
        "register_linter_set",
        "register_tactic_tag",
        "tactic_extension",
        "deprecated_module",
        "deprecated_syntax",
        "allow_unused_tactic",
        "unif_hint",
        "init_quot",
        "declare_aesop_rule_sets",
        "add_aesop_rules",
        # Lake, in lakefile.lean
        "package",
        "require",
        "lean_lib",
        "lean_exe",
        "extern_lib",
        "script",
        "target",
        "input_file",
        "input_dir",
        "post_update",
        "module_facet",
        "package_facet",
        "library_facet",
        # Lean 3 (mathlib)
        "localized",
        "restate_axiom",
        "add_tactic_doc",
        "mk_simp_attribute",
    }
)

_MODIFIERS = frozenset(
    {
        "private",
        "protected",
        "noncomputable",
        "partial",
        "unsafe",
        "nonrec",
        "scoped",
        "local",
        "public",
        "meta",
    }
)

#: Words that follow a keyword but are never a declaration's name.
_RESERVED = frozenset(
    {
        "where",
        "extends",
        "deriving",
        "with",
        "do",
        "fun",
        "by",
        "at",
        "from",
        "in",
        "if",
        "then",
        "else",
        "match",
        "let",
        "have",
        "show",
        "calc",
        "Type",
        "Sort",
        "Prop",
    }
)

#: Declarations whose proof ``proofs="drop"`` leaves out.
_DROPPABLE = frozenset({"theorem", "lemma", "example"})

#: Tokens read after a declaration keyword to find its name: enough for
#: `class inductive`, Lean 3 universes and two `(priority := n)` groups.
_NAME_LOOKAHEAD = 24

_KNOWN = _DECLARATIONS | _SCOPES | _CONTEXT | _IMPORTS | _OTHER | {"deriving"}

# ---------------------------------------------------------------------------
# Lexical patterns (each linear: no pattern scans to a closer past an opener)
# ---------------------------------------------------------------------------

_CODE_SPECIAL = re.compile(r'--|/-|"|\'|«|(?<![\w\'!?.])r#*"')
_CODE_SPECIAL_INTERP = re.compile(r'--|/-|"|\'|«|(?<![\w\'!?.])r#*"|\{|\}')
_COMMENT_TOKEN = re.compile(r"/-|-/")
_STRING_TOKEN = re.compile(r'\\.?|"')
_STRING_TOKEN_INTERP = re.compile(r'\\.?|"|\{')
_CHAR_LITERAL = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)|[^\\'])'")

_NAME = r"(?:«[^«»]*»|[^\W\d][\w'!?]*)(?:\.(?:«[^«»]*»|[^\W\d][\w'!?]*|\d+))*"
_HEAD_TOKEN = re.compile(rf"@\[|#[^\W\d][\w'!?]*|{_NAME}%?|:=|\S")
_FULL_NAME = re.compile(rf"{_NAME}\Z")
_COMPONENT = re.compile(r"«[^«»]*»|[^.«»]+")

_OPENERS = "([{⟨⦃⟦⟪⁅"
_OPEN_BRACKET = re.compile(r"[(\[{⟨⦃⟦⟪⁅]")
_CLOSE_BRACKET = re.compile(r"[)\]}⟩⦄⟧⟫⁆]")
_STATEMENT_TOKEN = re.compile(
    r"[(\[{⟨⦃⟦⟪⁅]|[)\]}⟩⦄⟧⟫⁆]|:=|(?<![\w'!?.])(?:let|have|where)(?![\w'!?])"
)
_ENDS_IN = re.compile(r"(?<![\w'!?.])in\s*\Z")
_LEAN3_BLOCK = re.compile(r"(?<![\w'!?.])(?:begin|match|end)(?![\w'!?])")


def _is_ident_char(ch: str) -> bool:
    """Return ``True`` for a character that continues a Lean identifier."""
    return ch.isalnum() or ch in "_'!?"


def _is_name(lexeme: str) -> bool:
    """Return ``True`` when *token* is an identifier that may name a declaration."""
    return (
        lexeme not in _RESERVED
        and lexeme != "_"
        and (lexeme[0] == "«" or lexeme[0] == "_" or lexeme[0].isalpha())
        and _FULL_NAME.match(lexeme) is not None
    )


def _token_pattern(tokens: Collection[str]) -> re.Pattern[str] | None:
    """Compile *tokens* into one pattern matching them as whole identifiers."""
    if not tokens:
        return None
    alternatives = "|".join(
        re.escape(lexeme) for lexeme in sorted(tokens, key=len, reverse=True)
    )
    return re.compile(rf"(?<![\w'!?])(?:{alternatives})(?![\w'!?])")


def _components(name: str) -> list[str]:
    """Split a dotted name into its components (``«a.b».c`` is two)."""
    return _COMPONENT.findall(name) if "«" in name else name.split(".")


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


class _Lexer:
    r"""
    Mask a Lean file line by line: comments, strings and char literals become spaces.

    The masked line has the length of the line, so columns still match;
    what remains is code, including identifiers quoted in «guillemets» and
    the expressions inside an interpolated string's braces. The state
    (an open comment, string, raw string or «name») runs on over lines.

    Notes
    -----
    The rules are Lean's lexical rules: ``--`` starts a line comment;
    block comments ``/- -/`` nest, and ``/--`` and ``/-!`` open doc
    comments; strings take escapes (``\\"``, ``\\{``) and may run over
    lines; ``s!"..{e}.."`` interpolates code; ``r"..."`` and ``r#"..."#``
    are raw strings; a ``'`` after an identifier character is a prime
    (``h'``), elsewhere it opens a character literal. Every step moves
    forward, so a line costs time linear in its length.
    """

    __slots__ = ("stack",)

    def __init__(self) -> None:
        self.stack: list[list[Any]] = [["code", None]]

    def in_code(self) -> bool:
        """Return ``True`` when the next line starts in code, not inside a literal or comment."""
        return len(self.stack) == 1

    def mask(self, line: str) -> str:  # noqa: PLR0912, PLR0915
        """Return *line* with every character that is not code replaced by a space."""
        stack = self.stack
        n = len(line)
        pos = 0
        out: list[str] = []
        while pos < n:
            top = stack[-1]
            kind = top[0]
            if kind == "code":
                pattern = _CODE_SPECIAL if top[1] is None else _CODE_SPECIAL_INTERP
                found = pattern.search(line, pos)
                if found is None:
                    out.append(line[pos:])
                    break
                start = found.start()
                lexeme = found.group()
                if start > pos:
                    out.append(line[pos:start])
                if lexeme == "--":
                    out.append(" " * (n - start))
                    break
                if lexeme == "/-":
                    # "/--" and "/-!" open doc comments: three characters.
                    width = 3 if line[start + 2 : start + 3] in ("-", "!") else 2
                    out.append(" " * width)
                    stack.append(["comment", 1])
                    pos = start + width
                elif lexeme == '"':
                    interpolated = (
                        start > 1
                        and line[start - 1] == "!"
                        and _is_ident_char(line[start - 2])
                    )
                    out.append(" ")
                    stack.append(["string", interpolated])
                    pos = start + 1
                elif lexeme == "'":
                    literal = None
                    if start == 0 or not _is_ident_char(line[start - 1]):
                        literal = _CHAR_LITERAL.match(line, start)
                    if literal is None:  # a prime ending an identifier
                        out.append("'")
                        pos = start + 1
                    else:
                        out.append(" " * (literal.end() - start))
                        pos = literal.end()
                elif lexeme == "«":
                    close = line.find("»", start + 1)
                    if close < 0:  # a «name» may run over lines
                        out.append(line[start:])
                        stack.append(["name"])
                        break
                    out.append(line[start : close + 1])
                    pos = close + 1
                elif lexeme == "{":
                    top[1] += 1
                    out.append("{")
                    pos = start + 1
                elif lexeme == "}":
                    if top[1] == 0:  # back into the interpolated string
                        stack.pop()
                        out.append(" ")
                    else:
                        top[1] -= 1
                        out.append("}")
                    pos = start + 1
                else:  # r#*" opens a raw string
                    out.append(" " * len(lexeme))
                    stack.append(["raw", '"' + "#" * (len(lexeme) - 2)])
                    pos = found.end()
            elif kind == "comment":
                found = _COMMENT_TOKEN.search(line, pos)
                if found is None:
                    out.append(" " * (n - pos))
                    break
                out.append(" " * (found.end() - pos))
                pos = found.end()
                if found.group() == "/-":
                    top[1] += 1
                else:
                    top[1] -= 1
                    if top[1] == 0:
                        stack.pop()
            elif kind == "string":
                pattern = _STRING_TOKEN_INTERP if top[1] else _STRING_TOKEN
                found = pattern.search(line, pos)
                if found is None:
                    out.append(" " * (n - pos))
                    break
                out.append(" " * (found.end() - pos))
                pos = found.end()
                if found.group() == '"':
                    stack.pop()
                elif found.group() == "{":
                    stack.append(["code", 0])
            elif kind == "raw":
                close = line.find(top[1], pos)
                if close < 0:
                    out.append(" " * (n - pos))
                    break
                end = close + len(top[1])
                out.append(" " * (end - pos))
                pos = end
                stack.pop()
            else:  # "name": a «name» running over lines
                close = line.find("»", pos)
                if close < 0:
                    out.append(line[pos:])
                    break
                out.append(line[pos : close + 1])
                pos = close + 1
                stack.pop()
        return "".join(out)

    def open_construct(self) -> str | None:
        """Name what is still open at the end of the file, or ``None``."""
        if self.in_code():
            return None
        return {"code": "string", "comment": "comment", "string": "string"}.get(
            self.stack[-1][0], "raw string" if self.stack[-1][0] == "raw" else "«name»"
        )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


class _Head:
    """What the first line of a command says about it."""

    __slots__ = (
        "attributes",
        "body_col",
        "category",
        "imports",
        "kind",
        "modifiers",
        "name",
        "open_depth",
        "scope",
    )

    def __init__(self, category: str, kind: str) -> None:
        self.category = category
        self.kind = kind
        self.name: str | None = None
        self.modifiers: list[str] = []
        self.attributes: list[str] = []
        self.imports: list[str] = []
        self.scope: tuple[str, str | None] | None = None
        self.body_col = 0
        self.open_depth = 0


class _Held:
    """A blank, comment, doc-comment or attribute line not yet given to a command."""

    __slots__ = ("indent", "kind", "masked", "number", "text")

    def __init__(self, number: int, text: str, masked: str, kind: str, indent: int):
        self.number = number
        self.text = text
        self.masked = masked
        self.kind = kind
        self.indent = indent


class _Statement:
    """Find where a theorem's statement ends and its proof begins, line by line."""

    __slots__ = ("arrow", "col", "depth", "number", "pending")

    def __init__(self, number: int, col: int, arrow: str) -> None:
        self.number = number
        self.col = col
        self.depth = 0
        self.pending = 0  # `let`/`have` waiting for their own `:=`
        self.arrow = (
            arrow  # what follows an equation's `|`: "=>" (Lean 4), ":=" (Lean 3)
        )

    def scan(self, number: int, masked: str) -> int | None:  # noqa: PLR0912
        """Return the column where the proof starts on this line, or ``None``."""
        if number < self.number:
            return None  # doc comment and attribute lines above the keyword
        start = self.col if number == self.number else 0
        if number != self.number and not self.depth:
            body = masked.lstrip()
            if body.startswith("|") and not body.startswith("||"):
                col = len(masked) - len(body)
                if self.arrow in masked[col:]:
                    return col  # `| pattern => proof`: equations start the proof
        for found in _STATEMENT_TOKEN.finditer(masked, start):
            lexeme = found.group()
            if lexeme == ":=":
                if self.depth:
                    continue
                if self.pending:
                    self.pending -= 1
                    continue
                return found.start()
            if lexeme in ("let", "have"):
                if not self.depth:
                    self.pending += 1
            elif lexeme == "where":
                if not self.depth:
                    return found.start()
            elif lexeme in _OPENERS:
                self.depth += 1
            elif self.depth:
                self.depth -= 1
        return None


class _Command:
    """One command being read: its lines, what it declares, and what it holds."""

    __slots__ = (
        "after_exit",
        "attributes",
        "boundary",
        "category",
        "consumed",
        "depth",
        "ends_in",
        "hit_flags",
        "hit_lines",
        "imports",
        "kind",
        "last",
        "lines",
        "modifiers",
        "name",
        "namespace",
        "parts",
        "size",
        "statement",
        "watch_lines",
        "watch_names",
    )

    def __init__(self, head: _Head, namespace: str, after_exit: bool) -> None:
        self.category = head.category
        self.kind = head.kind
        self.name: str | None = None
        self.namespace = namespace
        self.modifiers: list[str] = []
        self.attributes: list[str] = []
        self.imports: list[str] = []
        self.lines: list[tuple[int, str]] = []
        self.size = -1
        self.hit_lines: list[int] = []
        self.hit_flags: list[str] = []
        self.watch_lines: list[int] = []
        self.watch_names: list[str] = []
        self.consumed = 0  # lines before this one were emitted in earlier parts
        self.last = 0
        self.after_exit = after_exit
        self.ends_in = False
        self.statement: _Statement | None = None
        self.boundary: tuple[int, int] | None = None
        self.parts = 0
        self.depth = 0  # brackets open at the end of the last line


def _between(lines: list[int], values: list[str], start: int, end: int) -> set[str]:
    """Return the values whose line is in ``start..end`` (*lines* is sorted)."""
    low = bisect.bisect_left(lines, start)
    high = bisect.bisect_right(lines, end)
    return set(values[low:high])


def _scan_prefix(  # noqa: PLR0912
    masked: str, pos: int
) -> tuple[list[str], list[str], Any, int]:
    """
    Read attributes and modifiers from *pos*; return them and the next token.

    Returns
    -------
    attributes, modifiers, token, depth
        ``token`` is the first token that is neither (a match object, or
        ``None`` at the end of the line); ``depth`` is how many attribute
        brackets are still open at the end of the line.
    """
    attributes: list[str] = []
    modifiers: list[str] = []
    depth = 0
    want = False  # the next identifier names an attribute
    skip = 0  # brackets of `scoped[NS]`
    for found in _HEAD_TOKEN.finditer(masked, pos):
        lexeme = found.group()
        if depth:
            if lexeme in ("[", "@["):
                depth += 1
            elif lexeme == "]":
                depth -= 1
            elif depth == 1:
                if lexeme == ",":
                    want = True
                elif (
                    want and lexeme not in ("local", "scoped", "-") and _is_name(lexeme)
                ):
                    attributes.append(lexeme)
                    want = False
            continue
        if skip:
            if lexeme == "[":
                skip += 1
            elif lexeme == "]":
                skip -= 1
            continue
        if lexeme == "@[":
            depth, want = 1, True
        elif lexeme in _MODIFIERS:
            if lexeme not in modifiers:
                modifiers.append(lexeme)
        elif lexeme == "[" and modifiers and modifiers[-1] == "scoped":
            skip = 1
        else:
            return attributes, modifiers, found, 0
    return attributes, modifiers, None, depth


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


class _Splitter:
    """
    Turn a file's lines into command chunks; see :class:`LeanReader` for the rules.

    ``feed`` takes one line at a time and returns the chunks completed by
    it; ``finish`` returns the rest. Memory is one command's lines (a
    command longer than ``step`` lines is emitted in parts as it is read)
    and at most ``held_limit`` lines waiting for the next command.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        file_name: str,
        meta: dict[str, Any],
        step: int | None,
        cap: int | None,
        proofs: str,
        dialect: str,
        kinds: frozenset[str] | None,
        watch: tuple[str, ...],
        commands: frozenset[str],
        max_scope_depth: int,
        flag_tokens: Mapping[str, str],
    ) -> None:
        self._file_name = file_name
        self._meta = meta
        self._step = step
        self._cap = cap
        self._drop = proofs == "drop"
        self._lean3 = dialect == "lean3"
        self._kinds = kinds
        self._commands = commands
        self._max_depth = max_scope_depth
        self._flag_of = dict(flag_tokens)
        self._flags = _token_pattern(self._flag_of)
        self._watch = _token_pattern(watch)
        self._lexer = _Lexer()
        self._scopes: list[tuple[str, str | None]] = []
        self._namespace_cache: str | None = ""
        self._held: list[_Held] = []
        self._held_limit = max(step or 0, 256)
        self._prefix_depth = 0
        self._carry: tuple[str, int] | None = None
        self._cur: _Command | None = None
        self._exited = False
        self._block_depth = 0
        self._mutual_col: int | None = None
        self._mutuals = 0  # mutual blocks open
        self._anomalies = 0
        self._out: list[dict[str, Any]] = []

    # -- feeding ------------------------------------------------------------

    def feed(self, number: int, line: str) -> list[dict[str, Any]]:  # noqa: PLR0912
        """Read one line; return the chunks it completes."""
        starts_in_code = self._lexer.in_code()
        masked = self._lexer.mask(line)
        stripped = line.strip()
        held_kind: str | None = None
        indent = len(line) - len(line.lstrip())
        code = bool(masked.strip())
        if not starts_in_code:
            carry = self._carry
            if carry is not None and (carry[0] == "prefix" or not code):
                # A comment, doc comment or attribute running on over lines
                # (blank lines inside it included) stays with where it began.
                held_kind, indent = carry
                self._hold(number, line, masked, held_kind, indent)
                if held_kind == "prefix":
                    self._prefix_depth = max(
                        0, self._prefix_depth + masked.count("[") - masked.count("]")
                    )
            else:
                self._continue(number, line, masked)
        elif not stripped:
            self._hold(number, line, masked, "blank", 0)
        elif self._prefix_depth:
            held_kind = "prefix"  # an attribute list running on over lines
            self._hold(number, line, masked, held_kind, indent)
            self._prefix_depth = max(
                0, self._prefix_depth + masked.count("[") - masked.count("]")
            )
        elif not code:
            if stripped.startswith("/-!"):
                head = _Head("moduledoc", "moduledoc")
                self._start(number, line, masked, indent, head)
            else:
                held_kind = "doc" if stripped.startswith("/--") else "comment"
                self._hold(number, line, masked, held_kind, indent)
        else:
            if self._mutuals and self._mutual_col is None:
                self._mutual_col = indent
            head = self._classify(masked, indent)
            if head is None:
                self._continue(number, line, masked)
            elif head.category == "prefix":
                held_kind = "prefix"
                self._hold(number, line, masked, held_kind, indent)
                self._prefix_depth = head.open_depth
            else:
                self._start(number, line, masked, indent, head)
        self._carry = (
            (held_kind, indent)
            if held_kind is not None and not self._lexer.in_code()
            else None
        )
        if self._lean3:
            self._count_blocks(masked)
        out, self._out = self._out, []
        return out

    def finish(self) -> list[dict[str, Any]]:
        """Return the chunks still open at the end of the file."""
        cur = self._cur
        if cur is None and any(h.kind != "blank" for h in self._held):
            cur = self._cur = _Command(
                _Head("comment", "comment"), self._namespace, self._exited
            )
        if cur is not None:
            for held in self._held:
                self._add(cur, held.number, held.text, held.masked)
            self._held = []
            self._emit(cur, final=True)
            self._cur = None
        construct = self._lexer.open_construct()
        if construct is not None:
            logger.warning(
                "LeanReader: %s ends inside a %s; its lines are kept in the last chunk.",
                self._file_name,
                construct,
            )
        if self._scopes or self._anomalies:
            logger.debug(
                "LeanReader: %s: %d scope(s) left open, %d unmatched `end`.",
                self._file_name,
                len(self._scopes),
                self._anomalies,
            )
        out, self._out = self._out, []
        return out

    # -- classifying a line -------------------------------------------------

    def _classify(  # noqa: PLR0911, PLR0912
        self, masked: str, indent: int
    ) -> _Head | None:
        """Return the head of the command this code line starts, or ``None``."""
        attributes, modifiers, first, depth = _scan_prefix(masked, indent)
        if first is None:
            if not (attributes or modifiers):
                return None
            head = _Head("prefix", "prefix")
            head.open_depth = depth
            return head
        keyword = first.group()
        col0 = indent in (0, self._mutual_col)
        if keyword.startswith("#"):
            if not col0:
                return None
            head = _Head("command", keyword)
        elif not col0 and (
            (keyword not in _INDENTED and not (not self._lean3 and keyword in _SCOPES))
            or (self._cur is not None and self._cur.depth)
        ):
            # An indented keyword starts a command only outside brackets: a
            # declaration quoted in a metaprogram, `(command| def $f ...),
            # is code of the command around it.
            return None
        elif keyword not in _KNOWN and keyword not in self._commands:
            return None
        elif self._lean3 and keyword == "end" and self._block_depth:
            return None  # closes a `begin` or `match` block
        else:
            rest = _HEAD_TOKEN.finditer(masked, first.end())
            if keyword == "deriving":
                following = next(rest, None)
                if following is None or following.group() != "instance":
                    return None
                head = _Head("command", "deriving instance")
            elif keyword in _SCOPES:
                head = _Head("context", "context")
                following = next(rest, None)
                name = following.group() if following is not None else None
                head.scope = (keyword, name if name and _is_name(name) else None)
            elif keyword in _IMPORTS:
                head = _Head("import", "import")
                if keyword == "import":
                    head.imports = [
                        t.group()
                        for t in rest
                        if t.group() != "all" and _is_name(t.group())
                    ]
            elif keyword in _CONTEXT:
                head = _Head("context", "context")
            elif keyword in _OTHER:
                head = _Head("command", keyword)
            else:
                head = _Head("declaration", keyword)
                self._name_declaration(head, keyword, first.end(), rest, self._lean3)
        head.attributes = attributes
        head.modifiers = modifiers
        return head

    @staticmethod
    def _name_declaration(  # noqa: PLR0912
        head: _Head,
        keyword: str,
        keyword_end: int,
        rest: Iterator[Any],
        lean3: bool = False,
    ) -> None:
        """Set *head*'s name and where its body starts, from the tokens after *keyword*."""
        body_col = keyword_end
        tokens = []
        for found in rest:
            tokens.append(found)
            if len(tokens) >= _NAME_LOOKAHEAD:
                break
        i = 0
        if (
            keyword == "class"
            and tokens
            and tokens[0].group() in ("inductive", "abbrev")
        ):
            body_col = tokens[0].end()
            i = 1
        if lean3 and i < len(tokens) and tokens[i].group() == "{":
            # Lean 3 universe parameters before the name: `structure {u} part`
            close = next(
                (j for j in range(i, len(tokens)) if tokens[j].group() == "}"), None
            )
            if close is not None:
                body_col = tokens[close].end()
                i = close + 1
        # (priority := n), (name := x), (kind := k) before the name
        while (
            i + 2 < len(tokens)
            and tokens[i].group() == "("
            and tokens[i + 1].group() in ("priority", "name", "kind")
            and tokens[i + 2].group() == ":="
        ):
            depth = 0
            close = None
            for j in range(i, len(tokens)):
                if tokens[j].group() == "(":
                    depth += 1
                elif tokens[j].group() == ")":
                    depth -= 1
                    if not depth:
                        close = j
                        break
            if close is None:
                break
            if tokens[i + 1].group() == "name" and i + 3 < close:
                head.name = tokens[i + 3].group()
            body_col = tokens[close].end()
            i = close + 1
        if (
            keyword in _NAMED
            and head.name is None
            and i < len(tokens)
            and _is_name(tokens[i].group())
        ):
            head.name = tokens[i].group()
            body_col = tokens[i].end()
        head.body_col = body_col

    # -- assigning lines ----------------------------------------------------

    def _hold(
        self, number: int, text: str, masked: str, kind: str, indent: int
    ) -> None:
        self._held.append(_Held(number, text, masked, kind, indent))
        if len(self._held) > self._held_limit:
            # Far from any command: the older half joins what came before.
            half = len(self._held) // 2
            older, self._held = self._held[:half], self._held[half:]
            cur = self._cur
            if cur is None:
                cur = self._cur = _Command(
                    _Head("comment", "comment"), self._namespace, self._exited
                )
            for held in older:
                self._add(cur, held.number, held.text, held.masked)

    def _continue(self, number: int, line: str, masked: str) -> None:
        cur = self._cur
        if cur is None:  # code before any known command
            cur = self._cur = _Command(
                _Head("command", "command"), self._namespace, self._exited
            )
        for held in self._held:
            self._add(cur, held.number, held.text, held.masked)
        self._held = []
        self._add(cur, number, line, masked)

    def _start(  # noqa: PLR0912
        self, number: int, line: str, masked: str, indent: int, head: _Head
    ) -> None:
        cur = self._cur
        if cur is not None and cur.ends_in and head.category != "moduledoc":
            # `open Foo in`, `set_option o v in`, `#guard_msgs in`: one command
            # with what follows; it takes that command's kind and name.
            for held in self._held:
                self._take_prefix(cur, held)
                self._add(cur, held.number, held.text, held.masked)
            self._held = []
            self._become(cur, head, number)
            self._add(cur, number, line, masked)
            return
        if (
            cur is not None
            and head.category in ("context", "import")
            and cur.category == head.category
            and not _ENDS_IN.search(masked)
        ):
            for held in self._held:  # a run of context commands is one chunk
                self._add(cur, held.number, held.text, held.masked)
            self._held = []
            cur.imports.extend(head.imports)
            self._apply_scope(head)
            self._add(cur, number, line, masked)
            return
        attach, trailing = self._split_held(indent)
        if cur is not None:
            for held in trailing:
                self._add(cur, held.number, held.text, held.masked)
            self._emit(cur, final=True)
        elif any(held.kind != "blank" for held in trailing):
            # Comments before the first command, apart from it: their own chunk.
            lead = _Command(_Head("comment", "comment"), self._namespace, self._exited)
            for held in trailing:
                self._add(lead, held.number, held.text, held.masked)
            self._emit(lead, final=True)
        self._held = []
        cur = self._cur = _Command(head, self._namespace, self._exited)
        self._become(cur, head, number)
        for held in attach:
            self._take_prefix(cur, held)
            self._add(cur, held.number, held.text, held.masked)
        self._add(cur, number, line, masked)

    def _become(self, cur: _Command, head: _Head, number: int) -> None:
        """Give *cur* the kind, name and modifiers of the command *head* starts."""
        cur.category = head.category
        cur.kind = head.kind
        cur.ends_in = False
        if head.name is not None:
            name = head.name
            if name.startswith("_root_."):
                cur.name = name[len("_root_.") :]
            else:
                cur.name = f"{self._namespace}.{name}" if self._namespace else name
        for modifier in head.modifiers:
            if modifier not in cur.modifiers:
                cur.modifiers.append(modifier)
        cur.attributes.extend(head.attributes)
        cur.imports.extend(head.imports)
        if self._drop and head.kind in _DROPPABLE:
            cur.statement = _Statement(
                number, head.body_col, ":=" if self._lean3 else "=>"
            )
        self._apply_scope(head)
        if head.kind == "#exit":
            self._exited = True

    def _take_prefix(self, cur: _Command, held: _Held) -> None:
        """Record the attributes and modifiers of an attribute line before a command."""
        if held.kind != "prefix":
            return
        attributes, modifiers, _token, _depth = _scan_prefix(held.masked, 0)
        cur.attributes.extend(attributes)
        for modifier in modifiers:
            if modifier not in cur.modifiers:
                cur.modifiers.append(modifier)

    def _split_held(self, indent: int) -> tuple[list[_Held], list[_Held]]:
        """
        Split the held lines into those that open the next command and the rest.

        Lines directly above the command (no blank line between, a comment
        no deeper than the command) open it; so does everything from the
        first doc comment or attribute line on, since Lean gives those to
        the declaration that follows. The rest closes the previous command.
        """
        held = self._held
        i = len(held)
        while i > 0:
            line = held[i - 1]
            if line.kind == "blank" or (
                line.kind == "comment" and line.indent > indent
            ):
                break
            i -= 1
        for j in range(i):
            if held[j].kind in ("doc", "prefix"):
                i = j
                break
        return held[i:], held[:i]

    def _add(self, cur: _Command, number: int, text: str, masked: str) -> None:
        """Give one line to *cur*: record what it holds and keep its text."""
        if text.strip():
            cur.last = number  # the last line with content
        if masked.strip():
            if self._flags is not None:
                for found in self._flags.finditer(masked):
                    cur.hit_lines.append(number)
                    cur.hit_flags.append(self._flag_of[found.group()])
            if self._watch is not None:
                for found in self._watch.finditer(masked):
                    cur.watch_lines.append(number)
                    cur.watch_names.append(found.group())
            if cur.category != "declaration":
                cur.ends_in = _ENDS_IN.search(masked) is not None
            cur.depth = max(
                0,
                cur.depth
                + len(_OPEN_BRACKET.findall(masked))
                - len(_CLOSE_BRACKET.findall(masked)),
            )
        if cur.boundary is not None:
            return  # the proof is left out
        if cur.statement is not None and masked.strip():
            col = cur.statement.scan(number, masked)
            if col is not None:
                cur.boundary = (number, col)
                cur.statement = None
                text = text[:col].rstrip()
        cur.lines.append((number, text))
        cur.size += len(text) + 1
        step, cap = self._step, self._cap
        if (step is not None and len(cur.lines) > step) or (
            cap is not None and cur.size > cap and len(cur.lines) > 1
        ):
            # Emit what came before this line as a part; keep this line.
            newest = cur.lines.pop()
            self._emit(cur, final=False)
            cur.lines = [newest]
            cur.size = len(newest[1])

    def _emit(self, cur: _Command, *, final: bool) -> None:  # noqa: PLR0912
        """Emit *cur*'s kept lines as one chunk (a part, unless *final* and alone)."""
        lines = cur.lines
        low, high = 0, len(lines)
        while low < high and not lines[low][1].strip():
            low += 1
        while high > low and not lines[high - 1][1].strip():
            high -= 1
        if not final or cur.parts:
            cur.parts += 1
        if low == high or (self._kinds is not None and cur.kind not in self._kinds):
            cur.lines = []
            return
        start, end = lines[low][0], lines[high - 1][0]
        upto = cur.last if final else end
        flags = _between(cur.hit_lines, cur.hit_flags, start, upto)
        watched = _between(cur.watch_lines, cur.watch_names, start, upto)
        if cur.kind == "axiom":
            flags.add("axiom")
        if cur.after_exit:
            flags.add("after_exit")
        chunk: dict[str, Any] = {
            "text": "\n".join(text for _number, text in lines[low:high]),
            "section_type": (
                SectionType.TEXT.value
                if cur.kind in ("moduledoc", "comment")
                else SectionType.CODE.value
            ),
            "source_type": SourceType.CODE.value,
            "line_start": start,
            "line_end": end,
            **self._meta,
            "lean_kind": cur.kind,
        }
        if cur.name is not None:
            chunk["declaration"] = cur.name
        if cur.namespace:
            chunk["namespace"] = cur.namespace
        if cur.modifiers:
            chunk["modifiers"] = list(cur.modifiers)
        if cur.attributes:
            chunk["attributes"] = list(cur.attributes)
        if cur.imports:
            chunk["imports"] = list(cur.imports)
        if flags:
            chunk["flags"] = sorted(flags)
        if watched:
            chunk["watched"] = sorted(watched)
        if cur.parts:
            chunk["part"] = cur.parts
        if final and cur.boundary is not None:
            chunk["proof_line_start"] = cur.boundary[0]
            chunk["proof_line_end"] = cur.last
        self._out.append(chunk)
        cur.lines = []

    # -- scopes ---------------------------------------------------------------

    def _apply_scope(self, head: _Head) -> None:
        if head.scope is None:
            return
        keyword, name = head.scope
        scopes = self._scopes
        if keyword == "namespace":
            if name:
                scopes.extend(("namespace", part) for part in _components(name))
        elif keyword == "end":
            count = len(_components(name)) if name else 1
            if count > len(scopes):
                self._anomalies += 1
                count = len(scopes)
            if count:
                self._mutuals -= sum(kind == "mutual" for kind, _n in scopes[-count:])
                del scopes[-count:]
        else:  # section, mutual
            scopes.append((keyword, name))
            if keyword == "mutual":
                self._mutuals += 1
                self._mutual_col = None
        if len(scopes) > self._max_depth:
            raise ValueError(
                f"LeanReader: {self._file_name} nests more than"
                f" max_scope_depth={self._max_depth} namespaces and sections."
                " Increase max_scope_depth to read it."
            )
        if keyword == "end" and not self._mutuals:
            self._mutual_col = None
        self._namespace_cache = None  # rebuilt when a command needs it

    @property
    def _namespace(self) -> str:
        """The dotted namespace in force, built once per change of scope."""
        if self._namespace_cache is None:
            self._namespace_cache = ".".join(
                part for kind, part in self._scopes if kind == "namespace" and part
            )
        return self._namespace_cache

    def _count_blocks(self, masked: str) -> None:
        """Track Lean 3's `begin`/`match` ... `end` blocks, so their `end` closes no scope."""
        for found in _LEAN3_BLOCK.finditer(masked):
            if found.group() == "end":
                if self._block_depth:
                    self._block_depth -= 1
            else:
                self._block_depth += 1


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
        If ``kinds``, ``watch`` or ``commands`` is not a collection of str.
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

    def __post_init__(self) -> None:
        """Validate the options."""
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
