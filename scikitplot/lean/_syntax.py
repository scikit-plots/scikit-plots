# scikitplot/lean/_syntax.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Lean source text: commands, declarations, scopes, references and audit flags.

The language core every Lean tool of scikit-plots shares. A :class:`_Lexer`
tells code from comments and strings line by line; a :class:`_Splitter`
turns the lines of one file into its commands, each with its kind, its
fully qualified name, its namespace and ``open`` scope, the lines it spans,
the names its code uses and the audit flags its code raises: ``sorry`` and
``admit``, declared axioms, ``native_decide``, ``unsafe`` code,
``implemented_by`` and ``extern``, a disabled kernel check and text after
``#exit``. :data:`LEAN_FLAGS` maps each flagged token to its flag.

Both are streaming and linear in the text: one line in, the finished
commands out. :class:`~scikitplot.corpus.LeanReader` and
:class:`~scikitplot.lean.LeanIndex` are built on them. The flags say what
the source text contains; whether a theorem depends on them is what
``#print axioms`` in Lean answers.

Notes
-----
**Developer.** Standard library only, and nothing from
:mod:`scikitplot.corpus`: the corpus reader depends on this module, never
the reverse.
"""  # noqa: D205, D400

from __future__ import annotations

import bisect
import logging
import re
from types import MappingProxyType
from typing import Any, Collection, Iterator, Mapping

logger = logging.getLogger(__name__)

__all__ = ["LEAN_FLAGS"]

# ---------------------------------------------------------------------------
# Audit flags
# ---------------------------------------------------------------------------

_LEAN_FLAGS = {
    "sorry": "sorry",
    "sorryAx": "sorry",
    "mkSorry": "sorry",  # meta code that builds a `sorry` term
    "mkLabeledSorry": "sorry",
    "admit": "admit",
    "native_decide": "native",
    "ofReduceBool": "native",
    "ofReduceNat": "native",
    "+native": "native",
    "unsafe": "unsafe",
    "implemented_by": "implemented_by",
    "extern": "extern",
    "skipKernelTC": "skip_kernel_tc",
    "addDecl": "add_decl",
    "addAndCompile": "add_decl",
    "axiomDecl": "axiom",  # meta code that declares an axiom
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

#: Words of Lean's syntax that ``references`` leaves out: they never name
#: a declaration of the project.
_NOT_REFERENCES = frozenset(
    {
        "fun",
        "by",
        "have",
        "show",
        "let",
        "if",
        "then",
        "else",
        "match",
        "with",
        "do",
        "at",
        "from",
        "in",
        "where",
        "calc",
        "return",
        "for",
        "unless",
        "try",
        "catch",
        "finally",
        "mut",
        "rec",
        "this",
        "sorry",
        "Type",
        "Sort",
        "Prop",
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
        "extends",
        "deriving",
        "namespace",
        "section",
        "end",
        "open",
        "variable",
        "universe",
        "set_option",
        "import",
        "private",
        "protected",
        "noncomputable",
        "partial",
        "unsafe",
        "mutual",
        "local",
        "scoped",
        "attribute",
        "termination_by",
        "decreasing_by",
        "begin",
        "assume",
        "obtain",
        "suffices",
        "using",
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

# Identifier characters exactly as Lean's lexer has them (Init/Meta/Defs:
# isIdFirst, isIdRest, isLetterLike, isSubScriptAlnum). Char.isAlpha and
# isAlphanum are ASCII only there; Python's \w is far wider (it takes the
# postfix ᶜ of `sᶜ` and the ᵀ of `Aᵀ`, which Lean reads as notation).
_LETTER_LIKE = (
    "\u03b1-\u03ba\u03bc-\u03c9"  # lower Greek, but lambda
    "\u0391-\u039f\u03a1-\u03a2\u03a4-\u03a9"  # upper Greek, but Pi and Sigma
    "\u03ca-\u03fb"  # Coptic letters
    "\u1f00-\u1ffe"  # polytonic Greek extended
    "\u2100-\u214f"  # letter-like block
    "\U0001d49c-\U0001d59f"  # script, double-struck, Fraktur
    "\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff"  # Latin-1 letters, but U+00D7 and U+00F7 (times, divide)
    "\u0100-\u017f"  # Latin Extended-A
)
_SUBSCRIPT = "\u2080-\u2089\u2090-\u209c\u1d62-\u1d6a\u2c7c"
_ID_FIRST = f"A-Za-z_{_LETTER_LIKE}"
_ID_REST = f"A-Za-z0-9_'!?{_LETTER_LIKE}{_SUBSCRIPT}"
#: One identifier component, as Lean lexes it.
_IDENT = f"[{_ID_FIRST}][{_ID_REST}]*"
_ID_CHAR = re.compile(f"[{_ID_REST}]")

_CODE_SPECIAL = re.compile(rf'--|/-|"|\'|«|(?<![{_ID_REST}.])r#*"')
_CODE_SPECIAL_INTERP = re.compile(rf'--|/-|"|\'|«|(?<![{_ID_REST}.])r#*"|\{{|\}}')
_COMMENT_TOKEN = re.compile(r"/-|-/")
_STRING_TOKEN = re.compile(r'\\.?|"')
_STRING_TOKEN_INTERP = re.compile(r'\\.?|"|\{')
_CHAR_LITERAL = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)|[^\\'])'")

_NAME = rf"(?:«[^«»]*»|{_IDENT})(?:\.(?:«[^«»]*»|{_IDENT}|[0-9]+))*"
_HEAD_TOKEN = re.compile(rf"@\[|#{_IDENT}|{_NAME}%?|:=|\S")
_FULL_NAME = re.compile(rf"{_NAME}\Z")
#: A name the code uses. A ``.name`` after something that is not part of a
#: name is kept with its dot: field notation on an expression (``(e).card``,
#: ``s⁻¹.card``) or Lean 4's leading-dot identifier (``.inl h``). Its head is
#: unknown, so the index links it by its components' short names, as it
#: links ``h.card`` when ``h`` is bound.
_REFERENCE = re.compile(rf"(?<![{_ID_REST}.#])\.?{_NAME}")
_COMPONENT = re.compile(r"«[^«»]*»|[^.«»]+")

_OPENERS = "([{⟨⦃⟦⟪⁅"
_OPEN_BRACKET = re.compile(r"[(\[{⟨⦃⟦⟪⁅]")
_CLOSE_BRACKET = re.compile(r"[)\]}⟩⦄⟧⟫⁆]")
_STATEMENT_TOKEN = re.compile(
    rf"[(\[{{⟨⦃⟦⟪⁅]|[)\]}}⟩⦄⟧⟫⁆]|:=|(?<![{_ID_REST}.])(?:let|have|where)(?![{_ID_REST}])"
)
_ENDS_IN = re.compile(rf"(?<![{_ID_REST}.])in\s*\Z")
_LEAN3_BLOCK = re.compile(rf"(?<![{_ID_REST}.])(?:begin|match|end)(?![{_ID_REST}])")


def _is_ident_char(ch: str) -> bool:
    """Return ``True`` for a character that continues a Lean identifier."""
    return _ID_CHAR.match(ch) is not None


def _is_name(lexeme: str) -> bool:
    """Return ``True`` when *token* is an identifier that may name a declaration."""
    return (
        lexeme not in _RESERVED
        and lexeme != "_"
        and (lexeme[0] == "«" or re.match(f"[{_ID_FIRST}]", lexeme) is not None)
        and _FULL_NAME.match(lexeme) is not None
    )


def _token_pattern(tokens: Collection[str]) -> re.Pattern[str] | None:
    """Compile *tokens* into one pattern matching them as whole identifiers."""
    if not tokens:
        return None
    alternatives = "|".join(
        re.escape(lexeme) for lexeme in sorted(tokens, key=len, reverse=True)
    )
    return re.compile(rf"(?<![{_ID_REST}])(?:{alternatives})(?![{_ID_REST}])")


_QUOTED = re.compile(r"«[^«»]*»")


def _code_matches(pattern: re.Pattern[str], masked: str) -> list[re.Match[str]]:
    """
    Return *pattern*'s matches in *masked*, leaving out text inside a «name».

    ``theorem «a sorry b»`` names a declaration; the ``sorry`` in it is part
    of the name, not code. A match spanning a whole «name» (a watched
    ``«a b»``) is kept; one inside or overlapping it is not, so the escaped
    identifier ``«sorry»`` raises no flag either.
    """
    if "«" not in masked:
        return list(pattern.finditer(masked))
    spans = [(m.start(), m.end()) for m in _QUOTED.finditer(masked)]
    kept = []
    for found in pattern.finditer(masked):
        start, end = found.span()
        if any(a < end and start < b and (a, b) != (start, end) for a, b in spans):
            continue
        kept.append(found)
    return kept


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
        "opens",
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
        self.opens: list[str] = []


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
        "local_opens",
        "modifiers",
        "name",
        "namespace",
        "opens",
        "parts",
        "refs",
        "size",
        "statement",
        "statement_text",
        "watch_lines",
        "watch_names",
        "written",
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
        self.statement_text: str | None = None
        self.boundary: tuple[int, int] | None = None
        self.parts = 0
        self.depth = 0  # brackets open at the end of the last line
        self.refs: set[str] = set()  # identifiers its code uses
        self.opens: tuple[str, ...] = ()  # namespaces open for it
        self.local_opens: list[str] = []  # from its own `open .. in`
        self.written: str | None = None  # its name as written


def _opened(tokens: Iterator[Any]) -> list[str]:
    """
    Return the namespaces an ``open`` command opens, from the tokens after it.

    ``scoped`` is skipped; an explicit name list ``(x y)`` opens its
    namespace for those names, counted as opening it; reading stops at
    ``hiding``, ``renaming``, ``in`` or ``:=``.
    """
    names: list[str] = []
    depth = 0
    for found in tokens:
        word = found.group()
        if word == "(":
            depth += 1
        elif word == ")":
            depth = max(0, depth - 1)
        elif depth:
            continue
        elif word in ("hiding", "renaming", "in", ":="):
            break
        elif word != "scoped" and _is_name(word):
            names.append(word[len("_root_.") :] if word.startswith("_root_.") else word)
    return names


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
        references: bool = False,
        statements: bool = False,
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
        self._references = references
        # Record each result's statement (its text before the proof) while
        # keeping every line, so an index reads a file once.
        self._statements = statements and not self._drop
        if self._statements and step is not None:
            # A part holds only some of a command's lines: a statement
            # recorded from one part could be cut short.
            raise ValueError(
                "_Splitter: statements=True needs step=None (whole commands)."
            )
        self._file_opens: list[str] = []  # `open` outside any scope
        self._scope_opens: list[list[str]] = []  # one list per scope
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
        #: ``(line, problem)`` for text Lean refuses to read: an ``end`` that
        #: closes no scope or names another one, a file ending inside a
        #: comment, string or «name».
        self.problems: list[tuple[int, str]] = []
        self._last = 0
        self._out: list[dict[str, Any]] = []

    # -- feeding ------------------------------------------------------------

    def feed(self, number: int, line: str) -> list[dict[str, Any]]:  # noqa: PLR0912
        """Read one line; return the chunks it completes."""
        self._last = number
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
            self.problems.append((self._last, f"the file ends inside a {construct}"))
            logger.warning(
                "Lean: %s ends inside a %s; its lines are kept in the last chunk.",
                self._file_name,
                construct,
            )
        if self._scopes or self._anomalies:
            logger.debug(
                "Lean: %s: %d scope(s) left open, %d unmatched `end`.",
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
                if keyword == "open":
                    head.opens = _opened(rest)
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
            self._apply_opens(cur, head, masked)
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
            self._apply_scope(head, number)
            self._apply_opens(cur, head, masked)
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
        self._apply_opens(cur, head, masked)
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
        if head.name is not None:
            cur.written = head.name
        if self._references:
            # Namespaces open for it: the file's, each enclosing scope's, and
            # those of its own `open .. in` prefix.
            seen = dict.fromkeys(self._file_opens)
            for opened in self._scope_opens:
                seen.update(dict.fromkeys(opened))
            seen.update(dict.fromkeys(cur.local_opens))
            cur.opens = tuple(seen)
        if (self._drop or self._statements) and head.kind in _DROPPABLE:
            cur.statement = _Statement(
                number, head.body_col, ":=" if self._lean3 else "=>"
            )
        self._apply_scope(head, number)
        if head.kind == "#exit":
            self._exited = True

    def _apply_opens(self, cur: _Command, head: _Head, masked: str) -> None:
        """Record what an `open` opens: for its scope or, with `in`, its command."""
        if not head.opens:
            return
        if _ENDS_IN.search(masked):
            cur.local_opens.extend(head.opens)
        elif self._scope_opens:
            self._scope_opens[-1].extend(head.opens)
        else:
            self._file_opens.extend(head.opens)

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
                for found in _code_matches(self._flags, masked):
                    cur.hit_lines.append(number)
                    cur.hit_flags.append(self._flag_of[found.group()])
            if self._watch is not None:
                for found in _code_matches(self._watch, masked):
                    cur.watch_lines.append(number)
                    cur.watch_names.append(found.group())
            if self._references:
                cur.refs.update(_REFERENCE.findall(masked))
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
            text = self._end_of_statement(cur, number, masked, text)
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

    def _end_of_statement(
        self, cur: _Command, number: int, masked: str, text: str
    ) -> str:
        """
        Find where a result's statement ends on this line; return the line to keep.

        With ``proofs="drop"`` the line is cut there and the rest of the
        command left out. With *statements* the line is kept whole and the
        statement's text recorded.
        """
        col = cur.statement.scan(number, masked)
        if col is None:
            return text
        cur.statement = None
        if self._drop:
            cur.boundary = (number, col)
            return text[:col].rstrip()
        cur.statement_text = "\n".join(
            [t for _n, t in cur.lines] + [text[:col].rstrip()]
        )
        return text

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
            cur.refs = set()
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
            # Plain strings, equal to the corpus schema's SectionType.TEXT /
            # SectionType.CODE and SourceType.CODE values (pinned by a test
            # in scikitplot/corpus): the language core imports no corpus code.
            "section_type": "text" if cur.kind in ("moduledoc", "comment") else "code",
            "source_type": "code",
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
        if self._references:
            own = {cur.written, cur.name}
            chunk["references"] = sorted(
                ref for ref in cur.refs if ref not in _NOT_REFERENCES and ref not in own
            )
            if cur.opens:
                chunk["opens"] = list(cur.opens)
            cur.refs = set()
        if cur.parts:
            chunk["part"] = cur.parts
        if self._statements and final and cur.kind in _DROPPABLE:
            # No boundary found (a `where` or pattern-matching proof): the
            # whole command, as proofs="drop" keeps it.
            chunk["statement"] = (
                cur.statement_text if cur.statement_text is not None else chunk["text"]
            )
        if final and cur.boundary is not None:
            chunk["proof_line_start"] = cur.boundary[0]
            chunk["proof_line_end"] = cur.last
        self._out.append(chunk)
        cur.lines = []

    # -- scopes ---------------------------------------------------------------

    def _apply_scope(self, head: _Head, number: int) -> None:
        if head.scope is None:
            return
        keyword, name = head.scope
        scopes = self._scopes
        if keyword == "end":
            self._close_scopes(name, number)
        elif name and keyword in ("namespace", "section"):
            # `namespace A.B` and `section A.B` open a scope per component,
            # and `end A.B` closes both; only a namespace names declarations.
            parts = _components(name)
            scopes.extend((keyword, part) for part in parts)
            self._scope_opens.extend([] for _part in parts)
        elif keyword != "namespace":  # an anonymous section, mutual
            scopes.append((keyword, name))
            self._scope_opens.append([])
            if keyword == "mutual":
                self._mutuals += 1
                self._mutual_col = None
        if len(scopes) > self._max_depth:
            raise ValueError(
                f"Lean: {self._file_name} nests more than"
                f" max_scope_depth={self._max_depth} namespaces and sections."
                " Increase max_scope_depth to read it."
            )
        if keyword == "end" and not self._mutuals:
            self._mutual_col = None
        self._namespace_cache = None  # rebuilt when a command needs it

    def _close_scopes(self, name: str | None, number: int) -> None:
        """Close the scopes ``end [name]`` closes; record an ``end`` Lean refuses."""
        scopes = self._scopes
        parts = _components(name) if name else []
        count = len(parts) or 1
        written = f"`end {name}`" if name else "`end`"
        if count > len(scopes):
            self._anomalies += 1
            self.problems.append((number, f"{written} closes no open scope"))
            count = len(scopes)
        elif not self._lean3 and [n for _k, n in scopes[-count:]] != (parts or [None]):
            opened = " ".join(
                f"`{kind}{' ' + scope if scope else ''}`"
                for kind, scope in scopes[-count:]
            )
            self.problems.append((number, f"{written} does not match {opened}"))
        if count:
            self._mutuals -= sum(kind == "mutual" for kind, _n in scopes[-count:])
            del scopes[-count:]
            del self._scope_opens[-count:]

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
