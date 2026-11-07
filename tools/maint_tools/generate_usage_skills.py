#!/usr/bin/env python3
# tools/maint_tools/generate_usage_skills.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Generate, check and explain the usage skills of the ``scikitplot`` plugin.

A usage skill tells an agent how to use one public module of scikit-plots in
any project: what it is for, how to install it, its declared public API and
an example, all taken from the package itself. The skills live in
``plugins/scikitplot/skills/<name>/`` (``SKILL.md`` and
``references/API.md``), follow the Agent Skills specification, and are named
by Rule 164 (``scikitplot`` for the root package, ``scikitplot-<derived>``
for a subpackage).

Commands::

``check``    read-only: regenerate in memory, compare with disk, lint every skill.
``sync``     write (or with ``--dry-run`` only plan) the generated files.
``explain``  unified diff of every stale file, optionally for one module.
``list``     the modules that have a skill, the excluded ones and why.
``guide``    the complete error-handling guideline (every code, every step).

Usage::

    python tools/maint_tools/generate_usage_skills.py check            # exit 1 when stale
    python tools/maint_tools/generate_usage_skills.py explain          # why is it stale
    python tools/maint_tools/generate_usage_skills.py sync             # write the files
    python tools/maint_tools/generate_usage_skills.py sync --dry-run   # plan only
    python tools/maint_tools/generate_usage_skills.py guide            # what to do on error

Notes
-----
**User notes.** Edit the module's docstrings, its ``__all__``, or the policy
in ``[tool.scikitplot.maintenance.skills]`` of ``pyproject.toml``; never the
generated files. ``check`` needs the package importable with its optional
dependencies, because names are resolved to read their docstrings; it reports
every problem it found instead of writing a partial skill.

**Developer notes.** The public API is a module's ``__all__`` and nothing
else: names a module does not declare are not promised to users, and
``dir()`` depends on what happened to be imported. Every text in a skill is a
docstring, a signature or a policy string, so a skill cannot state what the
package does not. Names (:func:`derived_skill_names`) and frontmatter checks
are shared with ``check_records.py``, loaded from its file (from
``--repo-root``, not from the location of this script).

**Design.** Same shape as ``generate_pr_labeler.py``: importing has no side
effects, ``check`` / ``explain`` / ``list`` are read-only, ``sync`` is the only
writer (all-or-nothing planning, atomic per-file writes, verified afterwards),
``--json`` for automation, ``main(argv, stdout=, stderr=)`` returns a status.
``sync`` writes by default (as before); ``--dry-run`` is the labeler's
"without ``--apply``".

**Exit status.** ``0`` ok; ``1`` the check found errors (stale or invalid
skills) or a write failed; ``2`` the command could not be evaluated (usage,
policy, environment) - nothing was written.

**Error handling guideline.** Every message starts with a code
(``[E-...]`` error, ``[W-...]`` warning). On failure the relevant steps are
printed under ``How to fix``; ``guide`` prints all of them. The rules that
never change:

1. A generated file is never edited by hand; fix the *source* (docstring,
   ``__all__``, policy) and run ``sync``.
2. ``check`` failing with ``E-STALE``: run ``explain``. If the diff is the
   change you made, run ``sync`` and commit generated files together with
   the source change. If it is not, fix the source.
3. A *stale* file is one of: *missing* (a skill is wanted, no file), *outdated*
   (content differs) or *orphaned* (a file no skill produces any more:
   removed, excluded or renamed module). ``sync`` creates, rewrites and
   deletes accordingly; shared skills (``plugins/_shared``) are never touched.
4. When any module cannot be read (``E-IMPORT``, ``E-API``, ``E-NAME``,
   ``E-POLICY``, ``E-ENV``, ``E-RECORDS``) staleness is not evaluated and
   ``sync`` refuses to write: a partial result would delete good skills.
5. ``sync`` is idempotent. After writing it reads the tree again; a difference
   there is ``E-UNSTABLE`` (the output is not deterministic, Rule 168).
6. Warnings never fail ``check``; ``--strict`` turns them into errors.
"""

from __future__ import annotations

import argparse
import difflib
import doctest
import enum
import importlib
import importlib.util
import inspect
import json
import os
import pathlib
import re
import sys
import textwrap
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, TextIO, Tuple

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # reported as E-ENV when a policy is read
        tomllib = None  # type: ignore[assignment]


REPO = Path(__file__).resolve().parents[2]
RECORDS_RELATIVE = "maintenances/_repo/_maintenance/tools/check_records.py"
RECORDS = REPO / RECORDS_RELATIVE
SCRIPT = "tools/maint_tools/generate_usage_skills.py"
POLICY_KEYS = ("tool", "scikitplot", "maintenance", "skills")
GENERATED = (
    "Generated by `python tools/maint_tools/generate_usage_skills.py sync` from the"
    " installed package; edit the module's docstrings or `__all__`, not this file."
)
#: Rows of the API table in SKILL.md; the full table is references/API.md.
SHOWN_NAMES = 40
SIGNATURE_MAX = 110
DESCRIPTION_MAX = 1024
#: Agent Skills specification: lower-case letters, digits, single hyphens, <= 64.
SKILL_NAME_MAX = 64
#: Names listed inside one message before ``... (+N more)``.
MAX_LISTED = 10
_SECTION = re.compile(r"^([A-Z][A-Za-z ]+)\n-{3,}\s*$", re.MULTILINE)
_MODULE_RE = re.compile(r"scikitplot(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_SKILL_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_CODE_RE = re.compile(r"^\[([EW]-[A-Z]+)\]")
_LINK_RE = re.compile(r"\]\((references/[^)#]+)")

# --------------------------------------------------------------------------- codes

E_ENV = "E-ENV"
E_RECORDS = "E-RECORDS"
E_POLICY = "E-POLICY"
E_NAME = "E-NAME"
E_IMPORT = "E-IMPORT"
E_API = "E-API"
E_LINT = "E-LINT"
E_STALE = "E-STALE"
E_IO = "E-IO"
E_UNSTABLE = "E-UNSTABLE"
E_USAGE = "E-USAGE"
W_COVERAGE = "W-COVERAGE"
W_EXCLUDED = "W-EXCLUDED"
W_EMPTY = "W-EMPTY"
W_DOC = "W-DOC"
W_EXAMPLE = "W-EXAMPLE"
W_SIGNATURE = "W-SIGNATURE"
W_DESCRIPTION = "W-DESCRIPTION"

#: ``code -> (what it means, ordered steps)``; printed by ``guide`` and, for the
#: codes present, under ``How to fix`` after a failure.
GUIDE: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    E_ENV: (
        "The environment cannot import the package or read the policy",
        (
            "Activate the environment the project is developed in and install the package with"
            " its optional dependencies (for example `pip install -e .`).",
            "On Python < 3.11 also install `tomli`.",
            'Verify: `python -c "import scikitplot.experimental.enable_ann_imputer"`.',
            "Rerun the same command. Nothing was written; repeating it is safe.",
        ),
    ),
    E_RECORDS: (
        "check_records.py is missing or does not provide what this generator reads",
        (
            f"Confirm `{RECORDS_RELATIVE}` exists below `--repo-root`.",
            "It must define `USAGE_SKILLS_DIR`, `derived_skill_names`, `mirror_keys` and"
            " `shared_skill_names`. A message naming one of them means the two tools come from"
            " different revisions: update both together.",
            "Rerun the command. Nothing was written.",
        ),
    ),
    E_POLICY: (
        "`[tool.scikitplot.maintenance.skills]` in pyproject.toml is missing or invalid",
        (
            "Read the message: it names the key.",
            "`usage_modules` is a list of unique dotted module names starting with `scikitplot`;"
            " `usage_excluded` maps a module name to a non-empty reason; no module is in both.",
            "Two modules must not derive the same skill name; rename or exclude one.",
            "Rerun `check`, then `sync`.",
        ),
    ),
    E_NAME: (
        "A listed module is not a package that can carry a usage skill",
        (
            "A name is `scikitplot` or a package below `scikitplot/` that check_records knows"
            " (it has a development-skill mirror, Rule 164).",
            "Fix the spelling in `usage_modules`; or add the package's development skill first;"
            " or move the entry to `usage_excluded` with a reason.",
            "Rerun `check`.",
        ),
    ),
    E_IMPORT: (
        "A listed module does not import",
        (
            'Run `python -c "import <module>"` for the full traceback; the message holds only'
            " its last line.",
            "Fix the import error in the package, or install the missing optional dependency."
            " Do not delete the module from the policy to silence it.",
            "If the module was removed on purpose, remove it from `usage_modules`; the next"
            " `sync` deletes its skill.",
        ),
    ),
    E_API: (
        "A module's public API cannot be read",
        (
            "A module with no `__all__` has no declared API: add `__all__` (it is the only source).",
            "Every name in `__all__` must resolve: define it, import it, or remove it from `__all__`.",
            "Rerun `check`. A partial skill is never written.",
        ),
    ),
    E_LINT: (
        "A generated skill violates the Agent Skills layout",
        (
            "A skill name is lower-case letters, digits and single hyphens, at most 64"
            " characters. It derives from the package path: fix the path or the policy.",
            "A link to `references/...` without that file, or a frontmatter that does not parse,"
            " is a generator defect: do not edit the output; fix `render` and report it.",
            "Rerun `check`.",
        ),
    ),
    E_STALE: (
        "Generated files differ from what the package produces",
        (
            "Run `explain` (or `explain scikitplot.<module>`) to see a unified diff per file.",
            "If the diff is the change you made (docstring, `__all__`, policy): run `sync`,"
            " review `git diff plugins/`, and commit generated files with the source change.",
            "If the diff is not what you intended: fix the source. Never edit a generated"
            " file by hand; the next `sync` reverts it.",
            "`missing` = skill wanted, file absent; `outdated` = content differs; `orphaned` ="
            " no skill produces the file any more (module removed, excluded or renamed): `sync`"
            " deletes it. Shared skills (`plugins/_shared`) are never touched.",
            "Stale only in CI: CI imports another package version or lacks optional"
            " dependencies. Run `explain` in both and regenerate in the CI environment.",
            "Rerun `check`: it must pass. If it fails right after `sync`, see E-UNSTABLE.",
        ),
    ),
    E_IO: (
        "A file could not be read, written or removed",
        (
            "The message names the path: check permissions, free disk space and that the"
            " checkout is writable.",
            "Rerun `sync`: it is idempotent and skips files that are already correct.",
            "Never commit a half-synced tree: run `check` until it passes.",
        ),
    ),
    E_UNSTABLE: (
        "The tree still differs after `sync`: the output is not deterministic",
        (
            "Run `sync` twice and diff the results to find the unstable text (set order,"
            " memory address, platform class, Python-version-specific repr).",
            "Fix the text rendering in the generator's canonical-text functions (Rule 168:"
            " byte-identical on every platform); not the generated file.",
            "Do not commit until `check` passes in a fresh process.",
        ),
    ),
    E_USAGE: (
        "The command line selects nothing",
        (
            "`list` shows module and skill names; `explain` accepts a module name, a skill name"
            " or a generated path (or a folder prefix of one).",
            "A skill whose module was removed is still selectable by its skill name or path"
            " while its files are on disk.",
        ),
    ),
    W_COVERAGE: (
        "A package has a development skill but neither a usage skill nor an exclusion",
        (
            "Decide: add it to `usage_modules` and `sync`, or to `usage_excluded` with a reason.",
            "`--strict` turns this warning into an error.",
        ),
    ),
    W_EXCLUDED: (
        "`usage_excluded` names a module that has no development skill (renamed or removed)",
        ("Remove the entry from `usage_excluded`, or fix its spelling.",),
    ),
    W_EMPTY: (
        "A module has no public names and no public subpackages",
        (
            "Declare the public names in the module's `__all__`, or move the module to"
            " `usage_excluded` with a reason.",
        ),
    ),
    W_DOC: (
        "Public names have no docstring summary",
        (
            "Add a one-sentence docstring to each named function, class, enum or exception;"
            " the summary column of the skill is that sentence.",
        ),
    ),
    W_EXAMPLE: (
        "No docstring of the module or its public names has a doctest example",
        (
            "Add an `Examples` section with `>>>` lines to the module docstring or to a public"
            " function; the skill shows the first one found.",
        ),
    ),
    W_SIGNATURE: (
        "A signature could not be read and is shown as `name(...)`",
        (
            "`inspect.signature` failed: give the callable a Python-level signature or a"
            " correct `__signature__`.",
        ),
    ),
    W_DESCRIPTION: (
        "A frontmatter description was cut at the 1024-character limit",
        (
            "Shorten the first sentence of the module docstring (or the partial distribution's"
            " summary); names listed after `needs` are the first eight of `__all__`.",
        ),
    ),
}


def _tag(code: str, text: str) -> str:
    return f"[{code}] {text}"


def _code_of(message: str) -> str:
    match = _CODE_RE.match(message)
    return match.group(1) if match else E_POLICY


def _listed(items: Sequence[str], limit: int = MAX_LISTED) -> str:
    items = list(items)
    text = ", ".join(items[:limit])
    extra = len(items) - limit
    return text + (f", ... (+{extra} more)" if extra > 0 else "")


class SkillError(Exception):
    """
    A module cannot be turned into a skill as declared.

    Parameters
    ----------
    message : str
        User-facing text.
    code : str, default="E-POLICY"
        One of the ``E-*`` codes of :data:`GUIDE`.
    messages : sequence of str, optional
        Already tagged messages when one exception carries several problems.

    Attributes
    ----------
    code : str
    messages : list of str
        Tagged messages (``[E-...] text``); one entry unless *messages* is given.
    """

    def __init__(
        self,
        message: str,
        code: str = E_POLICY,
        messages: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.messages = list(messages) if messages else [self.tagged()]

    def tagged(self) -> str:
        """Return ``[code] message``."""
        return _tag(self.code, str(self))


@dataclass
class Report:
    # Annotations are written ``typing.X``: with ``from __future__ import
    # annotations`` they are strings, and ``@dataclass`` resolves a bare name
    # (``List[str]``) through ``sys.modules[__module__]``, which is absent when
    # this file is executed without registering it; a module-qualified name is
    # looked up only when that module is present (test_usage_skills.py).

    errors: typing.List[str]
    warnings: typing.List[str]
    stale: typing.List[str]
    counts: typing.Dict[str, int]

    @property
    def ok(self) -> bool:
        return not self.errors


def remediation(messages: Sequence[str]) -> List[Dict[str, object]]:
    """
    Return the guideline entries for the codes present in *messages*.

    Parameters
    ----------
    messages : sequence of str
        Tagged errors and warnings.

    Returns
    -------
    list of dict
        ``{"code", "title", "steps"}`` in first-seen order, one per code known to
        :data:`GUIDE`.
    """
    seen: List[str] = []
    for message in messages:
        match = _CODE_RE.match(message)
        if match and match.group(1) in GUIDE and match.group(1) not in seen:
            seen.append(match.group(1))
    return [{"code": c, "title": GUIDE[c][0], "steps": list(GUIDE[c][1])} for c in seen]


_RECORDS_CACHE: Dict[Path, object] = {}
_RECORDS_NEEDS = (
    "USAGE_SKILLS_DIR",
    "derived_skill_names",
    "mirror_keys",
    "shared_skill_names",
)


def _records(repo: Optional[Path] = None):
    """
    Load ``check_records.py`` of *repo* (default: this checkout), once per path.

    Raises
    ------
    SkillError
        ``E-RECORDS`` if the file is missing, fails to load, or lacks a name
        this generator reads.
    """
    path = ((REPO if repo is None else repo) / RECORDS_RELATIVE).resolve()
    if path in _RECORDS_CACHE:
        return _RECORDS_CACHE[path]
    if not path.is_file():
        raise SkillError(f"{path} does not exist", E_RECORDS)
    spec = importlib.util.spec_from_file_location(
        "_check_records_for_skills",
        path,
    )
    if spec is None or spec.loader is None:
        raise SkillError(f"cannot load {path}", E_RECORDS)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - reported with its cause
        raise SkillError(
            f"{path} fails to load: {type(exc).__name__}: {exc}",
            E_RECORDS,
        ) from exc
    missing = [name for name in _RECORDS_NEEDS if not hasattr(module, name)]
    if missing:
        raise SkillError(
            f"{path} does not define {', '.join(missing)}",
            E_RECORDS,
        )
    _RECORDS_CACHE[path] = module
    return module


def load_policy(repo: Path) -> dict:
    """
    Return ``[tool.scikitplot.maintenance.skills]`` of ``pyproject.toml``.

    Parameters
    ----------
    repo : pathlib.Path
        Repository root.

    Returns
    -------
    dict
        ``usage_modules`` (list of dotted module names) and
        ``usage_excluded`` (module name to the reason it has no skill).

    Raises
    ------
    SkillError
        ``E-ENV`` when no TOML reader is installed; ``E-POLICY`` when the file,
        the table or one of its keys is missing or malformed, a module name is
        not ``scikitplot[.name...]``, a module is listed twice or both used and
        excluded.
    """
    if tomllib is None:
        raise SkillError(
            "Python < 3.11 requires 'tomli' to read pyproject.toml",
            E_ENV,
        )
    path = repo / "pyproject.toml"
    try:
        with path.open("rb") as handle:
            table = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise SkillError(
            f"{path} does not exist",
            E_POLICY,
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise SkillError(
            f"{path.name} is not valid TOML: {exc}",
            E_POLICY,
        ) from exc
    for key in POLICY_KEYS:
        if not isinstance(table, dict) or key not in table:
            raise SkillError(
                f"pyproject.toml has no [{'.'.join(POLICY_KEYS)}]",
                E_POLICY,
            )
        table = table[key]
    modules = table.get("usage_modules")
    excluded = table.get("usage_excluded", {})
    if not isinstance(modules, list) or not all(isinstance(m, str) for m in modules):
        raise SkillError("skills.usage_modules must be a list of module names")
    if not isinstance(excluded, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in excluded.items()
    ):
        raise SkillError("skills.usage_excluded must map module names to reasons")
    badly_named = sorted(m for m in [*modules, *excluded] if not _MODULE_RE.fullmatch(m))
    if badly_named:
        raise SkillError(
            f"not a dotted module name starting with 'scikitplot': {badly_named}"
        )
    overlap = sorted(set(modules) & set(excluded))
    if overlap:
        raise SkillError(f"listed both as usage modules and excluded: {overlap}")
    if len(set(modules)) != len(modules):
        twice = sorted({m for m in modules if modules.count(m) > 1})
        raise SkillError(f"skills.usage_modules lists a module twice: {twice}")
    return {"usage_modules": modules, "usage_excluded": excluded}


def skill_name(
    module: str,
    derived: dict[str, str],
) -> str:
    """
    Return the usage skill name of a dotted module (Rule 164).

    Parameters
    ----------
    module : str
        ``"scikitplot"`` or ``"scikitplot.<path>"``.
    derived : dict
        :func:`check_records.derived_skill_names` of the package paths.

    Returns
    -------
    str
        ``"scikitplot"``, or ``"scikitplot-" + derived[path]``.
    """
    if module == "scikitplot":
        return "scikitplot"
    path = module[len("scikitplot.") :].replace(".", "/")
    if path not in derived:
        raise SkillError(f"{module} is not a package below scikitplot/", E_NAME)
    return f"scikitplot-{derived[path]}"


_TITLE = re.compile(r"\A[^\n]+\n([=\-~^*#\"'`+])\1{2,}[ \t]*(\n|\Z)")
_ROLE = re.compile(r":(?:[a-z]+:)?[a-z]+:`(~?)([^`<]*?)(?:\s*<([^>]+)>)?`")
_FOOTNOTE = re.compile(r"\s*\[(?:\d+|#\w*|\*)\]_")


def _plain(text: str) -> str:
    """
    Return reST text as plain Markdown-safe text.

    ``:py:class:`~numpy.random.Generator``` becomes ```Generator```, as Sphinx
    shows it (``~`` keeps the last part); a role with an explicit title keeps
    the title; footnote references (``[1]_``) are dropped.
    """

    def role(match: re.Match) -> str:
        tilde, target, explicit = match.groups()
        if explicit:
            return f"`{target.strip()}`"
        target = target.strip().lstrip(".")
        if tilde:
            target = target.rsplit(".", 1)[-1]
        return f"`{target}`"

    return _FOOTNOTE.sub("", _ROLE.sub(role, text))


def _first_paragraph(doc: str) -> str:
    """Return the first paragraph, after a reST title, before any numpydoc section."""
    doc = inspect.cleandoc(doc or "")
    doc = _TITLE.sub("", doc, count=1).lstrip("\n")
    head = _SECTION.split(doc, maxsplit=1)[0]
    return _plain(" ".join(head.strip().split("\n\n", 1)[0].split()))


# --------------------------------------------------------------------------- canonical text
#
# Every text below is rendered here from the object's facts (names, parameter
# kinds, defaults, annotations, the class's own docstring), never taken from a
# format CPython, ``typing`` or the operating system owns: ``str(Signature)``,
# ``repr`` of typing constructs, ``type(obj).__name__`` of a platform class, or
# a docstring inherited from a built-in. Those formats change between Python
# versions (``Enum`` signatures in 3.12, ``typing.`` prefixes, ``ForwardRef``,
# the lazy annotations of 3.14) and platforms (``PosixPath``), and the files
# are compared byte for byte (Rule 168).

_TYPING_PREFIX = re.compile(r"(?<![\w.])typing(?:_extensions)?\.")
_FORWARD_REF = re.compile(r"""ForwardRef\((['"])(.*?)\1(?:, [^)]*)?\)""")
_P = inspect.Parameter
#: ``int | str`` (3.10+) has its own origin; ``typing.Union`` is the same object from 3.14.
_UNION_TYPE = getattr(__import__("types"), "UnionType", None)
#: ``typing.Annotated`` exists from Python 3.9.
_ANNOTATED = getattr(typing, "Annotated", None)


def _own_doc(obj: object) -> str:
    """
    Return the docstring *obj* or a scikit-plots base of it defines, else ``""``.

    A class without one of its own would otherwise show a built-in base's text
    (``int([x]) -> integer``), which belongs to the interpreter and changes
    between versions.
    """
    if inspect.isclass(obj):
        for klass in obj.__mro__:
            if str(getattr(klass, "__module__", "")).split(".")[0] != "scikitplot":
                break
            doc = klass.__dict__.get("__doc__")
            if isinstance(doc, str) and doc.strip():
                return inspect.cleandoc(doc)
        return ""
    return inspect.getdoc(obj) or ""


def _summary(obj: object) -> str:
    """One line: the first sentence of a function's, class's or module's docstring."""
    if not (inspect.isroutine(obj) or inspect.isclass(obj) or inspect.ismodule(obj)):
        return ""
    doc = inspect.getdoc(obj) if inspect.ismodule(obj) else _own_doc(obj)
    paragraph = _first_paragraph(doc or "")
    sentence = re.split(r"(?<=[.!?])\s", paragraph, maxsplit=1)[0]
    return sentence.replace("|", "\\|")


def _is_protocol(obj: object) -> bool:
    return inspect.isclass(obj) and bool(getattr(obj, "_is_protocol", False))


def _kind(obj: object) -> str:
    if _is_protocol(obj):
        return "protocol"
    if inspect.isclass(obj):
        if issubclass(obj, BaseException):
            return "exception"
        return "enum" if issubclass(obj, enum.Enum) else "class"
    if type(obj).__module__ in ("typing", "typing_extensions"):
        return "type alias"
    if inspect.isroutine(obj):
        return "function"
    if inspect.ismodule(obj):
        return "module"
    if isinstance(obj, pathlib.PurePath):
        return "Path"  # PosixPath or WindowsPath, by the platform
    return type(obj).__name__


_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _default(value: object) -> str:
    """
    Return the text of a default value, the same in every process and on every platform.

    A set's repr follows string hashing, which differs per process; an
    object's repr may hold its memory address; a path's repr names the
    platform's class.
    """
    if isinstance(value, (set, frozenset)):
        items = ", ".join(sorted(_default(item) for item in value))
        return f"{type(value).__name__}({{{items}}})" if items else f"{type(value).__name__}()"
    if isinstance(value, pathlib.PurePath):
        return f"Path({value.as_posix()!r})"
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    return _ADDRESS.sub("", repr(value))


def _annotation(value: object) -> str:
    """
    Return the text of an annotation, built from its parts on every Python.

    A string (``from __future__ import annotations``, a quoted name) is shown
    quoted, as written; a ``ForwardRef`` as its string; ``None`` as ``None``;
    a union as ``A | B`` (3.14 prints ``typing.Optional[int]`` as
    ``int | None``, earlier versions as ``Optional[int]``); ``Annotated``,
    ``Literal``, ``Callable`` and any other generic as ``Name[args]`` with each
    argument rendered the same way; a class by its dotted name (only the name
    for a built-in); a type variable by its name. Only an unknown construct
    falls back to its repr, without ``typing.`` prefixes.
    """
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, typing.ForwardRef):
        return repr(value.__forward_arg__)
    if value is None or value is type(None):
        return "None"
    if value is Ellipsis:
        return "..."
    if isinstance(value, list):  # the parameter list of Callable[[A, B], R]
        return f"[{', '.join(_annotation(item) for item in value)}]"
    origin, args = typing.get_origin(value), typing.get_args(value)
    if origin is typing.Union or (_UNION_TYPE is not None and origin is _UNION_TYPE):
        rendered = [_annotation(a) for a in args if a is not type(None)]
        if len(rendered) < len(args):
            rendered.append("None")
        return " | ".join(rendered)
    if _ANNOTATED is not None and origin is _ANNOTATED:
        metadata = ", ".join(_ADDRESS.sub("", repr(m)) for m in value.__metadata__)
        return f"Annotated[{_annotation(args[0])}, {metadata}]"
    if origin is getattr(typing, "Literal", None):
        return f"Literal[{', '.join(repr(a) for a in args)}]"
    if origin is not None:
        name = getattr(value, "_name", None) or getattr(origin, "__qualname__", None) or repr(origin)
        if not args:
            return name
        if name == "Callable" and len(args) == 2:
            params, result = args
            return (
                f"Callable[{_annotation(params if params is Ellipsis else list(params))}, {_annotation(result)}]"
            )
        return f"{name}[{', '.join(_annotation(a) for a in args)}]"
    if isinstance(value, typing.TypeVar):
        return value.__name__
    if isinstance(value, type):
        module, name = value.__module__, value.__qualname__
        return name if module == "builtins" else f"{module}.{name}"
    text = _FORWARD_REF.sub(lambda m: repr(m.group(2)), repr(value))
    return _TYPING_PREFIX.sub("", text)


def _parameters(sig: inspect.Signature) -> str:
    """Render ``(a: T = d, /, b, *args, c, **kw) -> R`` from the parameters' facts."""
    parts: list[str] = []
    seen_keyword_only = seen_positional_only = False
    for param in sig.parameters.values():
        if param.kind is _P.POSITIONAL_ONLY:
            seen_positional_only = True
        elif seen_positional_only:
            parts.append("/")
            seen_positional_only = False
        if param.kind is _P.KEYWORD_ONLY and not seen_keyword_only:
            if not any(p.kind is _P.VAR_POSITIONAL for p in sig.parameters.values()):
                parts.append("*")
            seen_keyword_only = True
        prefix = {_P.VAR_POSITIONAL: "*", _P.VAR_KEYWORD: "**"}.get(param.kind, "")
        text = prefix + param.name
        if param.annotation is not _P.empty:
            text += f": {_annotation(param.annotation)}"
        if param.default is not _P.empty:
            text += (" = " if param.annotation is not _P.empty else "=") + _default(param.default)
        parts.append(text)
    if seen_positional_only:
        parts.append("/")
    text = f"({', '.join(parts)})"
    if sig.return_annotation is not _P.empty:
        text += f" -> {_annotation(sig.return_annotation)}"
    return text


def _signature(name: str, obj: object) -> str:
    """
    Return the call signature of a public name as Markdown code.

    An enum is called with one member value, ``Name(value)``, whatever its
    metaclass's signature looks like in this Python; a protocol is never
    called, so only its name is shown.
    """
    if not (inspect.isroutine(obj) or inspect.isclass(obj)):
        return f"`{name}`"
    if _is_protocol(obj):
        return f"`{name}`"
    if inspect.isclass(obj) and issubclass(obj, enum.Enum):
        return f"`{name}(value)`"
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return f"`{name}(...)`"
    text = " ".join(_parameters(sig).split())
    if len(text) > SIGNATURE_MAX:
        text = text[: SIGNATURE_MAX - 4] + " ...)"
    return f"`{name}{text}`".replace("|", "\\|")


_DOCTEST = doctest.DocTestParser()


def _examples(doc: str) -> str:
    """
    Return the executed doctest examples of a numpydoc ``Examples`` section.

    Parameters
    ----------
    doc : str
        A docstring, possibly empty.

    Returns
    -------
    str
        The section's examples as a ``pycon`` transcript, or ``""`` when the
        docstring has no ``Examples`` section or every example in it is
        skipped.

    Notes
    -----
    **Developer notes.** The section is read with :class:`doctest.DocTestParser`,
    the parser the test suite runs, so the skill shows exactly the examples a
    doctest run executes and checks. An example marked ``# doctest: +SKIP``
    never runs: its call and its output are unverified, and it is where a
    docstring keeps live network sources, files the reader does not have and
    optional backends. Prose between examples is commentary for the rendered
    documentation, not transcript, and is not copied into the code block.
    A ``>>>`` comment line is transcript and is kept; reST markup such as a
    ``.. plot::`` directive is not. Lines that were separated in the docstring
    stay separated by one blank line, and the block is dedented.
    """
    doc = inspect.cleandoc(doc or "")
    parts = _SECTION.split(doc)
    # parts: [before, title1, body1, title2, body2, ...]
    for index in range(1, len(parts) - 1, 2):
        if parts[index].strip() != "Examples":
            continue
        body = parts[index + 1].split("\n")
        role: Dict[int, bool] = {}  # line -> True kept example, False skipped
        for example in _DOCTEST.get_examples(parts[index + 1]):
            size = example.source.count("\n") + example.want.count("\n")
            for line in range(example.lineno, example.lineno + size):
                role[line] = not example.options.get(doctest.SKIP, False)
        if True not in role.values():
            continue
        kept: List[str] = []
        gap = False
        for number, text in enumerate(body):
            # A ">>>" line outside any example is a comment or a bare prompt.
            keep = role.get(number, text.lstrip().startswith(">>>"))
            if keep:
                if gap and kept:
                    kept.append("")
                kept.append(text.rstrip())
                gap = False
            else:
                gap = True
        return textwrap.dedent("\n".join(kept))
    return ""


def module_api(module: str) -> dict:
    """
    Read one module's declared public API from the installed package.

    Parameters
    ----------
    module : str
        Dotted module name.

    Returns
    -------
    dict
        ``doc`` (first paragraph), ``rows`` (one dict per public name in
        ``__all__`` order: ``name``, ``kind``, ``signature``, ``summary``),
        ``children`` (for an empty ``__all__``: public child packages and
        whether each defines a Sphinx ``setup``), ``example`` and its
        ``example_source``.

    Raises
    ------
    SkillError
        When the module does not import, has no ``__all__``, or a declared
        name does not resolve.
    """
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - reported with its cause
        raise SkillError(
            f"{module} does not import: {type(exc).__name__}: {exc}",
            E_IMPORT,
        ) from exc
    names = getattr(mod, "__all__", None)
    if names is None:
        raise SkillError(f"{module} declares no __all__", E_API)
    rows, failed = [], []
    for name in names:
        if name.startswith("_"):
            continue
        try:
            obj = getattr(mod, name)
        except Exception as exc:  # noqa: BLE001 - reported with its cause
            failed.append(f"{name} ({type(exc).__name__}: {exc})")
            continue
        if inspect.ismodule(obj):
            continue
        rows.append(
            {
                "name": name,
                "kind": _kind(obj),
                "signature": _signature(name, obj),
                "summary": _summary(obj),
                "doc": _own_doc(obj) if (inspect.isroutine(obj) or inspect.isclass(obj)) else "",
            }
        )
    if failed:
        raise SkillError(
            f"{module}: names in __all__ that do not resolve here: {failed}",
            E_API,
        )
    children = []
    if not rows and hasattr(mod, "__path__"):
        import pkgutil

        for info in sorted(
            pkgutil.iter_modules(mod.__path__),
            key=lambda i: i.name,
        ):
            if not info.ispkg or info.name.startswith("__") or info.name in ("tests", "test"):
                continue
            child = importlib.import_module(f"{module}.{info.name}")
            children.append(
                {
                    "module": f"{module}.{info.name}",
                    "summary": _summary(child),
                    "sphinx": callable(getattr(child, "setup", None)),
                }
            )
    example, source = _examples(mod.__doc__ or ""), module
    if not example:
        for row in rows:
            example = _examples(row["doc"])
            if example:
                source = f"{module}.{row['name']}"
                break
    return {
        "doc": _first_paragraph(mod.__doc__ or ""),
        "rows": rows,
        "children": children,
        "example": example,
        "example_source": source if example else "",
    }


#: The install guide. A skill links to it instead of naming another index, a
#: git URL or a distribution that is not on PyPI (see :func:`_install_lines`).
INSTALL_GUIDE = "https://scikit-plots.github.io/dev/install/installation.html"
#: Its section on the single-part distributions and development builds.
INSTALL_PARTS = INSTALL_GUIDE + "#installing-a-lightweight-version"


def _install_lines(module: str, dists: Sequence[str]) -> List[str]:
    """
    Return the ``## Install`` section of a usage skill.

    Parameters
    ----------
    module : str
        Dotted module name, such as ``"scikitplot.corpus"``.
    dists : sequence of str
        The partial distributions that ship *module*; may be empty.

    Returns
    -------
    list of str
        Markdown lines, ending with a blank line.

    Notes
    -----
    **Developer notes.** The only command is ``pip install scikit-plots``, the
    one name this project holds on PyPI. A skill is read from the checkout's
    version, which can be newer than that release; a module added since is
    missing from it, so the text says so. The partial distributions are not
    on PyPI, and an install from another index or from a git URL redirects
    the package source, which the Claude directory refuses in a plugin
    ("in documentation, link to the package manager's instructions instead");
    both are therefore linked, not written, and the guide keeps the commands.
    """
    lines = ["## Install", "", "```sh", "pip install scikit-plots", "```", ""]
    tail = f" or to install only `{dists[0]}`," if len(dists) == 1 else (
        " or to install only one part (" + ", ".join(f"`{d}`" for d in dists) + "),"
        if dists else ""
    )
    if module == "scikitplot":
        # The package root is in every release; only its contents can differ.
        text = (
            "The release on PyPI can be older than the version this skill was"
            " read from; a name below can be missing from it. For a development"
            f" build, follow the [install guide]({INSTALL_GUIDE})."
        )
    else:
        text = (
            "The release on PyPI can be older than the version this skill was read"
            f" from, and can lack `{module}`. If `import {module}` fails after"
            f" installing it,{tail} follow the"
            f" [install guide]({INSTALL_PARTS if dists else INSTALL_GUIDE})."
        )
    lines += textwrap.wrap(text, width=79, break_long_words=False, break_on_hyphens=False)
    return lines + [""]


def distributions_for(module: str) -> list[tuple[str, str]]:
    """Return ``(distribution, summary)`` for each partial distribution shipping *module*."""
    from scikitplot import _distributions

    path = module[len("scikitplot.") :].replace(".", "/") if module != "scikitplot" else ""
    found = []
    for dist in _distributions.DISTRIBUTIONS:
        if path and any(path == tree or path.startswith(tree + "/") for tree in dist.trees):
            found.append((dist.name, dist.summary))
    return found


def _dist_trees() -> list[tuple[str, str, tuple[str, ...]]]:
    from scikitplot import _distributions

    return [(d.name, d.summary, tuple(d.trees)) for d in _distributions.DISTRIBUTIONS]


def _quote(text: str) -> str:
    import json

    return json.dumps(text, ensure_ascii=False)


def render(
    module: str,
    name: str,
    api: dict,
    related: list[tuple[str, str]],
) -> dict[str, str]:
    """
    Render a usage skill's files.

    Parameters
    ----------
    module, name : str
        The module and its skill name.
    api : dict
        :func:`module_api`.
    related : list of tuple
        For the root skill, ``(module, skill name)`` of every usage skill.

    Returns
    -------
    dict
        Relative path inside the skill folder to file text.
    """
    dists = distributions_for(module)
    names = [r["name"] for r in api["rows"]] or [c["module"].rsplit(".", 1)[-1] for c in api["children"]]
    path = module[len("scikitplot.") :].replace(".", "/")
    # The main tree of a partial distribution has a one-line summary written
    # for users; otherwise the module docstring's first sentence describes it.
    own = [summary for dist, summary, trees in _dist_trees() if trees and trees[0] == path]
    what = api["doc"] or (own[0] if own else "")
    lead = own[0] if own else (re.split(r"(?<=[.!?])\s", what, maxsplit=1)[0] if what else "")
    description = f"Use {module} (scikit-plots) in any Python project."
    if lead:
        description += f" {lead}"
    if names:
        description += f" Use when code imports {module} or needs " + ", ".join(names[:8]) + "."
    if len(description) > DESCRIPTION_MAX:
        description = description[: DESCRIPTION_MAX - 3].rstrip() + "..."
    lines = [
        "---",
        f"name: {name}",
        f"description: {_quote(description)}",
        "license: BSD-3-Clause",
        "metadata:",
        f'  scikitplot-module: "{module}"',
        "---",
        "",
        f"# `{module}`",
        "",
        f"<!-- {GENERATED} -->",
        "",
    ]
    if what:
        lines += [what, ""]
    lines += _install_lines(module, [dist for dist, _ in dists])
    lines += [
        "## Before writing code",
        "",
        "The API below was read from one version of the package. Check the version the",
        "project has, and read a name's docstring and signature before calling it:",
        "",
        "```sh",
        'python -c "import scikitplot; print(scikitplot.__version__)"',
        f'python -c "import {module} as m, inspect; print(inspect.signature(m.<name>)); help(m.<name>)"',
        "```",
        "",
    ]
    if api["rows"]:
        shown = api["rows"][:SHOWN_NAMES]
        lines += ["## Public API", "", f"Declared in `{module}.__all__`, in that order.", ""]
        lines += ["| Name | Kind | Summary |", "|---|---|---|"]
        lines += [f"| `{r['name']}` | {r['kind']} | {r['summary']} |" for r in shown]
        rest = len(api["rows"]) - len(shown)
        tail = f"{rest} more names, and" if rest > 0 else "All names with"
        lines += ["", f"{tail} signatures: [references/API.md](references/API.md).", ""]
    if api["children"]:
        sphinx = any(c["sphinx"] for c in api["children"])
        head = "| Module | Summary | Defines `setup(app)` |" if sphinx else "| Module | Summary |"
        lines += ["## Subpackages", "", head, "|---|---|---|" if sphinx else "|---|---|"]
        for c in api["children"]:
            row = f"| `{c['module']}` | {c['summary']} |"
            lines.append(row + (" yes |" if c["sphinx"] else " |") if sphinx else row)
        if sphinx:
            lines += [
                "",
                "A subpackage that defines `setup(app)` is a Sphinx extension: add its module",
                "name to `extensions` in the project's `conf.py`.",
            ]
        lines.append("")
    if api["example"]:
        lines += [f"## Example (from the docstring of `{api['example_source']}`)", "", "```pycon"]
        lines += [api["example"], "```", ""]
    if related:
        lines += ["## Skills for the parts of scikit-plots", "", "| Module | Skill |", "|---|---|"]
        lines += [f"| `{m}` | `{s}` |" for m, s in related]
        lines.append("")
    files = {"SKILL.md": "\n".join(lines).rstrip("\n") + "\n"}
    if api["rows"]:
        ref = [f"# `{module}` public API", "", f"<!-- {GENERATED} -->", ""]
        ref += ["| Signature | Kind | Summary |", "|---|---|---|"]
        ref += [f"| {r['signature']} | {r['kind']} | {r['summary']} |" for r in api["rows"]]
        files["references/API.md"] = "\n".join(ref) + "\n"
    return files


# --------------------------------------------------------------------------- lint


def lint_api(module: str, api: dict) -> List[str]:
    """
    Return warnings about one module's API that make a weak skill.

    Parameters
    ----------
    module : str
        Dotted module name.
    api : dict
        :func:`module_api`.

    Returns
    -------
    list of str
        Tagged ``W-*`` messages: no public names or subpackages, callables
        without a docstring summary, no doctest example, signature unreadable.
    """
    rows, warnings = api["rows"], []
    if not rows and not api["children"]:
        warnings.append(
            _tag(
                W_EMPTY,
                (
                    f"{module}: no public names and no public subpackages; "
                    "the skill would say only how to install it"
                ),
            ),
        )
    callables = ("function", "class", "enum", "exception", "protocol")
    undocumented = [r["name"] for r in rows if r["kind"] in callables and not r["summary"]]
    if undocumented:
        warnings.append(
            _tag(
                W_DOC,
                (
                    f"{module}: {len(undocumented)} public name(s) without a docstring "
                    f"summary: {_listed(undocumented)}"
                ),
            ),
        )
    unreadable = [r["name"] for r in rows if r["signature"].endswith("(...)`")]
    if unreadable:
        warnings.append(
            _tag(
                W_SIGNATURE,
                f"{module}: signature unreadable for {_listed(unreadable)}",
            ),
        )
    if (rows or api["children"]) and not api["example"]:
        warnings.append(
            _tag(
                W_EXAMPLE,
                f"{module}: no doctest example in any docstring",
            ),
        )
    return warnings


def lint_files(
    module: str,
    name: str,
    files: Mapping[str, str],
) -> Tuple[List[str], List[str]]:
    """
    Check one rendered skill against the Agent Skills layout.

    Parameters
    ----------
    module, name : str
        The module and its skill name.
    files : mapping
        :func:`render`.

    Returns
    -------
    errors, warnings : list of str
        ``E-LINT`` for an invalid name, unparsable frontmatter, a name that
        differs from the folder, an empty description, a link to a file that
        was not rendered; ``W-DESCRIPTION`` when the description was cut.
    """
    errors: List[str] = []
    warnings: List[str] = []

    def bad(text: str) -> None:
        errors.append(_tag(E_LINT, f"{module} -> {name}: {text}"))

    if not _SKILL_NAME_RE.fullmatch(name) or len(name) > SKILL_NAME_MAX:
        bad(
            f"name must be lower-case letters, digits and single hyphens, <= {SKILL_NAME_MAX}",
        )
    text = files.get("SKILL.md")
    if text is None:
        bad("SKILL.md was not rendered")
        return errors, warnings
    front = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    if front is None:
        bad("SKILL.md has no frontmatter")
        return errors, warnings
    fields = dict(
        line.split(": ", 1)
        for line in front.group(1).splitlines()
        if ": " in line and line[0] != " "
    )
    if fields.get("name") != name:
        bad(f"frontmatter name {fields.get('name')!r} differs from the folder")
    try:
        description = json.loads(fields.get("description", ""))
    except ValueError:
        description = None
    if not isinstance(description, str) or not description.strip():
        bad("frontmatter description is missing, empty or not a quoted string")
    elif len(description) > DESCRIPTION_MAX:
        bad(f"description has {len(description)} characters (limit {DESCRIPTION_MAX})")
    elif len(description) == DESCRIPTION_MAX and description.endswith("..."):
        warnings.append(
            _tag(
                W_DESCRIPTION,
                f"{module} -> {name}: description was cut at {DESCRIPTION_MAX}",
            ),
        )
    for link in sorted(set(_LINK_RE.findall(text))):
        if link not in files:
            bad(f"SKILL.md links to {link}, which was not rendered")
    return errors, warnings


# --------------------------------------------------------------------------- build


@dataclass
class Build:
    """
    Everything one evaluation of the package produced.

    Attributes
    ----------
    wanted : dict
        Repository path to file text; complete only when *errors* is empty.
    errors, warnings : list of str
        Tagged messages; every problem found, not only the first.
    names : dict
        Module to skill name.
    entries : list of dict
        ``module``, ``skill``, ``public_names``, ``subpackages``, ``files``.
    excluded : dict
        Module to the reason it has no skill.
    counts : dict
    """

    # Annotations are written ``typing.X``: with ``from __future__ import
    # annotations`` they are strings, and ``@dataclass`` resolves a bare name
    # (``List[str]``) through ``sys.modules[__module__]``, which is absent when
    # this file is executed without registering it; a module-qualified name is
    # looked up only when that module is present (test_usage_skills.py).

    wanted: typing.Dict[str, str] = field(default_factory=dict)
    errors: typing.List[str] = field(default_factory=list)
    warnings: typing.List[str] = field(default_factory=list)
    names: typing.Dict[str, str] = field(default_factory=dict)
    entries: typing.List[typing.Dict[str, object]] = field(default_factory=list)
    excluded: typing.Dict[str, str] = field(default_factory=dict)
    counts: typing.Dict[str, int] = field(default_factory=dict)


def _usage_dir(repo: Path) -> str:
    return Path(_records(repo).USAGE_SKILLS_DIR).as_posix()


def build_all(repo: Path) -> Build:
    """
    Evaluate the package and policy; collect every error and warning.

    Parameters
    ----------
    repo : pathlib.Path
        Repository root.

    Returns
    -------
    Build
        With ``wanted`` filled only when ``errors`` is empty. Environment,
        records and policy problems stop the evaluation (nothing else can be
        judged); module problems are all collected.
    """
    result = Build()
    # Importing the package is deferred to here, so the renderer above can be
    # tested on every Python without a built scikit-plots. This import is a
    # side effect on purpose: it registers the experimental ANNImputer in
    # scikitplot.impute, whose skill lists it.
    try:
        importlib.import_module("scikitplot.experimental.enable_ann_imputer")
        records = _records(repo)
        policy = load_policy(repo)
        derived = records.derived_skill_names(records.mirror_keys(repo))
    except SkillError as exc:
        result.errors.append(exc.tagged())
        return result
    except Exception as exc:  # noqa: BLE001 - reported with its cause
        code = E_ENV if isinstance(exc, ImportError) else E_RECORDS
        result.errors.append(_tag(code, f"{type(exc).__name__}: {exc}"))
        return result
    modules: List[str] = policy["usage_modules"]
    result.excluded = dict(policy["usage_excluded"])
    for module in modules:
        try:
            result.names[module] = skill_name(module, derived)
        except SkillError as exc:
            result.errors.append(exc.tagged())
    by_skill: Dict[str, List[str]] = {}
    for module, skill in result.names.items():
        by_skill.setdefault(skill, []).append(module)
    for skill, owners in sorted(by_skill.items()):
        if len(owners) > 1:
            result.errors.append(
                _tag(
                    E_POLICY,
                    f"skill {skill!r} is derived for {', '.join(owners)}",
                ),
            )
    known = {"scikitplot." + path.replace("/", ".") for path in derived if path}
    for module in sorted(known - set(modules) - set(result.excluded)):
        result.warnings.append(
            _tag(
                W_COVERAGE,
                f"{module} has a development skill but is neither in usage_modules nor usage_excluded",
            ),
        )
    for module in sorted(set(result.excluded) - known - {"scikitplot"}):
        result.warnings.append(
            _tag(W_EXCLUDED, f"usage_excluded names {module}, which has no development skill")
        )
    apis: Dict[str, dict] = {}
    for module in modules:
        try:
            apis[module] = module_api(module)
        except SkillError as exc:
            result.errors.append(exc.tagged())
    for module, api in apis.items():
        result.warnings.extend(lint_api(module, api))
    if result.errors:
        return result
    base = _usage_dir(repo)
    related = [(m, result.names[m]) for m in modules if m != "scikitplot"]
    for module in modules:
        files = render(
            module,
            result.names[module],
            apis[module],
            related if module == "scikitplot" else [],
        )
        found_errors, found_warnings = lint_files(
            module,
            result.names[module],
            files,
        )
        result.errors.extend(found_errors)
        result.warnings.extend(found_warnings)
        for rel, text in files.items():
            result.wanted[f"{base}/{result.names[module]}/{rel}"] = text
        result.entries.append(
            {
                "module": module,
                "skill": result.names[module],
                "public_names": len(apis[module]["rows"]),
                "subpackages": len(apis[module]["children"]),
                "files": sorted(files),
            }
        )
    result.counts = {
        "modules": len(modules),
        "excluded": len(result.excluded),
        "files": len(result.wanted),
        "public_names": sum(len(a["rows"]) for a in apis.values()),
    }
    return result


def _raise_if_errors(result: Build) -> None:
    if result.errors:
        raise SkillError(
            "; ".join(result.errors),
            _code_of(result.errors[0]),
            messages=result.errors,
        )


def build(repo: Path) -> Dict[str, str]:
    """
    Return every generated file of the usage skills, by repository path.

    Parameters
    ----------
    repo : pathlib.Path
        Repository root.

    Returns
    -------
    dict
        Path relative to *repo* to text.

    Raises
    ------
    SkillError
        With every problem found (``.messages``), from :func:`build_all`.
    """
    result = build_all(repo)
    _raise_if_errors(result)
    return result.wanted


def current(repo: Path) -> Dict[str, str]:
    """
    Return the usage skill files on disk, by repository path.

    The copies of the shared skills (``plugins/_shared/skills/``) are left
    out: ``check_records.py --write-index`` owns them (Rule 167), so this
    generator neither reports nor removes them.

    Raises
    ------
    SkillError
        ``E-IO`` when a file cannot be read as UTF-8.
    """
    records = _records(repo)
    base = repo / records.USAGE_SKILLS_DIR
    if not base.is_dir():
        return {}
    shared = set(records.shared_skill_names(repo))
    found: Dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.parent == base or path.relative_to(base).parts[0] in shared:
            continue
        try:
            found[path.relative_to(repo).as_posix()] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SkillError(f"cannot read {path}: {exc}", E_IO) from exc
    return found


def classify_stale(
    wanted: Mapping[str, str],
    have: Mapping[str, str],
) -> Dict[str, List[str]]:
    """
    Split the differences between generated and on-disk files.

    Returns
    -------
    dict
        ``missing`` (wanted, absent), ``outdated`` (both, text differs),
        ``orphaned`` (on disk, no longer wanted); each sorted.
    """
    return {
        "missing": sorted(p for p in wanted if p not in have),
        "outdated": sorted(p for p in wanted if p in have and wanted[p] != have[p]),
        "orphaned": sorted(p for p in have if p not in wanted),
    }


# --------------------------------------------------------------------------- check / sync


def check(repo: Path, *, strict: bool = False) -> Report:
    """
    Validate policy, every module, every rendered skill and the files on disk.

    Parameters
    ----------
    repo : pathlib.Path
        Repository root.
    strict : bool, default=False
        Warnings become errors.

    Returns
    -------
    Report
        ``errors``: environment/policy/module/lint problems and ``E-STALE``;
        ``warnings``: weak skills; ``stale``: stale paths (evaluated only when
        the package could be read completely); ``counts``.
    """
    result = build_all(repo)
    errors, warnings = list(result.errors), list(result.warnings)
    stale: List[str] = []
    counts = dict(result.counts)
    if not errors:
        try:
            groups = classify_stale(result.wanted, current(repo))
        except SkillError as exc:
            errors.extend(exc.messages)
        else:
            stale = sorted(p for paths in groups.values() for p in paths)
            counts.update({key: len(paths) for key, paths in groups.items()})
            if stale:
                errors.append(
                    _tag(
                        E_STALE,
                        f"generated file(s) out of date: {len(groups['missing'])} missing, "
                        f"{len(groups['outdated'])} outdated, {len(groups['orphaned'])} orphaned "
                        f"({_listed(stale)}) (run: python {SCRIPT} sync)",
                    )
                )
    if strict:
        errors, warnings = errors + warnings, []
    return Report(errors, warnings, stale, counts)


def _target(repo: Path, path: str) -> Path:
    """Return *path* resolved, refusing anything outside the usage skills folder."""
    root = (repo / _usage_dir(repo)).resolve()
    target = (repo / path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SkillError(
            f"refusing to touch {path}: outside {root}",
            E_IO,
        ) from exc
    return target


def _write(target: Path, text: str) -> None:
    """Write *text* to *target* atomically, UTF-8 with ``\\n`` newlines."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _remove_empty_folders(root: Path) -> None:
    for folder, _dirs, _files in os.walk(root, topdown=False):
        if Path(folder) != root and not os.listdir(folder):
            os.rmdir(folder)


def sync(repo: Path, *, apply: bool = True) -> Report:
    """
    Create, rewrite and delete generated files so that ``check`` passes.

    Parameters
    ----------
    repo : pathlib.Path
        Repository root.
    apply : bool, default=True
        Write; with ``False`` only plan.

    Returns
    -------
    Report
        ``stale``: the files created, rewritten or deleted (or that would be);
        ``errors``: ``E-IO`` per failing file and ``E-UNSTABLE`` if the tree
        still differs after writing.

    Raises
    ------
    SkillError
        If the package or policy cannot be read completely (nothing is
        written: a partial result would delete good skills).
    """
    result = build_all(repo)
    _raise_if_errors(result)
    groups = classify_stale(result.wanted, current(repo))
    changed = sorted(p for paths in groups.values() for p in paths)
    errors: List[str] = []
    if apply and changed:
        for path in changed:
            try:
                target = _target(repo, path)
                if path in result.wanted:
                    _write(target, result.wanted[path])
                else:
                    target.unlink()
            except SkillError as exc:
                errors.extend(exc.messages)
            except OSError as exc:
                errors.append(_tag(E_IO, f"{path}: {type(exc).__name__}: {exc}"))
        _remove_empty_folders((repo / _usage_dir(repo)).resolve())
        if not errors:
            again = classify_stale(result.wanted, current(repo))
            left = sorted(p for paths in again.values() for p in paths)
            if left:
                errors.append(
                    _tag(E_UNSTABLE, f"still differs after sync: {_listed(left)}")
                )
    counts = dict(result.counts)
    counts.update({key: len(paths) for key, paths in groups.items()})
    return Report(errors, result.warnings, changed, counts)


def explain(
    repo: Path,
    selector: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Return a unified diff for every stale file.

    Parameters
    ----------
    repo : pathlib.Path
        Repository root.
    selector : str, optional
        A module name, a skill name, or a generated path (or folder prefix);
        default: every file.

    Returns
    -------
    list of dict
        ``path``, ``status`` (``missing``/``outdated``/``orphaned``), ``diff``
        (``on disk`` -> ``generated``); empty when nothing selected is stale.

    Raises
    ------
    SkillError
        ``E-USAGE`` when *selector* selects no file; the errors of
        :func:`build_all` otherwise.
    """
    result = build_all(repo)
    _raise_if_errors(result)
    have = current(repo)
    groups = classify_stale(result.wanted, have)
    status = {p: kind for kind, paths in groups.items() for p in paths}
    known = sorted(set(result.wanted) | set(have))
    if selector:
        base = _usage_dir(repo)
        skill = result.names.get(selector, selector)
        prefixes = (f"{base}/{skill}/", selector.rstrip("/") + "/")
        known = [p for p in known if p == selector or p.startswith(prefixes)]
        if not known:
            raise SkillError(
                f"{selector!r} matches no skill or generated file",
                E_USAGE,
            )
    out = []
    for path in known:
        if path not in status:
            continue
        old = have.get(path, "").splitlines()
        new = result.wanted.get(path, "").splitlines()
        diff = difflib.unified_diff(
            old,
            new,
            fromfile=f"a/{path} (on disk)",
            tofile=f"b/{path} (generated)",
            lineterm="",
        )
        out.append(
            {
                "path": path,
                "status": status[path],
                "diff": "\n".join(diff),
            },
        )
    return out


def list_skills(repo: Path) -> Dict[str, object]:
    """
    Return the skills (``entries``) and exclusions (``excluded``) of the policy.

    Raises
    ------
    SkillError
        The errors of :func:`build_all`.
    """
    result = build_all(repo)
    _raise_if_errors(result)
    return {"skills": result.entries, "excluded": dict(result.excluded)}


# --------------------------------------------------------------------------- CLI


def _dump(payload: object, out: TextIO) -> None:
    json.dump(
        payload,
        out,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    out.write("\n")


def _write_guide(
    entries: Sequence[Mapping[str, object]],
    out: TextIO,
    heading: str,
) -> None:
    if not entries:
        return
    out.write(f"\n{heading}\n")
    for entry in entries:
        out.write(f"\n[{entry['code']}] {entry['title']}\n")
        for number, step in enumerate(entry["steps"], 1):  # type: ignore[arg-type]
            out.write(f"  {number}. {step}\n")


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--repo-root",
        type=Path,
        default=argparse.SUPPRESS,
        help="checkout root (also accepted after the command)",
    )
    parser = argparse.ArgumentParser(
        description="Generate, check and explain the usage skills of the scikitplot plugin",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="checkout root",
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )
    check_p = sub.add_parser(
        "check",
        parents=[common],
        help="validate everything (read-only)",
    )
    check_p.add_argument(
        "--json",
        action="store_true",
    )
    check_p.add_argument(
        "--strict",
        action="store_true",
        help="warnings are errors",
    )
    sync_p = sub.add_parser(
        "sync",
        parents=[common],
        help="write the generated files",
    )
    sync_p.add_argument(
        "--dry-run",
        action="store_true",
        help="plan only; write nothing",
    )
    sync_p.add_argument(
        "--json",
        action="store_true",
    )
    explain_p = sub.add_parser(
        "explain",
        parents=[common],
        help="diff of every stale file",
    )
    explain_p.add_argument(
        "selector",
        nargs="?",
        default=None,
        help="module, skill name or path",
    )
    explain_p.add_argument(
        "--json",
        action="store_true",
    )
    list_p = sub.add_parser(
        "list",
        parents=[common],
        help="skills and exclusions of the policy",
    )
    list_p.add_argument(
        "--json",
        action="store_true",
    )
    guide_p = sub.add_parser(
        "guide",
        parents=[common],
        help="how to handle every error and warning",
    )
    guide_p.add_argument(
        "--json",
        action="store_true",
    )
    return parser


def _run(
    args: argparse.Namespace,
    repo: Path,
    out: TextIO,
    err: TextIO,
) -> int:
    if args.command == "guide":
        entries = [
            {"code": c, "title": t, "steps": list(s)}
            for c, (t, s) in GUIDE.items()
        ]
        if args.json:
            _dump(entries, out)
        else:
            _write_guide(
                entries,
                out,
                "Usage skills: error and warning guideline",
            )
        return 0
    if args.command == "check":
        report = check(repo, strict=args.strict)
        steps = remediation(report.errors + report.warnings)
        if args.json:
            _dump(
                {
                    "ok": report.ok,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "stale": report.stale,
                    "counts": report.counts,
                    "remediation": steps,
                },
                out,
            )
        else:
            for warning in report.warnings:
                out.write(f"warning: {warning}\n")
            if report.ok:
                counts = ", ".join(
                    f"{k}={v}" for k, v in sorted(report.counts.items())
                )
                out.write(
                    f"Usage skills check: PASS ({counts})\n",
                )
                if report.warnings:
                    out.write(
                        f"- steps for each warning code: python {SCRIPT} guide\n",
                    )
            else:
                for path in report.stale:
                    err.write(f"stale: {path}\n")
                err.write("Usage skills check: FAIL\n")
                for error in report.errors:
                    err.write(f"- {error}\n")
                _write_guide(remediation(report.errors), err, "How to fix:")
        return 0 if report.ok else 1
    if args.command == "sync":
        report = sync(repo, apply=not args.dry_run)
        steps = remediation(report.errors)
        if args.json:
            _dump(
                {
                    "applied": not args.dry_run,
                    "ok": report.ok,
                    "changed": report.stale,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "remediation": steps,
                },
                out,
            )
        else:
            mode = "DRY-RUN" if args.dry_run else "APPLIED"
            out.write(
                f"Usage skills sync: {mode}; changed: {len(report.stale)} file(s)\n",
            )
            for path in report.stale:
                out.write(f"- {path}\n")
            if report.stale and args.dry_run:
                out.write("- rerun without --dry-run to write these files\n")
            if not report.ok:
                err.write("Usage skills sync: FAIL\n")
                for error in report.errors:
                    err.write(f"- {error}\n")
                _write_guide(steps, err, "How to fix:")
        return 0 if report.ok else 1
    if args.command == "explain":
        items = explain(repo, args.selector)
        if args.json:
            _dump(items, out)
        elif not items:
            out.write(
                "No differences: the selected generated files are up to date.\n",
            )
        else:
            for item in items:
                out.write(f"== {item['path']} ({item['status']})\n{item['diff']}\n")
        return 0
    payload = list_skills(repo)
    if args.json:
        _dump(payload, out)
    else:
        for entry in payload["skills"]:  # type: ignore[union-attr]
            out.write(
                f"{entry['module']:50} {entry['skill']:40} {entry['public_names']} name(s)\n",
            )
        for module, reason in sorted(payload["excluded"].items()):  # type: ignore[union-attr]
            out.write(f"{module:50} (excluded) {reason}\n")
    return 0


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> int:
    """Run the CLI; return 0 (ok), 1 (check failed / stale / write failed) or 2 (cannot evaluate)."""
    args = _parser().parse_args(argv)
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    repo = (REPO if args.repo_root is None else args.repo_root).resolve()
    if args.command != "guide" and not (repo / "pyproject.toml").is_file():
        err.write(f"error: {repo} has no pyproject.toml\n")
        return 2
    try:
        return _run(args, repo, out, err)
    except SkillError as exc:
        err.write(
            "usage skills: nothing was written; the package could not be evaluated\n",
        )
        for message in exc.messages:
            err.write(f"- {message}\n")
        _write_guide(remediation(exc.messages), err, "How to fix:")
        return 2


if __name__ == "__main__":
    sys.exit(main())
