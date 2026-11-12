#!/usr/bin/env python
"""
Fail when a requirement list names a distribution this project must not require.

Some distributions cannot be installed next to the one the code needs: two
distributions that ship the same import package write their files into one
directory, and what remains depends on the installer and its order. Nothing
in a requirement list shows that, and the name of the wrong distribution is
usually the import name, so it is added again in good faith.

``REFUSED`` below is the list of such names, each with the distribution to
require instead and the reason. This check reads every requirement list in
the repository and fails when one of them names a refused distribution.

Usage
-----
``python tools/maint_tools/check_refused_requirements.py [ROOT]``

ROOT is the repository root (default: the checkout this file is in). Exit
status 0 when no list names a refused distribution, 1 otherwise with each
place listed, 2 when a file cannot be read.

Notes
-----
**What is read.** ``pyproject.toml`` and every ``libs/*/pyproject.toml``:
``[build-system] requires``, ``[project] dependencies``, every
``[project.optional-dependencies]`` extra and every ``[dependency-groups]``
group. Every ``requirements/*.txt``: one requirement per line; comment lines
and option lines (``-r other.txt``, ``--index-url ...``) are not
requirements.

**Developer notes.** A requirement's distribution name is the leading name
of the PEP 508 string, compared after PEP 503 normalization (lower case,
runs of ``-``, ``_`` and ``.`` become ``-``), so ``pdfminer.six`` is
``pdfminer-six`` and is not the refused ``pdfminer``. Add an entry to
``REFUSED`` only with a reproduced reason, and say in ``reason`` what was
observed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, NamedTuple, Sequence

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # refused where a pyproject.toml is read, never at import


class Refusal(NamedTuple):
    """Why a distribution is refused and what to require instead."""

    use: str
    reason: str


#: Normalized distribution name -> what to require instead, and why.
REFUSED: dict[str, Refusal] = {
    "pdfminer": Refusal(
        use="pdfminer.six",
        reason=(
            "pdfminer and pdfminer.six both install the import package"
            " 'pdfminer'. Installed together (uv, CPython 3.12, pdfminer"
            " 20191125 and pdfminer.six 20260107) they left a package that"
            " cannot import pdfminer.high_level, which"
            " scikitplot.corpus.PDFReader needs; pdfminer alone has no"
            " pdfminer.high_level at all."
        ),
    ),
}

_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_SEPARATORS = re.compile(r"[-_.]+")


class Finding(NamedTuple):
    """One refused requirement: where it is and what it says."""

    path: str
    where: str
    requirement: str
    name: str


def normalize(name: str) -> str:
    """Return a distribution name in its PEP 503 normalized form."""
    return _SEPARATORS.sub("-", name).lower()


def requirement_name(requirement: str) -> str | None:
    """
    Return the normalized distribution name a requirement string starts with.

    Parameters
    ----------
    requirement : str
        A PEP 508 requirement, such as ``"pdfminer.six>=20221105; extra == 'x'"``.

    Returns
    -------
    str or None
        The normalized name, or ``None`` when the string does not start with
        a distribution name (an empty line, a comment, an option, a URL).
    """
    text = requirement.strip()
    # A comment, an option or an empty line does not start with a letter or
    # a digit, so it does not match.
    match = _NAME.match(text)
    if match is None:
        return None
    rest = text[match.end() :]
    # A name is followed by nothing, an extra, a version, a marker, a URL
    # reference or a comment; anything else ("https://...") is not a name.
    if rest and not rest.lstrip().startswith(
        ("[", "<", ">", "=", "!", "~", ";", "@", "#", "(")
    ):
        return None
    return normalize(match.group(0))


def _strings(value: Any) -> Iterator[str]:
    """Yield the requirement strings of a TOML list; tables in it are skipped."""
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                yield item


def pyproject_lists(data: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """
    Yield ``(where, requirement)`` for every requirement in a parsed pyproject.

    Parameters
    ----------
    data : dict
        The parsed ``pyproject.toml``.

    Yields
    ------
    tuple of (str, str)
        The list's place in the file, such as
        ``project.optional-dependencies.corpus``, and one requirement of it.
    """
    project = data.get("project")
    project = project if isinstance(project, dict) else {}
    build = data.get("build-system")
    build = build if isinstance(build, dict) else {}
    for requirement in _strings(build.get("requires")):
        yield "build-system.requires", requirement
    for requirement in _strings(project.get("dependencies")):
        yield "project.dependencies", requirement
    for table, source in (
        ("project.optional-dependencies", project.get("optional-dependencies")),
        ("dependency-groups", data.get("dependency-groups")),
    ):
        if isinstance(source, dict):
            for key in sorted(source):
                for requirement in _strings(source[key]):
                    yield f"{table}.{key}", requirement


def requirement_files(root: Path) -> tuple[list[Path], list[Path]]:
    """Return the pyproject files and the requirement text files under *root*."""
    pyprojects = [root / "pyproject.toml"]
    pyprojects += sorted((root / "libs").glob("*/pyproject.toml"))
    texts = sorted((root / "requirements").glob("*.txt"))
    return [p for p in pyprojects if p.is_file()], texts


def findings(root: Path) -> list[Finding]:
    """
    Return every refused requirement under *root*.

    Parameters
    ----------
    root : pathlib.Path
        Repository root.

    Returns
    -------
    list of Finding
        In file order.

    Raises
    ------
    RuntimeError
        If a ``pyproject.toml`` has to be read and no TOML parser is
        available (Python < 3.11 without ``tomli``).
    OSError, ValueError
        If a file cannot be read or is not valid TOML or UTF-8.
    """
    pyprojects, texts = requirement_files(root)
    found: list[Finding] = []

    def consider(path: Path, where: str, requirement: str) -> None:
        name = requirement_name(requirement)
        if name in REFUSED:
            relative = path.relative_to(root).as_posix()
            found.append(Finding(relative, where, requirement.strip(), name))

    for path in pyprojects:
        if tomllib is None:  # pragma: no cover - Python < 3.11 without tomli
            raise RuntimeError(
                "Python < 3.11 requires 'tomli' to read pyproject.toml files."
            )
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        for where, requirement in pyproject_lists(data):
            consider(path, where, requirement)
    for path in texts:
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            consider(path, f"line {number}", line)
    return found


def report(found: Iterable[Finding]) -> list[str]:
    """Return the lines that describe *found*, one block per finding."""
    lines: list[str] = []
    for finding in found:
        refusal = REFUSED[finding.name]
        lines.append(
            f"{finding.path}: {finding.where}: {finding.requirement!r} is refused;"
            f" require {refusal.use!r} instead."
        )
        lines.append(f"    {refusal.reason}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Check one repository root; print findings; return 0, 1 or 2."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "root",
        nargs="?",
        default=str(Path(__file__).resolve().parents[2]),
        help="repository root (default: this checkout)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if not (root / "pyproject.toml").is_file():
        print(f"error: no pyproject.toml in {root}", file=sys.stderr)  # noqa: T201
        return 2
    try:
        found = findings(root)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    if found:
        for line in report(found):
            print(line)  # noqa: T201
        print(f"{len(found)} refused requirement(s).")  # noqa: T201
        return 1
    pyprojects, texts = requirement_files(root)
    print(  # noqa: T201
        f"ok: no refused requirement in {len(pyprojects)} pyproject file(s)"
        f" and {len(texts)} requirement file(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
