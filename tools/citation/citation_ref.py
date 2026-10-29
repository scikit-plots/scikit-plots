# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Generate the root ``CITATION.bib`` and ``CITATION.cff`` of scikit-plots.

Both files are rendered from the templates next to this script and from
facts read from the checkout, so a citation names the version that was
actually used:

* **latest** -- the Zenodo *concept* DOI, which stands for every version and
  resolves to the newest one (the ``@software`` entry kept first, with its
  historic key, so existing ``\cite`` keys keep working);
* **this version** -- the public version from ``scikitplot/__init__.py``
  (``0.5.dev0``, ``0.4.0.post11``), its kind (development, pre-release,
  post-release or release), the build version with the commit it was generated from
  (``0.5.dev0+git.20261028.b656b70``, the same form as
  ``scikitplot/_build_utils/gitversion.py`` writes into ``version.py``), and a
  URL to that exact source tree; a release's own Zenodo DOI when it is known.

Usage::

    python tools/citation/citation_ref.py render            # print, write nothing
    python tools/citation/citation_ref.py check             # exit 1 when stale
    python tools/citation/citation_ref.py sync              # preview as a diff
    python tools/citation/citation_ref.py sync --apply      # write both files
    python tools/citation/citation_ref.py sync --apply --commit <sha> --date 2026-10-28

Notes
-----
**User notes.** A committed file cannot name the commit that contains it, so
the files record the commit they were *generated from* (``git HEAD`` at
``sync`` time, or ``--commit``/``--date``), and say so. Regenerate when the
version changes and when a release is prepared; ``check`` fails when
``__version__`` no longer matches the files, or when they were edited by
hand. It does not fail merely because newer commits exist.

**Developer notes.** Read-only by default; ``--apply`` writes, atomically,
only the two root files. ``check`` re-renders from the provenance recorded
in ``CITATION.bib`` (version, commit, date), so its result does not depend
on the current ``HEAD``. Git is queried read-only (``git log -1``); git
configuration is never changed. The version kind follows the PEP 440
grammar (canonical regular expression of its appendix B); nothing is
inferred from substrings. The templates need Jinja2.
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, TextIO

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUTS = {"bib": "CITATION.bib", "cff": "CITATION.cff"}
TEMPLATES = {"bib": "CITATION.bib.template", "cff": "CITATION.cff.template"}

#: Where the public version is declared (the single source, Rule: no copies).
VERSION_FILE = Path("scikitplot") / "__init__.py"
_VERSION_LINE = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.MULTILINE)

#: PEP 440 appendix B, canonical public version grammar (no local segment).
_PEP440 = re.compile(
    r"""^v?(?:(?:(?P<epoch>[0-9]+)!)?(?P<release>[0-9]+(?:\.[0-9]+)*)
    (?P<pre>[-_\.]?(?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)[-_\.]?(?P<pre_n>[0-9]+)?)?
    (?P<post>(?:-(?P<post_n1>[0-9]+))|(?:[-_\.]?(?P<post_l>post|rev|r)[-_\.]?(?P<post_n2>[0-9]+)?))?
    (?P<dev>[-_\.]?(?P<dev_l>dev)[-_\.]?(?P<dev_n>[0-9]+)?)?)$""",
    re.VERBOSE | re.IGNORECASE,
)
#: A commit hash: full (40), or abbreviated (7 or more) as ``scikitplot -V`` shows.
_SHA = re.compile(r"^[0-9a-f]{7,40}$")
_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_PROVENANCE = re.compile(
    r"^% provenance: version=(?P<version>\S+) commit=(?P<commit>\S+) date=(?P<date>\S+)$",
    re.MULTILINE,
)

#: Zenodo DOIs of released versions, as listed in docs/source/project/citing.rst.
#: Add a version here when Zenodo archives it; a version not listed gets none.
VERSION_DOIS = {
    "0.4.0": "10.5281/zenodo.13367095",
    "0.3.8dev0": "10.5281/zenodo.13367001",
}

#: Facts that do not change from version to version.
PROJECT = {
    "title": "scikit-plots: Machine Learning Visualization in Python",
    "short_title": "scikit-plots",
    "author": "The scikit-plots developers",
    "license": "BSD-3-Clause",
    "concept_doi": "10.5281/zenodo.13367000",
    "repository": "https://github.com/scikit-plots/scikit-plots",
    "docs_dev": "https://scikit-plots.github.io/dev",
    "docs_stable": "https://scikit-plots.github.io/stable",
    "keywords": "Software, Python, scikit-plots, Machine Learning Visualization",
    # The first entry's historic key and date, kept so existing \cite keys work.
    "latest_key": "The-scikit-plots-developers_2024",
    "latest_year": "2024",
    "latest_month": "11",
    "message": (
        "If you use scikit-plots in a scientific publication,\n"
        "we respectfully request that you cite the reference provided below.\n"
        "A dedicated publication describing scikit-plots is currently in preparation;\n"
        "until it becomes available, the following citation is recommended:"
    ),
    "preferred_citation": {
        "type": "article",
        "title": "Scikit-plots: Machine Learning Visualization in Python",
        "family_names": "Çelik",
        "given_names": "Muhammed",
        "orcid": "https://orcid.org/0009-0001-2685-1263",
        "doi": "10.5281/zenodo.13367000",
        "url": "https://doi.org/10.5281/zenodo.13367000",
    },
}


class CitationError(RuntimeError):
    """A citation cannot be rendered or checked; the message says why."""


@dataclass(frozen=True)
class Provenance:
    """
    The version a citation names and the source it was generated from.

    Parameters
    ----------
    version : str
        Public version (PEP 440, no local segment), e.g. ``"0.5.dev0"``.
    commit : str or None
        Commit hash, full (what git gives) or abbreviated to 7 or more
        characters (what ``scikitplot -V`` shows); None without git.
    date : str or None
        Commit date, ``YYYY-MM-DD``; None with ``commit``.
    """

    version: str
    commit: str | None = None
    date: str | None = None

    def __post_init__(self):  # noqa: D105 - validation, documented in the class
        if not _PEP440.match(self.version):
            raise CitationError(
                f"version {self.version!r} is not a PEP 440 public version"
            )
        if (self.commit is None) != (self.date is None):
            raise CitationError("give both a commit and its date, or neither")
        if self.commit is not None and not _SHA.match(self.commit):
            raise CitationError(
                f"commit {self.commit!r} is not a hexadecimal hash of 7 to 40 characters"
            )
        if self.date is not None and not _DATE.match(self.date):
            raise CitationError(f"date {self.date!r} is not YYYY-MM-DD")

    @property
    def kind(self) -> str:
        """``"development"``, ``"pre-release"``, ``"post-release"`` or ``"release"`` (str)."""
        match = _PEP440.match(self.version)
        if match.group("dev") is not None:
            return "development"
        if match.group("pre") is not None:
            return "pre-release"
        if match.group("post") is not None:
            return "post-release"
        return "release"

    @property
    def build(self) -> str:
        """The build version, as ``version.py`` writes it (str)."""
        return _build_version(self.version, self.commit, self.date)


def _build_version(version, commit, date):
    """Return ``version`` with the git metadata of ``gitversion.py``, reusing it."""
    if commit is None:
        return version
    gitversion = _load_gitversion()
    return gitversion.GitVersionInfo(version, commit, f"{date}T00:00:00Z").full_version


def _load_gitversion():
    path = REPO / "scikitplot" / "_build_utils" / "gitversion.py"
    spec = importlib.util.spec_from_file_location("_skplt_gitversion", path)
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise CitationError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_version(repo: Path) -> str:
    """
    Return ``__version__`` as declared in ``scikitplot/__init__.py``.

    Parameters
    ----------
    repo : Path
        Repository root.

    Returns
    -------
    str
        The declared public version.

    Raises
    ------
    CitationError
        When the file or the line is missing.
    """
    path = repo / VERSION_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CitationError(f"cannot read {path}: {exc}") from exc
    match = _VERSION_LINE.search(text)
    if match is None:
        raise CitationError(f"no __version__ line in {path}")
    return match.group(1)


def read_git_head(repo: Path) -> tuple[str, str]:
    """
    Return the full hash and author date (``YYYY-MM-DD``) of ``HEAD``, read-only.

    Parameters
    ----------
    repo : Path
        Repository root.

    Returns
    -------
    tuple of str
        ``(commit, date)``, the same commit and date ``gitversion.py`` uses.

    Raises
    ------
    CitationError
        When git or the repository is unavailable; pass ``--commit`` and
        ``--date``, or ``--no-git``.
    """
    git = shutil.which("git")
    if git is None:
        raise CitationError(
            "git is not installed; pass --commit and --date, or --no-git"
        )
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, git by its full path
            [git, "-C", str(repo), "log", "-1", "--format=%H %aI"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=30,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CitationError(f"cannot run git: {exc}; pass --commit and --date") from exc
    parts = done.stdout.split()
    if done.returncode != 0 or len(parts) != 2:  # noqa: PLR2004 - "hash date"
        reason = done.stderr.strip() or "no commit"
        raise CitationError(f"git log failed: {reason}; pass --commit and --date")
    return parts[0], parts[1][:10]


def context(provenance: Provenance) -> dict:
    """
    Return the template variables for ``provenance``.

    Parameters
    ----------
    provenance : Provenance
        Version and source.

    Returns
    -------
    dict
        Every value the two templates use; deterministic for its input.
    """
    p = provenance
    if p.commit is not None:
        source_url = f"{PROJECT['repository']}/tree/{p.commit}"
        year, month = p.date[:4], p.date[5:7]
        released = p.date
    else:  # no commit, so no date can be claimed for this version
        source_url = PROJECT["repository"]
        year = month = released = None
    docs = PROJECT["docs_dev"] if p.kind == "development" else PROJECT["docs_stable"]
    if p.commit is not None:
        build_note = (
            f"build {p.build}, generated from commit {p.commit[:7]} of {p.date}"
        )
    else:
        build_note = "generated without git: no commit recorded"
    return {
        **PROJECT,
        "version": p.version,
        "kind": p.kind,
        "build": p.build,
        "commit": p.commit,
        "date": p.date,
        "version_key": f"scikit-plots_v{p.version}",
        "version_year": year,
        "version_month": month,
        "date_released": released,
        "source_url": source_url,
        "version_doi": VERSION_DOIS.get(p.version),
        "docs": docs,
        "build_note": build_note,
        "provenance": (
            f"version={p.version} commit={p.commit or '-'} date={p.date or '-'}"
        ),
    }


def render(provenance: Provenance) -> dict:
    """
    Render both files.

    Parameters
    ----------
    provenance : Provenance
        Version and source.

    Returns
    -------
    dict
        ``{"bib": text, "cff": text}``.

    Raises
    ------
    CitationError
        When Jinja2 is not installed.
    """
    try:
        import jinja2  # noqa: PLC0415 - only this command needs it
    except ImportError as exc:
        raise CitationError("rendering needs jinja2: pip install jinja2") from exc
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(HERE)),
        undefined=jinja2.StrictUndefined,  # a missing value is an error
        keep_trailing_newline=True,
        autoescape=False,  # noqa: S701 - BibTeX and YAML, not HTML
    )
    values = context(provenance)
    return {key: env.get_template(TEMPLATES[key]).render(values) for key in OUTPUTS}


def recorded_provenance(repo: Path) -> Provenance:
    """
    Return the provenance written in the root ``CITATION.bib``.

    Parameters
    ----------
    repo : Path
        Repository root.

    Returns
    -------
    Provenance
        What ``sync`` recorded.

    Raises
    ------
    CitationError
        When the file is missing or carries no provenance line.
    """
    path = repo / OUTPUTS["bib"]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CitationError(f"cannot read {path}: {exc}; run sync --apply") from exc
    match = _PROVENANCE.search(text)
    if match is None:
        raise CitationError(f"{path} has no provenance line; run sync --apply")
    commit = None if match.group("commit") == "-" else match.group("commit")
    date = None if match.group("date") == "-" else match.group("date")
    return Provenance(match.group("version"), commit, date)


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _provenance_from_args(args, repo: Path) -> Provenance:
    version = read_version(repo)
    if args.no_git:
        if args.commit or args.date:
            raise CitationError("--no-git cannot be combined with --commit/--date")
        return Provenance(version)
    if args.commit or args.date:
        return Provenance(version, args.commit, args.date)
    commit, date = read_git_head(repo)
    return Provenance(version, commit, date)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render, check or write CITATION.bib and CITATION.cff",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO, help="repository root")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (
        ("render", "print both files for the current version (writes nothing)"),
        ("sync", "show the update as a diff; --apply writes it"),
    ):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument(
            "--commit",
            help="commit hash to record instead of HEAD (7 to 40 hex digits)",
        )
        cmd.add_argument("--date", help="that commit's date, YYYY-MM-DD")
        cmd.add_argument("--no-git", action="store_true", help="record no commit")
        if name == "sync":
            cmd.add_argument("--apply", action="store_true", help="write the files")
    check = sub.add_parser("check", help="exit 1 when the files are stale or edited")
    check.add_argument("--json", action="store_true", help="machine-readable result")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """
    Run the command line; return its exit status.

    Parameters
    ----------
    argv : sequence of str or None, default=None
        Arguments; ``sys.argv[1:]`` when None.
    stdout, stderr : text stream or None, default=None
        Where output goes; the process streams when None.

    Returns
    -------
    int
        0 on success, 1 when ``check`` finds stale files, 2 on an error.
    """
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    args = _parser().parse_args(argv)
    repo = args.repo_root.resolve()
    try:
        if args.command == "check":
            return _check(repo, args, out)
        provenance = _provenance_from_args(args, repo)
        rendered = render(provenance)
        if args.command == "render":
            for key, name in OUTPUTS.items():
                print(f"# ---- {name} ----", file=out)
                out.write(rendered[key])
            return 0
        changed = []
        for key, name in OUTPUTS.items():
            path = repo / name
            current = path.read_text(encoding="utf-8") if path.is_file() else ""
            if current == rendered[key]:
                continue
            changed.append(name)
            if args.apply:
                _atomic_write(path, rendered[key])
            else:
                out.writelines(
                    difflib.unified_diff(
                        current.splitlines(keepends=True),
                        rendered[key].splitlines(keepends=True),
                        fromfile=name,
                        tofile=f"{name} (rendered)",
                    )
                )
        verb = "wrote" if args.apply else "would change"
        print(
            f"{verb}: {', '.join(changed) or 'nothing'} "
            f"(version {provenance.version}, {provenance.kind}, build {provenance.build})",
            file=out,
        )
        return 0
    except CitationError as exc:
        print(f"error: {exc}", file=err)
        return 2


def _check(repo: Path, args, out: TextIO) -> int:
    declared = read_version(repo)
    try:
        recorded = recorded_provenance(repo)
    except CitationError as exc:  # a missing or hand-written file is stale
        if args.json:
            print(json.dumps({"ok": False, "problems": [str(exc)]}, indent=2), file=out)
        else:
            print(f"STALE {exc}", file=out)
            print("citation: stale", file=out)
        return 1
    problems = []
    if recorded.version != declared:
        problems.append(
            f"CITATION files name version {recorded.version}, scikitplot/__init__.py "
            f"declares {declared}: run sync --apply"
        )
    rendered = render(recorded)
    for key, name in OUTPUTS.items():
        path = repo / name
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current is None:
            problems.append(f"{name} is missing: run sync --apply")
        elif current != rendered[key]:
            problems.append(
                f"{name} differs from its template rendering: run sync --apply"
            )
    if args.json:
        print(
            json.dumps(
                {"ok": not problems, "problems": problems, **asdict(recorded)}, indent=2
            ),
            file=out,
        )
    else:
        for line in problems:
            print(f"STALE {line}", file=out)
        print(
            f"citation: {'ok' if not problems else 'stale'} "
            f"(version {recorded.version}, commit {(recorded.commit or '-')[:7]})",
            file=out,
        )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
