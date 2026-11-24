#!/usr/bin/env python3
"""
Write ``upcoming_changes/`` notes from a JSON catalog whose claims are probed first.

Why this exists
---------------
A ledger note is only as good as its evidence. When many notes are drafted at
once (after a sweep, a benchmark or a capability review), each claim of the
form "the package has no X" or "X exists to build on" must be checked on the
checkout before it is written, and checked again later: once someone adds X,
the note is stale. This tool keeps the notes as data (one JSON catalog per
review), attaches machine-checkable probes to each, renders them in the shape
of ``upcoming_changes/TEMPLATE.md``, and re-runs the probes on demand.

Usage
-----
::

    python -B tools/maint_tools/ledger_notes.py example > /tmp/review.json
    python -B tools/maint_tools/ledger_notes.py snapshot CATALOG --python VENV/bin/python --out /tmp/snap
    python -B tools/maint_tools/ledger_notes.py check CATALOG --snapshot /tmp/snap
    python -B tools/maint_tools/ledger_notes.py render CATALOG --snapshot /tmp/snap          # plan only
    python -B tools/maint_tools/ledger_notes.py render CATALOG --snapshot /tmp/snap --apply  # write

``check`` runs every probe and exits 1 when one no longer holds (a gap was
closed, code moved): those notes need updating or closing. ``render`` refuses
to write while any probe fails; it creates missing notes, leaves identical
ones alone and reports differing ones, which it overwrites only with
``--force``. ``snapshot`` records, from an interpreter that has the package
installed, what the ``ext-*`` and ``api-*`` probes need. ``--json OUT``
writes the report as sorted JSON.

Catalog format
--------------
``{"schema": 1, "discovered": {"@name": "text"}, "notes": [NOTE, ...]}``.
A NOTE holds ``path`` (relative to ``upcoming_changes/``), ``title``,
``kind``, ``area``, ``summary``, ``why``, ``evidence`` (list), ``cause``,
``expected``, ``affected`` (list), ``constraints``, ``edges`` (list),
``direction``, ``accept`` (list), and optionally ``docs`` (default
``"none"``), ``release`` (``required``, ``not-required``, ``unknown``),
``ttype`` (Towncrier type), ``section`` (overrides the derived Towncrier
section), ``discovered`` (text, or an ``@name`` key of the top-level map;
``{ref}`` and ``{date}`` are replaced) and ``probes`` (list). Each probe's
evidence sentence is appended to ``evidence``. Probe kinds (``PROBES``):

* ``text``: ``{"text": ...}``, a fact checked by hand; always holds.
* ``absent`` / ``present``: ``{"pattern": REGEX, "path": "scikitplot",
  "public": false}``; a case-insensitive search of ``*.py`` files, tests
  and vendored code left out; ``public`` also leaves out the documentation
  extensions and build helpers.
* ``ext-unregistered``: ``{"ext": ".dcm"}``; no corpus reader for it.
* ``api-absent`` / ``api-present``: ``{"module": "corpus", "pattern": REGEX}``
  against ``python -m scikitplot api scikitplot.<module>``.

Notes
-----
**User notes.** A catalog is the reusable part of a review: copy the example,
write one entry per finding, attach the probe that would fail if the claim
were wrong, and let ``render`` refuse when one does. Keep catalogs in
``tools/maint_tools/catalogs/`` so ``check`` can tell later which notes have
gone stale. Name no private file, person or outside source in a catalog.

**Developer notes.** Standard library only; the package is never imported
here (lesson CI-12): ``snapshot`` asks a child interpreter. Python 3.8 and
newer (Rule 137). Probes search in a fixed, sorted file order, so two runs on
the same tree give the same notes. Paths are validated to stay inside
``upcoming_changes/``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 1
REQUIRED = (
    "path", "title", "kind", "area", "summary", "why", "evidence", "cause",
    "expected", "affected", "constraints", "edges", "direction", "accept",
)
LISTS = ("evidence", "affected", "edges", "accept")
OPTIONAL = ("docs", "release", "ttype", "section", "discovered", "probes")
RELEASES = ("required", "not-required", "unknown")
VENDORED = ("scikitplot/externals/", "scikitplot/cexternals/")
PRIVATE = ("scikitplot/_externals/", "scikitplot/_build_utils/")

Note = Dict[str, Any]
Probe = Dict[str, Any]


class CatalogError(ValueError):
    """A catalog, note or probe that cannot be used as written."""


# ------------------------------------------------------------------ probes
class Context:
    """
    What probes read: the checkout and an optional snapshot.

    Parameters
    ----------
    root : Path
        Repository checkout to search.
    snapshot : Path or None
        Directory written by ``snapshot``: ``registry.json`` and ``api/*.txt``.
    """

    def __init__(self, root: Path, snapshot: Optional[Path] = None) -> None:
        self.root = root
        self.snapshot = snapshot
        self._files: Dict[str, List[Tuple[str, str]]] = {}

    def files(self, path: str) -> List[Tuple[str, str]]:
        """Return sorted ``(relative path, text)`` of first-party ``*.py`` under *path*."""
        if path not in self._files:
            base = (self.root / path).resolve()
            if self.root.resolve() not in (base, *base.parents):
                raise CatalogError(f"probe path leaves the checkout: {path!r}")
            out = []
            for f in sorted(base.rglob("*.py")):
                rel = f.relative_to(self.root).as_posix()
                if "/tests/" in rel or f.name.startswith("test_") or rel.startswith(VENDORED):
                    continue
                out.append((rel, f.read_text(encoding="utf-8", errors="replace")))
            self._files[path] = out
        return self._files[path]

    def grep(self, pattern: str, path: str, public: bool) -> List[str]:
        """Return files under *path* matching *pattern* case-insensitively."""
        rx = _regex(pattern)
        return [r for r, text in self.files(path) if rx.search(text) and not (public and r.startswith(PRIVATE))]

    def registry(self) -> List[str]:
        """Return the reader extensions recorded by ``snapshot``."""
        return sorted(json.loads(self._need("registry.json").read_text(encoding="utf-8")))

    def api(self, module: str) -> str:
        """Return the recorded ``api`` listing of ``scikitplot.<module>``."""
        return self._need(f"api/{module}.txt").read_text(encoding="utf-8")

    def _need(self, rel: str) -> Path:
        if self.snapshot is None:
            raise CatalogError(f"this probe needs --snapshot (missing {rel}); run the snapshot command")
        p = self.snapshot / rel
        if not p.is_file():
            raise CatalogError(f"snapshot has no {rel}; rerun the snapshot command for this catalog")
        return p


def _regex(pattern: str) -> "re.Pattern[str]":
    try:
        return re.compile(pattern, re.I)
    except re.error as exc:
        raise CatalogError(f"bad probe pattern {pattern!r}: {exc}") from None


def _ticks(items: Sequence[str], limit: int) -> str:
    return ", ".join(f"`{i}`" for i in items[:limit])


def _p_text(p: Probe, ctx: Context) -> Tuple[bool, str]:
    return True, str(p["text"])


def _p_absent(p: Probe, ctx: Context) -> Tuple[bool, str]:
    path, public = p.get("path", "scikitplot"), bool(p.get("public", False))
    hits = ctx.grep(p["pattern"], path, public)
    if public:
        return not hits, (
            f"`grep -rEil '{p['pattern']}' {path} --include=*.py` finds nothing in the public submodules "
            "(the documentation extensions under `_externals/` and build helpers left out)."
        )
    return not hits, f"`grep -rEil '{p['pattern']}' {path} --include=*.py` (tests and vendored code excluded) finds nothing."


def _p_present(p: Probe, ctx: Context) -> Tuple[bool, str]:
    hits = ctx.grep(p["pattern"], p.get("path", "scikitplot"), bool(p.get("public", False)))
    return bool(hits), f"Related code exists: {_ticks(hits, 4)}."


def _p_ext(p: Probe, ctx: Context) -> Tuple[bool, str]:
    reg, ext = ctx.registry(), p["ext"]
    return ext not in reg, (
        f"`DocumentReader.create('file{ext}')` raises `ValueError: No DocumentReader registered for extension "
        f"'{ext}'`: the reader registry on main ({len(reg)} entries) has no `{ext}`."
    )


def _api_names(listing: str, pattern: str) -> List[str]:
    rx = _regex(pattern)
    return [
        line.split()[1].split("(")[0]
        for line in listing.splitlines()
        if line and not line.startswith(" ") and len(line.split()) > 1 and rx.search(line)
    ]


def _p_api_absent(p: Probe, ctx: Context) -> Tuple[bool, str]:
    names = _api_names(ctx.api(p["module"]), p["pattern"])
    return not names, f"`python -m scikitplot api scikitplot.{p['module']}` lists no public name matching `{p['pattern']}`."


def _p_api_present(p: Probe, ctx: Context) -> Tuple[bool, str]:
    names = _api_names(ctx.api(p["module"]), p["pattern"])
    return bool(names), f"Already public in `scikitplot.{p['module']}`: {_ticks(names, 6)} (the base to build on)."


PROBES: Dict[str, Tuple[Tuple[str, ...], Callable[[Probe, Context], Tuple[bool, str]]]] = {
    "text": (("text",), _p_text),
    "absent": (("pattern",), _p_absent),
    "present": (("pattern",), _p_present),
    "ext-unregistered": (("ext",), _p_ext),
    "api-absent": (("module", "pattern"), _p_api_absent),
    "api-present": (("module", "pattern"), _p_api_present),
}


def run_probe(p: Probe, ctx: Context) -> Tuple[bool, str]:
    """
    Run one probe.

    Parameters
    ----------
    p : dict
        ``{"kind": ..., <fields>}``; see ``PROBES``.
    ctx : Context
        Checkout and snapshot to read.

    Returns
    -------
    tuple of (bool, str)
        Whether the claim holds, and the evidence sentence for the note.

    Raises
    ------
    CatalogError
        Unknown kind, missing field, bad pattern or missing snapshot.
    """
    if not isinstance(p, dict) or p.get("kind") not in PROBES:
        raise CatalogError(f"unknown probe {p!r}; kinds: {', '.join(sorted(PROBES))}")
    fields, fn = PROBES[p["kind"]]
    missing = [f for f in fields if not isinstance(p.get(f), str) or not p[f]]
    if missing:
        raise CatalogError(f"probe {p['kind']!r} needs {', '.join(missing)}")
    return fn(p, ctx)


# ------------------------------------------------------------------ catalog
def _template_choices(root: Path, key: str) -> Tuple[str, ...]:
    text = (root / "upcoming_changes" / "TEMPLATE.md").read_text(encoding="utf-8")
    m = re.search(rf'^{key}: "<([^>]+)>"', text, re.M)
    if not m:
        raise CatalogError(f"upcoming_changes/TEMPLATE.md has no {key} choices")
    return tuple(m.group(1).split("|"))


def load_catalog(path: Path, root: Path) -> List[Note]:
    """
    Read and validate a catalog.

    Parameters
    ----------
    path : Path
        JSON catalog.
    root : Path
        Checkout whose ``upcoming_changes/TEMPLATE.md`` defines kinds and types.

    Returns
    -------
    list of dict
        Notes with defaults filled and ``discovered`` resolved (placeholders kept).

    Raises
    ------
    CatalogError
        On any schema violation, naming the note.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CatalogError(f"cannot read catalog {path}: {exc}") from None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or not isinstance(data.get("notes"), list):
        raise CatalogError(f"{path}: expected {{'schema': {SCHEMA}, 'notes': [...]}}")
    aliases = data.get("discovered", {})
    kinds, ttypes = _template_choices(root, "kind"), _template_choices(root, "towncrier_type")
    seen, notes = set(), []
    for i, raw in enumerate(data["notes"]):
        where = f"{path.name} note {i}"
        if not isinstance(raw, dict):
            raise CatalogError(f"{where}: not an object")
        where = f"{where} ({raw.get('path', '?')})"
        unknown = set(raw) - set(REQUIRED) - set(OPTIONAL)
        missing = [k for k in REQUIRED if k not in raw]
        if unknown or missing:
            raise CatalogError(f"{where}: unknown {sorted(unknown)}, missing {missing}")
        for k in REQUIRED:
            ok = isinstance(raw[k], list) and all(isinstance(x, str) for x in raw[k]) if k in LISTS else isinstance(raw[k], str) and raw[k].strip()
            if not ok:
                raise CatalogError(f"{where}: {k} must be {'a list of strings' if k in LISTS else 'non-empty text'}")
        n = {"docs": "none", "release": "required", "ttype": "enhancement", "section": None, "discovered": "", "probes": []}
        n.update(raw)
        rel = Path(n["path"])
        if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".md" or rel.name in ("README.md", "TEMPLATE.md"):
            raise CatalogError(f"{where}: path must be a relative .md path inside upcoming_changes/")
        if n["path"] in seen:
            raise CatalogError(f"{where}: duplicate path")
        seen.add(n["path"])
        if n["kind"] not in kinds or n["ttype"] not in ttypes or n["release"] not in RELEASES:
            raise CatalogError(f"{where}: kind in {kinds}, ttype in {ttypes}, release in {RELEASES}")
        if n["discovered"].startswith("@"):
            if n["discovered"] not in aliases:
                raise CatalogError(f"{where}: no discovered alias {n['discovered']!r}")
            n["discovered"] = aliases[n["discovered"]]
        notes.append(n)
    return notes


def towncrier_sections(root: Path) -> List[str]:
    """Return the configured Towncrier sections (``generate_towncrier_sections.py list``)."""
    out = subprocess.run(
        [sys.executable, "-B", str(root / "tools" / "maint_tools" / "generate_towncrier_sections.py"), "list"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    return sorted({m.group(1) for m in (re.match(r"\s*-?\s*([a-z_.]+\S*)", ln) for ln in out.splitlines()) if m})


def section_for(n: Note, sections: Sequence[str]) -> str:
    """Return the note's Towncrier section: explicit, else derived from ``area``."""
    if n["release"] != "required":
        return "unknown"
    if n["section"]:
        if n["section"] not in sections:
            raise CatalogError(f"{n['path']}: section {n['section']!r} is not configured")
        return n["section"]
    last = n["area"].rstrip("/").split("/")[-1]
    return next((c for c in (f"scikitplot.{last}", last) if c in sections), "unknown")


def _bullets(items: Sequence[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def render_note(n: Note, evidence: Sequence[str], section: str, ref: str, date: str) -> str:
    """
    Render one note in the shape of ``upcoming_changes/TEMPLATE.md``.

    Parameters
    ----------
    n : dict
        Validated note.
    evidence : sequence of str
        The note's evidence followed by its probe sentences.
    section : str
        Towncrier section (``unknown`` when not required).
    ref, date : str
        Values for ``{ref}`` and ``{date}`` in ``discovered``.

    Returns
    -------
    str
        The Markdown file content.
    """
    required = n["release"] == "required"
    ttype = n["ttype"] if required else "unknown"
    title = n["title"].replace('"', "'")
    discovered = n["discovered"].replace("{ref}", ref).replace("{date}", date)
    promo = (
        f"Required: `{section}` / `{ttype}`, one bullet, when the change lands." if required
        else "Not required (no shipped behaviour changes)." if n["release"] == "not-required"
        else "To decide with the change."
    )
    return f"""---
title: "{title}"
status: open
kind: "{n['kind']}"
area: "{n['area']}"
discovered_during: "{discovered}"
release_note: "{n['release']}"
towncrier_section: "{section}"
towncrier_type: "{ttype}"
towncrier_fragment: ""
---

# {title}

## Summary

{n['summary']}

## Why it matters

{n['why']}

## Current evidence

{_bullets(evidence)}

## Root cause / current understanding

{n['cause']}

## Expected behavior

{n['expected']}

## Affected paths and ownership

{_bullets(n['affected'])}

## Constraints and non-goals

{n['constraints']}

## Edge cases to cover

{_bullets(n['edges'])}

## Proposed direction

{n['direction']}

## Verification / acceptance criteria

{_bullets(n['accept'])}

## Documentation impact

{n['docs']}

## Release-note promotion

{promo}
"""


# ------------------------------------------------------------------ commands
def probe_all(notes: Sequence[Note], ctx: Context) -> Dict[str, List[Dict[str, Any]]]:
    """Run every note's probes; return ``{path: [{probe, holds, evidence}]}``."""
    return {
        n["path"]: [dict(probe=p, holds=ok, evidence=text) for p in n["probes"] for ok, text in [run_probe(p, ctx)]]
        for n in notes
    }


def plan(notes: Sequence[Note], results: Dict[str, List[Dict[str, Any]]], root: Path, ref: str, date: str) -> List[Dict[str, Any]]:
    """Return one ``{path, action, content}`` per note: create, unchanged or differs."""
    sections = towncrier_sections(root)
    out = []
    for n in notes:
        text = render_note(n, list(n["evidence"]) + [r["evidence"] for r in results[n["path"]]], section_for(n, sections), ref, date)
        target = root / "upcoming_changes" / n["path"]
        action = "create" if not target.exists() else "unchanged" if target.read_text(encoding="utf-8") == text else "differs"
        out.append(dict(path=n["path"], action=action, content=text))
    return out


def _git_ref(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def snapshot(notes: Sequence[Note], python: str, out: Path) -> Dict[str, Any]:
    """
    Record what ``ext-*`` and ``api-*`` probes need, using another interpreter.

    Parameters
    ----------
    notes : sequence of dict
        Validated notes; the ``api-*`` modules they name are listed.
    python : str
        Interpreter with scikit-plots installed.
    out : Path
        Directory to write ``registry.json`` and ``api/<module>.txt`` into.

    Returns
    -------
    dict
        ``{"registry": count, "api": {module: status}}``.
    """
    code = "import json,sys,scikitplot.corpus as c; json.dump(c.DocumentReader.supported_types(), sys.stdout)"
    # Run from an empty directory: from the checkout, ``-m scikitplot`` would
    # import the unbuilt source tree instead of the installed package.
    with tempfile.TemporaryDirectory() as cwd:
        return _snapshot(notes, python, out.resolve(), code, cwd)


def _snapshot(notes: Sequence[Note], python: str, out: Path, code: str, cwd: str) -> Dict[str, Any]:
    reg = subprocess.run([python, "-c", code], cwd=cwd, capture_output=True, text=True, timeout=600)
    if reg.returncode:
        raise CatalogError(f"reader registry: {reg.stderr.strip().splitlines()[-1:]}")
    (out / "api").mkdir(parents=True, exist_ok=True)
    (out / "registry.json").write_text(json.dumps(sorted(json.loads(reg.stdout))) + "\n", encoding="utf-8")
    modules = sorted({p["module"] for n in notes for p in n["probes"] if p.get("kind", "").startswith("api-")})
    status = {}
    for m in modules:
        r = subprocess.run([python, "-m", "scikitplot", "api", f"scikitplot.{m}"], cwd=cwd, capture_output=True, text=True, timeout=600)
        status[m] = "ok" if r.returncode == 0 else f"exit {r.returncode}"
        (out / "api" / f"{m}.txt").write_text(r.stdout, encoding="utf-8")
    return dict(registry=len(json.loads(reg.stdout)), api=status)


EXAMPLE = {
    "schema": SCHEMA,
    "discovered": {"@review": "Capability review, gap probed on main {ref} ({date})"},
    "notes": [
        {
            "path": "scikitplot/corpus/example-reader-gap.md",
            "title": "corpus: no reader for EXAMPLE files",
            "kind": "other",
            "area": "scikitplot/corpus",
            "summary": "What is missing or wrong, observed, in a few sentences.",
            "why": "Who is affected and how.",
            "evidence": ["A hand-checked observation, with the command that shows it."],
            "cause": "Current understanding.",
            "expected": "The behaviour to reach.",
            "affected": ["`scikitplot/corpus/_readers/`"],
            "constraints": "Optional extra, imported lazily; no new base dependency.",
            "edges": ["Empty file", "Huge file", "Wrong extension"],
            "direction": "The smallest change that reaches the expected behaviour.",
            "accept": ["A test that fails today and passes after the change"],
            "discovered": "@review",
            "probes": [{"kind": "ext-unregistered", "ext": ".example"}, {"kind": "absent", "pattern": "example_reader"}],
        }
    ],
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ledger_notes.py", description="Probe and render upcoming_changes notes from a JSON catalog.")
    p.add_argument("--root", type=Path, default=ROOT, help="checkout to read and write (default: this one)")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("example", help="print a minimal catalog")
    for name, text in (("check", "run every probe; exit 1 when one fails"), ("render", "plan or write the notes")):
        s = sub.add_parser(name, help=text)
        s.add_argument("catalog", type=Path)
        s.add_argument("--snapshot", type=Path, default=None, metavar="DIR")
        s.add_argument("--json", type=Path, default=None, metavar="OUT")
    r = sub.choices["render"]
    r.add_argument("--apply", action="store_true", help="write missing notes")
    r.add_argument("--force", action="store_true", help="with --apply, also overwrite differing notes")
    r.add_argument("--ref", default=None, help="value for {ref} (default: git short HEAD)")
    r.add_argument("--date", default=None, help="value for {date} (default: today, UTC)")
    s = sub.add_parser("snapshot", help="record the reader registry and api listings the probes need")
    s.add_argument("catalog", type=Path)
    s.add_argument("--python", required=True, help="interpreter with scikit-plots installed")
    s.add_argument("--out", type=Path, required=True, metavar="DIR")
    return p


def _dump(path: Optional[Path], payload: Any) -> None:
    if path is not None:
        path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the command line.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments; ``None`` uses ``sys.argv[1:]``.

    Returns
    -------
    int
        0 on success; 1 when a probe fails or a note differs without
        ``--force``; 2 on an unusable catalog, snapshot or argument.
    """
    args = _parser().parse_args(argv)
    if args.command == "example":
        print(json.dumps(EXAMPLE, indent=1))
        return 0
    if args.command == "render" and args.force and not args.apply:
        print("error: --force needs --apply", file=sys.stderr)
        return 2
    try:
        notes = load_catalog(args.catalog, args.root)
        if args.command == "snapshot":
            print(json.dumps(snapshot(notes, args.python, args.out), sort_keys=True))
            return 0
        results = probe_all(notes, Context(args.root, args.snapshot))
        failed = sorted((path, r["probe"]["kind"], r["evidence"]) for path, rs in results.items() for r in rs if not r["holds"])
        if args.command == "check":
            for path, kind, text in failed:
                print(f"STALE {path} [{kind}]: {text}")
            print(f"{len(notes)} notes, {sum(map(len, results.values()))} probes, {len(failed)} no longer hold")
            _dump(args.json, dict(notes=len(notes), stale=[dict(path=p, kind=k, evidence=t) for p, k, t in failed]))
            return 1 if failed else 0
        if failed:
            for path, kind, text in failed:
                print(f"PROBE FAILED {path} [{kind}]: {text}")
            print("nothing written: fix or drop the notes whose probes fail", file=sys.stderr)
            return 1
        ref = args.ref or _git_ref(args.root)
        date = args.date or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        steps = plan(notes, results, args.root, ref, date)
    except CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    counts: Dict[str, int] = {}
    for s in steps:
        counts[s["action"]] = counts.get(s["action"], 0) + 1
        if s["action"] == "differs":
            print(f"DIFFERS {s['path']}")
        write = args.apply and (s["action"] == "create" or (s["action"] == "differs" and args.force))
        if write:
            target = args.root / "upcoming_changes" / s["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(s["content"], encoding="utf-8")
    print(", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ("" if args.apply else " (plan only; --apply writes)"))
    _dump(args.json, dict(ref=ref, date=date, steps=[dict(path=s["path"], action=s["action"]) for s in steps]))
    return 1 if counts.get("differs") and not (args.apply and args.force) else 0


if __name__ == "__main__":
    raise SystemExit(main())
