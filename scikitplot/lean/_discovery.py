# scikitplot/lean/_discovery.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Discovery over a Lean project: rules, patterns, claims and anomalies, as helper documents.

A formal development is written to settle one problem. The ideas that carry
it are rarely written down as such: the statements everything else leans
on, the shapes that recur from file to file, the hypotheses it states as
definitions, the irregularities that point at something unstated. This
module reads them from what :class:`~scikitplot.lean.LeanIndex` recorded,
by stated rules, and writes them as documents for people and for language
models to reason from:

- **Rules**: theorems named by declarations in the most files, and the
  central concepts (definitions) most leaned on: candidates for the general
  statements a development rests on.
- **Patterns**: theorems whose statements have one shape up to the names of
  bound variables and the numerals, and numbered series of names: each a
  candidate for one general statement of which the members are instances.
- **Claims and assumptions**: what the project states without proving it
  (``Prop``-valued definitions, ``axiom``, ``proof_wanted``) and what it
  leaves unproven (``sorry``, ``admit``).
- **Anomalies**: public names declared in parallel (the same statement twice,
  or one name with two meanings), commands that add declarations no source
  spells, and results proved but named by nothing: questions to ask.
- **Headline results**: the theorems of modules nothing in the project
  imports, each with an evidence dossier.

Every item carries its file and lines. Nothing here is proved or disproved
by being listed: a pattern is a candidate generalisation, a rule is a
candidate principle, an anomaly is a question. Lean's kernel decides proofs;
a person (or a model, checked against the quoted sources) decides what the
candidates mean.
"""  # noqa: D205, D400

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
from array import array
from pathlib import Path
from typing import Any, Iterable, Sequence

from ._index import (
    _DEFINITIONS,
    PRODUCER,
    LeanIndex,
    _listing,
    _positive,
)
from ._shapes import _RESULTS, _shape
from ._syntax import _Splitter

__all__: list[str] = []

#: Identity of the rules that choose the items, recorded on every record.
DISCOVERY = "lean.discovery/1"


_SERIES = re.compile(r"^(.*?[^\W\d])_?(\d+)\Z")

#: Most members listed per pattern, series or parallel group.
_MEMBERS = 20

#: Fewest names that make a numbered series (two could be chance).
_MIN_SERIES = 3

#: Declarations of one name that make it declared in parallel.
_MIN_PARALLEL = 2


class _Discovery:
    """One discovery run over an index: passes, then the report."""

    def __init__(self, index: LeanIndex, *, top: int, min_instances: int) -> None:
        self.index = index
        self.top = top
        self.min_instances = min_instances
        n = len(index._name)
        # "q" is 8 bytes on every platform ("l" is 4 on Windows).
        self.in_degree = array("q", bytes(8 * n))
        self.in_files = array("q", bytes(8 * n))
        self.shapes: dict[bytes, list[int]] = {}
        self.claims: list[int] = []
        self.statements: dict[int, str] = {}

    # -- passes -------------------------------------------------------------

    def read_statements(self) -> None:
        """
        Group results by statement shape, and list the ``Prop`` definitions.

        Both were recorded while the files were indexed (one read per file):
        nothing is read again here.
        """
        index = self.index
        for node, digest in index._shape_of.items():
            self.shapes.setdefault(digest, []).append(node)
        self.claims.extend(sorted(index._claims))

    def count_uses(self) -> None:
        """
        Pass over the edges: how many declarations, and files, name each node.

        Edges link only into files the user can see (the index's
        ``visibility``), so a bound ``s`` is not counted as a use of a
        top-level ``s`` in a file nothing imports.
        """
        index = self.index
        if index._graph != "python":  # numpy or scipy: packed edges
            import numpy as np  # noqa: PLC0415

            from ._graph_numpy import use_counts  # noqa: PLC0415

            indptr, codes = index._pack()
            file_of = np.zeros(0, dtype=np.int64)
            if len(index._file):
                file_of = np.frombuffer(index._file, dtype=index._file.typecode)
            in_degree, in_files = use_counts(
                indptr, codes, file_of.astype(np.int64), len(index._name)
            )
            self.in_degree = array("q", in_degree.tolist())
            self.in_files = array("q", in_files.tolist())
            return
        last_file = array("q", [-1]) * len(index._name)
        for node in range(len(index._name)):
            file_id = index._file[node]
            for code in index._edges_of(node):
                target = code // 4
                self.in_degree[target] += 1
                if last_file[target] != file_id:  # nodes come file by file
                    last_file[target] = file_id
                    self.in_files[target] += 1

    def statement_of(self, nodes: Iterable[int]) -> None:
        """Read the statements of *nodes* (proofs left out), once per file."""
        index = self.index
        wanted: dict[int, set[int]] = {}
        for node in nodes:
            if node not in self.statements:
                wanted.setdefault(index._file[node], set()).add(node)
        for file_id, group in sorted(wanted.items()):
            path = index._paths[file_id]
            if path is None:
                continue
            starts = {index._start[node]: node for node in group}
            try:
                for chunk in _statement_chunks(index, path, file_id):
                    node = starts.get(chunk["line_start"])
                    if node is not None:
                        self.statements[node] = chunk["text"]
            except (OSError, ValueError):
                # Moved, deleted or changed since indexing: the items stay
                # listed (with file and lines) without a quoted statement.
                continue

    # -- the report ---------------------------------------------------------

    def item(self, node: int) -> dict[str, Any]:
        entry = self.index._location(node)
        entry["named_by"] = self.in_degree[node]
        entry["named_from_files"] = self.in_files[node]
        if node in self.statements:
            entry["statement"] = self.statements[node]
        if node in self.index._vacuous:
            entry["vacuous"] = self.index._vacuous[node]
        return entry

    def listing(self, nodes: list[int]) -> dict[str, Any]:
        """
        Return ``{"count", "items", "cut"}`` for *nodes*, items for the top only.

        ``count`` is every node, so a list cut to ``top`` says so; an item
        (location, uses, statement) is built only for a node that is listed.
        """
        listed = [self.item(n) for n in nodes[: self.top]]
        return {"count": len(nodes), "items": listed, "cut": len(nodes) > self.top}

    def ranked(self, kinds: frozenset[str]) -> list[int]:
        """Return the declarations of *kinds* that are named, most named first."""
        index = self.index
        nodes = [
            n
            for n in range(len(index._name))
            if index._kind[n] in kinds
            and index._name[n] is not None
            and self.in_degree[n]
        ]
        nodes.sort(
            key=lambda n: (-self.in_files[n], -self.in_degree[n], index._name[n])
        )
        return nodes

    def patterns(self) -> tuple[list[dict[str, Any]], int]:
        index = self.index
        families = [
            members
            for members in self.shapes.values()
            if len(members) >= self.min_instances
        ]
        families.sort(
            key=lambda m: (
                -len(m),
                -len({index._file[n] for n in m}),
                index._labels[index._file[m[0]]],
                index._start[m[0]],
            )
        )
        total = len(families)
        listed = families[: self.top]
        self.statement_of(n for m in listed for n in m[:_MEMBERS])
        out = []
        for members in listed:
            exemplar = members[0]
            shape = None
            if exemplar in self.statements:
                written = (index._name[exemplar] or "").rsplit(".", 1)[-1] or None
                shape = _shape(self.statements[exemplar], written)
            out.append(
                {
                    "instances": len(members),
                    "files": len({index._file[n] for n in members}),
                    "shape": shape,
                    "exemplar": self.statements.get(exemplar),
                    "members": _listing(
                        [index._location(n) for n in members], _MEMBERS
                    ),
                }
            )
        return out, total

    def series(self) -> tuple[list[dict[str, Any]], int]:
        index = self.index
        groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for node, name in enumerate(index._name):
            if name is None or index._kind[node] not in _RESULTS | _DEFINITIONS:
                continue
            head, _, last = name.rpartition(".")
            found = _SERIES.match(last)
            if found is not None:
                groups.setdefault((head, found.group(1)), []).append(
                    (int(found.group(2)), node)
                )
        families = [sorted(g) for g in groups.values() if len(g) >= _MIN_SERIES]
        families.sort(key=lambda g: (-len(g), index._name[g[0][1]]))
        out = []
        for members in families[: self.top]:
            head, _, last = index._name[members[0][1]].rpartition(".")
            stem = _SERIES.match(last).group(1)
            out.append(
                {
                    "series": f"{head}.{stem}" if head else stem,
                    "members": _listing(
                        [index._location(n) for _, n in members], _MEMBERS
                    ),
                    "numbers": [number for number, _ in members[:_MEMBERS]],
                }
            )
        return out, len(families)

    def parallel(self) -> tuple[list[dict[str, Any]], int]:
        """Public names declared more than once: the same statement, or two meanings."""
        index = self.index
        shape_of: dict[int, bytes] = {
            node: key for key, members in self.shapes.items() for node in members
        }
        groups = []
        for name, held in index._clashes():
            public = [n for n in held if n not in index._private]
            if len(public) < _MIN_PARALLEL:
                continue
            keys = {shape_of.get(n) for n in public}
            same = None not in keys and len(keys) == 1
            groups.append((name, public, same))
        groups.sort(key=lambda g: (-len(g[1]), g[0]))
        out = []
        for name, public, same in groups[: self.top]:
            files = sorted({index._file[n] for n in public})
            # -1 means the imports are unknown: not evidence that one meets another.
            meets = any(
                index._sees(f) != -1 and (index._sees(f) >> other) & 1
                for f in files
                for other in files
                if other != f
            )
            out.append(
                {
                    "declaration": name,
                    "same_statement": same,
                    "one_imports_another": meets,
                    "at": _listing([index._location(n) for n in public], _MEMBERS),
                }
            )
        return out, len(groups)

    def headlines(self) -> list[int]:
        """Results of the modules no project file imports."""
        index = self.index
        modules = index._module_files()
        imported = {
            modules[module]
            for imports in index._imports
            for module in imports
            if module in modules
        }
        return [
            n
            for n in range(len(index._name))
            if index._kind[n] in _RESULTS
            and index._name[n] is not None
            and index._file[n] not in imported
            and not self.in_degree[n]
        ]

    def report(self) -> dict[str, Any]:  # noqa: PLR0914
        index = self.index
        rules = self.ranked(_RESULTS)
        concepts = self.ranked(_DEFINITIONS)
        headline = self.headlines()
        unproven = [
            n for n in sorted(index._flags) if {"sorry", "admit"} & set(index._flags[n])
        ]
        axioms = [n for n in range(len(index._name)) if index._kind[n] == "axiom"]
        wanted = [
            n for n in range(len(index._name)) if index._kind[n] == "proof_wanted"
        ]
        claims = sorted(
            self.claims, key=lambda n: (-self.in_degree[n], index._name[n] or "")
        )
        # Most named first: a vacuous predicate many results use matters most.
        vacuous = sorted(index._vacuous, key=lambda n: -self.in_degree[n])
        headline_set = set(headline)
        isolated = [
            n
            for n in range(len(index._name))
            if index._kind[n] in _RESULTS
            and not self.in_degree[n]
            and n not in headline_set
        ]
        isolated.sort(
            key=lambda n: (-(index._end[n] - index._start[n]), index._name[n] or "")
        )
        listed = (
            rules[: self.top]
            + concepts[: self.top]
            + headline[: self.top]
            + claims[: self.top]
            + vacuous[: self.top]
            + axioms[: self.top]
            + wanted[: self.top]
            + unproven[: self.top]
        )
        self.statement_of(listed)
        patterns, pattern_count = self.patterns()
        series, series_count = self.series()
        parallel, parallel_count = self.parallel()
        return {
            "report": "discovery",
            "producer": DISCOVERY,
            "index": PRODUCER,
            "rules": {
                "load_bearing_theorems": self.listing(rules),
                "central_concepts": self.listing(concepts),
            },
            "patterns": {
                "statement_shapes": {
                    "count": pattern_count,
                    "items": patterns,
                    "cut": pattern_count > len(patterns),
                },
                "numbered_series": {
                    "count": series_count,
                    "items": series,
                    "cut": series_count > len(series),
                },
            },
            "claims": {
                "claim_definitions": self.listing(claims),
                "axioms": self.listing(axioms),
                "proof_wanted": self.listing(wanted),
                "unproven": self.listing(unproven),
                "vacuous_predicates": self.listing(vacuous),
            },
            "anomalies": {
                "parallel_declarations": {
                    "count": parallel_count,
                    "items": parallel,
                    "cut": parallel_count > len(parallel),
                },
                "generators": _listing(index._generators(), self.top),
                "isolated_results": self.listing(isolated),
            },
            "headline_results": self.listing(headline),
        }


def _statement_chunks(
    index: LeanIndex, path: Path, file_id: int
) -> Iterable[dict[str, Any]]:
    """Yield the file's chunks with proofs left out, as the index reads the file."""
    splitter = _Splitter(
        file_name=index._labels[file_id],
        meta={},
        step=None,
        cap=None,
        proofs="drop",
        dialect=index._dialect,
        kinds=None,
        watch=(),
        commands=index._commands,
        max_scope_depth=index._max_depth,
        flag_tokens={},
    )
    with open(path, encoding="utf-8", errors="replace", newline=None) as stream:
        for number, read in enumerate(stream, start=1):
            line = read[:-1] if read.endswith("\n") else read
            if number == 1 and line.startswith("﻿"):
                line = line[1:]
            yield from splitter.feed(number, line)
    yield from splitter.finish()


def discover(
    index: LeanIndex, *, top: int = 50, min_instances: int = 2, patterns: bool = True
) -> dict[str, Any]:
    """Run discovery; see :meth:`LeanIndex.discover`."""
    _positive(top, "top")
    _positive(min_instances, "min_instances")
    run = _Discovery(index, top=top, min_instances=min_instances)
    if patterns:
        run.read_statements()
    run.count_uses()
    return run.report()


# ---------------------------------------------------------------------------
# Helper documents
# ---------------------------------------------------------------------------

_README = """# Helper documents for a Lean project

Generated from the source by `scikitplot.lean.LeanIndex` ({producer}).
Every item names a file and its lines: check each one against the source.

## What is here

| file | for |
|---|---|
| `summary.md`, `summary.json` | the project: sizes, audit flags, imports, generators |
| `rules.md` | theorems named from the most files, and the central definitions |
| `patterns.md` | theorems of one statement shape, and numbered series of names |
| `claims.md` | what is stated without proof: claim definitions, axioms, `proof_wanted`, `sorry` |
| `anomalies.md` | names declared in parallel, code that adds declarations, results named by nothing |
| `dossiers/` | the evidence on each headline result (and each target asked for) |
| `discovery.json` | all of the above as data |
| `records.jsonl` | one record per item, for a corpus or an MCP server |

## How the items are chosen

- **Rules**: by the number of files whose declarations name them, then the
  number of declarations. A statement many separate arguments use behaves
  like a general rule of the development.
- **Patterns**: two theorems have one *shape* when their statements are
  the same tokens once bound variables are renamed in order and numerals
  are replaced; a numbered series is a run of names that differ only by a
  trailing number. Several instances of one shape are candidates for one
  general statement with the varying parts as parameters.
- **Claims**: definitions of type `Prop` (a claim written down to be
  proved or assumed), `axiom`, `proof_wanted`, and results flagged `sorry`
  or `admit`.
- **Anomalies**: one public name declared in several files (the same
  statement twice is repeated work; two statements under one name are two
  meanings); commands whose code adds declarations at build time (their
  source is nowhere); results no declaration names (final, or unused).
- **Headline results**: theorems of the modules no project file imports.

## What a listing means, and does not

A listing is a candidate, chosen by the rule above. A pattern is not
proved general by having instances; a rule is not proved central by being
named often; an anomaly is a question, not a fault. Whether a proof holds
is for Lean's kernel: `lake build`, `#print axioms`, an independent replay.

## Feeding these files to a corpus or a model

```python
from scikitplot.corpus import DocumentReader, to_mcp_resources

reader = DocumentReader.create(
    "{folder}/records.jsonl",
    text_fields=["title", "text"],
    metadata_fields=["section", "declaration", "file", "line_start", "line_end"],
)
documents = list(reader.get_documents())
resources = to_mcp_resources(documents)  # for an MCP server
```

A model reading them should cite `file:line_start` for each claim it
makes, and say which kernel check would settle it.
"""


def _where(item: dict[str, Any]) -> str:
    return f"`{item['file']}:{item['line_start']}`"


def _statement_block(text: str | None) -> list[str]:
    return ["", "```lean", text, "```", ""] if text else []


def _rules_md(report: dict[str, Any]) -> str:
    out = ["# Rules: what the development leans on", ""]
    for title, key in (
        ("Load-bearing theorems", "load_bearing_theorems"),
        ("Central concepts", "central_concepts"),
    ):
        listing = report["rules"][key]
        out += [f"## {title} ({listing['count']})", ""]
        for item in listing["items"]:
            out.append(
                f"- `{item['declaration']}` ({item['kind']}, {_where(item)}): named by"
                f" {item['named_by']:,} declarations in {item['named_from_files']:,} files"
            )
            out += _statement_block(item.get("statement"))
    return "\n".join(out) + "\n"


def _patterns_md(report: dict[str, Any]) -> str:
    shapes = report["patterns"]["statement_shapes"]
    out = [
        "# Patterns: candidates for general statements",
        "",
        f"## Statements of one shape ({shapes['count']} shapes with several instances)",
        "",
    ]
    for number, family in enumerate(shapes["items"], start=1):
        members = family["members"]
        out.append(
            f"### Pattern {number}: {family['instances']} instances in {family['files']} files"
        )
        if family["shape"]:
            out += ["", f"Shape: `{family['shape'][:300]}`"]
        out += _statement_block(family["exemplar"])
        out += [f"- `{m['declaration']}` at {_where(m)}" for m in members["items"]]
        if members["cut"]:
            out.append(f"- ... {members['count'] - len(members['items'])} more")
        out.append("")
    series = report["patterns"]["numbered_series"]
    out += [f"## Numbered series ({series['count']})", ""]
    for family in series["items"]:
        members = family["members"]
        numbers = ", ".join(str(n) for n in family["numbers"])
        out.append(
            f"- `{family['series']}`: {members['count']} members (numbers {numbers})"
        )
    return "\n".join(out) + "\n"


def _claims_md(report: dict[str, Any]) -> str:
    out = ["# Claims: stated, assumed or left unproven", ""]
    for title, key in (
        ("Claim definitions (`Prop`)", "claim_definitions"),
        ("Axioms", "axioms"),
        ("`proof_wanted`", "proof_wanted"),
        ("Unproven (`sorry`, `admit`)", "unproven"),
        ("Vacuous predicates (say nothing about their subject)", "vacuous_predicates"),
    ):
        listing = report["claims"][key]
        out += [f"## {title}: {listing['count']}", ""]
        for item in listing["items"]:
            why = f": {item['vacuous']}" if "vacuous" in item else ""
            out.append(
                f"- `{item['declaration']}` ({_where(item)}), named by"
                f" {item['named_by']:,}{why}"
            )
            out += _statement_block(item.get("statement"))
    return "\n".join(out) + "\n"


def _anomalies_md(report: dict[str, Any]) -> str:
    anomalies = report["anomalies"]
    parallel = anomalies["parallel_declarations"]
    out = [
        "# Anomalies: questions the structure raises",
        "",
        f"## Public names declared more than once ({parallel['count']})",
        "",
    ]
    for group in parallel["items"]:
        what = (
            "the same statement" if group["same_statement"] else "different statements"
        )
        clash = (
            "; one module imports another (Lean refuses that)"
            if group["one_imports_another"]
            else ""
        )
        places = ", ".join(_where(item) for item in group["at"]["items"])
        out.append(f"- `{group['declaration']}`: {what} at {places}{clash}")
    generators = anomalies["generators"]
    out += [
        "",
        f"## Code that adds declarations no source spells ({generators['count']})",
        "",
    ]
    for item in generators["items"]:
        adder = item["adds_declarations_at"]
        out.append(
            f"- `{item['invocation']['kind']}` at {_where(item['invocation'])}"
            f" runs `{adder['declaration']}` ({_where(adder)})"
        )
    isolated = anomalies["isolated_results"]
    out += [
        "",
        f"## Results no declaration names ({isolated['count']}), largest first",
        "",
    ]
    out += [
        f"- `{item['declaration']}` at {_where(item)}"
        f" ({item['line_end'] - item['line_start'] + 1} lines)"
        for item in isolated["items"]
    ]
    return "\n".join(out) + "\n"


def _records(report: dict[str, Any]) -> list[dict[str, Any]]:
    """One record per reported item: title, text, where, and its evidence."""
    records = []

    def add(
        section: str,
        title: str,
        item: dict[str, Any],
        text: str | None,
        **evidence: Any,
    ):
        records.append(
            {
                "section": section,
                "title": title,
                "text": text or "",
                "declaration": item.get("declaration"),
                "file": item.get("file"),
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end"),
                "evidence": evidence,
                "producer": DISCOVERY,
            }
        )

    for key, section in (
        ("load_bearing_theorems", "rule"),
        ("central_concepts", "concept"),
    ):
        for item in report["rules"][key]["items"]:
            add(
                section,
                f"{section}: {item['declaration']}",
                item,
                item.get("statement"),
                named_by=item["named_by"],
                named_from_files=item["named_from_files"],
            )
    for family in report["patterns"]["statement_shapes"]["items"]:
        first = family["members"]["items"][0]
        add(
            "pattern",
            f"pattern of {family['instances']} statements",
            first,
            family["exemplar"],
            instances=family["instances"],
            files=family["files"],
            shape=family["shape"],
            members=[m["declaration"] for m in family["members"]["items"]],
        )
    for family in report["patterns"]["numbered_series"]["items"]:
        first = family["members"]["items"][0]
        add(
            "series",
            f"numbered series {family['series']}",
            first,
            None,
            members=family["members"]["count"],
            numbers=family["numbers"],
        )
    for key, section in (
        ("claim_definitions", "claim"),
        ("axioms", "assumption"),
        ("proof_wanted", "open"),
        ("unproven", "open"),
        ("vacuous_predicates", "vacuous"),
    ):
        for item in report["claims"][key]["items"]:
            extra = {"vacuous": item["vacuous"]} if "vacuous" in item else {}
            add(
                section,
                f"{section}: {item['declaration']}",
                item,
                item.get("statement"),
                named_by=item["named_by"],
                **extra,
            )
    for group in report["anomalies"]["parallel_declarations"]["items"]:
        add(
            "anomaly",
            f"declared in parallel: {group['declaration']}",
            group["at"]["items"][0],
            None,
            same_statement=group["same_statement"],
            one_imports_another=group["one_imports_another"],
            places=[f"{m['file']}:{m['line_start']}" for m in group["at"]["items"]],
        )
    for item in report["anomalies"]["generators"]["items"]:
        add(
            "anomaly",
            f"adds declarations: {item['invocation']['kind']}",
            item["invocation"],
            None,
            runs=item["adds_declarations_at"]["declaration"],
        )
    for item in report["headline_results"]["items"]:
        add(
            "headline",
            f"headline result: {item['declaration']}",
            item,
            item.get("statement"),
        )
    return records


def _safe(name: str) -> str:
    """Return a file name for *name*: its safe characters and a digest (no collisions)."""
    keep = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:80].strip(".") or "item"
    return f"{keep}-{hashlib.blake2b(name.encode('utf-8'), digest_size=4).hexdigest()}"


def _atomic_write(target: Path, data: bytes) -> None:
    """
    Write *data* to *target* atomically: a reader sees the old file or the new one.

    A temporary file in the same folder is written, flushed to disk and
    renamed over *target* (``os.replace``, atomic on POSIX and Windows); on
    any error the temporary file is removed and the error raised.
    """
    handle, name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(name)
        raise


def _write(folder: Path, relative: str, text: str, written: list[str]) -> None:
    target = folder / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, text.encode("utf-8"))
    written.append(relative)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def export(
    index: LeanIndex,
    folder: str | Path,
    *,
    targets: Sequence[str] = (),
    top: int = 50,
    patterns: bool = True,
) -> list[str]:
    """Write the helper documents; see :meth:`LeanIndex.export`."""
    folder = Path(folder)
    if folder.exists() and not folder.is_dir():
        raise ValueError(f"LeanIndex: {folder} exists and is not a folder.")
    report = discover(index, top=top, patterns=patterns)
    summary = index.summary(max_items=top)
    written: list[str] = []
    _write(
        folder,
        "README.md",
        _README.format(producer=DISCOVERY, folder=folder.name),
        written,
    )
    _write(folder, "summary.json", _json(summary), written)
    _write(folder, "summary.md", LeanIndex.render(summary), written)
    _write(folder, "discovery.json", _json(report), written)
    _write(folder, "rules.md", _rules_md(report), written)
    _write(folder, "patterns.md", _patterns_md(report), written)
    _write(folder, "claims.md", _claims_md(report), written)
    _write(folder, "anomalies.md", _anomalies_md(report), written)
    records = _records(report)
    names = list(
        dict.fromkeys(
            [item["declaration"] for item in report["headline_results"]["items"]]
            + list(targets)
        )
    )
    for name in names:
        dossier = index.dossier(name, max_items=top)
        stem = f"dossiers/{_safe(name)}"
        _write(folder, f"{stem}.json", _json(dossier), written)
        _write(folder, f"{stem}.md", LeanIndex.render(dossier), written)
        if "target" in dossier:
            target = dossier["target"]
            title = f"evidence on {target['declaration']}: {dossier['verdict']}"
            records.append(
                {
                    "section": "dossier",
                    "title": title,
                    "text": LeanIndex.render(dossier),
                    "declaration": target["declaration"],
                    "file": target["file"],
                    "line_start": target["line_start"],
                    "line_end": target["line_end"],
                    "evidence": {
                        "verdict": dossier["verdict"],
                        "flags_found": dossier["flags_found"],
                    },
                    "producer": DISCOVERY,
                }
            )
    _write(
        folder,
        "records.jsonl",
        "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records
        ),
        written,
    )
    _prune(folder, written)
    return written


#: The export's own record of what it wrote, so a later export into the same
#: folder removes what it no longer writes and nothing else.
_MANIFEST = ".lean-export.json"


def _previous(folder: Path) -> list[str]:
    """Return the files the last export into *folder* wrote, read defensively."""
    try:
        data = json.loads((folder / _MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    files = data.get("files") if isinstance(data, dict) else None
    return [f for f in files if isinstance(f, str)] if isinstance(files, list) else []


def _prune(folder: Path, written: list[str]) -> None:
    """
    Remove what the previous export wrote and this one did not, then record this one.

    Only paths the previous manifest lists are candidates; each must be
    relative, free of ``..``, a regular file (not a link) and inside
    *folder* once resolved. A file the export never wrote is never touched,
    and a tampered manifest cannot reach outside the folder.
    """
    keep = set(written)
    base = folder.resolve()
    for relative in _previous(folder):
        parts = relative.split("/")
        if (
            relative in keep
            or relative.startswith("/")
            or "\\" in relative
            or ".." in parts
        ):
            continue
        path = folder.joinpath(*parts)
        try:
            inside = os.path.commonpath([str(base), str(path.resolve())]) == str(base)
        except (OSError, ValueError):
            continue
        if inside and path.is_file() and not path.is_symlink():
            path.unlink()
    _atomic_write(
        folder / _MANIFEST,
        _json({"producer": DISCOVERY, "files": sorted(written)}).encode("utf-8"),
    )


def render(report: dict[str, Any]) -> str:
    """Render a discovery report as one Markdown document."""
    return "\n".join(
        [
            _rules_md(report),
            _patterns_md(report),
            _claims_md(report),
            _anomalies_md(report),
        ]
    )
