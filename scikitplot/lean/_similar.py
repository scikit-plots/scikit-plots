# lean/_similar.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Near-duplicate statements: results whose shapes share most of their token pairs.

Two theorems of one statement shape are found by discovery
(``statement_shapes``). Two that differ in a few tokens (a hypothesis more,
an operator changed, a lemma restated with other names) are not: this
module finds every pair of results whose shapes have a Jaccard similarity
of at least a threshold, over the sets of adjacent token pairs.

The search is exact: every pair at or above the threshold is found, and
none below. Sets are compared by *prefix filtering*: with the tokens of
every set put in one global order (rarest first), two sets of similarity
at least ``t`` share a token among the first ``|x| - ceil(t |x|) + 1`` of
each, so only sets sharing such a token are compared, and those of sizes
that cannot reach ``t`` are not. No hashing decides what is found.
"""

from __future__ import annotations

from array import array
from collections import Counter
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Iterable

from ._index import PRODUCER, _batch_size, _listing, _positive, _reading_pool
from ._shapes import _RESULTS, shape_tokens
from ._syntax import _Splitter

if TYPE_CHECKING:  # pragma: no cover
    from ._index import LeanIndex

__all__: list[str] = []

#: What a near-duplicate report measures, recorded on every report.
MEASURE = "jaccard of adjacent token pairs of statement shapes"

#: Marks the start and end of a shape, so its first and last tokens count.
_EDGE = "\x02"


def _pairs_of(tokens: list[str]) -> frozenset[str]:
    """Return the adjacent token pairs of a shape, its two ends marked."""
    marked = [_EDGE, *tokens, _EDGE]
    return frozenset(f"{a}\x00{b}" for a, b in zip(marked, marked[1:]))


def _file_shapes(arguments: tuple[Any, ...]) -> list[tuple[int, frozenset[str]]]:
    """
    Return ``(line_start, token pairs)`` of the results a file states.

    The work a worker process does: the file is split as the index read
    it, proofs left out, and the results starting on the *wanted* lines
    (line to declaration name) are shaped. A file that cannot be read
    gives nothing.
    """
    path, label, dialect, commands, max_depth, wanted = arguments
    splitter = _Splitter(
        file_name=label,
        meta={},
        step=None,
        cap=None,
        proofs="drop",
        dialect=dialect,
        kinds=None,
        watch=(),
        commands=commands,
        max_scope_depth=max_depth,
        flag_tokens={},
    )
    found = []

    def take(chunks: Iterable[dict[str, Any]]) -> None:
        for chunk in chunks:
            start = chunk["line_start"]
            if start in wanted and chunk.get("lean_kind") in _RESULTS:
                tokens = shape_tokens(chunk["text"], chunk["lean_kind"], wanted[start])
                if tokens:
                    found.append((start, _pairs_of(tokens)))

    try:
        with open(path, encoding="utf-8", errors="replace", newline=None) as stream:
            for number, read in enumerate(stream, start=1):
                line = read[:-1] if read.endswith("\n") else read
                if number == 1 and line.startswith("﻿"):
                    line = line[1:]
                take(splitter.feed(number, line))
        take(splitter.finish())
    except (OSError, ValueError):
        # Moved, changed past max_scope_depth, or gone since indexing.
        return []
    return found


def similar_pairs(
    sets: list[array], threshold: Fraction
) -> list[tuple[int, int, int, int]]:
    """
    Return ``(i, j, shared, union)`` for every pair of *sets* of Jaccard >= *threshold*.

    Each set is an ``array`` of distinct token numbers, sorted by the
    global order (rarest first); ``i < j``. Exact: the same pairs as
    comparing every two sets, found by prefix filtering (see the module
    docstring).

    Sets are taken smallest first. A set ``x`` probes with its first
    ``|x| - ceil(t |x|) + 1`` tokens, and is indexed under its first
    ``|y| - ceil(2t / (1 + t) |y|) + 1``: a later set is at least as large,
    so a pair of Jaccard ``t`` overlaps by at least ``2t / (1 + t)`` of the
    smaller set, and shares a token of both prefixes (positions counted in
    the global order). Each candidate is then counted once, and compared
    only when the overlap still reachable from the first shared position
    can reach the overlap the pair needs.
    """
    num, den = threshold.numerator, threshold.denominator
    order = sorted(range(len(sets)), key=lambda i: (len(sets[i]), i))
    # token -> [(set, position of the token in it)], in the order indexed
    # (sizes ascending), and where its sets still large enough start.
    inverted: dict[int, list[tuple[int, int]]] = {}
    first: dict[int, int] = {}
    found = []
    for i in order:
        x = sets[i]
        size = len(x)
        if not size:
            continue
        seen = _candidates(x, sets, num, den, inverted=inverted, first=first)
        # Integers only (ceil(a / b) is -(-a // b)): t is num / den.
        for px in range(size + (-2 * num * size // (num + den)) + 1):
            token = x[px]
            if token not in inverted:
                inverted[token] = []
                first[token] = 0
            inverted[token].append((i, px))
        if not seen:
            continue
        mine = set(x)
        for j, count in seen.items():
            if count < 0:
                continue
            shared = len(mine.intersection(sets[j]))
            union = size + len(sets[j]) - shared
            if shared * den >= num * union:
                found.append((min(i, j), max(i, j), shared, union))
    return found


def _candidates(  # noqa: PLR0913
    x: array,
    sets: list[array],
    num: int,
    den: int,
    *,
    inverted: dict[int, list[tuple[int, int]]],
    first: dict[int, int],
) -> dict[int, int]:
    """
    Return the indexed sets *x* may reach: set to overlap so far, -1 when pruned.

    *x* probes with its first ``|x| - ceil(t |x|) + 1`` tokens (t is
    ``num / den``). A set too small for *x* is too small for every later,
    larger set, so it is skipped for good (``first``). A set whose overlap
    still reachable from the shared positions cannot reach the overlap a
    pair of Jaccard t needs, ``t (|x| + |y|) / (1 + t)``, is pruned.
    """
    size = len(x)
    seen: dict[int, int] = {}
    for px in range(size + (-num * size // den) + 1):
        token = x[px]
        held = inverted.get(token)
        if held is None:
            continue
        # Sizes ascend: |y| <= |x|, and Jaccard <= |y| / |x|.
        start = first[token]
        while start < len(held) and len(sets[held[start][0]]) * den < num * size:
            start += 1
        first[token] = start
        for k in range(start, len(held)):
            j, py = held[k]
            if seen.get(j, 0) < 0:
                continue
            other = len(sets[j])
            need = -(-num * (size + other) // (num + den))
            count = seen.get(j, 0) + 1
            if count + min(size - px - 1, other - py - 1) < need:
                seen[j] = -1
            else:
                seen[j] = count
    return seen


def _ranked(shapes: list[tuple[int, array]], names: list[str]) -> list[array]:
    """
    Return each set's pair numbers renumbered in the global order, sorted.

    Rarest first, ties by the pair's text: prefixes then hold rare pairs,
    so few sets share one, and the order does not depend on reading order.
    """
    counts = Counter(i for _node, ids in shapes for i in ids)
    rank = array("i", bytes(4 * len(names)))
    for r, i in enumerate(sorted(counts, key=lambda i: (counts[i], names[i]))):
        rank[i] = r
    return [array("i", sorted(rank[i] for i in ids)) for _node, ids in shapes]


def _threshold(value: Any) -> Fraction:
    """Validate ``threshold``: a number in (0, 1], kept as an exact fraction."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError(  # noqa: TRY004  (every option error is a ValueError)
            f"LeanIndex: threshold must be a number in (0, 1]; got {value!r}."
        )
    exact = Fraction(str(value)) if isinstance(value, float) else Fraction(value)
    if not 0 < exact <= 1:
        raise ValueError(
            f"LeanIndex: threshold must be a number in (0, 1]; got {value!r}."
        )
    return exact


def near_duplicates(  # noqa: PLR0913
    index: LeanIndex,
    *,
    threshold: float = 0.8,
    layer: str | None = None,
    against: str | None = None,
    min_pairs: int = 4,
    top: int = 50,
) -> dict[str, Any]:
    """Find near-duplicate statements; see :meth:`LeanIndex.near_duplicates`."""
    exact = _threshold(threshold)
    _positive(min_pairs, "min_pairs")
    _positive(top, "top")
    index._canonical()
    mine, theirs = index._scope(layer), index._scope(against)
    wanted_files = None if mine is None or theirs is None else mine | theirs
    shapes, names, unread = _read_shapes(index, wanted_files, min_pairs)
    nodes = [node for node, _ids in shapes]
    sets = _ranked(shapes, names)
    del shapes, names

    def side(node: int, scope: frozenset[int] | None) -> bool:
        return scope is None or index._file[node] in scope

    pairs = []
    for i, j, shared, union in similar_pairs(sets, exact):
        a, b = nodes[i], nodes[j]
        if index._shape_of.get(a) == index._shape_of.get(b):
            continue  # one shape: discovery's statement_shapes lists them
        if not (
            (side(a, mine) and side(b, theirs)) or (side(b, mine) and side(a, theirs))
        ):
            continue
        pairs.append((Fraction(shared, union), min(a, b), max(a, b), shared, union))
    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    listed = pairs[:top]
    statements = _statements(index, {n for p in listed for n in p[1:3]})
    items = []
    for similarity, a, b, shared, union in listed:
        items.append(
            {
                "similarity": round(float(similarity), 4),
                "shared_pairs": shared,
                "pairs_in_either": union,
                "a": {**index._location(a), "statement": statements.get(a)},
                "b": {**index._location(b), "statement": statements.get(b)},
            }
        )
    return {
        "report": "near_duplicates",
        "producer": PRODUCER,
        "measure": MEASURE,
        "threshold": float(exact),
        "layer": layer,
        "against": against,
        "min_pairs": min_pairs,
        "statements_compared": len(sets),
        "files_not_read": _listing(unread, top),
        "pairs": {"count": len(pairs), "items": items, "cut": len(pairs) > top},
    }


def _read_shapes(
    index: LeanIndex, files: frozenset[int] | None, min_pairs: int
) -> tuple[list[tuple[int, array]], list[str], list[str]]:
    """
    Return ``(node, pair numbers)`` per result, the pairs' text, and the files not read.

    Only results with a statement shape (``_shape_of``) and at least
    *min_pairs* token pairs are kept, only from *files* (``None``: every
    file); a file with no known path is listed as not read. Files are read
    with the index's ``workers``; each pair's text is kept once, and each
    set as an array of the pairs' numbers.
    """
    wanted: dict[int, dict[int, str | None]] = {}
    for node in sorted(index._shape_of):
        f = index._file[node]
        if files is None or f in files:
            wanted.setdefault(f, {})[index._start[node]] = index._name[node]
    jobs = []
    unread = []
    for f in sorted(wanted):
        path = index._paths[f]
        if path is None:
            unread.append(index._labels[f])
            continue
        commands = index._commands_for(index._layer_of[f])
        jobs.append(
            (
                f,
                (
                    path,
                    index._labels[f],
                    index._dialect,
                    commands,
                    index._max_depth,
                    wanted[f],
                ),
            )
        )
    shapes: list[tuple[int, array]] = []
    numbers: dict[str, int] = {}
    workers = index._workers
    with _reading_pool(workers, len(jobs)) as pool:
        arguments = [job for _f, job in jobs]
        results = (
            pool.map(_file_shapes, arguments, chunksize=_batch_size(len(jobs), workers))
            if pool is not None
            else map(_file_shapes, arguments)
        )
        for (f, _job), found in zip(jobs, results):
            starts = {index._start[n]: n for n in _nodes_of(index, f)}
            for start, pairs in found:
                node = starts.get(start)
                if (
                    node is None
                    or node not in index._shape_of
                    or len(pairs) < min_pairs
                ):
                    continue
                ids = array(
                    "i", [numbers.setdefault(p, len(numbers)) for p in sorted(pairs)]
                )
                shapes.append((node, ids))
    shapes.sort(key=lambda item: item[0])
    return shapes, list(numbers), unread


def _nodes_of(index: LeanIndex, file_id: int) -> range:
    first = index._first_node[file_id]
    last = (
        index._first_node[file_id + 1]
        if file_id + 1 < len(index._first_node)
        else len(index._name)
    )
    return range(first, last)


def _statements(index: LeanIndex, nodes: set[int]) -> dict[int, str]:
    """Quote the statements of *nodes* (proofs left out), reading their files once."""
    from ._discovery import _Discovery  # noqa: PLC0415  (imports this module's parent)

    run = _Discovery(index, top=1, min_instances=1)
    run.statement_of(nodes)
    return run.statements


def render(report: dict[str, Any]) -> str:
    """Render a near-duplicate report as Markdown, every item with ``file:line``."""
    scope = ""
    if report["layer"] is not None or report["against"] is not None:
        scope = f" (layer {report['layer']!r} against {report['against']!r})"
    out = [
        "# Near-duplicate statements" + scope,
        "",
        (
            f"{report['pairs']['count']:,} pair(s) of results whose statement shapes"
            f" share at least {report['threshold']:.0%} of their token pairs, among"
            f" {report['statements_compared']:,} statements. Results of one shape"
            " are listed by discovery (statement shapes), not here."
        ),
    ]
    for pair in report["pairs"]["items"]:
        a, b = pair["a"], pair["b"]
        out += [
            "",
            f"## {pair['similarity']:.0%}: `{a['declaration']}` and `{b['declaration']}`",
            f"- {a['file']}:{a['line_start']}",
            f"- {b['file']}:{b['line_start']}",
        ]
        for item in (a, b):
            if item.get("statement"):
                out += ["", "```lean", item["statement"], "```"]
    if report["pairs"]["cut"]:
        out += ["", "(list cut)"]
    unread = report["files_not_read"]
    if unread["count"]:
        out += ["", f"{unread['count']} file(s) not read (no known path)."]
    return "\n".join(out) + "\n"
