# scikitplot/lean/_parallel.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Resolve a :class:`~scikitplot.lean.LeanIndex`'s names in worker processes.

Name resolution reads the whole index (names, namespaces, what each file
sees) and writes one sorted array of edges per declaration; one
declaration's edges never depend on another's. Forked workers start from
the index as it is in memory, so nothing is sent to them; each resolves a run
of consecutive declarations and sends back its edges, which are taken in
order. The edges are therefore the same for any number of workers.

Notes
-----
**User.** Used on Linux when the index has ``workers > 1`` and enough
declarations to pay for starting the processes; elsewhere, and whenever
forking would not be safe (other threads running, inside a daemonic
process), names are resolved here, with the same result.

**Developer.** Memory: a worker copies the pages it reads, since Python's
reference counts write to the objects (the reference strings and tuples
of its declarations), and builds its own lookup cache. On a project of a
million declarations two workers added 1.7 GB to 2.2 GB; ``gc.freeze()``
before forking saved under 100 MB, so it is not used.

``fork`` is requested explicitly: Python 3.14 no longer
uses it by default on Linux, and other start methods would have to pickle
the index. It is not used on macOS, where forking a process that has used
system frameworks is unsafe.
"""

from __future__ import annotations

import multiprocessing
import sys
import threading
from array import array
from typing import Any

__all__: list[str] = []

#: Fewest declarations to resolve that make starting worker processes pay.
MIN_NODES = 20_000

#: The index the workers read: set in the parent just before forking.
_INDEX: Any = None


def usable(workers: int, nodes: int) -> bool:
    """Return whether *nodes* declarations are resolved in *workers* forked processes."""
    return (
        workers > 1
        and nodes >= MIN_NODES
        and sys.platform.startswith("linux")
        and "fork" in multiprocessing.get_all_start_methods()
        and threading.active_count() == 1
        and not multiprocessing.current_process().daemon
    )


def _resolve(bounds: tuple[int, int]) -> tuple[int, bytes, bytes]:
    """Resolve declarations ``start .. stop - 1`` (in a worker): lengths and edges."""
    start, stop = bounds
    lengths = array("q")
    edges = array("l")
    for node in range(start, stop):
        found = _INDEX._compute_edges(node)
        lengths.append(len(found))
        edges.extend(found)
    return start, lengths.tobytes(), edges.tobytes()


def resolve(index: Any, nodes: list[int], workers: int) -> dict[int, array]:
    """
    Return the edges of *nodes* (ascending), resolved in *workers* processes.

    The caller has checked :func:`usable` and warmed what every worker
    reads (each file's visibility, the short names), so that work is done
    once, before forking, instead of once per worker.
    """
    global _INDEX  # noqa: PLW0603
    runs = _runs(nodes, max(1, len(nodes) // (workers * 32)))
    found: dict[int, array] = {}
    _INDEX = index
    try:
        context = multiprocessing.get_context("fork")
        with context.Pool(workers) as pool:
            for start, raw_lengths, raw_edges in pool.imap(_resolve, runs):
                lengths = array("q")
                lengths.frombytes(raw_lengths)
                edges = array("l")
                edges.frombytes(raw_edges)
                at = 0
                for offset, length in enumerate(lengths):
                    found[start + offset] = edges[at : at + length]
                    at += length
    finally:
        _INDEX = None
    return found


def _runs(nodes: list[int], size: int) -> list[tuple[int, int]]:
    """Split ascending *nodes* into runs of consecutive numbers, at most *size* each."""
    runs: list[tuple[int, int]] = []
    start = previous = None
    for node in nodes:
        if start is not None and node == previous + 1 and node - start < size:
            previous = node
            continue
        if start is not None:
            runs.append((start, previous + 1))
        start = previous = node
    if start is not None:
        runs.append((start, previous + 1))
    return runs
